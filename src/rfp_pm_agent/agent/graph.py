"""LangGraph 에이전트 (이슈 #22, docs/adr/0010-langgraph-agent.md).

그래프: `prepare` → `agent` ⇄ `tools` → `finalize` → 끝.

- prepare: 사용자 턴마다 LLM 호출 수·이번 턴 기록을 0으로 되돌린다(대화 기록·청크 번호·대상 사업은 이어진다)
- agent: LLM 1회. 호출 수가 `max_steps`에 닿는 호출은 `tool_choice="none"`으로 불러 툴 없이 답을 끝낸다
- tools: 마지막 AI 메시지의 툴 호출을 순서대로 실행(`agent/tools.py`). 검색 청크 번호·대상 사업을 State에 쌓는다
- finalize: 답의 청크 번호를 출처 문자열로 바꾼다(#19 `extract_citation_numbers`·`format_citation`)

대화는 Checkpointer(thread_id 단위)로 이어진다. 기본은 `InMemorySaver`(프로세스가 끝나면 사라진다).
State의 청크 기록은 `DocumentHit`을 dict로 둔다 — 체크포인트 직렬화에 Pydantic 모델을 넣지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib import resources
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from rfp_pm_agent.agent.catalog import CatalogEntry, build_catalog, render_catalog
from rfp_pm_agent.agent.tools import TOOL_SPECS, ToolDeps, run_tool
from rfp_pm_agent.clients.chat_model import make_chat_model
from rfp_pm_agent.clients.cost import CostLogger
from rfp_pm_agent.config import AgentConfig, ClientsConfig
from rfp_pm_agent.schemas.agent import AgentAnswer, ToolCallRecord
from rfp_pm_agent.schemas.answer import LLMUsage
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search.grounded_answer import (
    extract_citation_numbers,
    format_citation,
    is_refusal,
)
from rfp_pm_agent.tools.pms_tools import PmsDeps
from rfp_pm_agent.tools.search_documents import SearchDeps

# 프롬프트 파일 이름(확장자 제외)이 곧 버전이다. 내용을 바꾸면 파일을 고치지 않고 _v2를 새로 만든다
PROMPT_VERSION = "agent_v1"

# 마지막(상한) 호출에 덧붙이는 system 메시지
FINAL_STEP_NOTE = (
    "툴 호출 상한에 닿았다. 더 이상 툴을 부를 수 없다. 지금까지의 툴 결과만으로 답한다. "
    "확인하지 못한 부분은 확인하지 못했다고 적는다."
)


def _merge_citations(left: dict[int, Any], right: dict[int, Any]) -> dict[int, Any]:
    return {**left, **right}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int  # 이번 턴의 LLM 호출 수
    citations: Annotated[
        dict[int, Any], _merge_citations
    ]  # 청크 번호 → DocumentHit dict (대화 전체)
    turn_tool_calls: list[dict[str, Any]]  # 이번 턴의 ToolCallRecord dict
    turn_usage: dict[str, int]  # 이번 턴 토큰 합계
    project_id: str | None  # 대화의 대상 프로젝트
    doc_ids: list[str]  # 대화의 대상 문서
    result: dict[str, Any] | None  # 이번 턴의 AgentAnswer dict(finalize가 채운다)


def load_system_prompt(as_of_date: date, catalog: list[CatalogEntry]) -> str:
    """`agent/prompts/<PROMPT_VERSION>.md`에 기준일과 사업 목록을 넣는다."""
    path = resources.files("rfp_pm_agent.agent").joinpath("prompts", f"{PROMPT_VERSION}.md")
    template = path.read_text("utf-8")
    return template.replace("{{AS_OF_DATE}}", as_of_date.isoformat()).replace(
        "{{CATALOG}}", render_catalog(catalog)
    )


def _usage_of(message: AIMessage) -> dict[str, int]:
    usage = message.usage_metadata
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    details = usage.get("output_token_details", {})
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": int(details.get("reasoning", 0) or 0),
    }


@dataclass(frozen=True)
class AgentDeps:
    """그래프가 쓰는 것 묶음. chat_model은 `clients.chat_model.make_chat_model`(테스트는 가짜 모델)."""

    tools: ToolDeps
    chat_model: BaseChatModel
    catalog: list[CatalogEntry]
    max_steps: int


def build_graph(
    deps: AgentDeps, *, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """그래프를 만든다. checkpointer를 주지 않으면 `InMemorySaver`."""
    if deps.max_steps < 1:
        raise ValueError(f"max_steps는 1 이상이어야 한다: {deps.max_steps}")
    system_prompt = load_system_prompt(deps.tools.pms.as_of_date, deps.catalog)
    tools = [spec.openai_tool() for spec in TOOL_SPECS]
    model_with_tools = deps.chat_model.bind_tools(tools)
    model_final = deps.chat_model.bind_tools(tools, tool_choice="none")

    def prepare(state: AgentState) -> dict[str, Any]:
        return {
            "llm_calls": 0,
            "turn_tool_calls": [],
            "turn_usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
            "result": None,
        }

    def agent(state: AgentState) -> dict[str, Any]:
        calls = state.get("llm_calls", 0) + 1
        messages: list[AnyMessage] = [SystemMessage(system_prompt), *state["messages"]]
        if calls >= deps.max_steps:
            messages.append(SystemMessage(FINAL_STEP_NOTE))
            reply = model_final.invoke(messages)
        else:
            reply = model_with_tools.invoke(messages)
        if not isinstance(reply, AIMessage):
            raise TypeError(f"채팅 모델이 AIMessage가 아닌 {type(reply).__name__}를 돌려줬다")
        usage = dict(state.get("turn_usage", {}))
        for key, value in _usage_of(reply).items():
            usage[key] = usage.get(key, 0) + value
        return {"messages": [reply], "llm_calls": calls, "turn_usage": usage}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if (
            isinstance(last, AIMessage)
            and last.tool_calls
            and state.get("llm_calls", 0) < deps.max_steps
        ):
            return "tools"
        return "finalize"

    def run_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        registry = {
            int(n): DocumentHit.model_validate(h) for n, h in state.get("citations", {}).items()
        }
        project_id = state.get("project_id")
        doc_ids = list(state.get("doc_ids", []))
        records = list(state.get("turn_tool_calls", []))
        tool_messages: list[ToolMessage] = []
        new_citations: dict[int, Any] = {}
        for call in last.tool_calls:
            run = run_tool(call["name"], call["args"], deps.tools, registry)
            registry.update(run.new_hits)
            new_citations.update({n: h.model_dump(mode="json") for n, h in run.new_hits.items()})
            if run.project_id is not None:
                project_id = run.project_id
            if run.doc_ids is not None:
                doc_ids = run.doc_ids
            records.append(
                ToolCallRecord(
                    name=call["name"],
                    args=call["args"],
                    status=run.status,
                    result_count=run.result_count,
                ).model_dump(mode="json")
            )
            tool_messages.append(
                ToolMessage(content=run.content, tool_call_id=call["id"] or "", name=call["name"])
            )
        return {
            "messages": tool_messages,
            "citations": new_citations,
            "turn_tool_calls": records,
            "project_id": project_id,
            "doc_ids": doc_ids,
        }

    def finalize(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        text = last.text.strip() if isinstance(last, AIMessage) else ""
        registry = {
            int(n): DocumentHit.model_validate(h) for n, h in state.get("citations", {}).items()
        }
        refused = is_refusal(text)
        citations: list[str] = []
        citation_map: dict[int, str] = {}
        cited_chunk_ids: list[str] = []
        invalid: list[int] = []
        if not refused:
            for number in extract_citation_numbers(text):
                hit = registry.get(number)
                if hit is None:
                    invalid.append(number)
                    continue
                citation = format_citation(hit)
                citation_map[number] = citation
                if citation not in citations:
                    citations.append(citation)
                if hit.chunk_id not in cited_chunk_ids:
                    cited_chunk_ids.append(hit.chunk_id)
        calls = state.get("llm_calls", 0)
        answer = AgentAnswer(
            answer=text,
            refused=refused,
            citations=citations,
            citation_map=citation_map,
            cited_chunk_ids=cited_chunk_ids,
            invalid_citations=invalid,
            tool_calls=[ToolCallRecord.model_validate(r) for r in state.get("turn_tool_calls", [])],
            llm_calls=calls,
            reached_max_steps=calls >= deps.max_steps,
            project_id=state.get("project_id"),
            doc_ids=list(state.get("doc_ids", [])),
            usage=LLMUsage(**state.get("turn_usage", {})),
        )
        return {"result": answer.model_dump(mode="json")}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "finalize"])
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class Agent:
    """질문 하나 → `AgentAnswer`. 같은 thread_id로 부르면 대화(대상 사업·청크 번호 포함)가 이어진다."""

    def __init__(
        self, deps: AgentDeps, *, checkpointer: BaseCheckpointSaver[Any] | None = None
    ) -> None:
        self.deps = deps
        self.graph = build_graph(deps, checkpointer=checkpointer)

    def ask(self, question: str, *, thread_id: str) -> AgentAnswer:
        # LangGraph 단계(노드 실행) 수 안전망: prepare 1 + agent·tools 왕복 최대 2·max_steps − 1 + finalize 1에 여유 2.
        # 실제 상한은 agent 노드가 지키는 max_steps(LLM 호출 수)다
        recursion_limit = 2 * self.deps.max_steps + 4
        state = self.graph.invoke(
            {"messages": [HumanMessage(question)]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit},
        )
        return AgentAnswer.model_validate(state["result"])


def agent_deps_from_env(cost_logger: CostLogger, *, model: str | None = None) -> AgentDeps:
    """환경변수 설정으로 실제 의존성을 만든다. 사업 목록을 만들려고 OpenSearch(집계 1회)와 PMS DB를 읽는다."""
    clients = ClientsConfig.from_env()
    tool_deps = ToolDeps(search=SearchDeps.from_env(), pms=PmsDeps.from_env())
    return AgentDeps(
        tools=tool_deps,
        chat_model=make_chat_model(clients, cost_logger, model=model),
        catalog=build_catalog(tool_deps.search.search_client, tool_deps.pms.reader),
        max_steps=AgentConfig.from_env().max_steps,
    )


def render_answer(answer: AgentAnswer) -> str:
    """사용자에게 보일 문자열: 답 + 출처 목록(번호 → 출처). 출처가 없으면 답만."""
    if not answer.citation_map:
        return answer.answer
    lines = [f"[{n}] {c}" for n, c in sorted(answer.citation_map.items())]
    return answer.answer + "\n\n출처\n" + "\n".join(lines)
