"""LangGraph 에이전트 그래프 (이슈 #22, ADR-0010). 가짜 채팅 모델·가짜 PMS 리더 — 네트워크 없음.

검색은 `agent.tools.search_documents`를 정해 둔 결과로 바꿔 끼운다(검색 자체는 test_search_documents.py가 본다).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rfp_pm_agent.agent import tools as agent_tools
from rfp_pm_agent.agent.catalog import CatalogEntry
from rfp_pm_agent.agent.graph import (
    FINAL_STEP_NOTE,
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    Agent,
    AgentDeps,
    load_system_prompt,
    render_answer,
)
from rfp_pm_agent.agent.tools import ToolDeps
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.grounded_answer import REFUSAL_TEXT
from rfp_pm_agent.tools.pms_tools import PmsDeps
from rfp_pm_agent.tools.search_documents import SearchDeps
from tests.fakes.fake_chat_model import ScriptedChatModel, answer, tool_call
from tests.fakes.fake_embedding import FakeEmbeddingClient
from tests.fakes.fake_pms_reader import FakePmsReader
from tests.fakes.fake_reranker import FakeRerankerClient
from tests.fakes.fake_search_client import RecordingSearchClient
from tests.fakes.pms_sample import AS_OF, build_sample

CATALOG = [
    CatalogEntry(doc_id="docA", bid_title="A 사업", project_id="PRJ-01"),
    CatalogEntry(doc_id="docB", bid_title="B 사업", project_id="PRJ-02"),
    CatalogEntry(doc_id="docC", bid_title="C 사업(PMS 없음)", project_id=None),
]


def _hit(chunk_id: str, rank: int = 1, **overrides: Any) -> DocumentHit:
    fields: dict[str, Any] = {
        "rank": rank,
        "chunk_id": chunk_id,
        "doc_id": "docA",
        "bid_title": "A 사업",
        "format": "pdf",
        "requirement_id": "SFR-001",
        "page": 14,
        "printed_page": 23,
        "block_type": None,
        "text": f"{chunk_id} 본문",
        "score": 0.9,
    }
    fields.update(overrides)
    return DocumentHit(**fields)


@pytest.fixture(scope="module")
def dataset() -> Any:
    return build_sample()[0]


def _tool_deps(dataset: Any) -> ToolDeps:
    return ToolDeps(
        search=SearchDeps(
            search_client=RecordingSearchClient(),
            embedder=FakeEmbeddingClient(),
            reranker=FakeRerankerClient(),
        ),
        pms=PmsDeps(reader=FakePmsReader(dataset), as_of_date=AS_OF),
    )


def _agent(
    dataset: Any, responses: list[Any], max_steps: int = 6
) -> tuple[Agent, ScriptedChatModel]:
    model = ScriptedChatModel(responses=responses)
    deps = AgentDeps(
        tools=_tool_deps(dataset), chat_model=model, catalog=CATALOG, max_steps=max_steps
    )
    return Agent(deps), model


def _patch_search(
    monkeypatch: pytest.MonkeyPatch, results: list[list[DocumentHit] | Exception]
) -> list[dict[str, Any]]:
    """호출마다 results를 차례로 돌려준다(예외면 던진다). 받은 인자 기록을 돌려준다."""
    calls: list[dict[str, Any]] = []

    def fake(query: str, top_k: int, doc_ids: list[str] | None = None, *, deps: Any) -> Any:
        calls.append({"query": query, "top_k": top_k, "doc_ids": doc_ids})
        result = results[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(agent_tools, "search_documents", fake)
    return calls


def _tool_messages(model: ScriptedChatModel, call_index: int) -> list[ToolMessage]:
    return [m for m in model.calls[call_index]["messages"] if isinstance(m, ToolMessage)]


# --- 경로 ---


def test_document_question_cites_chunk_numbers_as_sources(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    search_calls = _patch_search(
        monkeypatch, [[_hit("c1"), _hit("c2", 2, requirement_id=None, printed_page=None)]]
    )
    agent, model = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "사업 금액", "doc_ids": ["docA"]}),
            answer("사업 금액은 1억 원이다 [1][2]."),
        ],
    )

    result = agent.ask("A 사업 금액은?", thread_id="t1")

    assert search_calls == [{"query": "사업 금액", "top_k": 8, "doc_ids": ["docA"]}]
    assert result.answer == "사업 금액은 1억 원이다 [1][2]."
    assert result.citations == ["[A 사업 p.23 · SFR-001]", "[A 사업 PDF 15쪽]"]
    assert result.citation_map == {1: "[A 사업 p.23 · SFR-001]", 2: "[A 사업 PDF 15쪽]"}
    assert result.cited_chunk_ids == ["c1", "c2"]
    assert result.invalid_citations == []
    assert not result.refused
    assert result.llm_calls == 2
    assert not result.reached_max_steps
    assert [(c.name, c.status, c.result_count) for c in result.tool_calls] == [
        ("search_documents", "ok", 2)
    ]
    assert result.doc_ids == ["docA"]
    # 검색만 불러도 사업 목록으로 대상 프로젝트를 정한다
    assert result.project_id == "PRJ-01"
    # 사용량은 이번 턴 AI 메시지의 합(툴 호출 메시지는 사용량이 없다)
    assert result.usage.input_tokens == 100
    assert result.usage.reasoning_tokens == 3
    # 툴 결과가 v2 구분자 형식으로 다음 LLM 호출에 들어간다
    [tool_message] = _tool_messages(model, 1)
    assert str(tool_message.content).startswith('<tool_result name="search_documents" status="ok">')
    assert '<chunk id="1" doc="A 사업">\n[1] c1 본문\n</chunk>' in str(tool_message.content)


def test_system_prompt_is_first_and_tools_are_bound(dataset: Any) -> None:
    agent, model = _agent(dataset, [answer(REFUSAL_TEXT)])

    agent.ask("대한민국의 수도는?", thread_id="t1")

    [call] = model.calls
    system = call["messages"][0]
    assert isinstance(system, SystemMessage)
    assert "기준일: 2026-09-24" in system.content
    assert "| C 사업(PMS 없음) | docC | 없음(PMS 데이터 없음) |" in system.content
    names = [t["function"]["name"] for t in call["kwargs"]["tools"]]
    assert names == ["search_documents", "get_delayed_tasks", "get_open_risks", "get_deliverables"]
    assert call["kwargs"]["tool_choice"] is None


def test_refusal_answer_has_no_citations(dataset: Any) -> None:
    agent, _ = _agent(dataset, [answer("문서에서 찾을 수 없음.")])

    result = agent.ask("김치찌개 끓이는 법", thread_id="t1")

    assert result.refused
    assert result.citations == [] and result.citation_map == {}
    assert result.tool_calls == []
    assert result.llm_calls == 1


def test_pms_question_sets_project_and_splits_stage_delays(dataset: Any) -> None:
    agent, model = _agent(
        dataset,
        [tool_call("get_delayed_tasks", {"project_id": "PRJ-01"}), answer("지연 업무 …")],
    )

    result = agent.ask("A 사업 지연 업무", thread_id="t1")

    assert result.project_id == "PRJ-01"
    assert result.doc_ids == ["docA"]
    [record] = result.tool_calls
    assert record.status == "ok" and record.args == {"project_id": "PRJ-01"}
    [tool_message] = _tool_messages(model, 1)
    content = str(tool_message.content)
    assert '"stage_delays"' in content and '"task_delays"' in content
    assert '"items"' not in content


def test_missing_project_is_recorded_as_project_required(dataset: Any) -> None:
    agent, model = _agent(
        dataset,
        [tool_call("get_open_risks", {}), answer("어느 사업인지 알려 주세요: A 사업, B 사업")],
    )

    result = agent.ask("미해결 Risk 알려줘", thread_id="t1")

    assert [c.status for c in result.tool_calls] == ["project_required"]
    assert result.project_id is None and result.doc_ids == []
    [tool_message] = _tool_messages(model, 1)
    assert "available_projects" in str(tool_message.content)


# --- 반복 상한 ---


def test_max_steps_limits_llm_calls_and_last_call_has_no_tools(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_search(monkeypatch, [[_hit("c1")], [_hit("c2")]])
    agent, model = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}, "call-1"),
            tool_call("search_documents", {"query": "b", "doc_ids": ["docA"]}, "call-2"),
            answer("찾은 것은 c2다 [2]."),
        ],
        max_steps=3,
    )

    result = agent.ask("질문", thread_id="t1")

    assert result.llm_calls == 3
    assert result.reached_max_steps
    assert [c["kwargs"]["tool_choice"] for c in model.calls] == [None, None, "none"]
    last_messages = model.calls[2]["messages"]
    assert isinstance(last_messages[-1], SystemMessage)
    assert last_messages[-1].content == FINAL_STEP_NOTE
    assert result.citation_map == {2: "[A 사업 p.23 · SFR-001]"}


def test_tool_calls_on_last_step_are_not_executed(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    search_calls = _patch_search(monkeypatch, [[_hit("c1")]])
    agent, _ = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}, "call-1"),
            # 모델이 tool_choice="none"을 어기고 툴을 불러도 실행하지 않고 끝낸다
            tool_call("search_documents", {"query": "b", "doc_ids": ["docA"]}, "call-2"),
        ],
        max_steps=2,
    )

    result = agent.ask("질문", thread_id="t1")

    assert len(search_calls) == 1
    assert result.llm_calls == 2 and result.reached_max_steps
    assert len(result.tool_calls) == 1


def test_max_steps_one_answers_without_tools(dataset: Any) -> None:
    agent, model = _agent(dataset, [answer("답")], max_steps=1)

    result = agent.ask("질문", thread_id="t1")

    assert result.llm_calls == 1 and result.reached_max_steps
    assert model.calls[0]["kwargs"]["tool_choice"] == "none"


def test_max_steps_below_one_is_rejected(dataset: Any) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        _agent(dataset, [], max_steps=0)


# --- 출처 ---


def test_citation_number_not_given_by_tools_is_invalid(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_search(monkeypatch, [[_hit("c1")]])
    agent, _ = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}),
            answer("답 [1][7]"),
        ],
    )

    result = agent.ask("질문", thread_id="t1")

    assert result.citation_map == {1: "[A 사업 p.23 · SFR-001]"}
    assert result.invalid_citations == [7]


def test_render_answer_appends_sources(dataset: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(monkeypatch, [[_hit("c1")]])
    agent, _ = _agent(
        dataset,
        [tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}), answer("답 [1]")],
    )

    result = agent.ask("질문", thread_id="t1")

    assert render_answer(result) == "답 [1]\n\n출처\n[1] [A 사업 p.23 · SFR-001]"


# --- 대화 이어가기(Checkpointer) ---


def test_thread_keeps_history_project_and_chunk_numbers(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_search(monkeypatch, [[_hit("c1"), _hit("c2", 2)], [_hit("c2"), _hit("c3", 2)]])
    agent, model = _agent(
        dataset,
        [
            tool_call("get_delayed_tasks", {"project_id": "PRJ-01"}, "call-1"),
            tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}, "call-2"),
            answer("첫 답 [1]"),
            tool_call("search_documents", {"query": "b", "doc_ids": ["docA"]}, "call-3"),
            answer("둘째 답 [2][3]"),
        ],
    )

    first = agent.ask("A 사업 지연 업무와 요구사항", thread_id="t1")
    second = agent.ask("그 다음 요구사항은?", thread_id="t1")

    assert first.llm_calls == 3
    # 턴마다 LLM 호출 수·툴 기록은 새로 센다
    assert second.llm_calls == 2
    assert [c.name for c in second.tool_calls] == ["search_documents"]
    # 대상 사업은 이어진다(둘째 턴에는 PMS 툴을 부르지 않았다)
    assert second.project_id == "PRJ-01"
    # 같은 청크(c2)는 같은 번호, 새 청크(c3)는 다음 번호
    [tool_message] = _tool_messages(model, 4)[-1:]
    assert '<chunk id="2"' in str(tool_message.content)
    assert '<chunk id="3"' in str(tool_message.content)
    assert second.cited_chunk_ids == ["c2", "c3"]
    # 둘째 턴 LLM은 첫 턴 질문을 본다
    humans = [m.content for m in model.calls[3]["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["A 사업 지연 업무와 요구사항", "그 다음 요구사항은?"]


def test_other_thread_starts_fresh(dataset: Any) -> None:
    agent, model = _agent(dataset, [answer("하나"), answer("둘")])

    agent.ask("첫 질문", thread_id="t1")
    agent.ask("다른 질문", thread_id="t2")

    humans = [m.content for m in model.calls[1]["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["다른 질문"]


# --- 툴 오류는 LLM에 돌려준다 ---


def test_tool_errors_are_returned_to_llm(dataset: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(monkeypatch, [SearchUnavailableError("opensearch", "connection refused")])
    agent, model = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}, "call-1"),
            tool_call("search_documents", {"query": "a"}, "call-2"),
            tool_call("unknown_tool", {}, "call-3"),
            answer("검색 서버에 연결할 수 없습니다."),
        ],
    )

    result = agent.ask("질문", thread_id="t1")

    assert [c.status for c in result.tool_calls] == ["unavailable", "invalid_args", "invalid_args"]
    contents = [str(m.content) for m in _tool_messages(model, 3)]
    assert "인자를 바꿔도 해결되지 않는다" in contents[0]
    assert "doc_ids" in contents[1] and "인자를 고쳐 다시 부른다" in contents[1]
    assert "모르는 툴" in contents[2]


@pytest.mark.parametrize("version", PROMPT_VERSIONS)
def test_load_system_prompt_fills_placeholders(version: str) -> None:
    prompt = load_system_prompt(AS_OF, CATALOG, version)

    assert "{{" not in prompt
    assert "| A 사업 | docA | PRJ-01 |" in prompt


def test_load_system_prompt_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="agent_v9"):
        load_system_prompt(AS_OF, CATALOG, "agent_v9")


# --- agent_v2: 답 없음 거절 규칙 (#94) ---


def test_agent_v2_rules_2_and_7_split_partial_and_full_refusal() -> None:
    v1 = load_system_prompt(AS_OF, CATALOG, "agent_v1")
    v2 = load_system_prompt(AS_OF, CATALOG, "agent_v2")

    # 기본값은 agent_v1로 되돌렸다 — v2는 거절이 v1과 같아(#94, ADR-0013) 재현용으로만 남긴다
    assert PROMPT_VERSION == "agent_v1"
    # 규칙 2: 일부만 찾았을 때로 좁힌다 — 찾은 사실은 번호, 못 찾은 사실은 번호 없이
    assert "그중 일부만 청크에 직접 적혀 있으면" in v2
    assert '못 찾은 사실은 번호 없이 "문서에서 찾지 못했다"고 적는다' in v2
    assert "사업명·주제만 같고 묻는 사실이 없으면 그 부분은 문서에서 찾지 못했다고 적는다" not in v2
    # 규칙 7: 하나도 못 찾으면 주제가 같은 청크가 있어도 거절 문구만, 번호 없이
    assert "질문이 묻는 사실을 하나도 찾지 못하면" in v2
    assert "사업명·주제가 같은 청크가 있더라도" in v2
    assert f'정확히 "{REFUSAL_TEXT}"만 쓴다' in v2
    assert "이때 청크 번호를 붙이지 않는다" in v2
    # v1은 #22 결과 재현용으로 그대로 남는다
    assert "사업명·주제만 같고 묻는 사실이 없으면 그 부분은 문서에서 찾지 못했다고 적는다" in v1
    assert "하나도 찾지 못하면" not in v1


@pytest.mark.parametrize("version", PROMPT_VERSIONS)
def test_prompt_version_selects_system_prompt(dataset: Any, version: str) -> None:
    model = ScriptedChatModel(responses=[answer(REFUSAL_TEXT)])
    deps = AgentDeps(
        tools=_tool_deps(dataset),
        chat_model=model,
        catalog=CATALOG,
        max_steps=6,
        prompt_version=version,
    )

    Agent(deps).ask("김치찌개 끓이는 법", thread_id="t1")

    system = model.calls[0]["messages"][0]
    assert system.content == load_system_prompt(AS_OF, CATALOG, version)


def test_refusal_after_search_has_no_sources(dataset: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """주제가 같은 청크를 받고도 묻는 사실이 없어 거절하면 출처가 비어야 한다(#22 n1·n2 유형)."""
    _patch_search(monkeypatch, [[_hit("c1"), _hit("c2", 2)]])
    agent, _ = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "GPU 서버 대수", "doc_ids": ["docA"]}),
            answer(REFUSAL_TEXT),
        ],
    )

    result = agent.ask("A 사업 GPU 서버는 몇 대야?", thread_id="t1")

    assert result.refused
    assert result.citations == [] and result.citation_map == {} and result.cited_chunk_ids == []
    assert result.invalid_citations == []
    assert [(c.name, c.status) for c in result.tool_calls] == [("search_documents", "ok")]


def test_refusal_text_with_chunk_numbers_is_not_a_refusal(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """거절 문구에 번호를 붙이면(#22 n1 `[1]–[8]`) 거절로 세지 않고 출처가 남는다 — #94 A 기준의 실패."""
    _patch_search(monkeypatch, [[_hit("c1")]])
    agent, _ = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "GPU 서버 대수", "doc_ids": ["docA"]}),
            answer(f"{REFUSAL_TEXT} [1]"),
        ],
    )

    result = agent.ask("A 사업 GPU 서버는 몇 대야?", thread_id="t1")

    assert not result.refused
    assert result.citation_map == {1: "[A 사업 p.23 · SFR-001]"}


# --- 짝 없는 툴 호출이 대화에 남지 않는다 (PR 전 reviewer 권장 1) ---


def _unpaired_tool_calls(messages: list[Any]) -> list[str]:
    """모델이 받은 메시지에서 ToolMessage 짝이 없는 툴 호출 id."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    ids = [
        call["id"] or ""
        for m in messages
        if isinstance(m, AIMessage)
        for call in [*m.tool_calls, *m.invalid_tool_calls]
    ]
    return [i for i in ids if i not in answered]


def test_unexecuted_last_step_tool_call_is_closed_for_next_turn(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_search(monkeypatch, [[_hit("c1")]])
    agent, model = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}, "call-1"),
            tool_call("search_documents", {"query": "b", "doc_ids": ["docA"]}, "call-2"),
            answer("둘째 답"),
        ],
        max_steps=2,
    )

    agent.ask("첫 질문", thread_id="t1")
    agent.ask("둘째 질문", thread_id="t1")

    assert _unpaired_tool_calls(model.calls[2]["messages"]) == []


def test_invalid_tool_call_is_returned_as_invalid_args(dataset: Any) -> None:
    broken = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "get_open_risks",
                "args": '{"project_id": ',
                "id": "call-bad",
                "error": "JSON 해석 실패",
                "type": "invalid_tool_call",
            }
        ],
    )
    agent, model = _agent(dataset, [broken, answer("어느 사업인가요?")])

    result = agent.ask("Risk 알려줘", thread_id="t1")

    assert [(c.name, c.status) for c in result.tool_calls] == [("get_open_risks", "invalid_args")]
    assert result.llm_calls == 2
    assert _unpaired_tool_calls(model.calls[1]["messages"]) == []
    [tool_message] = _tool_messages(model, 1)
    assert "인자를 고쳐 다시 부른다" in str(tool_message.content)


def test_tool_bug_is_raised_and_thread_stays_usable(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_search(monkeypatch, [KeyError("bug")])
    agent, model = _agent(
        dataset,
        [
            tool_call("search_documents", {"query": "a", "doc_ids": ["docA"]}, "call-1"),
            answer("둘째 답"),
        ],
    )

    with pytest.raises(KeyError):
        agent.ask("첫 질문", thread_id="t1")
    result = agent.ask("둘째 질문", thread_id="t1")

    assert result.answer == "둘째 답"
    assert _unpaired_tool_calls(model.calls[1]["messages"]) == []


def test_search_only_switch_updates_project_from_catalog(
    dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_search(monkeypatch, [[_hit("c1", doc_id="docB")], [_hit("c2", doc_id="docC")]])
    agent, _ = _agent(
        dataset,
        [
            tool_call("get_delayed_tasks", {"project_id": "PRJ-01"}, "call-1"),
            tool_call("search_documents", {"query": "a", "doc_ids": ["docB"]}, "call-2"),
            answer("B 사업 답"),
            tool_call("search_documents", {"query": "b", "doc_ids": ["docC"]}, "call-3"),
            answer("C 사업 답"),
        ],
    )

    first = agent.ask("질문", thread_id="t1")
    second = agent.ask("C 사업은?", thread_id="t1")

    # 검색이 다른 사업으로 옮겨 가면 project_id도 사업 목록으로 다시 정한다(PMS 없는 사업은 None)
    assert (first.project_id, first.doc_ids) == ("PRJ-02", ["docB"])
    assert (second.project_id, second.doc_ids) == (None, ["docC"])
