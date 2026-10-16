"""에이전트 툴 래퍼 (이슈 #22, docs/adr/0010-langgraph-agent.md 결정 2·4·5).

`tools/`의 함수(`search_documents`·PMS 툴 3개)를 LLM 툴로 감싼다. 툴 함수 자체는 바꾸지 않는다(MCP #24와 공유).

- 툴 설명: 원래 함수의 docstring을 그대로 쓰되, LLM에게 숨기는 인자(`deps`·`as_of`)의 설명을 지우고 에이전트에서 달라지는
  인자(`doc_ids` 필수, `top_k`·`limit` 상한)의 설명만 바꾼다. 바꿀 문장이 docstring에 없으면 import 시점에 멈춘다
  (docstring이 바뀌었는데 래퍼가 모르고 지나가지 않게)
- 인자 검증: Pydantic 모델(`SearchArgs` 등). 검증 실패·ValueError는 "인자 오류" 툴 결과로 LLM에 돌려준다
- 결과: `<tool_result name="…">` 안에 넣는다. 검색 결과는 #86 v2 형식(`<chunk id doc>`)이고 번호는 대화 전체에서 이어 매긴다.
  PMS 결과는 JSON이다. 어느 쪽이든 본문의 `<chunk`·`<tool_result` 태그는 `&lt;`로 바꿔 구역을 닫거나 열지 못하게 한다
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rfp_pm_agent.pms.errors import PmsUnavailableError
from rfp_pm_agent.pms.rules import RiskLevel
from rfp_pm_agent.schemas.agent import ToolCallStatus
from rfp_pm_agent.schemas.pms_tools import DeliverableFilter, ToolResultBase
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.grounded_answer import escape_chunk_tags, escape_doc_attr
from rfp_pm_agent.tools import pms_tools
from rfp_pm_agent.tools.pms_tools import PmsDeps
from rfp_pm_agent.tools.search_documents import SearchDeps, search_documents

AGENT_DEFAULT_TOP_K = 8
AGENT_MAX_TOP_K = 10
AGENT_MAX_LIMIT = 20
# 청크 하나의 본문 상한(글자). 색인 청크 3,499건의 99%가 2,092자 이하, 최대 6,795자(2026-10-16)
MAX_CHUNK_CHARS = 3000
# PMS 툴 결과(JSON) 하나의 상한(글자). 넘으면 목록 뒤에서부터 덜어 내고 omitted_by_size에 센다
MAX_PMS_RESULT_CHARS = 12000
# Risk 설명·대응 방안 하나의 상한(글자)
MAX_TEXT_FIELD_CHARS = 300

_TOOL_RESULT_TAG_RE = re.compile(r"<(\s*/?\s*tool_result)", re.IGNORECASE)


def escape_tags(text: str) -> str:
    """`<chunk`·`</chunk`·`<tool_result`·`</tool_result`(대소문자·공백 포함)를 `&lt;`로 바꾼다. 그 밖은 그대로."""
    return escape_chunk_tags(_TOOL_RESULT_TAG_RE.sub(r"&lt;\1", text))


# --- 툴 설명(docstring 손보기) ---


def _bullet_span(lines: list[str], name: str) -> tuple[int, int]:
    """`- name:`으로 시작하는 항목의 [시작, 끝) 줄 번호. 이어지는 줄은 두 칸 들여 쓴 줄이다."""
    start = next((i for i, line in enumerate(lines) if line.startswith(f"- {name}:")), None)
    if start is None:
        raise RuntimeError(f"docstring에 '- {name}:' 항목이 없다 — 래퍼 설명을 다시 맞춰야 한다")
    end = start + 1
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return start, end


def tool_description(
    func: Callable[..., Any],
    *,
    drop: tuple[str, ...] = (),
    replace: dict[str, str] | None = None,
    drop_lines: tuple[str, ...] = (),
) -> str:
    """func의 docstring에서 인자 항목을 지우거나(drop) 바꾸고(replace), drop_lines로 시작하는 줄을 지운다."""
    lines = inspect.cleandoc(func.__doc__ or "").splitlines()
    for name in drop:
        start, end = _bullet_span(lines, name)
        del lines[start:end]
    for name, text in (replace or {}).items():
        start, end = _bullet_span(lines, name)
        lines[start:end] = [f"- {name}: {text}"]
    for prefix in drop_lines:
        index = next((i for i, line in enumerate(lines) if line.startswith(prefix)), None)
        if index is None:
            raise RuntimeError(f"docstring에 '{prefix}'로 시작하는 줄이 없다")
        del lines[index]
    return "\n".join(lines).strip()


_LIMIT_TEXT = (
    f"돌려받을 최대 건수. 1~{AGENT_MAX_LIMIT}(기본 {AGENT_MAX_LIMIT}). "
    "total에 전체 건수, truncated에 잘림 여부가 있다"
)

SEARCH_DESCRIPTION = (
    tool_description(
        search_documents,
        replace={
            "top_k": f"돌려받을 결과 수. 1 이상 {AGENT_MAX_TOP_K} 이하의 정수(기본 {AGENT_DEFAULT_TOP_K})",
            "doc_ids": "검색할 사업의 doc_id 목록(필수). system 메시지 사업 목록의 doc_id나 PMS 툴 결과의 doc_ids를 "
            "그대로 넣는다. 전체 문서 검색은 하지 않는다. 인덱스에 없는 doc_id면 오류",
        },
        drop_lines=("deps는",),
    )
    + '\n\n결과 형식: 청크마다 <chunk id="번호" doc="문서명">…</chunk>. 답에서 근거로 쓴 청크는 이 번호를 [번호]로 붙인다. '
    "번호는 대화 전체에서 이어지고, 같은 청크는 같은 번호다."
)

_PMS_REPLACE = {"limit": _LIMIT_TEXT}
DELAYED_DESCRIPTION = tool_description(
    pms_tools.get_delayed_tasks, drop=("as_of",), replace=_PMS_REPLACE
) + (
    "\n\n결과 형식: items 대신 stage_delays(단계, level 1)와 task_delays(그 아래 업무, level 2 이상)로 나눠 준다. "
    "두 목록의 건수 합이 total(잘렸으면 앞에서부터 limit개)이다."
)
RISKS_DESCRIPTION = tool_description(
    pms_tools.get_open_risks, drop=("as_of",), replace=_PMS_REPLACE
)
DELIVERABLES_DESCRIPTION = tool_description(
    pms_tools.get_deliverables, drop=("as_of",), replace=_PMS_REPLACE
)


# --- 인자 모델(LLM에게 보이는 스키마) ---


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Args):
    query: str = Field(description="찾을 내용을 담은 질문이나 키워드(한국어)")
    doc_ids: list[str] = Field(min_length=1, description="검색할 사업의 doc_id 목록(필수)")
    top_k: int = Field(default=AGENT_DEFAULT_TOP_K, ge=1, le=AGENT_MAX_TOP_K)


class DelayedTasksArgs(_Args):
    project_id: str | None = Field(default=None, description='프로젝트 ID(예: "PRJ-03")')
    scope: Literal["all", "this_week"] = "all"
    limit: int = Field(default=AGENT_MAX_LIMIT, ge=1, le=AGENT_MAX_LIMIT)


class OpenRisksArgs(_Args):
    project_id: str | None = Field(default=None, description='프로젝트 ID(예: "PRJ-03")')
    min_level: RiskLevel = "low"
    limit: int = Field(default=AGENT_MAX_LIMIT, ge=1, le=AGENT_MAX_LIMIT)


class DeliverablesArgs(_Args):
    project_id: str | None = Field(default=None, description='프로젝트 ID(예: "PRJ-03")')
    status_filter: DeliverableFilter = "all"
    wbs_id: str | None = None
    requirement_id: str | None = None
    limit: int = Field(default=AGENT_MAX_LIMIT, ge=1, le=AGENT_MAX_LIMIT)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[_Args]

    def openai_tool(self) -> dict[str, Any]:
        """`bind_tools`에 넘기는 OpenAI 함수 툴 정의."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("search_documents", SEARCH_DESCRIPTION, SearchArgs),
    ToolSpec("get_delayed_tasks", DELAYED_DESCRIPTION, DelayedTasksArgs),
    ToolSpec("get_open_risks", RISKS_DESCRIPTION, OpenRisksArgs),
    ToolSpec("get_deliverables", DELIVERABLES_DESCRIPTION, DeliverablesArgs),
)
_SPEC_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


# --- 실행 ---


@dataclass(frozen=True)
class ToolDeps:
    search: SearchDeps
    pms: PmsDeps


@dataclass
class ToolRun:
    """툴 호출 1건의 결과. content는 LLM에 돌려줄 문자열(`<tool_result>`로 감싼 것)."""

    content: str
    status: ToolCallStatus
    result_count: int = 0
    # 이번 호출로 새로 번호를 받은 청크(번호 → DocumentHit)
    new_hits: dict[int, DocumentHit] = field(default_factory=dict)
    # PMS 툴이 ok면 그 프로젝트(대화의 대상 사업이 된다). 검색이 ok면 project_id는 None이고 doc_ids만
    project_id: str | None = None
    doc_ids: list[str] | None = None


def _wrap(name: str, body: str, status: ToolCallStatus) -> str:
    return f'<tool_result name="{name}" status="{status}">\n{body}\n</tool_result>'


def _error(name: str, status: ToolCallStatus, message: str) -> ToolRun:
    if status == "invalid_args":
        hint = "인자를 고쳐 다시 부른다"
    else:
        hint = "인자를 바꿔도 해결되지 않는다. 사용자에게 서버에 연결할 수 없다고 알린다"
    return ToolRun(_wrap(name, escape_tags(f"오류: {message}. {hint}"), status), status)


def _validation_message(error: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()]
    return "인자 검증 실패 — " + "; ".join(parts)


def _number_hits(hits: list[DocumentHit], registry: dict[int, DocumentHit]) -> list[int]:
    """hits마다 번호를 정한다. 이 대화에서 이미 번호를 받은 청크(chunk_id)는 같은 번호, 새 청크는 다음 번호."""
    known = {hit.chunk_id: number for number, hit in registry.items()}
    next_number = max(registry, default=0) + 1
    numbers: list[int] = []
    for hit in hits:
        if hit.chunk_id in known:
            numbers.append(known[hit.chunk_id])
        else:
            known[hit.chunk_id] = next_number
            numbers.append(next_number)
            next_number += 1
    return numbers


def _format_chunk(number: int, hit: DocumentHit) -> str:
    text = hit.text
    if len(text) > MAX_CHUNK_CHARS:
        text = text[:MAX_CHUNK_CHARS] + f" …(이하 {len(hit.text) - MAX_CHUNK_CHARS}자 생략)"
    return f'<chunk id="{number}" doc="{escape_doc_attr(hit.bid_title)}">\n[{number}] {escape_tags(text)}\n</chunk>'


def _run_search(args: SearchArgs, deps: ToolDeps, registry: dict[int, DocumentHit]) -> ToolRun:
    hits = search_documents(args.query, args.top_k, doc_ids=args.doc_ids, deps=deps.search)
    if not hits:
        return ToolRun(
            _wrap("search_documents", "검색 결과 0건", "ok"), "ok", 0, doc_ids=args.doc_ids
        )
    numbers = _number_hits(hits, registry)
    body = f"검색 결과 {len(hits)}건\n\n" + "\n\n".join(
        _format_chunk(n, h) for n, h in zip(numbers, hits, strict=True)
    )
    new_hits = {n: h for n, h in zip(numbers, hits, strict=True) if n not in registry}
    return ToolRun(
        _wrap("search_documents", body, "ok"), "ok", len(hits), new_hits, doc_ids=args.doc_ids
    )


def _shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEXT_FIELD_CHARS:
        return value[:MAX_TEXT_FIELD_CHARS] + "…"
    return value


def pms_payload(name: str, result: ToolResultBase) -> dict[str, Any]:
    """PMS 툴 결과를 LLM에 넣을 dict로 바꾼다. 지연 업무는 단계(level 1)와 업무를 나눈다(ADR-0010 결정 5)."""
    data = result.model_dump(mode="json")
    if not data["available_projects"]:
        data.pop("available_projects")
    items: list[dict[str, Any]] = data.pop("items", [])
    if name == "get_open_risks":
        for item in items:
            item["description"] = _shorten(item["description"])
            item["mitigation"] = _shorten(item["mitigation"])
    if name == "get_delayed_tasks":
        data["stage_delays"] = [i for i in items if i["level"] == 1]
        data["task_delays"] = [i for i in items if i["level"] != 1]
    else:
        data["items"] = items
    return data


def _fit(data: dict[str, Any]) -> str:
    """JSON이 MAX_PMS_RESULT_CHARS를 넘으면 목록 뒤에서부터 덜어 내고 omitted_by_size에 센다."""
    list_keys = [k for k in ("task_delays", "stage_delays", "items") if k in data]
    omitted = 0
    text = json.dumps(data, ensure_ascii=False)
    while len(text) > MAX_PMS_RESULT_CHARS and any(data[k] for k in list_keys):
        key = next(k for k in list_keys if data[k])
        data[key].pop()
        omitted += 1
        data["omitted_by_size"] = omitted
        text = json.dumps(data, ensure_ascii=False)
    return text


def _run_pms(name: str, args: _Args, deps: ToolDeps) -> ToolRun:
    kwargs = args.model_dump()
    funcs: dict[str, Callable[..., ToolResultBase]] = {
        "get_delayed_tasks": pms_tools.get_delayed_tasks,
        "get_open_risks": pms_tools.get_open_risks,
        "get_deliverables": pms_tools.get_deliverables,
    }
    result = funcs[name](**kwargs, deps=deps.pms)
    status: ToolCallStatus = result.status
    body = escape_tags(_fit(pms_payload(name, result)))
    ok = status == "ok"
    return ToolRun(
        _wrap(name, body, status),
        status,
        result.total,
        project_id=result.project_id if ok else None,
        doc_ids=result.doc_ids if ok else None,
    )


def invalid_call(name: str, error: str | None) -> ToolRun:
    """형식이 깨진 툴 호출(인자 JSON을 읽지 못함)의 결과 — 인자 오류로 LLM에 돌려준다."""
    return _error(
        name or "알 수 없음", "invalid_args", f"인자를 읽을 수 없다({error or '형식 오류'})"
    )


def run_tool(
    name: str, raw_args: dict[str, Any], deps: ToolDeps, registry: dict[int, DocumentHit]
) -> ToolRun:
    """툴 호출 1건을 실행한다. 인자 오류·서버 연결 불가는 예외 대신 오류 툴 결과로 돌려준다(그 밖의 예외는 버그로 올린다)."""
    spec = _SPEC_BY_NAME.get(name)
    if spec is None:
        return _error(name, "invalid_args", f"모르는 툴 '{name}'. 툴 목록에 있는 이름만 쓴다")
    try:
        args = spec.args_model.model_validate(raw_args)
        if isinstance(args, SearchArgs):
            return _run_search(args, deps, registry)
        return _run_pms(name, args, deps)
    except ValidationError as e:
        return _error(name, "invalid_args", _validation_message(e))
    except ValueError as e:
        return _error(name, "invalid_args", str(e))
    except (SearchUnavailableError, PmsUnavailableError) as e:
        return _error(name, "unavailable", str(e))
