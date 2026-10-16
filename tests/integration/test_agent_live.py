"""에이전트 그래프를 실제 검색·PMS 서버로 돌린다 (이슈 #22, `-m tei`). LLM은 가짜(정해 둔 툴 호출) — 비용 없음.

실행: `docker compose up -d` (opensearch·postgres·tei-embed·tei-rerank, PMS v2 적재) 후
`uv run pytest tests/integration -m tei -q`. TEI를 쓰므로 마커는 `tei`다(test_search_documents_live.py와 같다).
서버가 꺼져 있으면 skip하지 않고 실패한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rfp_pm_agent.agent.catalog import build_catalog
from rfp_pm_agent.agent.graph import Agent, AgentDeps
from rfp_pm_agent.agent.tools import ToolDeps
from rfp_pm_agent.tools.pms_tools import PmsDeps
from rfp_pm_agent.tools.search_documents import SearchDeps
from tests.fakes.fake_chat_model import ScriptedChatModel, answer, tool_call

pytestmark = pytest.mark.tei

TRUTH = json.loads(Path("data/eval/pms_truth_v2.json").read_text("utf-8"))
PRJ03 = next(p for p in TRUTH["projects"] if p["project_id"] == "PRJ-03")
NO_PMS_DOC = "63d86447109810d1"  # 국가 망 보안 ISP — PMS 프로젝트가 없는 문서


@pytest.fixture(scope="module")
def tool_deps() -> ToolDeps:
    return ToolDeps(search=SearchDeps.from_env(), pms=PmsDeps.from_env())


def _agent(tool_deps: ToolDeps, responses: list[Any]) -> Agent:
    catalog = build_catalog(tool_deps.search.search_client, tool_deps.pms.reader)
    model = ScriptedChatModel(responses=responses)
    return Agent(AgentDeps(tools=tool_deps, chat_model=model, catalog=catalog, max_steps=6))


def test_catalog_lists_all_indexed_documents_with_projects(tool_deps: ToolDeps) -> None:
    catalog = build_catalog(tool_deps.search.search_client, tool_deps.pms.reader)

    by_doc = {e.doc_id: e for e in catalog}
    assert len(catalog) == 10
    assert by_doc[NO_PMS_DOC].project_id is None
    assert by_doc[PRJ03["doc_id"]].project_id == "PRJ-03"
    assert sum(e.project_id is not None for e in catalog) == 9


def test_pms_then_search_flow_on_real_services(tool_deps: ToolDeps) -> None:
    agent = _agent(
        tool_deps,
        [
            tool_call("get_delayed_tasks", {"project_id": "PRJ-03", "scope": "this_week"}, "c1"),
            tool_call(
                "search_documents",
                {"query": "DAR-002 비식별화", "doc_ids": [PRJ03["doc_id"]], "top_k": 5},
                "c2",
            ),
            answer("이번 주 지연은 WBS 2.5이고 관련 요구사항은 DAR-002다 [1]."),
        ],
    )

    result = agent.ask("AX 불균형 투자계약 점검 사업 이번 주 지연 업무와 요구사항", thread_id="t")

    assert [(c.name, c.status) for c in result.tool_calls] == [
        ("get_delayed_tasks", "ok"),
        ("search_documents", "ok"),
    ]
    assert result.tool_calls[0].result_count == 1  # 정답 파일: PRJ-03 이번 주 지연 WBS 2.5 한 건
    assert result.project_id == "PRJ-03"
    assert result.doc_ids == [PRJ03["doc_id"]]
    assert list(result.citation_map) == [1]
    assert result.citation_map[1].startswith("[AX를 통한 불균형 투자계약 점검 시스템 구축")
    assert result.invalid_citations == []


def test_unknown_doc_id_is_returned_as_invalid_args(tool_deps: ToolDeps) -> None:
    agent = _agent(
        tool_deps,
        [
            tool_call("search_documents", {"query": "예산", "doc_ids": ["0000000000000000"]}),
            answer("문서에서 찾을 수 없음"),
        ],
    )

    result = agent.ask("예산은?", thread_id="t")

    assert [c.status for c in result.tool_calls] == ["invalid_args"]
