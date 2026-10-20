"""시연 화면 API (이슈 #40, ADR-0015). 에이전트는 정해 둔 `AgentAnswer`를 돌려주는 가짜, OpenSearch는 기록용 가짜 — 네트워크 없음."""

from __future__ import annotations

from datetime import date
from typing import Any

import opensearchpy
import pytest
from fastapi.testclient import TestClient

from rfp_pm_agent.api.app import DemoDeps, cited_chunks, create_app
from rfp_pm_agent.schemas.agent import AgentAnswer, ToolCallRecord
from rfp_pm_agent.schemas.answer import LLMUsage
from rfp_pm_agent.search.grounded_answer import REFUSAL_TEXT
from tests.fakes.fake_search_client import RecordingSearchClient

AS_OF = date(2026, 9, 24)


class FakeAgent:
    def __init__(self, answer: AgentAnswer | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def ask(self, question: str, *, thread_id: str) -> AgentAnswer:
        self.calls.append((question, thread_id))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FailingSearchClient:
    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        raise opensearchpy.ConnectionError("N/A", "connection refused", None)


def _answer(text: str, **overrides: Any) -> AgentAnswer:
    fields: dict[str, Any] = {
        "answer": text,
        "refused": False,
        "citations": [],
        "citation_map": {},
        "cited_chunk_ids": [],
        "invalid_citations": [],
        "tool_calls": [],
        "llm_calls": 2,
        "reached_max_steps": False,
        "project_id": None,
        "doc_ids": [],
        "usage": LLMUsage(input_tokens=100, output_tokens=20, reasoning_tokens=0),
    }
    fields.update(overrides)
    return AgentAnswer(**fields)


def _source(chunk_id: str, **overrides: Any) -> dict[str, Any]:
    source: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": "docA",
        "bid_title": "A 사업",
        "format": "pdf",
        "requirement_id": "SFR-001",
        "page": 14,
        "printed_page": 23,
        "block_type": None,
        "text": f"{chunk_id} 원문",
    }
    source.update(overrides)
    return {"_id": chunk_id, "_score": 1.0, "_source": source}


CITED = _answer(
    "기둥은 15개다 [3]. 통합형 5개와 일반형 10개로 나뉜다 [1, 3][7].",
    citations=["[A 사업 p.23 · SFR-001]", "[A 사업 p.5]"],
    citation_map={3: "[A 사업 p.23 · SFR-001]", 1: "[A 사업 p.5]"},
    cited_chunk_ids=["c-three", "c-one"],
    invalid_citations=[7],
    tool_calls=[
        ToolCallRecord(name="search_documents", args={}, status="ok", result_count=8),
        ToolCallRecord(name="search_documents", args={}, status="ok", result_count=8),
    ],
)


def _client(agent: FakeAgent, search_client: Any) -> TestClient:
    app = create_app(DemoDeps(agent=agent, search_client=search_client, as_of_date=AS_OF))
    return TestClient(app)


def test_ask_returns_answer_sources_with_text_and_tools() -> None:
    search = RecordingSearchClient(
        {
            "hits": {
                "hits": [_source("c-one", printed_page=5, requirement_id=None), _source("c-three")]
            }
        }
    )
    agent = FakeAgent(CITED)
    with _client(agent, search) as client:
        response = client.post("/api/ask", json={"question": "  기둥은 몇 개?  "})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "refused", "sources", "tools_used", "elapsed_ms"}
    assert body["answer"] == CITED.answer
    assert body["refused"] is False
    assert body["tools_used"] == ["search_documents", "search_documents"]
    assert isinstance(body["elapsed_ms"], int) and body["elapsed_ms"] >= 0
    # 답에 처음 나온 순서, 유효하지 않은 번호(7)는 빠진다. 문서 필드·원문은 OpenSearch에서 채운다
    assert body["sources"] == [
        {
            "number": 3,
            "chunk_id": "c-three",
            "citation": "[A 사업 p.23 · SFR-001]",
            "bid_title": "A 사업",
            "format": "pdf",
            "printed_page": 23,
            "page": 14,
            "requirement_id": "SFR-001",
            "text": "c-three 원문",
        },
        {
            "number": 1,
            "chunk_id": "c-one",
            "citation": "[A 사업 p.5]",
            "bid_title": "A 사업",
            "format": "pdf",
            "printed_page": 5,
            "page": 14,
            "requirement_id": None,
            "text": "c-one 원문",
        },
    ]
    # 질문은 앞뒤 공백을 떼고, 질문마다 새 대화로 묻는다
    assert agent.calls[0][0] == "기둥은 몇 개?"
    assert agent.calls[0][1].startswith("demo-")


def test_sources_are_fetched_by_chunk_id_once_without_embedding() -> None:
    search = RecordingSearchClient({"hits": {"hits": []}})
    with _client(FakeAgent(CITED), search) as client:
        client.post("/api/ask", json={"question": "q"})

    assert len(search.calls) == 1
    call = search.calls[0]
    assert call["index"] == "rfp_chunks"
    assert call["body"] == {
        "size": 2,
        "query": {"terms": {"chunk_id": ["c-three", "c-one"]}},
        "_source": {"excludes": ["embedding"]},
    }


def test_chunk_missing_from_index_keeps_citation_and_leaves_text_empty() -> None:
    search = RecordingSearchClient({"hits": {"hits": [_source("c-three")]}})
    with _client(FakeAgent(CITED), search) as client:
        sources = client.post("/api/ask", json={"question": "q"}).json()["sources"]

    assert sources[1]["chunk_id"] == "c-one"
    assert sources[1]["citation"] == "[A 사업 p.5]"
    assert sources[1]["text"] is None
    assert sources[1]["bid_title"] is None


def test_opensearch_failure_returns_answer_without_source_text() -> None:
    with _client(FakeAgent(CITED), FailingSearchClient()) as client:
        response = client.post("/api/ask", json={"question": "q"})

    assert response.status_code == 200
    assert [s["text"] for s in response.json()["sources"]] == [None, None]
    assert [s["citation"] for s in response.json()["sources"]] == [
        "[A 사업 p.23 · SFR-001]",
        "[A 사업 p.5]",
    ]


def test_refusal_sets_flag_and_has_no_sources() -> None:
    refusal = _answer(
        REFUSAL_TEXT,
        refused=True,
        tool_calls=[ToolCallRecord(name="search_documents", args={}, status="ok", result_count=8)],
    )
    search = RecordingSearchClient()
    with _client(FakeAgent(refusal), search) as client:
        body = client.post("/api/ask", json={"question": "월간 목표 처리 건수는?"}).json()

    assert body["refused"] is True
    assert body["answer"] == REFUSAL_TEXT
    assert body["sources"] == []
    assert body["tools_used"] == ["search_documents"]
    assert search.calls == []  # 출처가 없으면 OpenSearch를 부르지 않는다


def test_pms_answer_without_citations_has_no_sources() -> None:
    pms = _answer(
        "이번 주 지연 업무는 WBS 2.5 한 건이다.",
        tool_calls=[ToolCallRecord(name="get_delayed_tasks", args={}, status="ok", result_count=1)],
    )
    with _client(FakeAgent(pms), RecordingSearchClient()) as client:
        body = client.post("/api/ask", json={"question": "이번 주 지연 업무"}).json()

    assert body["refused"] is False
    assert body["sources"] == []
    assert body["tools_used"] == ["get_delayed_tasks"]


@pytest.mark.parametrize("question", ["", "   ", "가" * 1001])
def test_blank_or_too_long_question_is_rejected(question: str) -> None:
    agent = FakeAgent(CITED)
    with _client(agent, RecordingSearchClient()) as client:
        response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422
    assert agent.calls == []


def test_agent_error_becomes_502_with_reason() -> None:
    with _client(FakeAgent(RuntimeError("LLM 연결 실패")), RecordingSearchClient()) as client:
        response = client.post("/api/ask", json={"question": "q"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "RuntimeError" in detail
    assert "LLM 연결 실패" not in detail  # 예외 메시지는 서버 로그에만 남긴다


def test_index_page_has_as_of_date_and_korean_settings() -> None:
    with _client(FakeAgent(CITED), RecordingSearchClient()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "기준일 2026-09-24" in html
    assert "{{AS_OF_DATE}}" not in html
    assert "pretendard" in html.lower()
    assert "word-break: keep-all" in html


def test_cited_chunks_pairs_numbers_with_chunk_ids_in_answer_order() -> None:
    assert cited_chunks(CITED) == [(3, "c-three"), (1, "c-one")]


def test_cited_chunks_rejects_mismatched_lists() -> None:
    broken = CITED.model_copy(update={"cited_chunk_ids": ["c-three"]})
    with pytest.raises(ValueError, match="개수가 다르다"):
        cited_chunks(broken)
