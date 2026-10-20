"""시연 화면 API의 출처 원문 조회를 실제 OpenSearch로 본다 (이슈 #40, `-m integration`).

에이전트는 가짜다 — 실제 LLM을 부르는 테스트는 만들지 않는다(CLAUDE.md 테스트 규칙, 비용).
색인에 있는 청크 2개를 골라 그 chunk_id를 인용한 답을 돌려주게 하고, `/api/ask`가 원문·문서 필드를
OpenSearch `_source`와 똑같이 채우는지 확인한다. 실행: `docker compose up -d opensearch` 후
`uv run pytest tests/integration -m integration -q`. 서버가 꺼져 있으면 skip하지 않고 실패한다.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from opensearchpy import OpenSearch

from rfp_pm_agent.api.app import DemoDeps, create_app
from rfp_pm_agent.config import OpenSearchConfig, SynthConfig
from rfp_pm_agent.schemas.agent import AgentAnswer
from rfp_pm_agent.schemas.answer import LLMUsage
from rfp_pm_agent.search.opensearch import search_alias

pytestmark = pytest.mark.integration


class ScriptedAgent:
    def __init__(self, answer: AgentAnswer) -> None:
        self.answer = answer

    def ask(self, question: str, *, thread_id: str) -> AgentAnswer:
        return self.answer


def test_ask_fills_source_text_from_opensearch() -> None:
    config = OpenSearchConfig.from_env()
    client = OpenSearch(hosts=[config.url], timeout=config.timeout_s)
    picked = client.search(
        index=search_alias(),
        body={
            "size": 2,
            "query": {"match_all": {}},
            "sort": [{"chunk_id": "asc"}],
            "_source": {"excludes": ["embedding"]},
        },
    )["hits"]["hits"]
    expected: list[dict[str, Any]] = [hit["_source"] for hit in picked]
    assert len(expected) == 2
    answer = AgentAnswer(
        answer="첫 근거 [2]와 둘째 근거 [1].",
        refused=False,
        citations=["[둘째]", "[첫째]"],
        citation_map={2: "[둘째]", 1: "[첫째]"},
        cited_chunk_ids=[expected[0]["chunk_id"], expected[1]["chunk_id"]],
        invalid_citations=[],
        tool_calls=[],
        llm_calls=1,
        reached_max_steps=False,
        project_id=None,
        doc_ids=[],
        usage=LLMUsage(input_tokens=0, output_tokens=0, reasoning_tokens=0),
    )
    deps = DemoDeps(
        agent=ScriptedAgent(answer),
        search_client=client,
        as_of_date=SynthConfig.from_env().as_of_date,
    )

    with TestClient(create_app(deps)) as api:
        response = api.post("/api/ask", json={"question": "근거"})

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert [s["number"] for s in sources] == [2, 1]
    for source, truth in zip(sources, expected, strict=True):
        assert source["chunk_id"] == truth["chunk_id"]
        assert source["text"] == truth["text"]
        assert source["bid_title"] == truth["bid_title"]
        assert source["format"] == truth["format"]
        assert source["printed_page"] == truth["printed_page"]
        assert source["page"] == truth["page"]
        assert source["requirement_id"] == truth["requirement_id"]
