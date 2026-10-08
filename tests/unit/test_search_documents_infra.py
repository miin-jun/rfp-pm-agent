"""search_documents의 기반 부품 단위 테스트 (#18 PR ③) — 오류 클래스·반환 모델·응답 변환·상수·SearchDeps.

search_documents 본문(소유자 구현)의 테스트는 tests/unit/test_search_documents.py에 있다.
"""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy import OpenSearch
from pydantic import ValidationError

from rfp_pm_agent.clients.embedding import TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import TEIRerankerClient
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K, MAX_TOP_K, SERVICE_RERANK_N
from rfp_pm_agent.search.opensearch import to_document_hits
from rfp_pm_agent.tools.search_documents import SearchDeps


def _source(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_id": "d1:block_requirement:SFR-001",
        "doc_id": "d1",
        "bid_title": "샘플 사업",
        "format": "pdf",
        "requirement_id": "SFR-001",
        "page": 3,
        "printed_page": 2,
        "block_type": None,
        "text": "요구사항 고유번호 SFR-001",
    }
    return base | overrides


def test_service_constants_follow_adr_0004() -> None:
    assert SERVICE_RERANK_N == 20
    assert DEFAULT_TOP_K == 10
    assert MAX_TOP_K == SERVICE_RERANK_N


def test_search_unavailable_error_keeps_service_and_message() -> None:
    err = SearchUnavailableError("tei-rerank", "connection refused")
    assert err.service == "tei-rerank"
    assert "tei-rerank" in str(err) and "connection refused" in str(err)


def test_search_unavailable_error_rejects_unknown_service() -> None:
    with pytest.raises(ValueError, match="알 수 없는"):
        SearchUnavailableError("postgres", "x")  # type: ignore[arg-type]


def test_to_document_hits_copies_citation_fields_in_response_order() -> None:
    response = {
        "hits": {
            "hits": [
                {"_id": "a", "_score": 0.03, "_source": _source()},
                {
                    "_id": "b",
                    "_score": 0.02,
                    "_source": _source(
                        chunk_id="d2:block_requirement:b0001",
                        doc_id="d2",
                        format="hwp",
                        requirement_id=None,
                        page=None,
                        printed_page=None,
                        block_type="table",
                    ),
                },
            ]
        }
    }

    hits = to_document_hits(response)

    assert [h.rank for h in hits] == [1, 2]
    assert [h.score for h in hits] == [0.03, 0.02]
    assert hits[0].model_dump() == {"rank": 1, "score": 0.03, **_source()}
    assert (hits[1].format, hits[1].page, hits[1].printed_page, hits[1].block_type) == (
        "hwp",
        None,
        None,
        "table",
    )


def test_to_document_hits_rejects_embedding_and_missing_score() -> None:
    with pytest.raises(ValueError, match="embedding"):
        to_document_hits({"hits": {"hits": [{"_score": 1.0, "_source": _source(embedding=[0.1])}]}})
    with pytest.raises(ValueError, match="_score"):
        to_document_hits({"hits": {"hits": [{"_score": None, "_source": _source()}]}})


def test_to_document_hits_fails_on_index_without_citation_fields() -> None:
    """#81 이전 매핑(v1)의 인덱스면 출처 필드가 없어 멈춘다 — null로 채워 넘기지 않는다."""
    old = _source()
    del old["bid_title"]
    with pytest.raises(KeyError, match="bid_title"):
        to_document_hits({"hits": {"hits": [{"_score": 1.0, "_source": old}]}})


def test_document_hit_rank_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        DocumentHit(rank=0, score=0.1, **_source())


def test_search_deps_from_env_builds_clients_with_opensearch_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENSEARCH_URL", "http://opensearch.test:9200")
    monkeypatch.setenv("OPENSEARCH_TIMEOUT_S", "3.5")
    monkeypatch.setenv("EMBED_BASE_URL", "http://embed.test:8080")
    monkeypatch.setenv("RERANK_BASE_URL", "http://rerank.test:8081")

    deps = SearchDeps.from_env()

    assert isinstance(deps.search_client, OpenSearch)
    assert deps.search_client.transport.kwargs["timeout"] == 3.5
    [host] = deps.search_client.transport.hosts
    assert (host["host"], host["port"]) == ("opensearch.test", 9200)
    assert isinstance(deps.embedder, TEIEmbeddingClient)
    assert deps.embedder.base_url == "http://embed.test:8080"
    assert isinstance(deps.reranker, TEIRerankerClient)
    assert deps.reranker.base_url == "http://rerank.test:8081"
