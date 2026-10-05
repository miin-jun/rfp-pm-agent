"""실제 OpenSearch로 검색 동작을 고정한다 (이슈 #18 PR ①, `-m integration`).

실행: `docker compose up -d opensearch`, 별칭(`OPENSEARCH_INDEX_ALIAS`)이 색인된 인덱스를
가리키는 상태에서 `uv run pytest tests/integration -m integration -q`.

- 읽기만 한다. 인덱스·별칭·검색 파이프라인을 만들거나 바꾸지 않는다(RRF는 요청 본문의
  임시 파이프라인으로만 보낸다)
- OpenSearch가 꺼져 있으면 skip하지 않고 실패한다 — test_tei_live.py와 같은 이유
- 질의 벡터는 TEI를 부르지 않고 인덱스에 저장된 문서의 embedding을 꺼내 쓴다
- 가짜(`tests/fakes/fake_search_client.py`)는 점수·분석기·k-NN을 흉내 내지 않으므로,
  여기 항목은 단위 테스트로 대신할 수 없다 (#18 결정 2)
"""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy import OpenSearch

from rfp_pm_agent.config import OpenSearchConfig
from rfp_pm_agent.search.opensearch import (
    EXCLUDED_SOURCE_FIELDS,
    bm25_search,
    knn_search,
    search_alias,
    to_search_hits,
)

pytestmark = pytest.mark.integration

OWNER_TODO = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="#18 학습 모드: 소유자 구현 전"
)

QUERY = "사업 기간은 언제까지인가"
TOP_K = 10
SOURCE_WITHOUT_EMBEDDING = {"excludes": list(EXCLUDED_SOURCE_FIELDS)}


@pytest.fixture(scope="module")
def client() -> OpenSearch:
    config = OpenSearchConfig.from_env()
    os_client = OpenSearch(hosts=[config.url])
    assert os_client.ping(), f"OpenSearch 응답 없음: {config.url}"
    return os_client


@pytest.fixture(scope="module")
def alias_target(client: OpenSearch) -> str:
    """별칭이 가리키는 실제 인덱스 이름. 하나가 아니면 실패한다."""
    alias = search_alias()
    targets: list[str] = sorted(client.indices.get_alias(name=alias))
    assert len(targets) == 1, f"별칭 {alias}가 가리키는 인덱스가 {targets}다"
    return targets[0]


@pytest.fixture(scope="module")
def stored_doc(client: OpenSearch) -> dict[str, Any]:
    """chunk_id가 가장 작은 문서 하나(embedding 포함). k-NN 질의 벡터로 쓴다."""
    response = client.search(
        index=search_alias(),
        body={"size": 1, "sort": [{"chunk_id": "asc"}], "_source": ["chunk_id", "embedding"]},
    )
    [hit] = response["hits"]["hits"]
    source: dict[str, Any] = hit["_source"]
    return source


def _hybrid_body(vector: list[float]) -> dict[str, Any]:
    return {
        "size": TOP_K,
        "_source": SOURCE_WITHOUT_EMBEDDING,
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"text": QUERY}},
                    {"knn": {"embedding": {"vector": vector, "k": 50}}},
                ]
            }
        },
    }


# --- 별칭 검색 결과 형태 (지금 통과해야 함) ---


def test_alias_resolves_to_one_index_with_documents(client: OpenSearch, alias_target: str) -> None:
    assert client.count(index=search_alias())["count"] > 0
    assert alias_target != search_alias()


def test_alias_search_response_converts_without_embedding(
    client: OpenSearch, alias_target: str
) -> None:
    """별칭으로 검색하면 _index에 실제 인덱스 이름이 오고, excludes로 embedding이 빠진다."""
    response = client.search(
        index=search_alias(),
        body={
            "size": TOP_K,
            "_source": SOURCE_WITHOUT_EMBEDDING,
            "query": {"match": {"text": QUERY}},
        },
    )
    raw_hits = response["hits"]["hits"]
    assert len(raw_hits) == TOP_K
    assert {h["_index"] for h in raw_hits} == {alias_target}
    assert all("embedding" not in h["_source"] for h in raw_hits)
    hits = to_search_hits(response)
    assert [h.chunk_id for h in hits] == [h["_id"] for h in raw_hits]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_without_source_filter_returns_embedding(client: OpenSearch) -> None:
    """_source를 지정하지 않으면 1024차원 벡터가 따라온다 — to_search_hits가 거절해야 하는 이유."""
    response = client.search(
        index=search_alias(), body={"size": 1, "query": {"match": {"text": QUERY}}}
    )
    [hit] = response["hits"]["hits"]
    assert len(hit["_source"]["embedding"]) == 1024
    with pytest.raises(ValueError, match="embedding"):
        to_search_hits(response)


# --- hybrid 쿼리와 파이프라인 (#18 결정 7, 2026-10-02 실측 고정) ---


def test_hybrid_without_pipeline_returns_200_with_negative_scores(
    client: OpenSearch, stored_doc: dict[str, Any]
) -> None:
    """검색 파이프라인 없이 hybrid를 보내면 에러 없이 내부 구분용 음수 점수가 섞여 돌아온다.

    이 동작이 바뀌면(OpenSearch 버전 변경 등) 서비스 경로의 안전장치를 다시 정해야 한다.
    """
    response = client.search(index=search_alias(), body=_hybrid_body(stored_doc["embedding"]))
    scores = [h["_score"] for h in response["hits"]["hits"]]
    assert any(score < 0 for score in scores), scores


def test_hybrid_with_temporary_rrf_pipeline_returns_rrf_scores(
    client: OpenSearch, stored_doc: dict[str, Any]
) -> None:
    """요청 본문의 임시 파이프라인(score-ranker-processor, rrf)이 적용되면 점수가 RRF 범위에 든다.

    rank_constant=60, 검색기 2개 → 한 문서의 최대 점수는 2/61. 클러스터에 파이프라인을 만들지 않는다.
    """
    body = _hybrid_body(stored_doc["embedding"])
    body["search_pipeline"] = {
        "phase_results_processors": [
            {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": 60}}}
        ]
    }
    response = client.search(index=search_alias(), body=body)
    scores = [h["_score"] for h in response["hits"]["hits"]]
    assert len(scores) == TOP_K
    assert all(0 < score <= 2 / 61 + 1e-9 for score in scores), scores


# --- 소유자 구현 함수 (구현 전에는 NotImplementedError로 xfail) ---


def test_bm25_search_via_alias(client: OpenSearch, alias_target: str) -> None:
    hits = bm25_search(client, QUERY, TOP_K)
    assert len(hits) == TOP_K
    assert all(h.score > 0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_knn_search_finds_stored_doc_with_its_own_vector(
    client: OpenSearch, stored_doc: dict[str, Any]
) -> None:
    """저장된 벡터로 검색하면 그 문서가 상위 TOP_K 안에 있다(HNSW 근사라 1위는 보장하지 않음)."""
    hits = knn_search(client, stored_doc["embedding"], TOP_K)
    assert len(hits) == TOP_K
    assert stored_doc["chunk_id"] in [h.chunk_id for h in hits]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
