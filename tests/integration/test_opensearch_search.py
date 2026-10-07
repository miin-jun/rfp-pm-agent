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
from rfp_pm_agent.search.hybrid import rrf_fuse
from rfp_pm_agent.search.opensearch import (
    EXCLUDED_SOURCE_FIELDS,
    KNN_CANDIDATES,
    REQUIREMENT_ID_BOOST,
    RRF_RANK_CONSTANT,
    bm25_search,
    hybrid_search,
    knn_search,
    search_alias,
    to_search_hits,
)

pytestmark = pytest.mark.integration

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


def _rrf_pipeline(combination: dict[str, Any]) -> dict[str, Any]:
    return {"phase_results_processors": [{"score-ranker-processor": {"combination": combination}}]}


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


# --- 별칭 검색 결과 형태 ---


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
    body["search_pipeline"] = _rrf_pipeline(
        {"technique": "rrf", "parameters": {"rank_constant": 60}}
    )
    response = client.search(index=search_alias(), body=body)
    scores = [h["_score"] for h in response["hits"]["hits"]]
    assert len(scores) == TOP_K
    assert all(0 < score <= 2 / 61 + 1e-9 for score in scores), scores


def _top_scores(
    client: OpenSearch, vector: list[float], combination: dict[str, Any]
) -> list[float]:
    body = _hybrid_body(vector)
    body["query"]["hybrid"]["pagination_depth"] = KNN_CANDIDATES
    body["search_pipeline"] = _rrf_pipeline(combination)
    response = client.search(index=search_alias(), body=body)
    return [h["_score"] for h in response["hits"]["hits"]]


def test_rrf_rank_constant_applies_only_inside_parameters(
    client: OpenSearch, stored_doc: dict[str, Any]
) -> None:
    """rank_constant는 combination.parameters 안에서만 적용된다 (2026-10-07 실측 고정).

    combination 바로 아래(2.19 공식 문서 예시 형식)에 두면 200을 주고 값을 무시해 기본값 60을 쓴다.
    결정값 60은 기본값과 같아 결과로 드러나지 않으므로, 60이 아닌 값(10)으로 위치 차이를 본다.
    이 테스트가 실패하면 OpenSearch 동작이 바뀐 것이다 — hybrid_search의 파이프라인 형식을 다시 확인한다.
    """
    vector = stored_doc["embedding"]
    default = _top_scores(client, vector, {"technique": "rrf"})
    top_level = _top_scores(client, vector, {"technique": "rrf", "rank_constant": 10})
    in_parameters = _top_scores(
        client, vector, {"technique": "rrf", "parameters": {"rank_constant": 10}}
    )
    assert top_level == default, "combination 바로 아래의 rank_constant가 적용되기 시작했다"
    assert in_parameters != default
    assert max(in_parameters) > 2 / 61, "rank_constant=10이면 1위 점수가 2/61을 넘을 수 있다"


# --- 소유자 구현 함수 (bm25_search·knn_search) ---


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


def test_bm25_requirement_id_adds_exactly_boost_to_match_score(client: OpenSearch) -> None:
    """질문에 요구사항 ID가 있으면 그 ID 청크의 점수는 match 점수 + 100이다 (#75, ADR-0003).

    `term`에 boost를 주면 keyword 필드도 BM25로 점수가 매겨져 100 × idf가 더해진다. `constant_score`를
    고른 이유가 실제 서버에서 지켜지는지 본다. 청크는 데이터에 묶이지 않도록 인덱스에서 고른다.
    """
    response = client.search(
        index=search_alias(),
        body={
            "size": 1,
            "sort": [{"chunk_id": "asc"}],
            "_source": ["chunk_id", "requirement_id"],
            "query": {"exists": {"field": "requirement_id"}},
        },
    )
    [hit] = response["hits"]["hits"]
    chunk_id, requirement_id = hit["_source"]["chunk_id"], hit["_source"]["requirement_id"]
    question = f"{requirement_id} 요구사항의 내용은?"

    # 가산 없는 match 점수 — filter는 점수에 들어가지 않는다
    plain = client.search(
        index=search_alias(),
        body={
            "size": 1,
            "_source": ["chunk_id"],
            "query": {
                "bool": {
                    "must": [{"match": {"text": question}}],
                    "filter": [{"term": {"chunk_id": chunk_id}}],
                }
            },
        },
    )
    [plain_hit] = plain["hits"]["hits"]
    match_score = float(plain_hit["_score"])
    assert match_score > 0

    hits = bm25_search(client, question, TOP_K)
    boosted = {h.chunk_id: h.score for h in hits}
    assert chunk_id in boosted, f"{chunk_id}({requirement_id})가 상위 {TOP_K}에 없다"
    assert boosted[chunk_id] == pytest.approx(match_score + REQUIREMENT_ID_BOOST, abs=1e-3)


# --- 소유자 구현 함수 (#18 PR ②: hybrid_search·rrf_fuse) ---


def test_hybrid_search_scores_are_rrf_scores(
    client: OpenSearch, stored_doc: dict[str, Any]
) -> None:
    hits = hybrid_search(client, QUERY, stored_doc["embedding"], TOP_K)
    assert len(hits) == TOP_K
    max_score = 2 / (RRF_RANK_CONSTANT + 1)
    assert all(0 < h.score <= max_score + 1e-9 for h in hits), [h.score for h in hits]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hybrid_search_matches_app_rrf_over_candidates(
    client: OpenSearch, stored_doc: dict[str, Any]
) -> None:
    """OpenSearch RRF 상위 10과 앱 RRF(bm25·knn 각 후보 50) 상위 10의 점수 목록이 같다.

    점수 목록으로 비교하면 동점끼리의 순서 차이에 흔들리지 않는다. pagination_depth를 빼면
    하위 쿼리마다 size개만 가져와 점수가 달라진다(2026-10-07 실측: chunk 5개만 일치).
    """
    vector = stored_doc["embedding"]
    service = hybrid_search(client, QUERY, vector, TOP_K)
    app = rrf_fuse(
        [bm25_search(client, QUERY, KNN_CANDIDATES), knn_search(client, vector, KNN_CANDIDATES)],
        top_k=TOP_K,
    )
    assert [h.score for h in service] == pytest.approx([h.score for h in app], rel=1e-6)
