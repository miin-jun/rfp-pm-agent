"""하이브리드 검색 단위 테스트 (이슈 #18 PR ②).

`hybrid_search`·`rrf_fuse`·`hybrid_rerank_search`와 `knn_search`의 후보 수 상한은 소유자가 구현한다
(학습 모드). 이 파일은 구현 전에 먼저 고정해 둔 것이다 — 껍데기 상태에서는 실패하는 것이 정상이다.

고정하는 것
- hybrid body: 하위 쿼리는 PR ①의 `bm25_query`·`knn_query` 그대로, `pagination_depth` = 후보 50,
  `rank_constant`는 `combination.parameters` 안(바로 아래에 두면 2.19.1이 에러 없이 무시한다)
- 앱 RRF: 손으로 계산한 점수, 동점 순서, 입력 검증
- 리랭크: hybrid를 rerank_n개로 한 번, 리랭커를 한 번 부르고 상위 top_k를 돌려준다
점수 계산·분석기·k-NN 자체는 가짜로 흉내 내지 않는다 — tests/integration/test_opensearch_search.py
"""

from __future__ import annotations

from typing import Any

import pytest

from rfp_pm_agent.schemas.eval import SearchHit
from rfp_pm_agent.search.hybrid import hybrid_rerank_search, rrf_fuse
from rfp_pm_agent.search.opensearch import (
    KNN_CANDIDATES,
    RRF_RANK_CONSTANT,
    bm25_query,
    hybrid_search,
    knn_query,
    knn_search,
    to_search_hits,
)
from tests.fakes.fake_search_client import RecordingSearchClient

TEST_ALIAS = "test_alias_for_hybrid"
VECTOR = [0.1, 0.2, 0.3]
QUESTION = "사업 기간은 언제까지인가"
QUESTION_WITH_ID = "SFR-013 요구사항은 어떤 업무를 지원하나"


@pytest.fixture(autouse=True)
def _alias_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_INDEX_ALIAS", TEST_ALIAS)


def _hit(chunk_id: str, score: float = 1.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id, doc_id=chunk_id.split(":")[0], score=score, text=f"본문 {chunk_id}"
    )


def _raw_hit(chunk_id: str, score: float) -> dict[str, Any]:
    source = {"chunk_id": chunk_id, "doc_id": chunk_id.split(":")[0], "text": f"본문 {chunk_id}"}
    return {"_index": "rfp_chunks_v1_kure", "_id": chunk_id, "_score": score, "_source": source}


def _response(n: int) -> dict[str, Any]:
    """RRF 점수 범위의 결과 n개 (1위부터)."""
    hits = [_raw_hit(f"d{i}:block_requirement:b{i:04d}", 1 / (61 + i)) for i in range(n)]
    return {"hits": {"total": {"value": n}, "hits": hits}}


def _hybrid_body(question: str = QUESTION, top_k: int = 10) -> dict[str, Any]:
    client = RecordingSearchClient(_response(top_k))
    hybrid_search(client, question, VECTOR, top_k=top_k)
    assert len(client.calls) == 1
    body: dict[str, Any] = client.calls[0]["body"]
    return body


# --- hybrid_search: 요청 ---


def test_hybrid_search_sends_one_request_to_alias() -> None:
    client = RecordingSearchClient(_response(10))
    hybrid_search(client, QUESTION, VECTOR, top_k=10)
    assert len(client.calls) == 1
    assert client.calls[0]["index"] == TEST_ALIAS


@pytest.mark.parametrize("top_k", [10, 20, KNN_CANDIDATES])
def test_hybrid_search_size_is_top_k(top_k: int) -> None:
    assert _hybrid_body(top_k=top_k)["size"] == top_k


def test_hybrid_search_source_excludes_embedding() -> None:
    source = _hybrid_body()["_source"]
    assert "embedding" in source["excludes"]


@pytest.mark.parametrize("question", [QUESTION, QUESTION_WITH_ID])
def test_hybrid_search_reuses_bm25_and_knn_query_in_order(question: str) -> None:
    """하위 쿼리는 PR ①의 query 절 그대로다 — 요구사항 ID 가산도 함께 들어간다."""
    hybrid = _hybrid_body(question)["query"]["hybrid"]
    assert hybrid["queries"] == [bm25_query(question), knn_query(VECTOR)]


def test_hybrid_search_pagination_depth_is_candidates() -> None:
    """pagination_depth가 없으면 하위 쿼리마다 size개만 가져와 "검색기별 후보 50"이 깨진다."""
    hybrid = _hybrid_body(top_k=10)["query"]["hybrid"]
    assert "pagination_depth" in hybrid, "pagination_depth를 빼면 하위 쿼리마다 size개만 가져온다"
    assert hybrid["pagination_depth"] == KNN_CANDIDATES
    knn_k = hybrid["queries"][1]["knn"]["embedding"]["k"]
    assert hybrid["pagination_depth"] == knn_k


def test_hybrid_search_sends_temporary_rrf_pipeline() -> None:
    body = _hybrid_body()
    processors = body["search_pipeline"]["phase_results_processors"]
    assert len(processors) == 1
    assert set(processors[0]) == {"score-ranker-processor"}
    assert processors[0]["score-ranker-processor"]["combination"]["technique"] == "rrf"


def test_hybrid_search_rank_constant_is_inside_parameters() -> None:
    """rank_constant는 combination.parameters 안에 있어야 적용된다.

    combination 바로 아래(2.19 공식 문서 예시 형식)에 두면 2.19.1은 200을 주고 값을 무시한다.
    결정값 60이 기본값과 같아서 결과로는 드러나지 않으므로 위치를 직접 검사한다.
    """
    combination = _hybrid_body()["search_pipeline"]["phase_results_processors"][0][
        "score-ranker-processor"
    ]["combination"]
    assert "rank_constant" not in combination, "combination 바로 아래의 rank_constant는 무시된다"
    assert combination == {"technique": "rrf", "parameters": {"rank_constant": RRF_RANK_CONSTANT}}
    assert RRF_RANK_CONSTANT == 60


def test_hybrid_search_returns_converted_hits() -> None:
    response = _response(10)
    client = RecordingSearchClient(response)
    assert hybrid_search(client, QUESTION, VECTOR, top_k=10) == to_search_hits(response)


def test_hybrid_search_rejects_top_k_over_candidates_without_request() -> None:
    client = RecordingSearchClient(_response(10))
    with pytest.raises(ValueError):
        hybrid_search(client, QUESTION, VECTOR, top_k=KNN_CANDIDATES + 1)
    assert client.calls == []


# --- knn_search: 후보 수 상한 (결정 5A) ---


def test_knn_search_rejects_top_k_over_candidates_without_request() -> None:
    client = RecordingSearchClient(_response(10))
    with pytest.raises(ValueError):
        knn_search(client, VECTOR, top_k=KNN_CANDIDATES + 1)
    assert client.calls == []


def test_knn_search_accepts_top_k_equal_to_candidates() -> None:
    client = RecordingSearchClient(_response(KNN_CANDIDATES))
    knn_search(client, VECTOR, top_k=KNN_CANDIDATES)
    assert client.calls[0]["body"]["size"] == KNN_CANDIDATES


# --- rrf_fuse (앱 RRF, 비교·검증용) ---

A, B, C, D = (f"d{n}:block_requirement:b000{n}" for n in range(1, 5))


def test_rrf_fuse_hand_computed_scores_and_order() -> None:
    """A=[a,b,c], B=[c,a,d], k=60 → a=1/61+1/62, c=1/63+1/61, b=1/62, d=1/63."""
    fused = rrf_fuse([[_hit(A), _hit(B), _hit(C)], [_hit(C), _hit(A), _hit(D)]], top_k=10)
    assert [h.chunk_id for h in fused] == [A, C, B, D]
    assert [h.score for h in fused] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63]
    )


def test_rrf_fuse_ignores_input_scores() -> None:
    """입력 score(BM25 점수·유사도)는 쓰지 않고 순위만 쓴다."""
    low_first = rrf_fuse([[_hit(A, 0.001), _hit(B, 999.0)]], top_k=2)
    assert [h.chunk_id for h in low_first] == [A, B]


def test_rrf_fuse_truncates_to_top_k() -> None:
    fused = rrf_fuse([[_hit(A), _hit(B), _hit(C)], [_hit(D)]], top_k=2)
    assert len(fused) == 2


def test_rrf_fuse_tie_keeps_first_appearance_order() -> None:
    """두 목록의 같은 순위는 동점(1/61) — rankings[0]에서 먼저 나온 것이 앞이다."""
    fused = rrf_fuse([[_hit(B)], [_hit(A)]], top_k=2)
    assert [h.chunk_id for h in fused] == [B, A]
    assert fused[0].score == fused[1].score


def test_rrf_fuse_uses_rank_constant() -> None:
    fused = rrf_fuse([[_hit(A), _hit(B)], [_hit(B)]], top_k=2, rank_constant=1)
    assert [h.chunk_id for h in fused] == [B, A]
    assert [h.score for h in fused] == pytest.approx([1 / 3 + 1 / 2, 1 / 2])


def test_rrf_fuse_keeps_doc_id_and_text_from_first_appearance() -> None:
    first = SearchHit(chunk_id=A, doc_id="d1", score=5.0, text="BM25 쪽 본문")
    second = SearchHit(chunk_id=A, doc_id="d1", score=0.9, text="k-NN 쪽 본문")
    [fused] = rrf_fuse([[first], [second]], top_k=1)
    assert (fused.doc_id, fused.text) == ("d1", "BM25 쪽 본문")


@pytest.mark.parametrize("rankings", [[], [[], []]])
def test_rrf_fuse_empty(rankings: list[list[SearchHit]]) -> None:
    assert rrf_fuse(rankings, top_k=10) == []


@pytest.mark.parametrize("rank_constant", [0, -1])
def test_rrf_fuse_rejects_rank_constant_below_one(rank_constant: int) -> None:
    with pytest.raises(ValueError):
        rrf_fuse([[_hit(A)]], top_k=1, rank_constant=rank_constant)


def test_rrf_fuse_rejects_duplicate_chunk_in_one_ranking() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([[_hit(A), _hit(A)]], top_k=2)


# --- hybrid_rerank_search (리랭크 연결) ---


class RecordingReranker:
    """질의·문서를 기록하고 정해 둔 점수(문서 text → 점수)를 돌려준다."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [self.scores.get(doc, 0.0) for doc in documents]


def _texts(response: dict[str, Any]) -> list[str]:
    return [h["_source"]["text"] for h in response["hits"]["hits"]]


@pytest.mark.parametrize("rerank_n", [20, 50])
def test_hybrid_rerank_requests_rerank_n_candidates_once(rerank_n: int) -> None:
    response = _response(rerank_n)
    client = RecordingSearchClient(response)
    reranker = RecordingReranker({})
    hybrid_rerank_search(client, reranker, QUESTION, VECTOR, rerank_n=rerank_n, top_k=10)
    assert len(client.calls) == 1
    assert client.calls[0]["body"]["size"] == rerank_n
    assert reranker.calls == [(QUESTION, _texts(response))]


def test_hybrid_rerank_orders_by_reranker_score_and_truncates() -> None:
    response = _response(20)
    texts = _texts(response)
    # hybrid 19위·5위·12위가 리랭크 1·2·3위가 되게 한다
    reranker = RecordingReranker({texts[19]: 0.9, texts[5]: 0.8, texts[12]: 0.7})
    client = RecordingSearchClient(response)
    hits = hybrid_rerank_search(client, reranker, QUESTION, VECTOR, rerank_n=20, top_k=10)
    assert len(hits) == 10
    assert [h.text for h in hits[:3]] == [texts[19], texts[5], texts[12]]
    assert [h.score for h in hits[:3]] == pytest.approx([0.9, 0.8, 0.7])


def test_hybrid_rerank_tie_keeps_hybrid_order() -> None:
    response = _response(20)
    client = RecordingSearchClient(response)
    hits = hybrid_rerank_search(
        client, RecordingReranker({}), QUESTION, VECTOR, rerank_n=20, top_k=10
    )
    assert [h.text for h in hits] == _texts(response)[:10]


def test_hybrid_rerank_empty_hybrid_skips_reranker() -> None:
    client = RecordingSearchClient({"hits": {"hits": []}})
    reranker = RecordingReranker({})
    assert hybrid_rerank_search(client, reranker, QUESTION, VECTOR, rerank_n=20, top_k=10) == []
    assert reranker.calls == []


@pytest.mark.parametrize(
    ("rerank_n", "top_k"), [(KNN_CANDIDATES + 1, 10), (20, 21)], ids=["n_over_50", "k_over_n"]
)
def test_hybrid_rerank_rejects_out_of_range_without_calls(rerank_n: int, top_k: int) -> None:
    client = RecordingSearchClient(_response(10))
    reranker = RecordingReranker({})
    with pytest.raises(ValueError):
        hybrid_rerank_search(client, reranker, QUESTION, VECTOR, rerank_n=rerank_n, top_k=top_k)
    assert client.calls == []
    assert reranker.calls == []
