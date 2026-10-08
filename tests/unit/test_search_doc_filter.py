"""doc_ids 필터의 요청 본문 (#18 PR ③).

필터는 BM25·k-NN 하위 쿼리마다 넣는다. hybrid 최상위 filter(2.19.1이 400)와 post_filter(후보를 뽑은 뒤에
걸러 순위가 달라짐)는 쓰지 않는다 — docs/data-design.md 5절 "검색 툴 search_documents".
"""

from __future__ import annotations

from typing import Any

import pytest

from rfp_pm_agent.search.opensearch import (
    KNN_CANDIDATES,
    REQUIREMENT_ID_BOOST,
    bm25_query,
    hybrid_body,
    hybrid_search,
    knn_query,
    search_alias,
)
from tests.fakes.fake_search_client import RecordingSearchClient

VECTOR = [0.1, 0.2, 0.3]
DOCS = ["d3e23c59f151e223", "722630192c1100a1"]
TERMS = {"terms": {"doc_id": DOCS}}


def test_bm25_query_without_id_wraps_match_with_filter() -> None:
    assert bm25_query("사업 기간", DOCS) == {
        "bool": {"must": [{"match": {"text": "사업 기간"}}], "filter": [TERMS]}
    }


def test_bm25_query_with_id_keeps_boost_and_adds_filter() -> None:
    query = "SFR-013 요구사항"
    assert bm25_query(query, DOCS) == {
        "bool": {
            "must": [{"match": {"text": query}}],
            "should": [
                {
                    "constant_score": {
                        "filter": {"term": {"requirement_id": "SFR-013"}},
                        "boost": REQUIREMENT_ID_BOOST,
                    }
                }
            ],
            "filter": [TERMS],
        }
    }


def test_knn_query_puts_filter_inside_knn() -> None:
    assert knn_query(VECTOR, DOCS) == {
        "knn": {"embedding": {"vector": VECTOR, "k": KNN_CANDIDATES, "filter": TERMS}}
    }


@pytest.mark.parametrize("query", ["사업 기간", "SFR-013 요구사항"])
def test_none_doc_ids_keeps_queries_unchanged(query: str) -> None:
    """기존 호출(bm25_search·knn_search·평가)은 doc_ids를 주지 않는다 — 본문이 바뀌면 안 된다."""
    assert bm25_query(query, None) == bm25_query(query)
    assert knn_query(VECTOR, None) == knn_query(VECTOR)
    assert hybrid_body(query, VECTOR, 20, None) == hybrid_body(query, VECTOR, 20)


@pytest.mark.parametrize("build", [lambda: bm25_query("q", []), lambda: knn_query(VECTOR, [])])
def test_empty_doc_ids_is_rejected(build: Any) -> None:
    with pytest.raises(ValueError, match="doc_ids"):
        build()


def test_hybrid_body_filters_each_subquery() -> None:
    body = hybrid_body("사업 기간", VECTOR, 20, DOCS)

    assert body["query"]["hybrid"]["queries"] == [
        bm25_query("사업 기간", DOCS),
        knn_query(VECTOR, DOCS),
    ]
    # 하위 쿼리 말고는 필터 없는 본문과 같다
    plain = hybrid_body("사업 기간", VECTOR, 20)
    assert {k: v for k, v in body.items() if k != "query"} == {
        k: v for k, v in plain.items() if k != "query"
    }
    assert (
        body["query"]["hybrid"]["pagination_depth"] == plain["query"]["hybrid"]["pagination_depth"]
    )


def test_hybrid_body_never_uses_post_filter_or_top_level_hybrid_filter() -> None:
    body = hybrid_body("사업 기간", VECTOR, 20, DOCS)
    assert "post_filter" not in body
    assert "filter" not in body["query"]["hybrid"]
    assert set(body["query"]) == {"hybrid"}


def test_hybrid_search_sends_filtered_body_to_alias() -> None:
    client = RecordingSearchClient()

    hybrid_search(client, "사업 기간", VECTOR, 10, DOCS)

    [call] = client.calls
    assert call["index"] == search_alias()
    assert call["body"] == hybrid_body("사업 기간", VECTOR, 10, DOCS)
