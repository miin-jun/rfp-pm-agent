"""OpenSearch 단일 검색 단위 테스트 (이슈 #18 PR ①).

`bm25_search`·`knn_search`의 본문은 소유자가 구현한다(학습 모드). 그 전에는 이 파일의
body 검사 테스트가 NotImplementedError로 실패하는 것이 정상이다.

body 검사는 쿼리 작성 방식의 차이(축약형 `{"match": {"text": "..."}}` / 전체형
`{"match": {"text": {"query": "..."}}}`, `_source`의 excludes / includes)를 모두 받아들이고,
#18 결정(별칭만 사용, embedding 제외, size = top_k)만 고정한다.
"""

from __future__ import annotations

from typing import Any

import pytest

from rfp_pm_agent.schemas.eval import SearchHit
from rfp_pm_agent.search.opensearch import bm25_search, knn_search, search_alias, to_search_hits
from tests.fakes.fake_search_client import RecordingSearchClient

# 소유자 구현 전 표시. NotImplementedError로 실패할 때만 xfail이고, 다른 예외(AssertionError 등)는
# 그대로 실패다. strict=True라 구현 뒤 통과하면 XPASS가 실패로 보고된다 — 그때 이 표시를 지운다
OWNER_TODO = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="#18 학습 모드: 소유자 구현 전"
)

# 코드가 별칭 이름을 하드코딩하지 않고 설정에서 읽는지 보려고 기본값과 다른 이름을 쓴다
TEST_ALIAS = "test_alias_for_search"


@pytest.fixture(autouse=True)
def _alias_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_INDEX_ALIAS", TEST_ALIAS)


def _raw_hit(chunk_id: str, score: float | None, **extra: Any) -> dict[str, Any]:
    source = {"chunk_id": chunk_id, "doc_id": chunk_id.split(":")[0], "text": f"본문 {chunk_id}"}
    source.update(extra)
    return {"_index": "rfp_chunks_v1_kure", "_id": chunk_id, "_score": score, "_source": source}


def _response(*hits: dict[str, Any]) -> dict[str, Any]:
    return {"hits": {"total": {"value": len(hits)}, "hits": list(hits)}}


RESPONSE = _response(
    _raw_hit("d1:block_requirement:b0002", 9.5), _raw_hit("d2:block_requirement:b0001", 3.25)
)


def _assert_source_excludes_embedding(body: dict[str, Any]) -> None:
    """`_source`가 embedding을 돌려받지 않고, 변환에 필요한 필드는 돌려받는 모양인지."""
    assert "_source" in body, "_source를 지정하지 않으면 embedding까지 돌아온다"
    source = body["_source"]
    needed = {"chunk_id", "doc_id", "text"}
    if isinstance(source, dict) and "excludes" in source:
        assert "embedding" in source["excludes"]
        includes = source.get("includes")
        if includes is not None:
            assert needed <= set(includes)
            assert "embedding" not in includes
    elif isinstance(source, dict) and "includes" in source:
        assert needed <= set(source["includes"])
        assert "embedding" not in source["includes"]
    elif isinstance(source, list):
        assert needed <= set(source)
        assert "embedding" not in source
    else:
        pytest.fail(f"_source 모양을 해석할 수 없다: {source!r}")


# --- to_search_hits (변환) ---


def test_to_search_hits_keeps_order_and_fields() -> None:
    hits = to_search_hits(RESPONSE)
    assert hits == [
        SearchHit(
            chunk_id="d1:block_requirement:b0002",
            doc_id="d1",
            score=9.5,
            text="본문 d1:block_requirement:b0002",
        ),
        SearchHit(
            chunk_id="d2:block_requirement:b0001",
            doc_id="d2",
            score=3.25,
            text="본문 d2:block_requirement:b0001",
        ),
    ]


def test_to_search_hits_empty_response() -> None:
    assert to_search_hits(_response()) == []


def test_to_search_hits_rejects_embedding_in_source() -> None:
    with pytest.raises(ValueError, match="embedding"):
        to_search_hits(_response(_raw_hit("d1:block_requirement:b0001", 1.0, embedding=[0.1] * 4)))


def test_to_search_hits_rejects_null_score() -> None:
    with pytest.raises(ValueError, match="_score"):
        to_search_hits(_response(_raw_hit("d1:block_requirement:b0001", None)))


def test_to_search_hits_keeps_negative_score_as_is() -> None:
    """점수 부호는 판단하지 않는다 — 음수 점수 검출은 hybrid 쪽 책임이다 (#18 결정 7)."""
    [hit] = to_search_hits(_response(_raw_hit("d1:block_requirement:b0001", -9549512000.0)))
    assert hit.score == -9549512000.0


def test_search_alias_reads_env() -> None:
    assert search_alias() == TEST_ALIAS


# --- bm25_search (소유자 구현 전에는 NotImplementedError로 실패) ---


@OWNER_TODO
def test_bm25_search_sends_one_request_to_alias() -> None:
    client = RecordingSearchClient(RESPONSE)
    bm25_search(client, "사업 기간", top_k=10)
    assert len(client.calls) == 1
    assert client.calls[0]["index"] == TEST_ALIAS


@OWNER_TODO
def test_bm25_search_body_size_and_source() -> None:
    client = RecordingSearchClient(RESPONSE)
    bm25_search(client, "사업 기간", top_k=7)
    body = client.calls[0]["body"]
    assert body["size"] == 7
    _assert_source_excludes_embedding(body)


@OWNER_TODO
def test_bm25_search_body_is_match_on_text() -> None:
    client = RecordingSearchClient(RESPONSE)
    bm25_search(client, "사업 기간은 언제까지인가", top_k=10)
    query = client.calls[0]["body"]["query"]
    assert set(query) == {"match"}
    assert set(query["match"]) == {"text"}
    text_query = query["match"]["text"]
    sent = text_query["query"] if isinstance(text_query, dict) else text_query
    assert sent == "사업 기간은 언제까지인가"


@OWNER_TODO
def test_bm25_search_returns_converted_hits() -> None:
    client = RecordingSearchClient(RESPONSE)
    assert bm25_search(client, "사업 기간", top_k=10) == to_search_hits(RESPONSE)


# --- knn_search (소유자 구현 전에는 NotImplementedError로 실패) ---

VECTOR = [0.0, 0.6, 0.8]


@OWNER_TODO
def test_knn_search_sends_one_request_to_alias() -> None:
    client = RecordingSearchClient(RESPONSE)
    knn_search(client, VECTOR, top_k=10)
    assert len(client.calls) == 1
    assert client.calls[0]["index"] == TEST_ALIAS


@OWNER_TODO
def test_knn_search_body_size_and_source() -> None:
    client = RecordingSearchClient(RESPONSE)
    knn_search(client, VECTOR, top_k=7)
    body = client.calls[0]["body"]
    assert body["size"] == 7
    _assert_source_excludes_embedding(body)


@OWNER_TODO
def test_knn_search_body_is_knn_on_embedding() -> None:
    client = RecordingSearchClient(RESPONSE)
    knn_search(client, VECTOR, top_k=10)
    query = client.calls[0]["body"]["query"]
    assert set(query) == {"knn"}
    assert set(query["knn"]) == {"embedding"}
    knn = query["knn"]["embedding"]
    assert knn["vector"] == VECTOR
    assert knn["k"] >= 10


@OWNER_TODO
def test_knn_search_returns_converted_hits() -> None:
    client = RecordingSearchClient(RESPONSE)
    assert knn_search(client, VECTOR, top_k=10) == to_search_hits(RESPONSE)
