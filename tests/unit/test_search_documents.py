"""search_documents 단위 테스트 (#18 PR ③).

규칙은 docs/data-design.md 5절 "검색 툴 search_documents". 서버 셋은 모두 가짜다:
- FakeSearch: 집계 요청(`aggs`가 있는 본문)에는 KNOWN_DOCS 중 요청한 doc_id의 bucket을, 그 밖의 요청에는
  정해 둔 hybrid 결과를 돌려준다. 부른 본문을 모두 남긴다
- FakeEmbedder·FakeReranker: 부른 인자를 남기고, 지정한 예외를 낼 수 있다
"""

from __future__ import annotations

import copy
from typing import Any

import httpx
import opensearchpy
import pytest

from rfp_pm_agent.clients.embedding import InputType
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.hybrid import SERVICE_RERANK_N
from rfp_pm_agent.search.opensearch import hybrid_body, search_alias
from rfp_pm_agent.tools.search_documents import SearchDeps, search_documents

VECTOR = [0.5] * 4
KNOWN_DOCS = ["docA", "docB"]


def _source(i: int) -> dict[str, Any]:
    """hybrid 결과 i번째(0부터). 짝수는 PDF 요구사항 청크, 홀수는 HWP 표 블록."""
    if i % 2 == 0:
        return {
            "chunk_id": f"docA:block_requirement:SFR-{i:03d}",
            "doc_id": "docA",
            "bid_title": "A 사업",
            "format": "pdf",
            "requirement_id": f"SFR-{i:03d}",
            "page": i,
            "printed_page": i + 1,
            "block_type": None,
            "text": f"text-{i}",
        }
    return {
        "chunk_id": f"docB:block_requirement:b{i:04d}",
        "doc_id": "docB",
        "bid_title": "B 사업",
        "format": "hwp",
        "requirement_id": None,
        "page": None,
        "printed_page": None,
        "block_type": "table",
        "text": f"text-{i}",
    }


def _response(n: int) -> dict[str, Any]:
    return {
        "hits": {
            "hits": [
                {"_id": _source(i)["chunk_id"], "_score": 0.03 - i * 0.001, "_source": _source(i)}
                for i in range(n)
            ]
        }
    }


class FakeSearch:
    def __init__(self, n_hits: int = SERVICE_RERANK_N, error: Exception | None = None) -> None:
        self.n_hits = n_hits
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"index": index, "body": copy.deepcopy(body)})
        if self.error is not None:
            raise self.error
        if "aggs" in body or "aggregations" in body:
            aggs = body.get("aggs") or body["aggregations"]
            asked = body.get("query", {}).get("terms", {}).get("doc_id", KNOWN_DOCS)
            buckets = [{"key": d, "doc_count": 1} for d in KNOWN_DOCS if d in asked]
            return {
                "hits": {"hits": []},
                "aggregations": {name: {"buckets": buckets} for name in aggs},
            }
        return _response(self.n_hits)

    @property
    def search_bodies(self) -> list[dict[str, Any]]:
        """집계가 아닌 검색 요청 본문."""
        return [
            c["body"]
            for c in self.calls
            if "aggs" not in c["body"] and "aggregations" not in c["body"]
        ]


class FakeEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[list[str], InputType]] = []

    def embed(self, texts: list[str], *, input_type: InputType = "passage") -> list[list[float]]:
        self.calls.append((list(texts), input_type))
        if self.error is not None:
            raise self.error
        return [list(VECTOR) for _ in texts]


class FakeReranker:
    """text-{i}에 scores[i]를 준다. 기본은 i가 클수록 높은 점수(hybrid 순서를 뒤집는다)."""

    def __init__(
        self, scores: dict[int, float] | None = None, error: Exception | None = None
    ) -> None:
        self.scores = scores
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self.error is not None:
            raise self.error
        idx = [int(d.split("-")[1]) for d in documents]
        if self.scores is None:
            return [0.01 * (i + 1) for i in idx]
        return [self.scores[i] for i in idx]


def _deps(
    search: FakeSearch | None = None,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
) -> tuple[SearchDeps, FakeSearch, FakeEmbedder, FakeReranker]:
    s, e, r = search or FakeSearch(), embedder or FakeEmbedder(), reranker or FakeReranker()
    return SearchDeps(search_client=s, embedder=e, reranker=r), s, e, r


def _http_error(status: int, body: dict[str, Any] | str, path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"http://tei.test{path}")
    response = (
        httpx.Response(status, json=body, request=request)
        if isinstance(body, dict)
        else httpx.Response(status, text=body, request=request)
    )
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


TOO_LONG_422 = {
    "error": "Input validation error: `inputs` must have less than 8192 tokens. Given: 30006",
    "error_type": "Validation",
}
# 같은 422지만 입력 길이가 아니라 설정(클라이언트 배치 크기 > 서버 max_client_batch_size)이 원인
BATCH_SIZE_422 = {
    "error": "batch size 64 > maximum allowed batch size 32",
    "error_type": "Validation",
}


# --- 정상 동작 ---


def test_default_call_embeds_once_searches_once_reranks_once_and_returns_top_10() -> None:
    deps, search, embedder, reranker = _deps()

    hits = search_documents("사업 기간", deps=deps)

    assert embedder.calls == [(["사업 기간"], "query")]
    [call] = search.calls  # doc_ids가 없으면 집계 요청도 없다
    assert call["index"] == search_alias()
    assert call["body"] == hybrid_body("사업 기간", VECTOR, SERVICE_RERANK_N)
    assert reranker.calls == [("사업 기간", [f"text-{i}" for i in range(SERVICE_RERANK_N)])]
    # 가짜 리랭커는 뒤쪽일수록 높은 점수 → hybrid 19위가 1위
    assert len(hits) == 10
    assert [h.chunk_id for h in hits] == [_source(i)["chunk_id"] for i in range(19, 9, -1)]
    assert [h.rank for h in hits] == list(range(1, 11))
    assert [h.score for h in hits] == pytest.approx([0.01 * (i + 1) for i in range(19, 9, -1)])


def test_hits_carry_citation_fields_from_index() -> None:
    deps, *_ = _deps(reranker=FakeReranker({i: 1.0 - i * 0.01 for i in range(20)}))

    hits = search_documents("사업 기간", top_k=2, deps=deps)

    assert hits[0].model_dump() == {"rank": 1, "score": 1.0, **_source(0)}
    assert hits[1].model_dump() == {"rank": 2, "score": pytest.approx(0.99), **_source(1)}


@pytest.mark.parametrize("top_k", [1, 3, 20])
def test_top_k_limits_result_count(top_k: int) -> None:
    deps, search, *_ = _deps()
    assert len(search_documents("사업 기간", top_k=top_k, deps=deps)) == top_k
    # 후보 수(리랭크 N=20)는 top_k와 상관없이 같다
    assert search.search_bodies[0]["size"] == SERVICE_RERANK_N


def test_fewer_hits_than_top_k_returns_what_exists() -> None:
    deps, *_ = _deps(search=FakeSearch(n_hits=4))
    assert len(search_documents("사업 기간", top_k=10, deps=deps)) == 4


def test_no_hits_returns_empty_without_calling_reranker() -> None:
    deps, _, _, reranker = _deps(search=FakeSearch(n_hits=0))
    assert search_documents("드론 배송", deps=deps) == []
    assert reranker.calls == []


def test_reranker_ties_keep_hybrid_order() -> None:
    deps, *_ = _deps(reranker=FakeReranker({i: 0.5 for i in range(20)}))
    hits = search_documents("사업 기간", top_k=5, deps=deps)
    assert [h.chunk_id for h in hits] == [_source(i)["chunk_id"] for i in range(5)]


def test_doc_ids_checks_existence_then_sends_filtered_hybrid_body() -> None:
    deps, search, embedder, _ = _deps()

    search_documents("사업 기간", doc_ids=["docB"], deps=deps)

    agg_call, search_call = search.calls
    assert "aggs" in agg_call["body"] or "aggregations" in agg_call["body"]
    assert agg_call["body"].get("size") == 0 and agg_call["index"] == search_alias()
    assert search_call["body"] == hybrid_body("사업 기간", VECTOR, SERVICE_RERANK_N, ["docB"])
    assert "post_filter" not in search_call["body"]
    assert len(embedder.calls) == 1


def test_duplicate_doc_ids_are_checked_once_and_filter_keeps_them() -> None:
    deps, search, *_ = _deps()

    hits = search_documents("사업 기간", doc_ids=["docA", "docA"], deps=deps)

    agg_call, search_call = search.calls
    agg_body = agg_call["body"]
    aggs = agg_body.get("aggs") or agg_body["aggregations"]
    # 확인 요청은 고유 ID 기준: 질의 terms·집계 size 모두 1개
    assert agg_body["query"] == {"terms": {"doc_id": ["docA"]}}
    assert [a["terms"]["size"] for a in aggs.values()] == [1]
    # 검색 필터는 받은 그대로(terms는 값 목록을 집합으로 다루므로 결과가 같다)
    assert search_call["body"] == hybrid_body(
        "사업 기간", VECTOR, SERVICE_RERANK_N, ["docA", "docA"]
    )
    assert len(hits) == 10


def test_unknown_doc_id_is_rejected_before_embedding_or_hybrid_search() -> None:
    deps, search, embedder, reranker = _deps()

    with pytest.raises(ValueError, match="docZ"):
        search_documents("사업 기간", doc_ids=["docA", "docZ"], deps=deps)

    assert len(search.calls) == 1 and search.search_bodies == []  # 집계 요청 1회뿐
    assert embedder.calls == [] and reranker.calls == []


# --- 입력 검증: 서버를 하나도 부르지 않는다 ---


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"query": ""}, "query"),
        ({"query": "   "}, "query"),
        ({"query": "q", "top_k": 0}, "top_k"),
        ({"query": "q", "top_k": 21}, "top_k"),
        ({"query": "q", "top_k": 2.5}, "top_k"),
        ({"query": "q", "top_k": True}, "top_k"),  # bool은 int의 하위 클래스지만 1로 받지 않는다
        ({"query": None}, "query"),
        ({"query": 123}, "query"),
        ({"query": "q", "doc_ids": []}, "doc_ids"),
        ({"query": "q", "doc_ids": [""]}, "doc_ids"),
        ({"query": "q", "doc_ids": ["docA", " "]}, "doc_ids"),
        ({"query": "q", "doc_ids": "docA"}, "doc_ids"),  # 문자열을 글자 목록으로 읽지 않는다
        ({"query": "q", "doc_ids": ("docA",)}, "doc_ids"),
        ({"query": "q", "doc_ids": [1]}, "doc_ids"),
        ({"query": "q", "doc_ids": ["docA", None]}, "doc_ids"),
    ],
)
def test_invalid_input_raises_value_error_without_calling_servers(
    kwargs: dict[str, Any], match: str
) -> None:
    deps, search, embedder, reranker = _deps()
    with pytest.raises(ValueError, match=match):
        search_documents(**kwargs, deps=deps)
    assert search.calls == [] and embedder.calls == [] and reranker.calls == []


# --- 서버 오류 ---


@pytest.mark.parametrize(
    "error",
    [
        opensearchpy.ConnectionError("N/A", "Connection refused", Exception("refused")),
        opensearchpy.ConnectionTimeout("TIMEOUT", "Read timed out", Exception("timeout")),
        opensearchpy.TransportError(500, "internal", {}),
        opensearchpy.TransportError(503, "unavailable", {}),
    ],
    ids=["refused", "timeout", "500", "503"],
)
def test_opensearch_unavailable_is_wrapped(error: Exception) -> None:
    deps, *_ = _deps(search=FakeSearch(error=error))
    with pytest.raises(SearchUnavailableError) as exc:
        search_documents("사업 기간", deps=deps)
    assert exc.value.service == "opensearch"
    assert exc.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        opensearchpy.ConnectionError("N/A", "Connection refused", Exception("refused")),
        opensearchpy.TransportError(503, "unavailable", {}),
    ],
    ids=["refused", "503"],
)
def test_opensearch_unavailable_during_doc_id_check_is_wrapped(error: Exception) -> None:
    deps, search, embedder, _ = _deps(search=FakeSearch(error=error))
    with pytest.raises(SearchUnavailableError) as exc:
        search_documents("사업 기간", doc_ids=["docA"], deps=deps)
    assert exc.value.service == "opensearch" and exc.value.__cause__ is error
    assert len(search.calls) == 1 and embedder.calls == []  # 확인 요청에서 끝난다


@pytest.mark.parametrize(
    "error",
    [
        opensearchpy.NotFoundError(404, "index_not_found_exception", {}),
        opensearchpy.RequestError(400, "parsing_exception", {}),
        # ConnectionError가 아닌데 상태 코드가 정수가 아닌 경우: 크기 비교 없이 그대로 올린다
        opensearchpy.TransportError("N/A", "sniff failed", {}),
    ],
    ids=["404", "400", "status-not-int"],
)
def test_opensearch_4xx_propagates_unwrapped(error: Exception) -> None:
    deps, *_ = _deps(search=FakeSearch(error=error))
    with pytest.raises(type(error)) as exc:
        search_documents("사업 기간", deps=deps)
    assert exc.value is error


TEI_UNAVAILABLE = [
    httpx.ConnectError("Connection refused"),
    httpx.ReadTimeout("timed out"),
    _http_error(503, {"error": "overloaded"}, "/x"),
]


@pytest.mark.parametrize("error", TEI_UNAVAILABLE, ids=["refused", "timeout", "503"])
def test_embed_unavailable_is_wrapped(error: Exception) -> None:
    deps, search, _, _ = _deps(embedder=FakeEmbedder(error=error))
    with pytest.raises(SearchUnavailableError) as exc:
        search_documents("사업 기간", deps=deps)
    assert exc.value.service == "tei-embed" and exc.value.__cause__ is error
    assert search.calls == []


@pytest.mark.parametrize("error", TEI_UNAVAILABLE, ids=["refused", "timeout", "503"])
def test_rerank_unavailable_fails_even_after_hybrid_succeeded(error: Exception) -> None:
    deps, search, _, _ = _deps(reranker=FakeReranker(error=error))
    with pytest.raises(SearchUnavailableError) as exc:
        search_documents("사업 기간", deps=deps)
    assert exc.value.service == "tei-rerank" and exc.value.__cause__ is error
    assert len(search.search_bodies) == 1  # hybrid는 성공했지만 리랭크 없는 결과를 돌려주지 않는다


@pytest.mark.parametrize(
    ("make_deps", "path"),
    [
        (lambda e: _deps(embedder=FakeEmbedder(error=e))[0], "/embed"),
        (lambda e: _deps(reranker=FakeReranker(error=e))[0], "/rerank"),
    ],
    ids=["embed", "rerank"],
)
@pytest.mark.parametrize(
    ("status", "body", "detail"),
    [
        (422, TOO_LONG_422, "8192"),
        (413, "Failed to buffer the request body: length limit exceeded", "length limit"),
    ],
    ids=["422-tokens", "413-body"],
)
def test_too_long_input_becomes_value_error(
    make_deps: Any, path: str, status: int, body: Any, detail: str
) -> None:
    error = _http_error(status, body, path)
    with pytest.raises(ValueError, match="질의가 너무 깁니다") as exc:
        search_documents("사업 기간", deps=make_deps(error))
    assert detail in str(exc.value)
    assert not isinstance(exc.value, SearchUnavailableError)
    assert exc.value.__cause__ is error


@pytest.mark.parametrize(
    "make_deps",
    [
        lambda e: _deps(embedder=FakeEmbedder(error=e))[0],
        lambda e: _deps(reranker=FakeReranker(error=e))[0],
    ],
    ids=["embed", "rerank"],
)
@pytest.mark.parametrize(
    ("status", "body"),
    [(400, {"error": "bad request", "error_type": "Validation"}), (422, BATCH_SIZE_422)],
    ids=["400", "422-not-length"],
)
def test_other_tei_4xx_propagates_unwrapped(
    make_deps: Any, status: int, body: dict[str, Any]
) -> None:
    error = _http_error(status, body, "/x")
    with pytest.raises(httpx.HTTPStatusError) as exc:
        search_documents("사업 기간", deps=make_deps(error))
    assert exc.value is error
