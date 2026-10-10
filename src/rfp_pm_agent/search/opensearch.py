"""OpenSearch 단일 검색 — BM25(nori)와 k-NN (이슈 #18 PR ①).

#16까지의 `search/bm25.py`(rank_bm25)·`search/dense.py`(메모리 코사인)는 #13·#16 재현용으로
남기고, 서비스 검색은 여기서 색인된 OpenSearch 인덱스를 부른다. 하이브리드(RRF)와 리랭크는
PR ②에서 이 두 함수 위에 만든다.

규칙 (#18 "결정 (2026-10-03)", docs/data-design.md 5절)
- 인덱스 이름은 쓰지 않고 별칭만 부른다. 별칭은 `search_alias()`(= `OPENSEARCH_INDEX_ALIAS`)다.
  실제 인덱스(`rfp_chunks_v1_kure`)를 부르면 모델 교체(별칭 전환) 뒤에도 옛 인덱스를 검색한다
- `_source`에서 `embedding`을 뺀다. 빼지 않으면 결과 1건마다 실수 1024개가 응답에 실린다.
  `to_search_hits`는 `embedding`이 든 결과를 받으면 ValueError로 멈춘다
- 결과는 `schemas/eval.py`의 `SearchHit` 목록이다. 순위는 목록 안 위치, `score`는 OpenSearch
  `_score`(BM25 점수 또는 k-NN 유사도 — 두 척도는 서로 비교할 수 없다)
- 질문에 요구사항 ID(예: SFR-013)가 있으면 BM25에 `requirement_id` 정확 일치 가산(+100)을 붙인다
  (#75, docs/adr/0003-requirement-id-boost.md). nori가 `SFR-013`을 `sfr`/`013`으로 나눠 BM25만으로는
  ID를 구분하지 못하기 때문이다

bm25_search·knn_search는 별칭으로 OpenSearch에 검색 요청을 보내고 SearchHit 목록을 돌려준다.
테스트가 기대하는 본문 모양은 tests/unit/test_search_opensearch.py와
tests/unit/test_search_requirement_ids.py(ID 가산)에 있다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from rfp_pm_agent.config import OpenSearchConfig
from rfp_pm_agent.schemas.eval import SearchHit
from rfp_pm_agent.schemas.search import DocumentHit

# 검색 응답에서 빼는 필드. 벡터는 검색에 쓰고 돌려받지 않는다
EXCLUDED_SOURCE_FIELDS = ("embedding",)
# 검색기별 후보 수 — knn의 k이자 hybrid의 pagination_depth (#18 결정 4: 측정 전 고정)
KNN_CANDIDATES = 50
# RRF 순위 상수 (#18 결정 4: 측정 전 고정). OpenSearch 기본값도 60이라, 파이프라인에서 잘못된 위치에
# 써서 값이 무시돼도 결과가 같다 — 위치는 테스트로 고정한다(hybrid_search docstring)
RRF_RANK_CONSTANT = 60
# 요구사항 ID 패턴과 ID 일치 가산값 — ADR-0003: 측정 전 고정, 조정 금지
REQUIREMENT_ID_PATTERN = re.compile(r"(?<![A-Za-z])[A-Za-z]{2,5}-[0-9]{2,4}(?![0-9])")
REQUIREMENT_ID_BOOST = 100


class SearchClient(Protocol):
    """`opensearchpy.OpenSearch.search`의 쓰는 부분만. 단위 테스트는 기록용 가짜를 넣는다."""

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]: ...


def search_alias() -> str:
    """검색이 부르는 별칭 이름 (`OPENSEARCH_INDEX_ALIAS`, 기본 `rfp_chunks`)."""
    return OpenSearchConfig.from_env().index_alias


def to_search_hits(response: Mapping[str, Any]) -> list[SearchHit]:
    """OpenSearch 검색 응답의 `hits.hits`를 순서 그대로 `SearchHit` 목록으로 바꾼다.

    - chunk_id·doc_id·text는 `_source`에서, score는 `_score`에서 읽는다
    - `_source`에 `embedding`이 있으면 ValueError — 요청에서 `_source` 제외를 빠뜨렸다는 뜻이다
    - `_score`가 null이면 ValueError — 정렬(sort)을 지정한 요청처럼 점수를 계산하지 않은
      응답이라 순위 점수로 쓸 수 없다
    - 결과가 없으면 빈 목록
    """
    hits: list[SearchHit] = []
    for raw in response["hits"]["hits"]:
        source = raw["_source"]
        leaked = [f for f in EXCLUDED_SOURCE_FIELDS if f in source]
        if leaked:
            raise ValueError(f"검색 응답 _source에 {leaked}가 들어 있다 (_id={raw.get('_id')})")
        if raw.get("_score") is None:
            raise ValueError(f"검색 응답에 _score가 없다 (_id={raw.get('_id')})")
        hits.append(
            SearchHit(
                chunk_id=source["chunk_id"],
                doc_id=source["doc_id"],
                score=float(raw["_score"]),
                text=source["text"],
            )
        )
    return hits


def to_document_hits(response: Mapping[str, Any]) -> list[DocumentHit]:
    """OpenSearch 검색 응답의 `hits.hits`를 순서 그대로 `DocumentHit` 목록으로 바꾼다 (#18 PR ③).

    `to_search_hits`와 같은 검사(`embedding`이 섞였거나 `_score`가 null이면 ValueError)를 하고,
    출처 필드(bid_title·format·requirement_id·page·printed_page·block_type)를 `_source`에서 옮긴다.
    rank는 응답 안 위치(1부터), score는 `_score`(hybrid면 RRF 점수)다 — 리랭크한 뒤에는 호출하는 쪽이
    rank·score를 다시 매긴다. `_source`에 출처 필드가 없으면(#81 이전 매핑의 인덱스) KeyError.
    """
    to_search_hits(response)  # 같은 검사를 한 곳에서 한다
    return [
        DocumentHit(
            rank=rank,
            chunk_id=raw["_source"]["chunk_id"],
            doc_id=raw["_source"]["doc_id"],
            bid_title=raw["_source"]["bid_title"],
            format=raw["_source"]["format"],
            requirement_id=raw["_source"]["requirement_id"],
            page=raw["_source"]["page"],
            printed_page=raw["_source"]["printed_page"],
            block_type=raw["_source"]["block_type"],
            text=raw["_source"]["text"],
            score=float(raw["_score"]),
        )
        for rank, raw in enumerate(response["hits"]["hits"], start=1)
    ]


def extract_requirement_ids(query: str) -> list[str]:
    """질문에서 요구사항 ID를 찾아 대문자로 바꾼 목록을 돌려준다 (이슈 #75).

    - 패턴: 영문 2~5자 + 하이픈 + ASCII 숫자 2~4자리 (예: SFR-013, SER-001). 전각 숫자는 제외
    - 덩어리 전체에 적용한다: `ABCDEF-013`·`SFR-01234`에서 일부만 떼어 ID로 보지 않는다
    - 한국어 조사·괄호·문장부호가 붙어도 찾는다 (`SFR-013의` → `SFR-013`)
    - 중복은 빼고 질문에 처음 나온 순서를 지킨다. ID가 없으면 빈 목록
    - 패턴만 보고 뽑으며 인덱스에 그 ID가 있는지는 확인하지 않는다. `COVID-19`·`ISO-9001`도 뽑히지만
      일치하는 청크가 없어 검색 결과는 바뀌지 않는다
    - 뽑지 않는 표기: 하이픈 변형(`SFR–013` en dash, `SFR－013` 전각, `SFR 013`, `SFR_013`).
      범위 표기 `SFR-013~015`·`SFR-013, 014`는 앞 ID(`SFR-013`)만 뽑는다. `SFR-013-01`은 `SFR-013`으로 뽑는다
    - 규칙 근거: #75 "결정 (2026-10-06)", docs/adr/0003-requirement-id-boost.md
    """
    ids: list[str] = []
    for match in REQUIREMENT_ID_PATTERN.finditer(query):
        req_id = match.group().upper()
        if req_id not in ids:
            ids.append(req_id)
    return ids


def bm25_query(query: str, doc_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """질문 문자열로 BM25(nori `korean` 분석기) 검색 요청의 `query` 절을 만든다.

    `bm25_search`와 하이브리드 검색(PR ②)의 BM25 하위 쿼리가 함께 쓴다. 요청은 보내지 않는다.
    - query: 사용자 질문 그대로. 질문 어미 처리(#70)는 여기서 하지 않는다
    - 질문에 요구사항 ID가 없으면(`extract_requirement_ids`가 빈 목록) `text` 필드의 `match` —
      #75 이전과 같다
    - ID가 있으면 `bool`: must = `text` 필드의 `match`(질문 원문 그대로), should = ID마다
      `constant_score`(filter = `requirement_id` term, boost `REQUIREMENT_ID_BOOST`=100). ID가 일치하는
      청크는 match 점수에 정확히 100이 더해진다. must라서 text match가 0인 청크는 ID가 맞아도 나오지 않는다
    - 가산은 문서를 가리지 않는다. 같은 ID가 여러 문서에 있으면(예: SFR-013은 4개 문서) 다른 문서의
      같은 ID 청크도 함께 올라온다(ADR-0003 한계)
    - hybrid 쿼리의 하위 쿼리로 넣어도 그대로 동작한다(2026-10-07 OpenSearch 2.19.1 실측)

    doc_ids (#18 PR ③)
    - None이면 위와 같다(필터 없음). 기존 호출(bm25_search·knn_search·평가)은 모두 None이다
    - 목록이면 `{"terms": {"doc_id": list(doc_ids)}}` 필터를 붙인다. 빈 목록이면 ValueError —
      빈 terms는 아무것도 맞지 않아 "결과 없음"과 구별되지 않는다
      - ID 없음: `{"bool": {"must": [{"match": {"text": query}}], "filter": [필터]}}`
      - ID 있음: 위 `bool`(must·should)에 `"filter": [필터]`를 더한다
      filter 절은 점수에 영향을 주지 않으므로, 필터를 통과한 청크의 점수는 필터가 없을 때와 같다
    """
    if doc_ids is not None:
        if not doc_ids:
            raise ValueError("doc_ids가 빈 목록이다. 전체 검색은 None을 넘긴다")
        doc_filter = {"terms": {"doc_id": list(doc_ids)}}
        base = bm25_query(query)
        if "match" in base:
            return {"bool": {"must": [base], "filter": [doc_filter]}}
        return {"bool": {**base["bool"], "filter": [doc_filter]}}
    requirement_ids = extract_requirement_ids(query)
    if not requirement_ids:
        return {"match": {"text": query}}
    return {
        "bool": {
            "must": [{"match": {"text": query}}],
            "should": [
                {
                    "constant_score": {
                        "filter": {"term": {"requirement_id": req_id}},
                        "boost": REQUIREMENT_ID_BOOST,
                    }
                }
                for req_id in requirement_ids
            ],
        }
    }


def knn_query(query_vector: list[float], doc_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """질의 벡터로 k-NN 검색 요청의 `query` 절을 만든다. `k`는 `KNN_CANDIDATES`(50) 고정.

    `knn_search`와 하이브리드 검색(PR ②)의 k-NN 하위 쿼리가 함께 쓴다. 요청은 보내지 않는다.
    - query_vector: 색인과 같은 임베딩 모델로 만든 질의 벡터(`input_type="query"`, 1024차원)

    doc_ids (#18 PR ③)
    - None이면 위와 같다(필터 없음). 기존 호출(bm25_search·knn_search·평가)은 모두 None이다
    - 목록이면 `{"terms": {"doc_id": list(doc_ids)}}` 필터를 붙인다. 빈 목록이면 ValueError —
      빈 terms는 아무것도 맞지 않아 "결과 없음"과 구별되지 않는다
      - `{"knn": {"embedding": {"vector": query_vector, "k": KNN_CANDIDATES, "filter": 필터}}}`
      lucene 엔진은 필터를 HNSW 탐색 안에서 적용해(efficient filtering) 필터를 통과한 문서 중 k개를 찾는다.
      요청 `post_filter`는 순위가 달라져 쓰지 않는다 — 필터가 하위 쿼리의 후보 선정 뒤에 적용되는 것으로
      보이나 확인하지 않음(2026-10-08 hybrid 요청 실측: 57문항 모두 10건 반환, 순서가 기준과 같은 문항
      20/57 — docs/adr/0005-search-documents-contract.md)
    """
    knn: dict[str, Any] = {"vector": query_vector, "k": KNN_CANDIDATES}
    if doc_ids is not None:
        if not doc_ids:
            raise ValueError("doc_ids가 빈 목록이다. 전체 검색은 None을 넘긴다")
        knn["filter"] = {"terms": {"doc_id": list(doc_ids)}}
    return {"knn": {"embedding": knn}}


def bm25_search(client: SearchClient, query: str, top_k: int) -> list[SearchHit]:
    """질문 문자열로 `text` 필드를 BM25(nori `korean` 분석기) 검색해 상위 top_k개를 돌려준다.

    인자
    - client: OpenSearch 클라이언트(`opensearchpy.OpenSearch` 또는 같은 `search`를 가진 가짜)
    - query: 사용자 질문 그대로. 질문 어미 처리(#70)는 여기서 하지 않는다
    - top_k: 돌려받을 결과 수(= 요청 `size`)

    요청: `client.search(index=search_alias(), body=...)` 한 번. body는 `size=top_k`,
    `_source`에서 `embedding` 제외, `query`는 `bm25_query(query)` — 요구사항 ID가 있으면
    `requirement_id` 일치 청크에 +100을 더하는 `bool`, 없으면 `text` 필드의 `match`.
    반환: `to_search_hits(응답)` — `_score` 내림차순.
    """
    body = {
        "size": top_k,
        "_source": {"excludes": list(EXCLUDED_SOURCE_FIELDS)},
        "query": bm25_query(query),
    }
    response = client.search(index=search_alias(), body=body)
    return to_search_hits(response)


def knn_search(client: SearchClient, query_vector: list[float], top_k: int) -> list[SearchHit]:
    """질의 벡터로 `embedding` 필드를 k-NN(lucene HNSW, cosinesimil) 검색해 상위 top_k개를 돌려준다.

    인자
    - client: OpenSearch 클라이언트(`opensearchpy.OpenSearch` 또는 같은 `search`를 가진 가짜)
    - query_vector: 색인과 같은 임베딩 모델로 만든 질의 벡터(`input_type="query"`, 1024차원).
      임베딩은 호출하는 쪽에서 한다 — 이 함수는 TEI를 부르지 않는다
    - top_k: 돌려받을 결과 수(= 요청 `size`). `KNN_CANDIDATES`(50) 이하. 넘으면 요청을 보내지
      않고 ValueError — knn의 `k`는 50 고정(#18 결정 4)이라 size가 k보다 크면 후보 수를 넘는 결과를
      요구하게 된다. k를 top_k에 맞춰 늘리면 HNSW 후보가 바뀌어 상위 10까지 달라질 수 있어 그렇게 하지 않는다
      (#18 PR ② 결정 5A)

    요청: `client.search(index=search_alias(), body=...)` 한 번. body는 `size=top_k`,
    `_source`에서 `embedding` 제외, `query`는 `knn_query(query_vector)`.
    반환: `to_search_hits(응답)` — `_score` 내림차순.
    """
    if top_k > KNN_CANDIDATES:
        raise ValueError(f"top_k({top_k})는 KNN_CANDIDATES({KNN_CANDIDATES}) 이하여야 한다")
    body = {
        "size": top_k,
        "_source": {"excludes": list(EXCLUDED_SOURCE_FIELDS)},
        "query": knn_query(query_vector),
    }
    response = client.search(index=search_alias(), body=body)
    return to_search_hits(response)


def hybrid_body(
    query: str,
    query_vector: list[float],
    top_k: int,
    doc_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """hybrid 검색 요청 본문을 만든다. 요청은 보내지 않는다.

    `hybrid_search`와 `search_documents`(#18 PR ③ — 응답의 출처 필드가 필요해 본문만 받아 직접 보낸다)가
    함께 쓴다. 인자는 `hybrid_search`와 같다. top_k가 `KNN_CANDIDATES`(50)를 넘으면 ValueError.

    본문
    - `size=top_k`, `_source`에서 `embedding` 제외
    - `query.hybrid.queries` = [`bm25_query(query)`, `knn_query(query_vector)`] (이 순서)
    - `query.hybrid.pagination_depth` = `KNN_CANDIDATES` — 하위 쿼리마다 가져올 후보 수. 빼면 하위
      쿼리마다 size개만 가져와 "검색기별 후보 50"이 지켜지지 않는다(2026-10-07 실측: 상위 10 중 5개만 일치)
    - `search_pipeline`(요청마다 보내는 임시 파이프라인, 클러스터에 만들지 않는다) =
      `{"phase_results_processors": [{"score-ranker-processor": {"combination":
      {"technique": "rrf", "parameters": {"rank_constant": RRF_RANK_CONSTANT}}}}]}`.
      `rank_constant`는 반드시 `combination.parameters` 안에 둔다 — `combination` 바로 아래에 두면
      (2.19 공식 문서 예시 형식) 2.19.1은 에러 없이 무시하고 기본값 60을 쓴다(2026-10-07 실측)

    doc_ids (#18 PR ③)
    - None이면 위와 같다
    - 목록이면 하위 쿼리 **각각에** 필터를 넣는다: `queries` = [`bm25_query(query, doc_ids)`,
      `knn_query(query_vector, doc_ids)`]. 나머지 본문은 같다
    - 본문에 `post_filter`를 넣지 않는다 — 200을 주지만 순위가 달라진다(필터가 하위 쿼리의 후보 선정 뒤에
      적용되는 것으로 보이나 확인하지 않음).
      `query.hybrid.filter`도 넣지 않는다 — 2.19.1은 400("Field is not supported by [hybrid] query")
      (둘 다 2026-10-08 실측, data-design.md 5절)
    """
    if top_k > KNN_CANDIDATES:
        raise ValueError(f"top_k({top_k})는 KNN_CANDIDATES({KNN_CANDIDATES}) 이하여야 한다")
    return {
        "size": top_k,
        "_source": {"excludes": list(EXCLUDED_SOURCE_FIELDS)},
        "query": {
            "hybrid": {
                "queries": [bm25_query(query, doc_ids), knn_query(query_vector, doc_ids)],
                "pagination_depth": KNN_CANDIDATES,
            }
        },
        "search_pipeline": {
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "parameters": {"rank_constant": RRF_RANK_CONSTANT},
                        }
                    }
                }
            ]
        },
    }


def hybrid_search(
    client: SearchClient,
    query: str,
    query_vector: list[float],
    top_k: int,
    doc_ids: Sequence[str] | None = None,
) -> list[SearchHit]:
    """BM25와 k-NN을 OpenSearch hybrid 쿼리로 한 번에 보내고 RRF로 합친 상위 top_k개를 돌려준다.

    인자
    - client: OpenSearch 클라이언트(`opensearchpy.OpenSearch` 또는 같은 `search`를 가진 가짜)
    - query: 사용자 질문 그대로. BM25 하위 쿼리는 `bm25_query(query, doc_ids)`(요구사항 ID 가산 포함)
    - query_vector: 색인과 같은 임베딩 모델로 만든 질의 벡터. k-NN 하위 쿼리는
      `knn_query(query_vector, doc_ids)`
    - top_k: 돌려받을 결과 수(= 요청 `size`). `KNN_CANDIDATES`(50) 이하.
      넘으면 요청을 보내지 않고 ValueError — 후보 50개를 합친 결과보다 많이 받을 수 없다
    - doc_ids: 검색할 문서(None이면 전체). `hybrid_body`에 그대로 넘긴다 — 필터는 하위 쿼리마다 들어간다

    요청: `client.search(index=search_alias(), body=hybrid_body(query, query_vector, top_k, doc_ids))` 한 번.
    본문 규칙은 `hybrid_body`.
    반환: `to_search_hits(응답)` — RRF 점수 내림차순. 점수는 0 초과 2/(RRF_RANK_CONSTANT+1) 이하.
    """
    body = hybrid_body(query, query_vector, top_k, doc_ids)
    response = client.search(index=search_alias(), body=body)
    return to_search_hits(response)
