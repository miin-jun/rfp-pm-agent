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
from collections.abc import Mapping
from typing import Any, Protocol

from rfp_pm_agent.config import OpenSearchConfig
from rfp_pm_agent.schemas.eval import SearchHit

# 검색 응답에서 빼는 필드. 벡터는 검색에 쓰고 돌려받지 않는다
EXCLUDED_SOURCE_FIELDS = ("embedding",)
KNN_CANDIDATES = 50
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


def bm25_search(client: SearchClient, query: str, top_k: int) -> list[SearchHit]:
    """질문 문자열로 `text` 필드를 BM25(nori `korean` 분석기) 검색해 상위 top_k개를 돌려준다.

    인자
    - client: OpenSearch 클라이언트(`opensearchpy.OpenSearch` 또는 같은 `search`를 가진 가짜)
    - query: 사용자 질문 그대로. 질문 어미 처리(#70)는 여기서 하지 않는다
    - top_k: 돌려받을 결과 수(= 요청 `size`)

    요청: `client.search(index=search_alias(), body=...)` 한 번. body는 `size=top_k`,
    `_source`에서 `embedding` 제외, `query`는 아래 둘 중 하나다.
    - 질문에 요구사항 ID가 없으면(`extract_requirement_ids`가 빈 목록) `text` 필드의 `match` —
      #75 이전과 body 전체가 같다
    - ID가 있으면 `bool`: must = `text` 필드의 `match`(질문 원문 그대로), should = ID마다
      `constant_score`(filter = `requirement_id` term, boost `REQUIREMENT_ID_BOOST`=100). ID가 일치하는
      청크는 match 점수에 정확히 100이 더해진다. must라서 text match가 0인 청크는 ID가 맞아도 나오지 않는다
    - 가산은 문서를 가리지 않는다. 같은 ID가 여러 문서에 있으면(예: SFR-013은 4개 문서) 다른 문서의
      같은 ID 청크도 함께 올라온다(ADR-0003 한계)
    반환: `to_search_hits(응답)` — `_score` 내림차순.
    """
    body = {
        "size": top_k,
        "_source": {"excludes": list(EXCLUDED_SOURCE_FIELDS)},
        "query": {"match": {"text": query}},
    }
    requirement_ids = extract_requirement_ids(query)
    if requirement_ids:
        body["query"] = {
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
    response = client.search(index=search_alias(), body=body)
    return to_search_hits(response)


def knn_search(client: SearchClient, query_vector: list[float], top_k: int) -> list[SearchHit]:
    """질의 벡터로 `embedding` 필드를 k-NN(lucene HNSW, cosinesimil) 검색해 상위 top_k개를 돌려준다.

    인자
    - client: OpenSearch 클라이언트(`opensearchpy.OpenSearch` 또는 같은 `search`를 가진 가짜)
    - query_vector: 색인과 같은 임베딩 모델로 만든 질의 벡터(`input_type="query"`, 1024차원).
      임베딩은 호출하는 쪽에서 한다 — 이 함수는 TEI를 부르지 않는다
    - top_k: 돌려받을 결과 수(= 요청 `size`). knn의 `k`는 top_k 이상이어야 한다

    요청: `client.search(index=search_alias(), body=...)` 한 번. body는 `size=top_k`,
    `_source`에서 `embedding` 제외, `query`는 `{"knn": {"embedding": {"vector": ..., "k": ...}}}`.
    반환: `to_search_hits(응답)` — `_score` 내림차순.
    """
    body = {
        "size": top_k,
        "_source": {"excludes": list(EXCLUDED_SOURCE_FIELDS)},
        "query": {"knn": {"embedding": {"vector": query_vector, "k": KNN_CANDIDATES}}},
    }
    response = client.search(index=search_alias(), body=body)
    return to_search_hits(response)
