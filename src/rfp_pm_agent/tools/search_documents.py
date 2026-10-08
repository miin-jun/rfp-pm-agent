"""문서 검색 툴 `search_documents` (이슈 #18 PR ③).

Agent(#22)·MCP 서버(#24)·`/search`(#23)가 함께 부르는 RFP 문서 검색 함수다. 인자·반환·오류 규칙은
docs/data-design.md 5절 "검색 툴 search_documents"가 기준이고, 서비스 기본값(하이브리드 RRF k=60·후보 50
+ 리랭크 N=20)은 docs/adr/0004-hybrid-rerank-default.md다.

- `SearchDeps`: 검색 클라이언트·임베더·리랭커 묶음. LLM에게 보이는 인자(query·top_k·doc_ids)와 분리해
  Agent·MCP 어댑터가 미리 묶어 넘긴다. 테스트는 가짜 클라이언트를 넣는다

`search_documents`의 docstring은 LLM에게 보이는 툴 설명이다. 내부 처리 순서와 예외 바꾸기 규칙은
함수 안 주석과 이 모듈의 `_call_tei`·`_validate`·`_check_doc_ids`에 둔다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx
import opensearchpy
from opensearchpy import OpenSearch

from rfp_pm_agent.clients.embedding import EmbeddingClient, TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import RerankerClient, TEIRerankerClient
from rfp_pm_agent.config import ClientsConfig, OpenSearchConfig
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K, MAX_TOP_K, SERVICE_RERANK_N
from rfp_pm_agent.search.opensearch import (
    SearchClient,
    hybrid_body,
    search_alias,
    to_document_hits,
)


@dataclass(frozen=True)
class SearchDeps:
    """`search_documents`가 부르는 세 서버의 클라이언트.

    - search_client: OpenSearch(`search`만 쓴다). 별칭(`OPENSEARCH_INDEX_ALIAS`)으로 검색한다
    - embedder: 질의 임베딩(TEI `/embed`, `input_type="query"`). 색인과 같은 모델이어야 한다
    - reranker: 리랭커(TEI `/rerank`)
    """

    search_client: SearchClient
    embedder: EmbeddingClient
    reranker: RerankerClient

    @classmethod
    def from_env(cls) -> SearchDeps:
        """환경변수 설정으로 실제 클라이언트를 만든다. 서버에 연결하지는 않는다(첫 요청 때 연결).

        OpenSearch는 `OPENSEARCH_URL`·`OPENSEARCH_TIMEOUT_S`, TEI는 `EMBED_*`·`RERANK_*`(`ClientsConfig`).
        """
        os_config = OpenSearchConfig.from_env()
        clients = ClientsConfig.from_env()
        return cls(
            search_client=OpenSearch(hosts=[os_config.url], timeout=os_config.timeout_s),
            embedder=TEIEmbeddingClient(clients),
            reranker=TEIRerankerClient(clients),
        )


# TEI 1.9.4가 토큰 한도를 넘는 입력에 주는 422 본문 문구(2026-10-08 실측). 같은 422라도 이 문구가 없으면
# (예: 배치 크기 초과) 입력이 아니라 설정이 원인이라 ValueError로 바꾸지 않는다
_TOO_LONG_MARKER = "must have less than"


def _call_tei[T](service: Literal["tei-embed", "tei-rerank"], call: Callable[[], T]) -> T:
    """TEI 호출의 예외를 data-design.md 5절 오류 표대로 바꾼다.

    - 413(요청 본문 한도 초과), 422 중 토큰 한도 초과 → ValueError("질의가 너무 깁니다: " + TEI 본문)
    - 5xx, 연결 거절·시간 초과(`httpx.TransportError`) → SearchUnavailableError(service)
    - 그 밖의 4xx(토큰 한도가 아닌 422 포함) → 원래 예외 그대로
    """
    try:
        return call()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 413 or (status == 422 and _TOO_LONG_MARKER in e.response.text):
            raise ValueError(f"질의가 너무 깁니다: {e.response.text}") from e
        if status >= 500:
            raise SearchUnavailableError(service, str(e)) from e
        raise
    except httpx.TransportError as e:
        raise SearchUnavailableError(service, str(e)) from e


def _validate(query: str, top_k: int, doc_ids: list[str] | None) -> None:
    """입력 검증. 실패하면 ValueError — 서버를 부르기 전에 끝난다.

    LLM이 만든 인자는 타입 힌트와 다를 수 있어 타입도 직접 확인한다. 타입 오류도 TypeError가 아니라
    ValueError로 통일한다(Agent·MCP가 입력 오류 하나로 LLM에 돌려주도록, data-design.md 5절).
    """
    if not isinstance(query, str):
        raise ValueError(f"query는 문자열이어야 한다: {query!r}")  # noqa: TRY004 — 입력 오류는 ValueError로 통일(data-design 5절)
    if not query.strip():
        raise ValueError("query가 비어 있다")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k는 정수여야 한다: {top_k!r}")  # noqa: TRY004 — 입력 오류는 ValueError로 통일(data-design 5절)
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k({top_k})는 1 이상 {MAX_TOP_K} 이하여야 한다")
    if doc_ids is not None:
        if not isinstance(doc_ids, list):
            raise ValueError(f"doc_ids는 문자열 목록이어야 한다: {doc_ids!r}")
        if not doc_ids:
            raise ValueError("doc_ids가 빈 목록이다. 전체 검색은 None을 넘긴다")
        if any(not isinstance(d, str) for d in doc_ids):
            raise ValueError(f"doc_ids의 원소는 문자열이어야 한다: {doc_ids!r}")
        if any(not d.strip() for d in doc_ids):
            raise ValueError("doc_ids에 빈 문자열이 있다")


def _check_doc_ids(client: SearchClient, doc_ids: list[str]) -> None:
    """인덱스에 없는 doc_id가 있으면 ValueError(없는 ID를 메시지에). 요청 1회(`size: 0` + terms 집계).

    중복은 지우고 고유 ID 기준으로 묻는다. 질의가 그 ID들로 제한되므로 버킷 수는 고유 ID 수를 넘지 않아
    집계 size = 고유 ID 수면 잘리지 않는다.
    """
    unique_ids = list(dict.fromkeys(doc_ids))
    body = {
        "size": 0,
        "query": {"terms": {"doc_id": unique_ids}},
        "aggs": {"found": {"terms": {"field": "doc_id", "size": len(unique_ids)}}},
    }
    response = client.search(index=search_alias(), body=body)
    found = {b["key"] for b in response["aggregations"]["found"]["buckets"]}
    missing = [d for d in unique_ids if d not in found]
    if missing:
        raise ValueError(f"인덱스에 없는 doc_id: {missing}")


def search_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_ids: list[str] | None = None,
    *,
    deps: SearchDeps,
) -> list[DocumentHit]:
    """RFP(제안요청서) 문서에서 질문과 관련된 부분(청크)을 찾아 관련도 순으로 돌려준다.

    인자
    - query: 찾을 내용을 담은 질문이나 키워드(한국어 문자열). 빈 문자열·공백만이면 ValueError.
      요구사항 ID(예: SFR-013)를 넣으면 그 ID의 요구사항 청크가 위로 온다
    - top_k: 돌려받을 결과 수. 1 이상 20 이하의 정수(기본 10). 범위를 벗어나면 ValueError(잘라서 맞추지 않는다)
    - doc_ids: 검색할 문서를 좁힐 때 넣는 doc_id 문자열 목록. 이전 검색 결과의 `doc_id` 값을 그대로 넣는다.
      None(기본)이면 전체 문서. 빈 목록·빈 문자열·인덱스에 없는 doc_id가 있으면 ValueError(없는 ID를
      메시지에 적는다)

    반환: 결과 목록 — 관련도(score, 0~1) 내림차순, rank 1부터. 각 결과에 출처(doc_id·bid_title·format·
    requirement_id·page·printed_page·block_type)와 본문 text가 있다. 결과가 top_k보다 적거나 0개일 수 있다
    (0개는 오류가 아니라 "찾지 못함"이다). score는 한 번의 호출 안에서만 비교한다.

    오류
    - ValueError: 인자가 잘못됐다. "질의가 너무 깁니다"면 질의를 줄여 다시 부른다
    - SearchUnavailableError: 검색 서버가 응답하지 않는다. 잠시 뒤 다시 시도하거나 사용자에게 알린다.
      인자를 바꿔도 해결되지 않는다

    deps는 서버 클라이언트 묶음으로, LLM에게 보이지 않는다(Agent·MCP 어댑터가 넘긴다).
    """
    # 처리 순서와 오류 규칙: docs/data-design.md 5절 "검색 툴 search_documents"
    # 1) 입력 검증 2) doc_ids 존재 확인(집계 1회) 3) 질의 임베딩 1회 4) hybrid 검색 1회(후보 SERVICE_RERANK_N,
    # 필터는 하위 질의마다) 5) 결과가 있으면 리랭크 1회(동점은 hybrid 순서) 6) 상위 top_k
    _validate(query, top_k, doc_ids)

    # OpenSearch: ConnectionError(ConnectionTimeout 포함)를 TransportError보다 먼저 잡는다(상속 관계라
    # 순서가 바뀌면 상태 코드 비교로 간다). 상태 코드가 정수인 5xx만 감싸고 4xx·"N/A"는 그대로 올린다
    try:
        if doc_ids is not None:
            _check_doc_ids(deps.search_client, doc_ids)
        [query_vector] = _call_tei(
            "tei-embed", lambda: deps.embedder.embed([query], input_type="query")
        )
        body = hybrid_body(query, query_vector, SERVICE_RERANK_N, doc_ids)
        response = deps.search_client.search(index=search_alias(), body=body)
    except opensearchpy.ConnectionError as e:
        raise SearchUnavailableError("opensearch", str(e)) from e
    except opensearchpy.TransportError as e:
        if isinstance(e.status_code, int) and e.status_code >= 500:
            raise SearchUnavailableError("opensearch", str(e)) from e
        raise

    hits = to_document_hits(response)
    if not hits:
        return []

    scores = _call_tei("tei-rerank", lambda: deps.reranker.rerank(query, [h.text for h in hits]))
    order = sorted(range(len(hits)), key=lambda i: -scores[i])
    return [
        hits[i].model_copy(update={"score": float(scores[i]), "rank": rank})
        for rank, i in enumerate(order[:top_k], start=1)
    ]
