"""문서 검색 툴 `search_documents` (이슈 #18 PR ③).

Agent(#22)·MCP 서버(#24)·`/search`(#23)가 함께 부르는 RFP 문서 검색 함수다. 인자·반환·오류 규칙은
docs/data-design.md 5절 "검색 툴 search_documents"가 기준이고, 서비스 기본값(하이브리드 RRF k=60·후보 50
+ 리랭크 N=20)은 docs/adr/0004-hybrid-rerank-default.md다.

- `SearchDeps`: 검색 클라이언트·임베더·리랭커 묶음. LLM에게 보이는 인자(query·top_k·doc_ids)와 분리해
  Agent·MCP 어댑터가 미리 묶어 넘긴다. 테스트는 가짜 클라이언트를 넣는다
- `search_documents` 본문은 소유자 구현(학습 모드 #18)
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


def _call_tei[T](service: Literal["tei-embed", "tei-rerank"], call: Callable[[], T]) -> T:
    try:
        return call()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (413, 422):
            raise ValueError(f"질의가 너무 깁니다: {e.response.text}") from e
        if status >= 500:
            raise SearchUnavailableError(service, str(e)) from e
        raise
    except httpx.TransportError as e:
        raise SearchUnavailableError(service, str(e)) from e


def _validate(query: str, top_k: int, doc_ids: list[str] | None) -> None:
    if not query.strip():
        raise ValueError("query가 비어 있다")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k는 정수여야 한다: {top_k!r}")  # noqa: TRY004 — 입력 오류는 ValueError로 통일(data-design 5절)
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k({top_k})는 1 이상 {MAX_TOP_K} 이하여야 한다")
    if doc_ids is not None:
        if not doc_ids:
            raise ValueError("doc_ids가 빈 목록이다. 전체 검색은 None을 넘긴다")
        if any(not d.strip() for d in doc_ids):
            raise ValueError("doc_ids에 빈 문자열이 있다")


def _check_doc_ids(client: SearchClient, doc_ids: list[str]) -> None:
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
    """RFP 문서에서 질문과 관련된 청크를 찾아 관련도 순으로 돌려준다.

    인자 (LLM이 정하는 것)
    - query: 찾을 내용을 담은 질문이나 키워드(한국어). 빈 문자열·공백만이면 ValueError.
      요구사항 ID(예: SFR-013)를 넣으면 그 ID의 요구사항 청크가 위로 온다
    - top_k: 돌려받을 결과 수. 1 이상 20 이하(기본 10). 범위를 벗어나면 ValueError(잘라서 맞추지 않는다)
    - doc_ids: 검색할 문서의 doc_id 목록. None이면 전체 문서. 빈 목록·빈 문자열·인덱스에 없는 doc_id가
      있으면 ValueError(없는 ID를 메시지에 적는다)
    deps: 서버 클라이언트 묶음(LLM에게 보이지 않는다)

    반환: `DocumentHit` 목록 — 리랭커 점수(0~1) 내림차순, rank 1부터. 결과가 top_k보다 적거나 0개일 수
    있다(0개는 오류가 아니다). score는 한 번의 호출 안에서만 비교한다.

    동작·오류 규칙은 docs/data-design.md 5절 "검색 툴 search_documents". 구현 순서(소유자):
    1. 입력 검증 — 실패하면 서버를 부르지 않는다
    2. doc_ids가 있으면 없는 ID 확인: 별칭에 `size: 0` + `terms` 집계(field `doc_id`) 요청 1회.
       응답 `aggregations.<이름>.buckets[].key`에 없는 ID가 있으면 ValueError(그 ID를 메시지에)
    3. `deps.embedder.embed([query], input_type="query")` 1회
    4. `deps.search_client.search(index=search_alias(), body=hybrid_body(query, 벡터, SERVICE_RERANK_N, doc_ids))`
       1회 → `to_document_hits(응답)`
    5. 결과가 없으면 리랭커를 부르지 않고 빈 목록. 있으면 `deps.reranker.rerank(query, [hit.text …])` 1회
       (hybrid 순서 그대로) → 리랭커 점수 내림차순(동점이면 hybrid 순서), score를 리랭커 점수로, rank를 1부터 다시
    6. 상위 top_k개
    예외 바꾸기
    - OpenSearch: `opensearchpy.ConnectionError`(ConnectionTimeout 포함)·`TransportError` 5xx →
      `SearchUnavailableError("opensearch", …) from 원래예외`. 4xx는 그대로
    - TEI(embed→"tei-embed", rerank→"tei-rerank"): `httpx.TransportError`(연결 거절·시간 초과)·5xx
      `httpx.HTTPStatusError` → `SearchUnavailableError`. 422(입력 토큰 한도 초과)·413(본문 한도 초과) →
      `ValueError("질의가 너무 깁니다: " + TEI 오류 문구)`. 그 밖의 4xx는 그대로
    """
    _validate(query, top_k, doc_ids)

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
