"""문서 검색 툴 `search_documents` (이슈 #18 PR ③).

Agent(#22)·MCP 서버(#24)·`/search`(#23)가 함께 부르는 RFP 문서 검색 함수다. 인자·반환·오류 규칙은
docs/data-design.md 5절 "검색 툴 search_documents"가 기준이고, 서비스 기본값(하이브리드 RRF k=60·후보 50
+ 리랭크 N=20)은 docs/adr/0004-hybrid-rerank-default.md다.

- `SearchDeps`: 검색 클라이언트·임베더·리랭커 묶음. LLM에게 보이는 인자(query·top_k·doc_ids)와 분리해
  Agent·MCP 어댑터가 미리 묶어 넘긴다. 테스트는 가짜 클라이언트를 넣는다
- `search_documents` 본문은 소유자 구현(학습 모드 #18)
"""

from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import OpenSearch

from rfp_pm_agent.clients.embedding import EmbeddingClient, TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import RerankerClient, TEIRerankerClient
from rfp_pm_agent.config import ClientsConfig, OpenSearchConfig
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K
from rfp_pm_agent.search.opensearch import SearchClient


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

    동작·오류 규칙은 docs/data-design.md 5절 "검색 툴 search_documents".
    """
    raise NotImplementedError("#18 PR ③ — 소유자 구현")
