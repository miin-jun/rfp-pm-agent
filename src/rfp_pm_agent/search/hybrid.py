"""하이브리드 검색의 앱 쪽 로직 — 비교용 RRF와 리랭크 연결 (이슈 #18 PR ②).

서비스 경로의 RRF는 OpenSearch가 한다(`search/opensearch.py`의 `hybrid_search`, 요청 본문의 임시
파이프라인). 여기 `rrf_fuse`는 같은 계산을 앱 코드로 다시 해서 두 결과의 상위 10 겹침 수를 기록하는
검증용이다(#18 결정 3). `hybrid_rerank_search`는 hybrid 결과 상위 N개를 크로스인코더로 다시 정렬한다.
"""

from __future__ import annotations

from collections.abc import Sequence

from rfp_pm_agent.clients.reranker import RerankerClient
from rfp_pm_agent.schemas.eval import SearchHit
from rfp_pm_agent.search.dense import rerank_hits
from rfp_pm_agent.search.opensearch import (
    KNN_CANDIDATES,
    RRF_RANK_CONSTANT,
    SearchClient,
    hybrid_search,
)

# 서비스 검색 기본값 (ADR-0004, #18 PR ③). 측정으로 고정한 값이라 환경변수로 두지 않는다 —
# 바꾸면 docs/hybrid-search-measurement.md 기준으로 다시 잰다.
# 평가 도구의 DEFAULT_RERANK_N(eval/run_retrieval.py, #16 규칙)도 20이지만 따로 둔다
SERVICE_RERANK_N = 20
DEFAULT_TOP_K = 10
MAX_TOP_K = SERVICE_RERANK_N


def rrf_fuse(
    rankings: Sequence[Sequence[SearchHit]],
    top_k: int,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> list[SearchHit]:
    """검색기별 순위 목록을 RRF로 합쳐 상위 top_k개를 돌려준다. OpenSearch를 부르지 않는다.

    인자
    - rankings: 검색기마다 1위부터 정렬된 결과 목록(예: [bm25_search 50개, knn_search 50개])
    - top_k: 돌려줄 결과 수
    - rank_constant: RRF 상수 k. 1 이상(OpenSearch와 같은 제약), 아니면 ValueError

    계산
    - chunk_id마다 점수 = 그 chunk가 들어 있는 목록마다 1 / (rank_constant + 순위)의 합. 순위는 1부터.
      목록에 없으면 그 목록 몫은 0이다
    - 반환 SearchHit의 score는 RRF 점수로 바꾸고, doc_id·text는 그 chunk가 처음 나온 결과에서 가져온다
    - RRF 점수 내림차순. 동점이면 처음 나온 순서(rankings[0]을 1위부터 끝까지, 다음 rankings[1] …)
    - 한 목록 안에 같은 chunk_id가 두 번 나오면 ValueError — 검색 결과가 잘못 만들어졌다는 뜻이다
    - rankings가 비었거나 모든 목록이 비면 빈 목록
    """
    if rank_constant < 1:
        raise ValueError(f"rank_constant({rank_constant})는 1 이상이어야 한다")

    scores: dict[str, float] = {}
    first_hit: dict[str, SearchHit] = {}

    for ranking in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk_id in seen:
                raise ValueError(f"한 목록 안에 같은 chunk_id가 두 번 있다: {hit.chunk_id}")
            seen.add(hit.chunk_id)

            if hit.chunk_id not in first_hit:
                first_hit[hit.chunk_id] = hit
                scores[hit.chunk_id] = 0.0
            scores[hit.chunk_id] += 1 / (rank_constant + rank)

    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        first_hit[chunk_id].model_copy(update={"score": scores[chunk_id]})
        for chunk_id in ordered[:top_k]
    ]


def hybrid_rerank_search(
    client: SearchClient,
    reranker: RerankerClient,
    query: str,
    query_vector: list[float],
    rerank_n: int,
    top_k: int,
) -> list[SearchHit]:
    """hybrid_search 상위 rerank_n개를 리랭커로 다시 정렬해 상위 top_k개를 돌려준다.

    인자
    - client·query·query_vector: `hybrid_search`와 같다
    - reranker: `RerankerClient`(TEI `/rerank`, 32개씩 나눠 보내는 처리는 클라이언트가 한다)
    - rerank_n: 리랭크할 후보 수(#18 결정 4: 20 또는 50). `KNN_CANDIDATES`(50) 이하
    - top_k: 돌려줄 결과 수. rerank_n 이하
    - 범위를 벗어나면 OpenSearch·리랭커를 부르지 않고 ValueError

    동작
    - `hybrid_search(client, query, query_vector, top_k=rerank_n)`를 한 번 부른다
    - `reranker.rerank(query, [그 결과의 text …])`를 한 번 부른다(hybrid 순서 그대로, 질문은 원문)
    - 리랭커 점수 내림차순으로 정렬해 상위 top_k개. 반환 SearchHit의 score는 리랭커 점수로 바꾼다
    - 리랭커 점수가 같으면 hybrid 순서를 지킨다
    - hybrid 결과가 비면 리랭커를 부르지 않고 빈 목록
    """
    if not 1 <= rerank_n <= KNN_CANDIDATES:
        raise ValueError(
            f"rerank_n({rerank_n})은 1 이상 KNN_CANDIDATES({KNN_CANDIDATES}) 이하여야 한다"
        )
    if not 1 <= top_k <= rerank_n:
        raise ValueError(f"top_k({top_k})는 1 이상 rerank_n({rerank_n}) 이하여야 한다")

    candidates = hybrid_search(client, query, query_vector, top_k=rerank_n)
    reranked = rerank_hits(query, candidates, reranker, n=rerank_n)
    return reranked[:top_k]
