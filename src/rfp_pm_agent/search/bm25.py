"""BM25 희소 검색 (이슈 #13 청킹 비교 평가용) — rank_bm25 라이브러리를 감싼다.

이 파일은 실제 서비스의 검색이 아니다. 서비스 검색은 #18에서 OpenSearch의
BM25(nori)로 따로 만든다. 여기 것은 청킹 방식 3개를 같은 조건에서 줄 세우기
위한 실험 도구라, 점수식을 직접 짜는 대신 널리 쓰이는 구현을 그대로 쓴다.

감싸는 층이 하는 일은 세 가지다.
- 토큰화 함수를 인자로 받는다 (bigram / whitespace를 갈아끼우며 비교해야 한다)
- k1·b를 인자로 받아 runs.jsonl에 기록할 수 있게 한다
- 점수 배열을 Chunk와 다시 이어 붙여 SearchHit으로 돌려주고, 동점일 때 순서가
  실행할 때마다 달라지지 않게 고정한다

rank_bm25의 BM25Okapi는 ATIRE 변형이라, 절반이 넘는 청크에 나오는 토큰의 IDF가
음수가 되는 것을 막으려고 작은 양수(epsilon × 평균 IDF)로 바꾼다. 그래서 아주
흔한 토큰도 점수에 조금은 기여한다. bigram 토큰화에서는 "사업" 같은 조각이 거의
모든 청크에 나오므로, 순위를 가르는 것은 드문 토큰 쪽이다.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import SearchHit
from rfp_pm_agent.search.tokenize import Tokenizer

# rank_bm25의 기본값과 같은 값. 이 두 값이 방식 간 순위를 바꿀 수 있어서
# runs.jsonl에 함께 기록한다. b는 문서 길이 보정 계수인데, 청크 길이 분포가
# 방식마다 크게 다르다 (requirement 청크는 길고 block 청크에는 "목\n차\n"
# 같은 것이 섞여 있다).
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class Bm25Index:
    """청크 묶음 하나에 대한 BM25 색인.

    색인 한 개가 한 방식(block / requirement / block_requirement)의 청크
    전체를 담는다. 문서별로 나누지 않고 10개 문서의 청크를 한 번에 검색한다.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        tokenizer: Tokenizer,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        corpus = [tokenizer(chunk.text) for chunk in chunks]
        # 청크마다 어떤 토큰이 들어 있는지. "질문 토큰이 하나라도 나오는가"를
        # 점수 부호가 아니라 이 집합으로 판정한다 (아래 search 설명 참고)
        self._token_sets = [set(tokens) for tokens in corpus]
        # BM25Okapi는 청크가 하나도 없으면 평균 길이를 구하다 0으로 나눈다.
        # 빈 색인은 검색 결과가 늘 비어 있으므로 색인 자체를 만들지 않는다.
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b) if chunks else None

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """질문을 토큰화해 점수를 매기고 상위 top_k개를 점수 내림차순으로 돌려준다.

        질문 토큰이 하나도 들어 있지 않은 청크는 결과에 넣지 않는다. 판정 기준을
        점수 부호가 아니라 토큰 포함 여부로 삼는 이유는, BM25Okapi가 흔한 토큰의
        IDF를 epsilon × 평균 IDF로 바꾸는데 모든 토큰이 흔하면 이 값이 음수가 되어
        실제로 일치한 청크의 점수까지 음수로 내려가기 때문이다. 점수로 걸러내면
        그런 청크가 결과에서 빠져 hit@k가 실제보다 낮게 나온다.

        질문 토큰이 어느 청크에도 없으면 빈 목록이 된다. 점수가 같으면 chunk_id
        오름차순으로 정렬해, 같은 입력을 다시 돌렸을 때 순위가 달라지지 않게 한다.
        """
        if self._bm25 is None:
            return []
        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []
        unique_tokens = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        scored = [
            (float(score), chunk)
            for score, chunk, tokens in zip(scores, self.chunks, self._token_sets, strict=True)
            if unique_tokens & tokens
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            SearchHit(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, score=score, text=chunk.text)
            for score, chunk in scored[:top_k]
        ]
