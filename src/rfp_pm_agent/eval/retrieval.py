"""검색 평가의 정답 판정과 점수 집계 (이슈 #13).

판정 규칙(확정):
- 청크의 doc_id가 질문의 doc_id와 같고,
- 공백을 모두 지운 뒤 evidence가 청크 text에 들어 있으면 정답이다.

공백을 지우는 이유는 PDF·HWP 추출 텍스트의 줄바꿈과 띄어쓰기가 원문과 다르게
들어가기 때문이다. evidence("□사업기간: 계약일로부터3개월이내")와 청크 text의
공백이 다르다는 이유로 오답이 되면, 청킹 방식이 아니라 추출 잡음을 재게 된다.

evidence가 두 청크에 걸쳐 잘리면 어느 청크도 정답이 아니다. 이것은 판정의
허점이 아니라 그 청킹 방식의 실제 약점이라 보정하지 않는다.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import (
    EvalQuestion,
    EvalQuestionV2,
    NoAnswerRecord,
    QuestionResult,
    RetrievalRunScore,
    RunScore,
    SearchHit,
    SliceScore,
    TypeScore,
)

_WHITESPACE = re.compile(r"\s+")

# 지시된 점수 지표. hit@k의 k 목록이자 검색해 올 상위 개수다
K_VALUES = (1, 3, 5)
TOP_K = max(K_VALUES)

# 질문 하나를 받아 상위 top_k개 검색 결과를 돌려주는 함수
SearchFn = Callable[[str, int], list[SearchHit]]


def normalize(text: str) -> str:
    """공백(띄어쓰기·줄바꿈·탭)을 모두 지운다. 정답 판정의 유일한 정규화 규칙이다."""
    return _WHITESPACE.sub("", text)


def evidence_in_text(evidence: str, text: str) -> bool:
    """공백을 모두 지운 evidence가 공백을 모두 지운 text에 들어 있으면 참."""
    return normalize(evidence) in normalize(text)


def _matches(*, chunk_doc_id: str, chunk_text: str, question: EvalQuestion) -> bool:
    if chunk_doc_id != question.doc_id:
        return False
    return evidence_in_text(question.evidence, chunk_text)


def is_correct(chunk: Chunk, question: EvalQuestion) -> bool:
    """청크가 질문의 정답 청크인지 판정한다."""
    return _matches(chunk_doc_id=chunk.doc_id, chunk_text=chunk.text, question=question)


def hit_is_correct(hit: SearchHit, question: EvalQuestion) -> bool:
    """검색 결과 한 건에 같은 판정 규칙을 적용한다."""
    return _matches(chunk_doc_id=hit.doc_id, chunk_text=hit.text, question=question)


def hit_at_k(hits: Sequence[SearchHit], question: EvalQuestion, k: int) -> bool:
    """상위 k개 안에 정답이 하나라도 있으면 참. 결과가 k개보다 적어도 그대로 센다."""
    return any(hit_is_correct(hit, question) for hit in hits[:k])


def ceiling(chunks: Iterable[Chunk], question: EvalQuestion) -> bool:
    """검색과 무관하게, 정답 조건을 만족하는 청크가 하나라도 있으면 참.

    이 값이 거짓인 질문은 그 청킹 방식이 검색을 아무리 잘해도 맞힐 수 없다.
    """
    return any(is_correct(chunk, question) for chunk in chunks)


# --- v2 (이슈 #15): evidence 목록 판정 ---


def gold_chunk_ids(chunks: Iterable[Chunk], question: EvalQuestionV2) -> list[list[str]]:
    """evidence마다 그 evidence를 담은 청크 ID 목록을 evidence 순서대로 돌려준다.

    판정은 v1과 같다 — doc_id가 같고, 공백을 지운 evidence가 청크 text에 들어
    있으면 그 청크가 그 evidence를 담은 것이다. 검색 결과는 쓰지 않는다.
    답 없음 문항은 evidence가 없으므로 빈 목록이다.
    """
    same_doc = [c for c in chunks if c.doc_id == question.doc_id]
    return [
        [c.chunk_id for c in same_doc if evidence_in_text(evidence, c.text)]
        for evidence in question.evidence
    ]


def reachable(chunks: Iterable[Chunk], question: EvalQuestionV2) -> bool:
    """모든 evidence가 청크 하나 이상에 담겨 있으면 참 (v2의 최고 점수 판정).

    evidence가 하나라도 어느 청크에도 없으면, 그 청킹 방식으로는 검색을 아무리
    잘해도 문항 전체를 맞힐 수 없다. 답 없음 문항은 도달할 정답이 없어 이 판정의
    대상이 아니므로 ValueError를 낸다 — 호출하는 쪽이 먼저 걸러야 최고 점수의
    분모에 섞이지 않는다.
    """
    if not question.evidence:
        raise ValueError(f"{question.question_id}: 답 없음 문항은 최고 점수 판정 대상이 아니다")
    return all(gold_chunk_ids(chunks, question))


def _tally(results: list[tuple[EvalQuestion, list[SearchHit], bool]]) -> TypeScore:
    return TypeScore(
        total=len(results),
        hit_at_1=sum(1 for q, hits, _ in results if hit_at_k(hits, q, 1)),
        hit_at_3=sum(1 for q, hits, _ in results if hit_at_k(hits, q, 3)),
        hit_at_5=sum(1 for q, hits, _ in results if hit_at_k(hits, q, 5)),
        ceiling=sum(1 for _, _, reachable in results if reachable),
    )


def score_run(
    questions: Sequence[EvalQuestion],
    chunks: Sequence[Chunk],
    search: SearchFn,
    *,
    top_k: int = TOP_K,
) -> tuple[RunScore, dict[str, list[SearchHit]]]:
    """질문 전체를 돌려 점수를 집계한다.

    돌려주는 값은 (집계, hit@5를 못 맞힌 질문의 상위 검색 결과)다. 두 번째 값은
    failures 파일로 남겨, 왜 틀렸는지(예: 목차 줄이 상위를 차지했는지)를 나중에
    볼 수 있게 한다.
    """
    results: list[tuple[EvalQuestion, list[SearchHit], bool]] = []
    for question in questions:
        hits = search(question.question, top_k)
        results.append((question, hits, ceiling(chunks, question)))

    missed = {q.question_id: hits for q, hits, _ in results if not hit_at_k(hits, q, 5)}
    scores = RunScore(
        overall=_tally(results),
        requirement=_tally([r for r in results if r[0].type == "요구사항"]),
        general=_tally([r for r in results if r[0].type == "일반"]),
        missed_question_ids=sorted(missed),
    )
    return scores, missed


def ceiling_only(questions: Sequence[EvalQuestion], chunks: Sequence[Chunk]) -> RunScore:
    """검색 없이 최고 점수만 집계한다. hit@k는 모두 0으로 남는다.

    BM25 구현과 무관하게 "이 방식으로 도달 가능한 상한"을 먼저 확인할 때 쓴다.
    """
    empty: list[SearchHit] = []
    results = [(q, empty, ceiling(chunks, q)) for q in questions]
    return RunScore(
        overall=_tally(results),
        requirement=_tally([r for r in results if r[0].type == "요구사항"]),
        general=_tally([r for r in results if r[0].type == "일반"]),
        missed_question_ids=[],
    )


# --- v2 검색 지표 (이슈 #16) — 정답을 evidence 묶음으로 본다 ---
#
# 문항의 정답은 gold_groups = [evidence i를 담은 청크 ID 집합 for i in evidence]다.
# q021처럼 evidence 하나를 청크 여럿이 담으면 묶음 하나에 청크가 여럿 들어가고,
# multi_chunk 문항은 묶음이 2개 이상이다. 결정 규칙(이슈 #16)의 주 지표는
# recall_all_at_k(모든 묶음을 찾아야 적중)이다.

# 이슈 #16 주 지표의 k와 참고 지표의 k
V2_K_VALUES = (5, 10)
V2_TOP_K = max(V2_K_VALUES)


def gold_groups(chunks: Iterable[Chunk], question: EvalQuestionV2) -> list[set[str]]:
    """evidence마다 그 evidence를 담은 청크 ID 집합. 답 없음 문항은 빈 목록."""
    return [set(ids) for ids in gold_chunk_ids(chunks, question)]


def _covered(ranked_ids: Sequence[str], groups: Sequence[Collection[str]], k: int) -> int:
    top = set(ranked_ids[:k])
    return sum(1 for group in groups if top & set(group))


def recall_all_at_k(ranked_ids: Sequence[str], groups: Sequence[Collection[str]], k: int) -> bool:
    """상위 k 안에 모든 묶음의 청크가 하나 이상 있으면 참. groups가 비면 ValueError."""
    if not groups:
        raise ValueError("gold_groups가 비어 있다 — 답 없음 문항은 검색 지표 대상이 아니다")
    return _covered(ranked_ids, groups, k) == len(groups)


def recall_fraction_at_k(
    ranked_ids: Sequence[str], groups: Sequence[Collection[str]], k: int
) -> float:
    """상위 k 안에서 찾은 묶음의 비율(0~1). groups가 비면 ValueError."""
    if not groups:
        raise ValueError("gold_groups가 비어 있다 — 답 없음 문항은 검색 지표 대상이 아니다")
    return _covered(ranked_ids, groups, k) / len(groups)


def reciprocal_rank(
    ranked_ids: Sequence[str], gold_groups: Sequence[Collection[str]], k: int | None = None
) -> float:
    """처음으로 어느 묶음이든 적중한 순위의 역수. 상위 k 안에 없으면 0.

    - ranked_ids: 검색 결과 청크 ID (1위부터)
    - gold_groups: evidence별 정답 청크 ID 집합의 목록
    - k: None이면 ranked_ids 전체를 본다
    gold_groups가 비면(답 없음 문항) ValueError를 낸다.
    MRR은 호출하는 쪽에서 문항별 값을 평균한다.

    소유자가 직접 구현했다 (학습 대상, 이슈 #16). 테스트: tests/unit/test_eval_rank_metrics.py
    """
    if not gold_groups:
        raise ValueError("gold_groups가 비어 있다 (답 없음 문항)")

    gold = {chunk_id for group in gold_groups for chunk_id in group}
    candidates = ranked_ids if k is None else ranked_ids[:k]

    for rank, chunk_id in enumerate(candidates, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str], gold_groups: Sequence[Collection[str]], k: int = 10
) -> float:
    """상위 k의 NDCG. relevance는 evidence 묶음 단위로 준다.

    - gain: 청크가 아직 덮지 않은 묶음을 새로 덮을 때 "새로 덮은 묶음 수"만큼 준다.
      이미 덮은 묶음의 중복 청크(q021의 나머지 청크 등)는 gain 0이다. 청크 하나가
      두 묶음을 함께 덮으면(q038형) gain 2다.
    - DCG = Σ gain_r / log2(r + 1) (r = 1부터)
    - IDCG: 같은 gain 정의로 계산한 이상적 순서의 DCG. 정답 청크를 "덮는 묶음 번호의
      조합"으로 줄이고(같은 조합의 청크는 하나로 친다), 조합들의 가능한 순서를 상위 k개까지
      모두 탐색해 가장 큰 DCG를 쓴다. 묶음이 청크를 겹쳐 가질 때 탐욕법(매번 가장 많이
      덮는 청크)은 최적이 아닐 수 있어 NDCG가 1을 넘을 수 있었다 — 그래서 전체 탐색을 한다.
      조합 수 n에 대해 순서 수가 n!/(n−k)!로 늘어나므로 조합이 많은 문항에서는 느려진다
      (qa_v2 답 있는 51문항은 문항당 조합 2개 이하, 2026-09-26 확인).
    - 도달 불가 문항: 어떤 evidence도 청크에 담기지 않아 IDCG가 0이면 0을 돌려준다.
    gold_groups가 비면(답 없음 문항) ValueError를 낸다.

    소유자가 직접 구현했다 (학습 대상, 이슈 #16). 테스트: tests/unit/test_eval_rank_metrics.py
    """
    if not gold_groups:
        raise ValueError("gold_groups가 비어 있다 (답 없음 문항)")

    groups = [set(group) for group in gold_groups]

    # 1~2단계: 실제 점수(DCG)
    found: set[int] = set()
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        new = {i for i, group in enumerate(groups) if i not in found and chunk_id in group}
        found |= new
        dcg += len(new) / math.log2(rank + 1)

    # 3단계: 만점(IDCG)
    # 청크를 "덮는 묶음 번호 조합"으로 줄인다. 같은 조합의 청크는 하나로 친다.
    signatures = {
        frozenset(i for i, group in enumerate(groups) if chunk_id in group)
        for group in groups
        for chunk_id in group
    }
    idcg = 0.0
    for order in itertools.permutations(signatures, min(k, len(signatures))):
        covered: set[int] = set()
        score = 0.0
        for rank, sig in enumerate(order, start=1):
            added = sig - covered
            covered |= added
            score += len(added) / math.log2(rank + 1)
        idcg = max(idcg, score)

    # 4단계
    if idcg == 0:
        return 0.0  # 도달할 수 있는 묶음이 없는 문항
    return dcg / idcg


def mcnemar_exact_p(b: int, c: int) -> float:
    """McNemar 정확검정(양측)의 p값. 이슈 #16 결정 규칙에 쓴다.

    b = 모델 A만 맞힌 문항 수, c = 모델 B만 맞힌 문항 수. 둘 다 맞히거나 둘 다 틀린
    문항은 검정에 들어가지 않는다. 귀무가설(두 모델의 적중 확률이 같다)에서 b는
    이항분포 B(b + c, 0.5)를 따르므로 p = min(1, 2 × P(X ≤ min(b, c)))다.
    b + c = 0이면 두 모델이 문항별로 같으므로 1.0이다.
    """
    if b < 0 or c < 0:
        raise ValueError(f"b와 c는 0 이상이어야 한다: b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    tail: float = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def score_question_v2(
    question: EvalQuestionV2, hits: Sequence[SearchHit], groups: Sequence[Collection[str]]
) -> QuestionResult:
    """답 있는 문항 하나를 채점한다."""
    ranked = [hit.chunk_id for hit in hits]
    return QuestionResult(
        question_id=question.question_id,
        tags=list(question.tags),
        recall_all_at_5=recall_all_at_k(ranked, groups, 5),
        recall_all_at_10=recall_all_at_k(ranked, groups, 10),
        recall_frac_at_5=recall_fraction_at_k(ranked, groups, 5),
        recall_frac_at_10=recall_fraction_at_k(ranked, groups, 10),
        reciprocal_rank=reciprocal_rank(ranked, groups, 10),
        ndcg_at_10=ndcg_at_k(ranked, groups, 10),
        top10_chunk_ids=ranked[:10],
    )


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    """비어 있거나(문항 0개) 하나라도 None(MRR·NDCG 구현 전 기록)이면 None, 아니면 평균."""
    present = [v for v in values if v is not None]
    if not values or len(present) != len(values):
        return None
    return sum(present) / len(present)


def _slice(label: str, results: Sequence[QuestionResult]) -> SliceScore:
    return SliceScore(
        label=label,
        total=len(results),
        recall_all_at_5=sum(r.recall_all_at_5 for r in results),
        recall_all_at_10=sum(r.recall_all_at_10 for r in results),
        recall_frac_at_5=sum(r.recall_frac_at_5 for r in results),
        recall_frac_at_10=sum(r.recall_frac_at_10 for r in results),
        mrr=_mean_or_none([r.reciprocal_rank for r in results]),
        ndcg_at_10=_mean_or_none([r.ndcg_at_10 for r in results]),
    )


# v1 30문항(#13 기준선과 비교용)과 추가 문항을 가르는 경계
V1_LAST_QUESTION_ID = "q030"


def aggregate_v2(
    results: Sequence[QuestionResult], no_answer: Sequence[NoAnswerRecord]
) -> RetrievalRunScore:
    """전체 / v1 / 추가 / 태그별 행으로 집계한다. 문항이 0개인 태그 행은 만들지 않는다."""
    slices = [
        _slice("전체", results),
        _slice("v1", [r for r in results if r.question_id <= V1_LAST_QUESTION_ID]),
        _slice("추가", [r for r in results if r.question_id > V1_LAST_QUESTION_ID]),
    ]
    tags = sorted({t for r in results for t in r.tags})
    slices += [_slice(f"tag:{t}", [r for r in results if t in r.tags]) for t in tags]
    return RetrievalRunScore(
        answerable=len(results),
        excluded_no_answer=len(no_answer),
        slices=slices,
        no_answer=list(no_answer),
    )


def score_run_v2(
    questions: Sequence[EvalQuestionV2],
    chunks: Sequence[Chunk],
    rankings: Mapping[str, Sequence[SearchHit]],
) -> tuple[RetrievalRunScore, list[QuestionResult]]:
    """문항별 검색 결과(rankings: question_id → 1위부터의 SearchHit)를 채점·집계한다.

    답 없음 문항(evidence == [])은 검색 지표에서 빼고, 1위 점수·상위 5개 점수·1위
    chunk_id만 기록한다. 정답 묶음은 chunks에서 evidence로 다시 뽑는다 — 색인한 청크
    파일과 정답이 어긋나지 않게 하기 위해서다.
    """
    results: list[QuestionResult] = []
    no_answer: list[NoAnswerRecord] = []
    for question in questions:
        hits = list(rankings[question.question_id])
        if not question.evidence:
            no_answer.append(
                NoAnswerRecord(
                    question_id=question.question_id,
                    top1_chunk_id=hits[0].chunk_id if hits else None,
                    top1_score=hits[0].score if hits else None,
                    top5_scores=[h.score for h in hits[:5]],
                )
            )
            continue
        results.append(score_question_v2(question, hits, gold_groups(chunks, question)))
    return aggregate_v2(results, no_answer), results
