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

import re
from collections.abc import Callable, Iterable, Sequence

from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import (
    EvalQuestion,
    EvalQuestionV2,
    RunScore,
    SearchHit,
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
