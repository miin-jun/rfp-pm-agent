"""v2 검색 지표(묶음 기반 Recall, McNemar, 집계) 단위 테스트 (이슈 #16).

MRR·NDCG 계산 함수 자체의 테스트는 tests/unit/test_eval_rank_metrics.py에 있다.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rfp_pm_agent.eval.retrieval import (
    aggregate_v2,
    gold_groups,
    mcnemar_exact_p,
    recall_all_at_k,
    recall_fraction_at_k,
    score_run_v2,
)
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import EvalQuestionV2, SearchHit


def _chunk(source_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"d:block_requirement:{source_id}",
        doc_id="d",
        method="block_requirement",
        source_ids=[source_id],
        text=text,
    )


def _q(question_id: str, evidence: list[str], tags: list[str] | None = None) -> EvalQuestionV2:
    fields: dict[str, Any] = {
        "question_id": question_id,
        "question": "질문",
        "answer": "답" if evidence else None,
        "evidence": evidence,
        "doc_id": "d",
        "tags": tags if tags is not None else ([] if evidence else ["no_answer"]),
    }
    return EvalQuestionV2(**fields)


def _hits(*source_ids: str) -> list[SearchHit]:
    return [
        SearchHit(chunk_id=f"d:block_requirement:{s}", doc_id="d", score=1.0 - i / 100, text="")
        for i, s in enumerate(source_ids)
    ]


# --- 묶음 기반 Recall ---


def test_모든_묶음을_찾아야_recall_all이_참이고_비율은_찾은_묶음_수로_센다() -> None:
    groups = [{"a"}, {"b", "c"}]
    assert recall_all_at_k(["a", "x", "c"], groups, 3) is True
    assert recall_all_at_k(["a", "x", "y"], groups, 3) is False
    assert recall_fraction_at_k(["a", "x", "y"], groups, 3) == pytest.approx(0.5)


def test_recall은_상위_k만_본다() -> None:
    groups = [{"a"}]
    assert recall_all_at_k(["x", "a"], groups, 1) is False
    assert recall_all_at_k(["x", "a"], groups, 2) is True


def test_같은_묶음의_청크가_여럿_나와도_묶음_하나로_센다() -> None:
    groups = [{"a", "b"}, {"c"}]
    assert recall_fraction_at_k(["a", "b"], groups, 10) == pytest.approx(0.5)


def test_묶음이_비면_recall은_ValueError() -> None:
    with pytest.raises(ValueError):
        recall_all_at_k(["a"], [], 10)
    with pytest.raises(ValueError):
        recall_fraction_at_k(["a"], [], 10)


def test_gold_groups는_evidence마다_담은_청크를_집합으로_모은다() -> None:
    chunks = [
        _chunk("b1", "사업기간 3개월"),
        _chunk("b2", "사업 기간 3개월 반복"),
        _chunk("b3", "예산"),
    ]
    groups = gold_groups(chunks, _q("q031", ["사업기간 3개월", "예산"]))
    assert groups == [
        {"d:block_requirement:b1", "d:block_requirement:b2"},
        {"d:block_requirement:b3"},
    ]


# --- McNemar 정확검정 (기대값은 이항분포로 손 계산) ---


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 0, 1.0),  # 불일치 문항 없음
        (0, 6, 2 * (1 / 64)),  # 2 × P(X ≤ 0; n=6) = 0.03125
        (6, 0, 0.03125),  # 불일치가 한쪽만 6건: 2 × 0.5^6
        (1, 5, 2 * (1 + 6) / 64),  # 2 × P(X ≤ 1; n=6) = 0.21875
        (3, 3, 1.0),  # 2 × 42/64 > 1 → 1
        (5, 1, 2 * (1 + 6) / 64),  # 대칭
        (0, 5, 2 * (1 / 32)),  # 0.0625 — 5문항 차이로는 p < 0.05가 안 된다
    ],
)
def test_mcnemar_exact_p(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact_p(b, c) == pytest.approx(expected)


def test_mcnemar_음수는_거부한다() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact_p(-1, 2)


# --- 채점·집계 ---


def test_score_run_v2는_답_없음을_빼고_1위_점수를_기록한다() -> None:
    chunks = [_chunk("b1", "사업기간 3개월"), _chunk("b2", "예산 1억")]
    questions = [
        _q("q001", ["사업기간 3개월"]),
        _q("q031", ["예산 1억"], tags=["paraphrase"]),
        _q("q032", []),
    ]
    rankings = {
        "q001": _hits("b1", "b2"),
        "q031": _hits("b1"),  # 예산 청크를 못 찾음
        "q032": _hits("b2", "b1"),
    }

    scores, results = score_run_v2(questions, chunks, rankings)

    assert scores.answerable == 2
    assert scores.excluded_no_answer == 1
    assert [r.question_id for r in results] == ["q001", "q031"]
    by_label = {s.label: s for s in scores.slices}
    assert by_label["전체"].recall_all_at_10 == 1
    assert by_label["v1"].total == 1 and by_label["v1"].recall_all_at_10 == 1
    assert by_label["추가"].total == 1 and by_label["추가"].recall_all_at_10 == 0
    assert by_label["tag:paraphrase"].total == 1
    [na] = scores.no_answer
    assert na.question_id == "q032"
    assert na.top1_chunk_id == "d:block_requirement:b2"
    assert na.top1_score == pytest.approx(1.0)
    assert na.top5_scores == pytest.approx([1.0, 0.99])


def test_문항이_없는_태그는_행을_만들지_않는다() -> None:
    scores = aggregate_v2([], [])
    assert [s.label for s in scores.slices] == ["전체", "v1", "추가"]
    assert scores.slices[0].total == 0


def test_행_단위_mrr·ndcg는_문항별_값의_평균이다() -> None:
    chunks = [_chunk("b1", "사업기간 3개월"), _chunk("b2", "예산 1억"), _chunk("b3", "무관")]
    questions = [
        _q("q001", ["사업기간 3개월"]),  # v1
        _q("q031", ["예산 1억"], tags=["paraphrase"]),  # 추가
        _q("q032", []),  # 답 없음 — 평균에서 빠진다
    ]
    rankings = {
        "q001": _hits("b1", "b3"),  # 정답 1위 → RR 1, NDCG 1
        "q031": _hits("b3", "b2"),  # 정답 2위 → RR 1/2, NDCG = (1/log2 3)/1 ≈ 0.630930
        "q032": _hits("b3"),
    }

    scores, _ = score_run_v2(questions, chunks, rankings)

    by_label = {s.label: s for s in scores.slices}
    # 전체: MRR = (1 + 0.5) / 2 = 0.75, NDCG = (1 + 0.630930) / 2 ≈ 0.815465
    assert by_label["전체"].mrr == pytest.approx(0.75)
    assert by_label["전체"].ndcg_at_10 == pytest.approx((1 + 1 / math.log2(3)) / 2)
    # v1은 q001만: 1.0 / 1.0
    assert by_label["v1"].mrr == pytest.approx(1.0)
    assert by_label["v1"].ndcg_at_10 == pytest.approx(1.0)
    # 추가·tag:paraphrase는 q031만: 0.5 / ≈ 0.630930
    assert by_label["추가"].mrr == pytest.approx(0.5)
    assert by_label["tag:paraphrase"].ndcg_at_10 == pytest.approx(1 / math.log2(3))


def test_문항이_0개인_행의_mrr·ndcg는_None이다() -> None:
    scores = aggregate_v2([], [])
    assert scores.slices[0].mrr is None
    assert scores.slices[0].ndcg_at_10 is None
