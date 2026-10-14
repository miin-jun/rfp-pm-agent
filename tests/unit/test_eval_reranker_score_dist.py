"""리랭커 1위 점수 분포 측정의 문항별 계산 (#19). 검색 서버는 부르지 않는다."""

from __future__ import annotations

from rfp_pm_agent.eval.reranker_score_dist import (
    GOLD_METHOD,
    ScoreDistRow,
    build_row,
    threshold_table,
)
from rfp_pm_agent.schemas.eval import EvalQuestionV2
from rfp_pm_agent.schemas.search import DocumentHit


def _hit(rank: int, source_id: str, score: float) -> DocumentHit:
    return DocumentHit(
        rank=rank,
        chunk_id=f"docA:{GOLD_METHOD}:{source_id}",
        doc_id="docA",
        bid_title="문서A",
        format="hwp",
        requirement_id=None,
        page=None,
        printed_page=None,
        block_type="paragraph",
        text=f"{source_id} 본문\n둘째 줄",
        score=score,
    )


def _question(gold: list[list[str]], *, answerable: bool = True) -> EvalQuestionV2:
    return EvalQuestionV2(
        question_id="q001",
        question="질문",
        answer="답" if answerable else None,
        evidence=["근거"] * len(gold) if answerable else [],
        doc_id="docA",
        tags=[] if answerable else ["no_answer"],
        gold_chunk_ids={GOLD_METHOD: [[f"docA:{GOLD_METHOD}:{s}" for s in g] for g in gold]},
    )


HITS = [_hit(1, "b0001", 0.9), _hit(2, "b0002", 0.5), _hit(3, "b0003", 0.1)]


def test_group_a_when_every_evidence_is_in_results() -> None:
    row = build_row(_question([["b0003"], ["b0002", "b0009"]]), HITS, 12.34)

    assert row.group == "A"
    assert row.gold_all_in_top10 is True
    assert [e.rank for e in row.gold_per_evidence] == [3, 2]
    # 가장 높은 순위의 정답 청크
    assert (row.gold_best_rank, row.gold_best_score) == (2, 0.5)
    assert row.top1_score == 0.9
    assert row.top1_text_head == "b0001 본문 둘째 줄"
    assert row.latency_ms == 12.3


def test_group_b_when_one_evidence_is_missing() -> None:
    row = build_row(_question([["b0001"], ["b0009"]]), HITS, 1.0)

    assert row.group == "B"
    assert row.gold_all_in_top10 is False
    assert row.gold_any_in_top10 is True
    assert [e.rank for e in row.gold_per_evidence] == [1, None]


def test_evidence_in_several_chunks_takes_the_highest_rank() -> None:
    row = build_row(_question([["b0003", "b0002"]]), HITS, 1.0)

    assert row.gold_per_evidence[0].rank == 2


def test_group_c_for_no_answer_question() -> None:
    row = build_row(_question([], answerable=False), HITS, 1.0)

    assert row.group == "C"
    assert row.gold_all_in_top10 is False
    assert row.gold_best_rank is None


def test_empty_results_leave_top1_empty() -> None:
    row = build_row(_question([["b0001"]]), [], 1.0)

    assert row.group == "B"
    assert row.top1_score is None
    assert row.top1_chunk_id is None


def _row(group: str, top1: float) -> ScoreDistRow:
    return ScoreDistRow.model_validate(
        {
            "qid": "q",
            "group": group,
            "answerable": group != "C",
            "tags": [],
            "question": "질문",
            "top1_score": top1,
            "top10_scores": [top1],
            "top10_chunk_ids": ["c"],
            "gold_all_in_top10": group == "A",
            "gold_any_in_top10": group == "A",
            "gold_best_rank": None,
            "gold_best_score": None,
            "gold_per_evidence": [],
            "top1_chunk_id": "c",
            "top1_text_head": "",
            "latency_ms": 1.0,
        }
    )


def test_threshold_table_counts_rejected_and_passed() -> None:
    rows = [_row("A", 0.2), _row("A", 0.8), _row("B", 0.4), _row("C", 0.5), _row("C", 0.1)]

    [low, high] = threshold_table(rows, thresholds=(0.5, 0.9))

    # 점수가 임계값과 같으면 통과한다(거절은 "미만")
    assert (low.a_rejected, low.b_rejected, low.c_passed) == (1, 1, 1)
    assert (high.a_rejected, high.b_rejected, high.c_passed) == (2, 1, 0)
