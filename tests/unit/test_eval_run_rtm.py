"""`eval.run_rtm_eval` 채점 규칙 (이슈 #39, ADR-0012) — 가짜 점수표로만 확인한다(실제 시험 50건은 보지 않는다)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfp_pm_agent.eval.run_rtm_eval import (
    JudgmentRow,
    check_sets,
    grade_practice,
    grade_test,
    load_judgments,
    run_rtm,
    summarize,
)
from rfp_pm_agent.rtm.report import ScoreTable

PID = "PRJ-01"


def table() -> ScoreTable:
    # T1~T4: 시험 문제(injected), P1~P3: 연습, O1: llm_omitted
    best = {"T1": 0.2, "T2": 0.8, "T3": 0.3, "T4": 0.1, "P1": 0.9, "P2": 0.4, "P3": 0.6, "O1": 0.2}
    return ScoreTable(
        project_id=PID,
        requirement_ids=list(best),
        wbs_ids=["W1", "W2"],
        scores={(r, w): (s if w == "W1" else 0.0) for r, s in best.items() for w in ("W1", "W2")},
        chunk_ids={r: f"d:block_requirement:{r}" for r in best},
        injected={"T1", "T2", "T3", "T4"},
    )


def judgment(no: int, rid: str, value: str) -> JudgmentRow:
    return JudgmentRow.model_validate(
        {
            "no": no,
            "project_id": PID,
            "requirement_id": rid,
            "judgment": value,
            "matched_wbs_ids": [],
            "note": "",
        }
    )


JUDGMENTS = [
    judgment(1, "T1", "없음"),  # 누락으로 나옴 → 정답
    judgment(2, "T2", "애매"),  # 연결 있음 → 정답
    judgment(3, "T3", "있음"),  # 누락으로 나옴 → 오답
    judgment(4, "T4", "해당없음"),  # 제외
]


def test_grade_test_applies_fixed_rules() -> None:
    items = grade_test(JUDGMENTS, run_rtm({PID: table()}))

    assert [(i.requirement_id, i.expected, i.predicted_missing, i.correct) for i in items] == [
        ("T1", "missing", True, True),
        ("T2", "linked", False, True),
        ("T3", "linked", True, False),
        ("T4", "excluded", True, None),
    ]
    assert items[1].links == [("W1", 0.8)]


def test_summarize_counts_by_expected_and_excludes_not_applicable() -> None:
    summary = summarize(grade_test(JUDGMENTS, run_rtm({PID: table()})))

    assert summary["missing"] == {"total": 1, "correct": 1}
    assert summary["linked"] == {"total": 2, "correct": 1}
    assert summary["excluded"] == 1
    assert summary["overall"] == {"total": 3, "correct": 2}


def test_grade_practice_skips_injected_and_reports_llm_omitted_separately() -> None:
    tables = {PID: table()}

    practice = grade_practice(tables, run_rtm(tables), llm_omitted={(PID, "O1")})

    assert practice["total"] == 3
    assert practice["false_missing"] == 1
    assert practice["false_missing_ids"] == ["PRJ-01/P2"]
    assert practice["llm_omitted"] == [
        {"project_id": PID, "requirement_id": "O1", "predicted_missing": True}
    ]


def test_check_sets_requires_same_test_questions() -> None:
    truth = {
        "projects": [
            {
                "project_id": PID,
                "missing_requirements": [
                    *(
                        {"requirement_id": r, "reason": "injected", "use_for_scoring": True}
                        for r in ("T1", "T2", "T3", "T4")
                    ),
                    {"requirement_id": "O1", "reason": "llm_omitted", "use_for_scoring": False},
                ],
            }
        ]
    }

    assert check_sets(JUDGMENTS, {PID: table()}, truth) == {(PID, "O1")}
    with pytest.raises(ValueError, match="injected"):
        check_sets(JUDGMENTS[:3], {PID: table()}, truth)


def test_load_judgments_splits_semicolon_ids(tmp_path: Path) -> None:
    path = tmp_path / "j.csv"
    path.write_text(
        "no,project_id,requirement_id,judgment,matched_wbs_ids,note\r\n"
        "1,PRJ-01,T1,애매,2.1;3.4,메모\r\n"
        "2,PRJ-01,T2,없음,,\r\n",
        encoding="utf-8-sig",
    )

    rows = load_judgments(path)

    assert rows[0].matched_wbs_ids == ["2.1", "3.4"]
    assert rows[1].matched_wbs_ids == []
    assert rows[1].judgment == "없음"


def test_load_judgments_rejects_unknown_judgment(tmp_path: Path) -> None:
    path = tmp_path / "j.csv"
    path.write_text(
        "no,project_id,requirement_id,judgment,matched_wbs_ids,note\n1,PRJ-01,T1,모름,,\n",
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError):
        load_judgments(path)
