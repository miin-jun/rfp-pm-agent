"""`rtm.report` (이슈 #39) — 점수표 읽기, 사업 RTM 만들기(지연·산출물·근거), CSV·Markdown 출력. DB·네트워크 없음."""

from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path

import pytest

from rfp_pm_agent.rtm.report import (
    ScoreTable,
    build_all,
    build_project_rtm,
    load_scores,
    render_csv,
    render_md,
)
from rfp_pm_agent.schemas.pms import (
    DeliverableRow,
    PmsDataset,
    ProjectRow,
    RequirementRow,
    WbsItemRow,
)
from tests.fakes.fake_pms_reader import FakePmsReader

AS_OF = date(2026, 9, 24)
PID = "PRJ-01"
DOC = "d0c"


def project() -> ProjectRow:
    return ProjectRow(
        project_id=PID,
        name="테스트 사업",
        notice_no=None,
        agency=None,
        contractor="가람소프트(가상)",
        start_date=date(2026, 4, 1),
        end_date=date(2027, 1, 31),
        status="in_progress",
    )


def requirement(rid: str, title: str) -> RequirementRow:
    return RequirementRow(
        project_id=PID,
        requirement_id=rid,
        category=None,
        title=title,
        description=title,
        source_doc_id=DOC,
        source_chunk_id=f"{DOC}:block_requirement:{rid}",
    )


def task(
    wbs_id: str, name: str, status: str, planned_end: date, parent: str | None = "2"
) -> WbsItemRow:
    done = status == "done"
    return WbsItemRow.model_validate(
        {
            "project_id": PID,
            "wbs_id": wbs_id,
            "parent_wbs_id": parent,
            "name": name,
            "phase": "설계",
            "assignee_user_id": "U001",
            "planned_start": date(2026, 6, 1),
            "planned_end": planned_end,
            "actual_start": date(2026, 6, 1) if status != "not_started" else None,
            "actual_end": planned_end if done else None,
            "progress_pct": 100 if done else 40,
            "status": status,
        }
    )


def deliverable(did: str, wbs_id: str, status: str, due: date) -> DeliverableRow:
    return DeliverableRow.model_validate(
        {
            "deliverable_id": did,
            "project_id": PID,
            "name": did,
            "phase": "설계",
            "wbs_id": wbs_id,
            "due_date": due,
            "submitted_date": None if status == "not_submitted" else due,
            "status": status,
            "doc_id": None,
        }
    )


SCORES = {
    # 요구사항 → 업무 → 점수. 9.9는 어느 요구사항과도 0.5 미만인 무관한 업무(범위 외 후보)
    "SFR-001": {"2.1": 0.92, "2.2": 0.50, "9.9": 0.03},
    "SER-001": {"2.1": 0.4999, "2.2": 0.20, "9.9": 0.10},
}
TASKS = [
    task("2", "설계 단계", "in_progress", date(2026, 9, 30), parent=None),
    task("2.1", "화면 설계", "in_progress", date(2026, 9, 21)),  # 지연 3일
    task("2.2", "DB 설계", "done", date(2026, 9, 1)),
    task("9.9", "사내 체육대회 준비", "not_started", date(2026, 12, 1)),
]
REQUIREMENTS = [requirement("SFR-001", "화면 | 구성"), requirement("SER-001", "보안 교육")]
DELIVERABLES = [
    deliverable("PRJ-01-D002", "2.1", "not_submitted", date(2026, 9, 20)),
    deliverable("PRJ-01-D001", "2.1", "approved", date(2026, 9, 1)),
    deliverable("PRJ-01-D003", "2.2", "submitted", date(2026, 9, 1)),
]


def write_scores(path: Path, scores: dict[str, dict[str, float]], injected: set[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["project_id", "requirement_id", "chunk_id", "wbs_id", "score", "is_injected"])
        for rid, row in scores.items():
            for wid, s in row.items():
                chunk = f"{DOC}:block_requirement:{rid}"
                w.writerow([PID, rid, chunk, wid, repr(s), str(rid in injected).lower()])


def table() -> ScoreTable:
    return ScoreTable(
        project_id=PID,
        requirement_ids=list(SCORES),
        wbs_ids=["2.1", "2.2", "9.9"],
        scores={(r, w): s for r, row in SCORES.items() for w, s in row.items()},
        chunk_ids={r: f"{DOC}:block_requirement:{r}" for r in SCORES},
        injected=set(),
    )


def test_load_scores_reads_full_matrix(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    write_scores(path, SCORES, injected={"SER-001"})

    tables = load_scores(path)

    t = tables[PID]
    assert t.requirement_ids == ["SFR-001", "SER-001"]
    assert t.wbs_ids == ["2.1", "2.2", "9.9"]
    assert t.scores[("SER-001", "2.1")] == 0.4999
    assert t.chunk_ids["SFR-001"] == f"{DOC}:block_requirement:SFR-001"
    assert t.injected == {"SER-001"}


def test_load_scores_rejects_missing_pair(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    partial = {"SFR-001": {"2.1": 0.9, "2.2": 0.1}, "SER-001": {"2.1": 0.2}}
    write_scores(path, partial, injected=set())

    with pytest.raises(ValueError, match="빠진 쌍"):
        load_scores(path)


def test_build_project_rtm_links_delay_deliverables_and_evidence() -> None:
    rtm = build_project_rtm(table(), project(), REQUIREMENTS, TASKS, DELIVERABLES, AS_OF)

    sfr = rtm.requirements[0]
    assert sfr.requirement_id == "SFR-001"
    assert sfr.chunk_id == f"{DOC}:block_requirement:SFR-001"
    assert [(link.task.wbs_id, link.score) for link in sfr.links] == [("2.1", 0.92), ("2.2", 0.5)]
    first = sfr.links[0]
    assert (first.task.delayed, first.task.delay_days) == (True, 3)
    assert first.deliverables == ["PRJ-01-D001 approved", "PRJ-01-D002 not_submitted 기한 지남"]
    assert (sfr.links[1].task.delayed, sfr.links[1].task.delay_days) == (False, None)
    assert [r.requirement_id for r in rtm.missing] == ["SER-001"]


def test_build_project_rtm_excludes_phase_tasks_and_finds_planted_out_of_scope() -> None:
    rtm = build_project_rtm(table(), project(), REQUIREMENTS, TASKS, DELIVERABLES, AS_OF)

    assert [t.wbs_id for t in rtm.out_of_scope] == ["9.9"]
    assert "2" not in {link.task.wbs_id for r in rtm.requirements for link in r.links}


def test_build_project_rtm_rejects_scores_for_other_tasks() -> None:
    with pytest.raises(ValueError, match="업무"):
        build_project_rtm(table(), project(), REQUIREMENTS, TASKS[:3], DELIVERABLES, AS_OF)


def test_build_project_rtm_rejects_chunk_id_mismatch() -> None:
    other = [
        requirement("SFR-001", "화면"),
        REQUIREMENTS[1].model_copy(update={"source_chunk_id": "x"}),
    ]

    with pytest.raises(ValueError, match="chunk_id"):
        build_project_rtm(table(), project(), other, TASKS, DELIVERABLES, AS_OF)


def test_render_csv_has_one_row_per_link_missing_and_out_of_scope() -> None:
    rtm = build_project_rtm(table(), project(), REQUIREMENTS, TASKS, DELIVERABLES, AS_OF)

    rows = list(csv.DictReader(io.StringIO(render_csv(rtm))))

    assert [(r["section"], r["requirement_id"], r["wbs_id"]) for r in rows] == [
        ("연결", "SFR-001", "2.1"),
        ("연결", "SFR-001", "2.2"),
        ("누락 후보", "SER-001", ""),
        ("범위 외 후보", "", "9.9"),
    ]
    assert rows[0]["score"] == "0.9200"
    assert rows[0]["delay_days"] == "3"
    assert rows[2]["chunk_id"] == f"{DOC}:block_requirement:SER-001"


def test_render_md_escapes_pipes_and_lists_sections() -> None:
    rtm = build_project_rtm(table(), project(), REQUIREMENTS, TASKS, DELIVERABLES, AS_OF)

    md = render_md(rtm)

    assert "화면 \\| 구성" in md
    assert "2.1 화면 설계 (0.9200)" in md
    assert "지연 3일" in md
    assert "## 누락 후보" in md and "| SER-001 | 보안 교육 |" in md
    assert "| 9.9 | 사내 체육대회 준비 |" in md


def test_build_all_requires_same_projects_in_db_and_scores() -> None:
    ds = PmsDataset.model_validate(
        {
            "as_of_date": AS_OF,
            "seed": 42,
            "projects": [project()],
            "users": [],
            "project_members": [],
            "requirements": REQUIREMENTS,
            "documents": [],
            "wbs_items": TASKS,
            "wbs_requirements": [],
            "deliverables": DELIVERABLES,
            "risks": [],
            "issues": [],
        }
    )
    reader = FakePmsReader(ds)

    assert [r.project_id for r in build_all(reader, {PID: table()}, AS_OF)] == [PID]
    with pytest.raises(ValueError, match="사업"):
        build_all(reader, {PID: table(), "PRJ-02": table()}, AS_OF)
