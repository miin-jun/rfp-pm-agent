"""합성 PMS 생성·검증·정답 (이슈 #20). 네트워크·DB 없이 가짜 문서와 정해 둔 초안으로 만든다."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pytest

from rfp_pm_agent.pms.rules import is_delayed, is_delayed_this_week
from rfp_pm_agent.pms.synth import (
    MISSING_RATIO,
    Source,
    build_requirements,
    build_truth,
    check_as_of,
    generate,
    pick_missing,
    project_rng,
    select_sources,
    stats,
    unmapped_requirements,
    validate,
)
from rfp_pm_agent.schemas.document import Requirement
from rfp_pm_agent.schemas.pms import PmsDataset
from tests.fakes.pms_sample import AS_OF, SEED, build_sample, document, draft


@pytest.fixture(scope="module")
def built() -> tuple[PmsDataset, dict[str, object]]:
    return build_sample()


# --- 생성 결과 ---


def test_generated_dataset_passes_validation(built: tuple[PmsDataset, dict[str, object]]) -> None:
    ds, _ = built
    assert validate(ds) == []


def test_generation_is_deterministic() -> None:
    first, truth1 = build_sample()
    second, truth2 = build_sample()
    assert first.model_dump_json() == second.model_dump_json()
    assert truth1 == truth2


def test_different_seed_changes_schedule() -> None:
    first, _ = build_sample(seed=1)
    second, _ = build_sample(seed=2)
    assert [w.planned_end for w in first.wbs_items] != [w.planned_end for w in second.wbs_items]


def test_every_project_has_delayed_open_risk_and_unsubmitted(
    built: tuple[PmsDataset, dict[str, object]],
) -> None:
    ds, _ = built
    for p in ds.projects:
        items = [w for w in ds.wbs_items if w.project_id == p.project_id]
        assert any(is_delayed(w.status, w.planned_end, AS_OF) for w in items)
        assert any(is_delayed_this_week(w.status, w.planned_end, AS_OF) for w in items)
        assert any(r.status == "open" for r in ds.risks if r.project_id == p.project_id)
        assert any(
            d.status == "not_submitted" and d.due_date < AS_OF
            for d in ds.deliverables
            if d.project_id == p.project_id
        )


def test_delay_injection_is_about_fifteen_percent_of_leaves(
    built: tuple[PmsDataset, dict[str, object]],
) -> None:
    ds, _ = built
    for p in ds.projects:
        leaves = [w for w in ds.wbs_items if w.project_id == p.project_id and w.parent_wbs_id]
        delayed = [w for w in leaves if is_delayed(w.status, w.planned_end, AS_OF)]
        assert len(delayed) == max(2, round(len(leaves) * 0.15))


def test_risk_and_issue_counts_and_required_statuses(
    built: tuple[PmsDataset, dict[str, object]],
) -> None:
    ds, _ = built
    for p in ds.projects:
        risks = [r.status for r in ds.risks if r.project_id == p.project_id]
        issues = [i.status for i in ds.issues if i.project_id == p.project_id]
        assert 5 <= len(risks) <= 10 and 5 <= len(issues) <= 10
        assert risks.count("open") >= 2 and "mitigating" in risks and "closed" in risks
        assert issues.count("open") >= 2 and "in_progress" in issues and "resolved" in issues


def test_missing_requirements_are_unmapped_and_marked_injected(
    built: tuple[PmsDataset, dict[str, object]],
) -> None:
    ds, truth = built
    unmapped = unmapped_requirements(ds)
    for p in truth["projects"]:  # type: ignore[attr-defined]
        missing = p["missing_requirements"]
        assert len(missing) == max(1, round(25 * MISSING_RATIO))
        assert {m["reason"] for m in missing} == {"injected"}
        assert [m["requirement_id"] for m in missing] == unmapped[p["project_id"]]


def test_llm_omitted_requirement_is_marked_separately() -> None:
    sources = [Source(document("docA"), None, "")]
    reqs = build_requirements("PRJ-01", sources[0].document)
    missing = pick_missing(SEED, "docA", [r.requirement_id for r in reqs])
    given = [r.requirement_id for r in reqs if r.requirement_id not in missing]
    omitted = given[-1]
    # 요구사항이 업무 수보다 적어 모두 연결되도록, 빠뜨릴 1건만 뺀 목록으로 초안을 만든다
    ds, injected = generate(sources, {"docA": draft(given[:-1])}, as_of=AS_OF, seed=SEED)

    truth = build_truth(
        ds, excluded={"PRJ-01": missing}, injected=injected, doc_ids={"PRJ-01": "docA"}
    )

    reasons = {
        m["requirement_id"]: m["reason"] for m in truth["projects"][0]["missing_requirements"]
    }
    assert reasons[omitted] == "llm_omitted"
    assert sum(r == "injected" for r in reasons.values()) == len(missing)


def test_truth_rtm_lists_tasks_and_their_deliverables(
    built: tuple[PmsDataset, dict[str, object]],
) -> None:
    ds, truth = built
    project = truth["projects"][0]  # type: ignore[index]
    entry = project["rtm"][0]
    deliverables = {d.deliverable_id: d.wbs_id for d in ds.deliverables}
    assert entry["wbs_ids"]
    assert all(deliverables[d] in entry["wbs_ids"] for d in entry["deliverable_ids"])
    assert project["out_of_scope_tasks"] == ["1.1", "1.2"]  # 요구사항 없는 사업관리 업무


def test_open_risks_sorted_by_probability_times_impact(
    built: tuple[PmsDataset, dict[str, object]],
) -> None:
    _, truth = built
    for p in truth["projects"]:  # type: ignore[attr-defined]
        scores = [r["score"] for r in p["open_risks"]]
        assert scores == sorted(scores, reverse=True)


def test_monday_as_of_has_no_this_week_delay_but_still_valid() -> None:
    ds, truth = build_sample(as_of=date(2026, 9, 21))
    assert validate(ds) == []
    assert truth["this_week_range"] is None
    assert stats(ds, truth)["delayed_this_week"] == 0
    assert stats(ds, truth)["delayed_leaf_tasks"] > 0


def test_users_are_fake(built: tuple[PmsDataset, dict[str, object]]) -> None:
    ds, _ = built
    assert all(u.email.endswith("@example.com") for u in ds.users)
    assert all("(가상)" in u.org for u in ds.users)
    assert len({u.user_id for u in ds.users}) == len(ds.users)


def test_stats_counts_match_truth(built: tuple[PmsDataset, dict[str, object]]) -> None:
    ds, truth = built
    s = stats(ds, truth)
    assert s["tables"]["projects"] == 2
    assert (
        s["missing_requirements"] == s["missing_injected"] == 2 * max(1, round(25 * MISSING_RATIO))
    )
    assert (
        s["rtm_coverage_given_to_llm"]
        == f"{50 - s['missing_injected']}/{50 - s['missing_injected']}"
    )


# --- 요구사항 추출 ---


def test_build_requirements_falls_back_to_raw_fields_and_text() -> None:
    doc = document("docA", 0)
    doc.requirements = [
        Requirement(
            requirement_id="SFR-001",
            prefix="SFR",
            fields={},
            raw_fields={"요구사항 명": "공정 스케줄 관리", "요구사항 내용": "스케줄 조회"},
            text="본문",
            source_order=0,
        ),
        Requirement(
            requirement_id="SFR-002",
            prefix="SFR",
            fields={},
            raw_fields={},
            text="본문만",
            source_order=1,
        ),
        Requirement(
            requirement_id="SFR-001",
            prefix="SFR",
            fields={},
            raw_fields={},
            text="중복",
            source_order=2,
        ),
    ]

    rows = build_requirements("PRJ-01", doc)

    assert [(r.requirement_id, r.title, r.description, r.category) for r in rows] == [
        ("SFR-001", "공정 스케줄 관리", "스케줄 조회", "SFR"),
        ("SFR-002", "SFR-002", "본문만", "SFR"),
    ]
    assert rows[0].source_chunk_id == "docA:block_requirement:SFR-001"


def test_pick_missing_is_deterministic_and_about_twelve_percent() -> None:
    ids = [f"SFR-{i:03d}" for i in range(100)]
    assert pick_missing(SEED, "docA", ids) == pick_missing(SEED, "docA", ids)
    assert len(pick_missing(SEED, "docA", ids)) == 12
    assert len(pick_missing(SEED, "docA", ids[:3])) == 1  # 최소 1건


def test_select_sources_needs_indexed_requirement_chunks(tmp_path: Path) -> None:
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for doc_id, n in (("indexed", 20), ("notindexed", 20), ("small", 5)):
        (parsed / f"{doc_id}.json").write_text(document(doc_id, n).model_dump_json(), "utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunk_lines = [
        json.dumps({"chunk_id": f"{d}:block_requirement:SFR-{i:03d}"})
        for d in ("indexed", "small")
        for i in range(1, 21)
    ]
    chunks.write_text("\n".join(chunk_lines) + "\n", "utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"doc_id": "indexed", "notice_no": "R1", "sha256": "ab"}) + "\n", "utf-8"
    )

    sources = select_sources(parsed, manifest, chunks)

    assert [(s.document.doc_id, s.notice_no, s.sha256) for s in sources] == [
        ("indexed", "R1", "ab")
    ]


# --- 검증: 망가뜨린 데이터를 잡는다 ---


def _broken(mutate: Callable[[PmsDataset], None]) -> list[str]:
    ds, _ = build_sample()
    mutate(ds)
    return validate(ds)


def _leaf(ds: PmsDataset, status: str | None = None):  # type: ignore[no-untyped-def]
    return next(
        w for w in ds.wbs_items if w.parent_wbs_id and (status is None or w.status == status)
    )


def test_validate_catches_reversed_planned_dates() -> None:
    def mutate(ds: PmsDataset) -> None:
        w = _leaf(ds)
        w.planned_start = w.planned_end + timedelta(days=1)

    assert any("planned_start > planned_end" in e for e in _broken(mutate))


def test_validate_catches_child_outside_parent() -> None:
    def mutate(ds: PmsDataset) -> None:
        w = _leaf(ds)
        w.planned_end = next(
            p for p in ds.wbs_items if p.wbs_id == w.parent_wbs_id
        ).planned_end + timedelta(days=1)

    assert any("상위 업무 기간 밖" in e for e in _broken(mutate))


def test_validate_catches_done_without_actual_end() -> None:
    def mutate(ds: PmsDataset) -> None:
        _leaf(ds, "done").actual_end = None

    assert any("done ⇔ progress 100" in e for e in _broken(mutate))


def test_validate_catches_assignee_not_member() -> None:
    def mutate(ds: PmsDataset) -> None:
        _leaf(ds).assignee_user_id = "U999"

    assert any("프로젝트 멤버가 아니다" in e for e in _broken(mutate))


def test_validate_catches_dangling_links() -> None:
    def mutate(ds: PmsDataset) -> None:
        ds.wbs_requirements[0].requirement_id = "XXX-001"
        ds.deliverables[0].wbs_id = "9.9"
        ds.issues[0].related_risk_id = (
            "PRJ-02-R01" if ds.issues[0].project_id == "PRJ-01" else "PRJ-01-R01"
        )

    errors = _broken(mutate)
    assert any("없는 요구사항" in e for e in errors)
    assert any("없는 업무 9.9" in e for e in errors)
    assert any("같은 프로젝트에 없는 Risk" in e for e in errors)


def test_validate_catches_future_actual_and_submission_dates() -> None:
    def mutate(ds: PmsDataset) -> None:
        _leaf(ds, "done").actual_end = AS_OF
        next(d for d in ds.deliverables if d.submitted_date).submitted_date = AS_OF

    errors = _broken(mutate)
    assert any("기준일 당일 또는 이후" in e and e.startswith("wbs") for e in errors)
    assert any("제출일이 기준일 당일 또는 이후" in e for e in errors)


def test_validate_catches_duplicate_keys() -> None:
    def mutate(ds: PmsDataset) -> None:
        ds.projects.append(ds.projects[0])
        ds.risks.append(ds.risks[0])

    errors = _broken(mutate)
    assert any("projects 키 중복" in e for e in errors)
    assert any("risks 키 중복" in e for e in errors)


def test_validate_catches_resolved_issue_without_date() -> None:
    def mutate(ds: PmsDataset) -> None:
        next(i for i in ds.issues if i.status == "resolved").resolved_date = None

    assert any("해결일과 상태" in e for e in _broken(mutate))


# --- reviewer 보강: 고정 기대값·새 검증 규칙·기준일 확인 ---


def test_fixed_seed_produces_known_values(built: tuple[PmsDataset, dict[str, object]]) -> None:
    """같은 프로세스 안의 두 번 비교만으로는 seed 방식이 바뀌어도(예: hash() 기반) 잡히지 않는다. 값을 고정한다."""
    ds, truth = built
    project = ds.projects[0]
    task = next(w for w in ds.wbs_items if w.project_id == "PRJ-01" and w.wbs_id == "1.3")
    assert (project.start_date, project.end_date, project.contractor) == (
        date(2026, 6, 5),
        date(2027, 3, 19),
        "누리시스템(가상)",
    )
    assert (task.planned_start, task.planned_end, task.status) == (
        date(2026, 7, 1),
        date(2026, 7, 14),
        "done",
    )
    assert ds.users[0].name == "송재소"
    missing = truth["projects"][0]["missing_requirements"]  # type: ignore[index]
    assert [m["requirement_id"] for m in missing] == ["SFR-004", "SFR-010", "SFR-015"]


def test_risk_and_issue_titles_are_unique_within_project(
    built: tuple[PmsDataset, dict[str, object]],
) -> None:
    ds, _ = built
    for p in ds.projects:
        for rows in (
            [r.title for r in ds.risks if r.project_id == p.project_id],
            [i.title for i in ds.issues if i.project_id == p.project_id],
        ):
            assert len(rows) == len(set(rows))


def test_truth_records_draft_hash_when_given() -> None:
    ds, _ = build_sample()
    truth = build_truth(
        ds,
        excluded={},
        injected={},
        doc_ids={"PRJ-01": "docA", "PRJ-02": "docB"},
        draft_sha256={"PRJ-01": "aa"},
    )
    assert [p["draft_sha256"] for p in truth["projects"]] == ["aa", None]


def test_check_as_of_rejects_mismatch() -> None:
    check_as_of(AS_OF, AS_OF, "x")
    with pytest.raises(ValueError, match="AS_OF_DATE"):
        check_as_of(date(2026, 9, 21), AS_OF, "data/eval/pms_truth_v1.json")


def test_validate_catches_not_started_with_actual_start() -> None:
    def mutate(ds: PmsDataset) -> None:
        _leaf(ds, "not_started").actual_start = date(2026, 9, 1)

    assert any("실적 시작일이 맞지 않는다" in e for e in _broken(mutate))


def test_validate_catches_done_parent_with_unfinished_child() -> None:
    def mutate(ds: PmsDataset) -> None:
        parent = next(w for w in ds.wbs_items if w.parent_wbs_id is None and w.status == "done")
        child = next(
            w
            for w in ds.wbs_items
            if w.project_id == parent.project_id and w.parent_wbs_id == parent.wbs_id
        )
        child.status = "in_progress"
        child.progress_pct = 50
        child.actual_end = None

    assert any("done인데 하위 업무가 미완료" in e for e in _broken(mutate))


def test_validate_catches_deliverable_phase_mismatch() -> None:
    def mutate(ds: PmsDataset) -> None:
        d = ds.deliverables[0]
        d.phase = "이행" if d.phase != "이행" else "분석"

    assert any("업무 단계" in e for e in _broken(mutate))


def test_validate_catches_requirement_without_document_and_member_outside_project() -> None:
    def mutate(ds: PmsDataset) -> None:
        ds.requirements[0].source_doc_id = "nodoc"
        ds.project_members[0].end_date = ds.projects[0].end_date + timedelta(days=1)

    errors = _broken(mutate)
    assert any("없는 문서 nodoc" in e for e in errors)
    assert any(e.startswith("project_members") and "프로젝트 기간 밖" in e for e in errors)


def test_validate_catches_duplicate_links_and_members() -> None:
    def mutate(ds: PmsDataset) -> None:
        ds.wbs_requirements.append(ds.wbs_requirements[0])
        ds.project_members.append(ds.project_members[0])

    errors = _broken(mutate)
    assert any("wbs_requirements 키 중복" in e for e in errors)
    assert any("project_members 키 중복" in e for e in errors)


# --- v2: 누락 후보는 사업관리 요구사항을 고르지 않는다 (ADR-0008) ---


def test_pick_missing_skips_management_requirements() -> None:
    ids = (
        [f"PMR-{i:03d}" for i in range(50)]
        + [f"PSR-{i:03d}" for i in range(30)]
        + [f"SFR-{i:03d}" for i in range(20)]
    )
    picked = pick_missing(SEED, "docA", ids)
    assert len(picked) == 12  # 개수는 전체(100건) 기준
    assert all(rid.startswith("SFR-") for rid in picked)


def test_pick_missing_takes_what_is_left_when_few_non_management() -> None:
    ids = [f"PMR-{i:03d}" for i in range(40)] + ["SFR-001", "DAR-001"]
    assert pick_missing(SEED, "docA", ids) == ["SFR-001", "DAR-001"]
    assert pick_missing(SEED, "docA", ["PMR-001", "PSR-001"]) == []


def test_this_week_delay_is_made_even_when_current_phase_has_only_future_tasks() -> None:
    """버그(v2 PRJ-08): 기준일이 단계 초반이고 그 단계 업무가 모두 기준일 뒤에 시작하면 이번 주 지연을 못 만들었다."""
    from rfp_pm_agent.pms.synth import _inject_delays
    from rfp_pm_agent.schemas.pms import WbsItemRow

    def leaf(wbs_id: str, parent: str, start: date, end: date, status: str) -> WbsItemRow:
        done = status == "done"
        return WbsItemRow(
            project_id="PRJ-01",
            wbs_id=wbs_id,
            parent_wbs_id=parent,
            name=wbs_id,
            phase="설계" if parent == "2" else "구현",
            assignee_user_id="U001",
            planned_start=start,
            planned_end=end,
            actual_start=start if done else None,
            actual_end=end if done else None,
            progress_pct=100 if done else 0,
            status="done" if done else "not_started",
        )

    leaves = [
        leaf("2.1", "2", date(2026, 8, 1), date(2026, 9, 6), "done"),
        leaf("2.2", "2", date(2026, 8, 1), date(2026, 9, 10), "done"),
        leaf("3.1", "3", date(2026, 10, 19), date(2026, 11, 4), "not_started"),
    ]
    spans = {"2": (date(2026, 7, 8), date(2026, 9, 11)), "3": (date(2026, 9, 12), date(2027, 1, 4))}

    injected = _inject_delays(project_rng(SEED, "docA", "delay"), leaves, spans, AS_OF)

    this_week = [w for w in leaves if is_delayed_this_week(w.status, w.planned_end, AS_OF)]
    assert [w.wbs_id for w in this_week] == ["3.1"]
    task = this_week[0]
    assert task.wbs_id in injected
    assert date(2026, 9, 12) <= task.planned_start <= task.planned_end  # 단계 안
    assert task.actual_start == task.planned_start and task.status == "in_progress"
