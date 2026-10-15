"""PMS 조회 툴 단위 테스트 (이슈 #21). DB 없이 #20 샘플 데이터셋을 읽는 가짜 리더로 입력 검증·결과 조립을 본다.

정답은 같은 샘플로 `build_truth`가 계산한 값(합성 정답 파일과 같은 함수)이다. 실제 DB·정답 파일 대조는
tests/integration/test_pms_tools_db.py.
"""

from __future__ import annotations

import inspect
from datetime import date, timedelta
from typing import Any

import pytest

from rfp_pm_agent.pms.rules import risk_level
from rfp_pm_agent.schemas.pms import PmsDataset
from rfp_pm_agent.tools.pms_tools import (
    MAX_LIMIT,
    PmsDeps,
    get_delayed_tasks,
    get_deliverables,
    get_open_risks,
)
from tests.fakes.fake_pms_reader import FakePmsReader
from tests.fakes.pms_sample import AS_OF, build_sample

TOOLS = [get_delayed_tasks, get_open_risks, get_deliverables]
# LLM이 목록 밖 값을 넘기는 경우 — 타입 힌트(Literal)는 실행 시 막지 않으므로 함수 안 검사로 막는다
BAD_SCOPE: Any = "week"
BAD_LEVEL: Any = "critical"
BAD_FILTER: Any = "late"


@pytest.fixture(scope="module")
def sample() -> tuple[PmsDataset, dict[str, Any]]:
    return build_sample()


@pytest.fixture
def deps(sample: tuple[PmsDataset, dict[str, Any]]) -> PmsDeps:
    return PmsDeps(reader=FakePmsReader(sample[0]), as_of_date=AS_OF)


def _truth(sample: tuple[PmsDataset, dict[str, Any]], pid: str) -> dict[str, Any]:
    project: dict[str, Any] = next(p for p in sample[1]["projects"] if p["project_id"] == pid)
    return project


# --- 프로젝트 지정 ---


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("project_id", [None, "", "   "])
def test_missing_project_returns_project_required_with_candidates(
    tool: Any, project_id: str | None, deps: PmsDeps
) -> None:
    result = tool(project_id, deps=deps)

    assert result.status == "project_required"
    assert result.items == [] and result.total == 0
    assert "물어본" in result.message
    assert [(p.project_id, p.doc_ids) for p in result.available_projects] == [
        ("PRJ-01", ["docA"]),
        ("PRJ-02", ["docB"]),
    ]


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("project_id", ["PRJ-99", "PRJ-01' OR '1'='1"])
def test_unknown_project_returns_project_not_found(
    tool: Any, project_id: str, deps: PmsDeps
) -> None:
    result = tool(project_id, deps=deps)

    assert result.status == "project_not_found"
    assert result.project_id == project_id and result.items == []
    assert len(result.available_projects) == 2


def test_unknown_project_reads_no_project_rows(deps: PmsDeps) -> None:
    get_delayed_tasks("PRJ-99", deps=deps)

    reader = deps.reader
    assert isinstance(reader, FakePmsReader)
    assert {name for name, _ in reader.calls} == {"projects", "documents"}


@pytest.mark.parametrize("tool", TOOLS)
def test_ok_result_has_project_header_and_doc_ids(tool: Any, deps: PmsDeps) -> None:
    result = tool("PRJ-01", deps=deps)

    assert result.status == "ok"
    assert (result.project_id, result.project_name, result.doc_ids) == (
        "PRJ-01",
        "docA 사업",
        ["docA"],
    )
    assert result.as_of_date == AS_OF and result.available_projects == []


# --- 입력 검증 ---


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("limit", [0, MAX_LIMIT + 1, -1, True, "5", 2.0])
def test_limit_out_of_range_is_value_error(tool: Any, limit: Any, deps: PmsDeps) -> None:
    with pytest.raises(ValueError, match="limit"):
        tool("PRJ-01", limit=limit, deps=deps)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda d: get_delayed_tasks("PRJ-01", scope=BAD_SCOPE, deps=d), "scope"),
        (lambda d: get_open_risks("PRJ-01", min_level=BAD_LEVEL, deps=d), "min_level"),
        (lambda d: get_deliverables("PRJ-01", status_filter=BAD_FILTER, deps=d), "status_filter"),
        (lambda d: get_deliverables("PRJ-01", wbs_id="9.9", deps=d), "9.9"),
    ],
)
def test_bad_enum_or_unknown_task_is_value_error(call: Any, match: str, deps: PmsDeps) -> None:
    with pytest.raises(ValueError, match=match):
        call(deps)


def test_bad_argument_is_checked_before_project(deps: PmsDeps) -> None:
    """project_id가 없어도 잘못된 인자는 먼저 ValueError다(되묻기 전에 인자부터 고친다)."""
    with pytest.raises(ValueError):
        get_delayed_tasks(None, scope=BAD_SCOPE, deps=deps)
    with pytest.raises(ValueError):
        get_open_risks(None, limit=0, deps=deps)


# --- get_delayed_tasks ---


@pytest.mark.parametrize("pid", ["PRJ-01", "PRJ-02"])
def test_delayed_tasks_match_truth(pid: str, deps: PmsDeps, sample: Any) -> None:
    truth = _truth(sample, pid)["delayed_tasks"]

    result = get_delayed_tasks(pid, limit=MAX_LIMIT, deps=deps)

    assert result.total == len(truth)
    assert {t.wbs_id for t in result.items} == {t["wbs_id"] for t in truth}
    by_id = {t["wbs_id"]: t for t in truth}
    for item in result.items:
        assert item.delay_days == by_id[item.wbs_id]["delay_days"]
        assert item.this_week == by_id[item.wbs_id]["this_week"]


@pytest.mark.parametrize("pid", ["PRJ-01", "PRJ-02"])
def test_this_week_scope_matches_truth(pid: str, deps: PmsDeps, sample: Any) -> None:
    truth = [t["wbs_id"] for t in _truth(sample, pid)["delayed_tasks"] if t["this_week"]]

    result = get_delayed_tasks(pid, scope="this_week", deps=deps)

    assert sorted(t.wbs_id for t in result.items) == sorted(truth) and truth
    assert result.this_week_range == (date(2026, 9, 21), date(2026, 9, 23))


def test_delayed_tasks_sorted_and_enriched(deps: PmsDeps) -> None:
    result = get_delayed_tasks("PRJ-01", limit=MAX_LIMIT, deps=deps)

    days = [t.delay_days for t in result.items]
    assert days == sorted(days, reverse=True)
    leaf = next(t for t in result.items if t.level == 2)
    assert leaf.parent is not None and leaf.parent.wbs_id == leaf.wbs_id.split(".")[0]
    assert leaf.assignee.name  # 멤버 이름이 붙는다
    # 샘플(seed 42)에서 PRJ-01은 단계 "2"가 하위 업무 지연으로 지연이다
    phase = next(t for t in result.items if t.level == 1)
    assert phase.wbs_id == "2" and phase.parent is None


def test_delayed_task_requirement_ids_come_from_links(deps: PmsDeps, sample: Any) -> None:
    ds: PmsDataset = sample[0]
    result = get_delayed_tasks("PRJ-01", limit=MAX_LIMIT, deps=deps)

    for item in result.items:
        expected = sorted(
            x.requirement_id
            for x in ds.wbs_requirements
            if x.project_id == "PRJ-01" and x.wbs_id == item.wbs_id
        )
        assert item.requirement_ids == expected


def test_limit_truncates_but_total_counts_all(deps: PmsDeps) -> None:
    full = get_delayed_tasks("PRJ-01", limit=MAX_LIMIT, deps=deps)
    cut = get_delayed_tasks("PRJ-01", limit=2, deps=deps)

    assert full.total > 2
    assert (cut.total, len(cut.items), cut.truncated) == (full.total, 2, True)
    assert [t.wbs_id for t in cut.items] == [t.wbs_id for t in full.items[:2]]
    assert full.truncated is False


def test_as_of_argument_overrides_default(deps: PmsDeps) -> None:
    earlier = get_delayed_tasks("PRJ-01", as_of=date(2026, 6, 10), limit=MAX_LIMIT, deps=deps)
    default = get_delayed_tasks("PRJ-01", limit=MAX_LIMIT, deps=deps)

    assert earlier.as_of_date == date(2026, 6, 10)
    assert earlier.total < default.total
    assert all(t.planned_end < date(2026, 6, 10) for t in earlier.items)


def test_this_week_on_monday_is_empty_with_note(deps: PmsDeps) -> None:
    result = get_delayed_tasks("PRJ-01", scope="this_week", as_of=date(2026, 9, 21), deps=deps)

    assert result.items == [] and result.this_week_range is None
    assert "월요일" in result.message


# --- get_open_risks ---


@pytest.mark.parametrize("pid", ["PRJ-01", "PRJ-02"])
def test_open_risks_match_truth_order(pid: str, deps: PmsDeps, sample: Any) -> None:
    truth = _truth(sample, pid)["open_risks"]

    result = get_open_risks(pid, limit=MAX_LIMIT, deps=deps)

    assert [r.risk_id for r in result.items] == [r["risk_id"] for r in truth]
    assert [r.score for r in result.items] == [r["score"] for r in truth]
    assert all(r.status in ("open", "mitigating") for r in result.items)


def test_min_level_filters_by_score(deps: PmsDeps) -> None:
    every = get_open_risks("PRJ-01", limit=MAX_LIMIT, deps=deps)
    high = get_open_risks("PRJ-01", min_level="high", limit=MAX_LIMIT, deps=deps)
    medium = get_open_risks("PRJ-01", min_level="medium", limit=MAX_LIMIT, deps=deps)

    assert [r.risk_id for r in high.items] == [r.risk_id for r in every.items if r.score >= 15]
    assert [r.risk_id for r in medium.items] == [r.risk_id for r in every.items if r.score >= 8]
    assert all(r.level == risk_level(r.probability, r.impact) for r in every.items)


def test_open_risk_links_issues_and_task(deps: PmsDeps, sample: Any) -> None:
    ds: PmsDataset = sample[0]
    result = get_open_risks("PRJ-01", limit=MAX_LIMIT, deps=deps)

    for risk in result.items:
        expected = sorted(i.issue_id for i in ds.issues if i.related_risk_id == risk.risk_id)
        assert [i.issue_id for i in risk.issues] == expected
    linked = [r for r in result.items if r.related_task]
    assert linked  # 샘플에서 1번 Risk는 지연 업무와 연결된다
    # 샘플(seed 42)에서 PRJ-01-R01은 open이고 지연 업무 2.3과 연결된다
    r01 = next(r for r in result.items if r.risk_id == "PRJ-01-R01")
    assert r01.related_task is not None and r01.related_task.wbs_id == "2.3"
    assert r01.related_task_delayed is True


@pytest.mark.parametrize(
    ("p", "i", "level"),
    [(5, 3, "high"), (3, 5, "high"), (2, 4, "medium"), (2, 3, "low"), (1, 1, "low")],
)
def test_risk_level_boundaries(p: int, i: int, level: str) -> None:
    assert risk_level(p, i) == level


# --- get_deliverables ---


@pytest.mark.parametrize("pid", ["PRJ-01", "PRJ-02"])
def test_overdue_filter_matches_truth(pid: str, deps: PmsDeps, sample: Any) -> None:
    truth = _truth(sample, pid)["overdue_unsubmitted_deliverables"]

    result = get_deliverables(pid, status_filter="overdue", limit=MAX_LIMIT, deps=deps)

    assert sorted(d.deliverable_id for d in result.items) == sorted(truth) and truth
    assert all(d.overdue and d.days_overdue > 0 for d in result.items)


def test_status_filters_partition_all(deps: PmsDeps) -> None:
    every = get_deliverables("PRJ-01", limit=MAX_LIMIT, deps=deps)
    parts = [
        get_deliverables("PRJ-01", status_filter=f, limit=MAX_LIMIT, deps=deps).total
        for f in ("not_submitted", "submitted", "approved")
    ]
    assert sum(parts) == every.total


def test_deliverables_sort_overdue_first(deps: PmsDeps) -> None:
    result = get_deliverables("PRJ-01", limit=MAX_LIMIT, deps=deps)

    flags = [d.overdue for d in result.items]
    assert flags == sorted(flags, reverse=True)
    overdue_days = [d.days_overdue for d in result.items if d.overdue]
    assert overdue_days == sorted(overdue_days, reverse=True)


def test_wbs_id_returns_related_deliverables_via_shared_requirements(
    deps: PmsDeps, sample: Any
) -> None:
    ds: PmsDataset = sample[0]
    links = [x for x in ds.wbs_requirements if x.project_id == "PRJ-01"]
    task = links[0].wbs_id
    shared = {x.requirement_id for x in links if x.wbs_id == task}
    related = {task} | {x.wbs_id for x in links if x.requirement_id in shared}

    result = get_deliverables("PRJ-01", wbs_id=task, limit=MAX_LIMIT, deps=deps)

    expected = sorted(
        d.deliverable_id
        for d in ds.deliverables
        if d.project_id == "PRJ-01" and d.wbs_id in related
    )
    assert sorted(d.deliverable_id for d in result.items) == expected
    assert len(related) > 1  # 같은 요구사항의 다른 업무 산출물도 들어온다


def test_requirement_id_filter_and_unmapped_requirement(deps: PmsDeps, sample: Any) -> None:
    ds: PmsDataset = sample[0]
    rid = next(x.requirement_id for x in ds.wbs_requirements if x.project_id == "PRJ-01")
    tasks = {
        x.wbs_id
        for x in ds.wbs_requirements
        if x.project_id == "PRJ-01" and x.requirement_id == rid
    }

    result = get_deliverables("PRJ-01", requirement_id=rid, limit=MAX_LIMIT, deps=deps)
    assert result.items and all(
        d.task.wbs_id in tasks and rid in d.requirement_ids for d in result.items
    )

    missing = sample[1]["projects"][0]["missing_requirements"][0]["requirement_id"]
    none = get_deliverables("PRJ-01", requirement_id=missing, deps=deps)
    assert none.items == [] and "누락 후보" in none.message


def test_deliverable_as_of_changes_overdue(deps: PmsDeps) -> None:
    later = get_deliverables(
        "PRJ-01",
        status_filter="overdue",
        as_of=AS_OF + timedelta(days=60),
        limit=MAX_LIMIT,
        deps=deps,
    )
    now = get_deliverables("PRJ-01", status_filter="overdue", limit=MAX_LIMIT, deps=deps)
    assert later.total > now.total


# --- docstring = 툴 설명 ---


@pytest.mark.parametrize("tool", TOOLS)
def test_docstring_explains_arguments_rules_and_search_example(tool: Any) -> None:
    doc = inspect.getdoc(tool) or ""
    for needle in (
        "project_id",
        "project_required",
        "as_of",
        "AS_OF_DATE",
        "limit",
        "1~50",
        "기본 20",
        "search_documents(",
        "doc_ids",
    ):
        assert needle in doc, f"{tool.__name__} docstring에 {needle!r}가 없다"


@pytest.mark.parametrize("tool", TOOLS)
def test_deps_is_keyword_only_and_hidden_from_llm_arguments(tool: Any) -> None:
    params = inspect.signature(tool).parameters
    assert params["deps"].kind is inspect.Parameter.KEYWORD_ONLY
    assert next(iter(params)) == "project_id" and params["project_id"].default is None


# --- reviewer 차단 1: "누락 후보" 문구는 요구사항 단독으로 연결 업무가 없을 때만 ---


def test_empty_intersection_of_wbs_and_requirement_is_not_called_missing(
    deps: PmsDeps, sample: Any
) -> None:
    ds: PmsDataset = sample[0]
    links = [x for x in ds.wbs_requirements if x.project_id == "PRJ-01"]
    rid = links[0].requirement_id
    linked = {x.wbs_id for x in links if x.requirement_id == rid}
    shared_reqs = {x.requirement_id for x in links if x.wbs_id in linked}
    other = next(
        w.wbs_id
        for w in ds.wbs_items
        if w.project_id == "PRJ-01"
        and w.parent_wbs_id
        and w.wbs_id not in linked
        and not {x.requirement_id for x in links if x.wbs_id == w.wbs_id} & shared_reqs
    )

    result = get_deliverables("PRJ-01", wbs_id=other, requirement_id=rid, deps=deps)

    assert result.items == []
    assert "누락 후보" not in result.message


def test_unknown_requirement_id_is_value_error(deps: PmsDeps) -> None:
    with pytest.raises(ValueError, match="XXX-999"):
        get_deliverables("PRJ-01", requirement_id="XXX-999", deps=deps)


# --- as_of 인자: date 또는 "YYYY-MM-DD" ---


def test_as_of_accepts_iso_string(deps: PmsDeps) -> None:
    by_str = get_delayed_tasks("PRJ-01", as_of="2026-06-10", limit=MAX_LIMIT, deps=deps)
    by_date = get_delayed_tasks("PRJ-01", as_of=date(2026, 6, 10), limit=MAX_LIMIT, deps=deps)
    assert by_str == by_date


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("bad", ["2026/09/24", "어제", 20260924])
def test_bad_as_of_is_value_error(tool: Any, bad: Any, deps: PmsDeps) -> None:
    with pytest.raises(ValueError, match="as_of"):
        tool("PRJ-01", as_of=bad, deps=deps)


def test_level_is_wbs_depth(deps: PmsDeps) -> None:
    result = get_delayed_tasks("PRJ-01", limit=MAX_LIMIT, deps=deps)
    assert all(t.level == t.wbs_id.count(".") + 1 for t in result.items)
