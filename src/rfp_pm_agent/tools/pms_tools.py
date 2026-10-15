"""PMS 조회 툴 `get_delayed_tasks`·`get_open_risks`·`get_deliverables` (이슈 #21).

Agent(#22)·MCP 서버(#24)가 함께 부르는 읽기 전용 툴이다. 판단 규칙은 docs/data-design.md 6절이고 코드는
`pms.rules` 한 곳이다(합성 정답 `data/eval/pms_truth_v2.json`도 같은 함수로 계산했다). 인자·반환·오류 계약은
data-design.md 6절 "PMS 조회 툴".

- `PmsDeps`: DB 읽기(`PmsReader`)와 기본 기준일. LLM에게 보이는 인자와 분리해 Agent·MCP 어댑터가 미리 묶어 넘긴다
- 툴 함수의 docstring은 LLM에게 보이는 툴 설명이다. 내부 처리는 `_resolve_project`·`_header`와 함수 안 주석에 둔다
- 쓰기 동작은 없다. DB 계정도 SELECT만 된다(agent_ro)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, get_args

from rfp_pm_agent.pms.reader import PmsReader, SqlPmsReader, readonly_engine
from rfp_pm_agent.pms.rules import (
    RISK_LEVEL_ORDER,
    RiskLevel,
    delay_days,
    is_delayed,
    is_delayed_this_week,
    is_open_risk,
    is_overdue_deliverable,
    risk_level,
    risk_score,
    this_week_delay_range,
)
from rfp_pm_agent.schemas.pms import ProjectRow, WbsItemRow
from rfp_pm_agent.schemas.pms_tools import (
    DelayedTask,
    DelayedTasksResult,
    DeliverableFilter,
    DeliverableItem,
    DeliverablesResult,
    LinkedIssue,
    OpenRisk,
    OpenRisksResult,
    PersonRef,
    ProjectRef,
    TaskRef,
)

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
DELAY_SCOPES = ("all", "this_week")
DELIVERABLE_FILTERS: tuple[str, ...] = get_args(DeliverableFilter)


@dataclass(frozen=True)
class PmsDeps:
    """조회 툴이 쓰는 DB 읽기와 기본 기준일.

    - reader: 프로젝트 단위 조회(`PmsReader`). 실서비스는 읽기 전용 계정의 `SqlPmsReader`
    - as_of_date: 툴 인자 `as_of`를 주지 않을 때의 기준일(`AS_OF_DATE`). 시스템 날짜를 쓰지 않는다
    """

    reader: PmsReader
    as_of_date: date

    @classmethod
    def from_env(cls) -> PmsDeps:
        """`POSTGRES_READONLY_DSN`(agent_ro)과 `AS_OF_DATE`로 만든다. DB 연결은 첫 조회 때 연다."""
        from rfp_pm_agent.config import PostgresConfig, SynthConfig

        engine = readonly_engine(PostgresConfig.from_env().readonly_dsn)
        return cls(reader=SqlPmsReader(engine), as_of_date=SynthConfig.from_env().as_of_date)


# --- 공통 ---


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit은 1 이상 {MAX_LIMIT} 이하의 정수여야 한다: {limit!r}")


def _as_of_date(as_of: date | str | None, default: date) -> date:
    """기준일 인자: date 또는 "YYYY-MM-DD" 문자열(LLM 툴 호출은 문자열로 올 수 있다). 생략하면 default. 그 밖은 ValueError."""
    if as_of is None:
        return default
    if isinstance(as_of, date):
        return as_of
    if isinstance(as_of, str):
        try:
            return date.fromisoformat(as_of.strip())
        except ValueError as e:
            raise ValueError(f"as_of는 YYYY-MM-DD 날짜여야 한다: {as_of!r}") from e
    raise ValueError(f"as_of는 YYYY-MM-DD 날짜여야 한다: {as_of!r}")


def _wbs_sort_key(wbs_id: str) -> tuple[int, ...]:
    """ "3.10"이 "3.9" 뒤에 오도록 숫자로 비교한다."""
    return tuple(int(part) for part in re.findall(r"\d+", wbs_id))


@dataclass(frozen=True)
class _Resolved:
    project: ProjectRow | None
    doc_ids: list[str]
    header: dict[str, Any]


def _resolve_project(
    deps: PmsDeps, project_id: str | None, as_of: date | str | None, limit: int
) -> _Resolved:
    """project_id를 확인한다. 없거나 모르는 ID면 status·message·available_projects를 채운 머리를 돌려준다."""
    _validate_limit(limit)
    as_of_date = _as_of_date(as_of, deps.as_of_date)
    header: dict[str, Any] = {
        "status": "ok",
        "message": "",
        "project_id": None,
        "project_name": None,
        "doc_ids": [],
        "as_of_date": as_of_date,
        "total": 0,
        "limit": limit,
        "truncated": False,
        "available_projects": [],
    }
    projects = deps.reader.projects()
    wanted = (project_id or "").strip()
    match = next((p for p in projects if p.project_id == wanted), None)
    if match is None:
        header["available_projects"] = [
            ProjectRef(
                project_id=p.project_id,
                name=p.name,
                doc_ids=[d.doc_id for d in deps.reader.documents(p.project_id)],
            )
            for p in projects
        ]
        if not wanted:
            header["status"] = "project_required"
            header["message"] = (
                "대상 프로젝트(project_id)를 지정해야 한다. 사용자에게 어느 사업인지 물어본 뒤 "
                "available_projects의 project_id로 다시 부른다"
            )
        else:
            header["status"] = "project_not_found"
            header["project_id"] = wanted
            header["message"] = (
                f"project_id '{wanted}'인 프로젝트가 없다. available_projects에서 고르거나 사용자에게 확인한다"
            )
        return _Resolved(None, [], header)
    doc_ids = [d.doc_id for d in deps.reader.documents(match.project_id)]
    header.update(project_id=match.project_id, project_name=match.name, doc_ids=doc_ids)
    return _Resolved(match, doc_ids, header)


def _person(user_id: str, names: dict[str, str]) -> PersonRef:
    return PersonRef(user_id=user_id, name=names.get(user_id))


def _task_ref(task: WbsItemRow) -> TaskRef:
    return TaskRef(wbs_id=task.wbs_id, name=task.name, status=task.status)


def _requirements_by_task(links: list[Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for link in links:
        out[link.wbs_id].append(link.requirement_id)
    return {k: sorted(v) for k, v in out.items()}


# --- get_delayed_tasks ---


def get_delayed_tasks(
    project_id: str | None = None,
    scope: Literal["all", "this_week"] = "all",
    as_of: date | str | None = None,
    limit: int = DEFAULT_LIMIT,
    *,
    deps: PmsDeps,
) -> DelayedTasksResult:
    """프로젝트의 지연 업무(WBS)를 지연 일수가 긴 순서로 돌려준다.

    판단 규칙(data-design.md 6절)
    - 지연 업무: 상태가 done이 아니고 계획 종료일(planned_end)이 기준일보다 앞선 업무. 지연 일수 = 기준일 − planned_end
    - 이번 주 지연: 지연 업무 중 planned_end가 이번 주 월요일 ~ 기준일 전날인 것(이번 주에 새로 지연된 것).
      기준일이 월요일이면 이 구간이 비어 결과가 0건이다
    - 단계(level 1, 예: "2")와 그 아래 업무(level 2, 예: "2.3")를 모두 본다. 하위 업무가 늦으면 단계도 지연일 수 있다

    인자
    - project_id: 조회할 프로젝트 ID(예: "PRJ-03"). 프로젝트 1개 = RFP 1건이다. 없으면 오류 대신 status="project_required"와
      프로젝트 후보(available_projects: project_id·사업명·doc_ids)를 돌려준다 — 사용자에게 어느 사업인지 물어본 뒤 다시 부른다.
      모르는 ID면 status="project_not_found"
    - scope: "all"(기본, 누적된 지연 전부) 또는 "this_week"(이번 주에 새로 지연된 것만). 그 밖의 값은 ValueError
    - as_of: 기준일(YYYY-MM-DD). 생략하면 설정값 AS_OF_DATE. 시스템 날짜("오늘")를 쓰지 않는다
    - limit: 돌려받을 최대 건수. 1~50(기본 20). total에 전체 건수, truncated에 잘림 여부가 있다

    반환: 머리(status·message·project_id·project_name·doc_ids·as_of_date·total·limit·truncated) + scope·this_week_range +
    items(지연 일수 내림차순). 업무마다 wbs_id·name·level·phase·status·progress_pct·planned_start/end·delay_days·this_week·
    담당자(assignee: user_id·name)·상위 업무(parent)·연결 요구사항(requirement_ids).

    search_documents와 함께 쓰는 예 — "이번 주 지연된 업무랑 관련 요구사항 알려줘":
    1) get_delayed_tasks(project_id="PRJ-03", scope="this_week")
    2) 결과 업무의 requirement_ids(예: "SFR-004")와 업무 이름으로 search_documents(query="SFR-004 Graph RAG 엔진",
       doc_ids=결과의 doc_ids)를 불러 요구사항 원문과 출처를 붙인다

    오류: ValueError(scope·limit이 잘못됨), PmsUnavailableError(PMS DB에 연결할 수 없음 — 인자를 바꿔도 해결되지 않는다)
    """
    if scope not in DELAY_SCOPES:
        raise ValueError(f"scope는 {DELAY_SCOPES} 중 하나여야 한다: {scope!r}")
    scope_value: Literal["all", "this_week"] = "this_week" if scope == "this_week" else "all"
    resolved = _resolve_project(deps, project_id, as_of, limit)
    header = resolved.header
    as_of_date: date = header["as_of_date"]
    window = this_week_delay_range(as_of_date)
    if resolved.project is None:
        return DelayedTasksResult(**header, scope=scope_value, this_week_range=window, items=[])

    pid = resolved.project.project_id
    items = deps.reader.wbs_items(pid)
    by_id = {w.wbs_id: w for w in items}
    names = deps.reader.user_names(pid)
    reqs = _requirements_by_task(deps.reader.wbs_requirements(pid))
    picked = [
        w
        for w in items
        if (
            is_delayed_this_week(w.status, w.planned_end, as_of_date)
            if scope == "this_week"
            else is_delayed(w.status, w.planned_end, as_of_date)
        )
    ]
    picked.sort(key=lambda w: (-delay_days(w.planned_end, as_of_date), _wbs_sort_key(w.wbs_id)))
    rows = [
        DelayedTask(
            wbs_id=w.wbs_id,
            name=w.name,
            level=w.wbs_id.count(".") + 1,
            phase=w.phase,
            status=w.status,
            progress_pct=w.progress_pct,
            planned_start=w.planned_start,
            planned_end=w.planned_end,
            delay_days=delay_days(w.planned_end, as_of_date),
            this_week=is_delayed_this_week(w.status, w.planned_end, as_of_date),
            assignee=_person(w.assignee_user_id, names),
            parent=_task_ref(by_id[w.parent_wbs_id])
            if w.parent_wbs_id and w.parent_wbs_id in by_id
            else None,
            requirement_ids=reqs.get(w.wbs_id, []),
        )
        for w in picked
    ]
    label = "이번 주 지연" if scope == "this_week" else "지연"
    note = (
        " (기준일이 월요일이라 이번 주 구간이 비어 있다)"
        if scope == "this_week" and window is None
        else ""
    )
    header.update(
        total=len(rows),
        truncated=len(rows) > limit,
        message=f"{resolved.project.name}: {label} 업무 {len(rows)}건, 기준일 {as_of_date}{note}",
    )
    return DelayedTasksResult(
        **header, scope=scope_value, this_week_range=window, items=rows[:limit]
    )


# --- get_open_risks ---


def get_open_risks(
    project_id: str | None = None,
    min_level: RiskLevel = "low",
    as_of: date | str | None = None,
    limit: int = DEFAULT_LIMIT,
    *,
    deps: PmsDeps,
) -> OpenRisksResult:
    """프로젝트의 미해결 Risk를 점수(확률 × 영향)가 높은 순서로 돌려준다.

    판단 규칙(data-design.md 6절)
    - 미해결 Risk: status가 open 또는 mitigating(closed 제외)
    - 점수(score) = probability(1~5) × impact(1~5). 등급(level): 15 이상 high, 8~14 medium, 7 이하 low

    인자
    - project_id: 조회할 프로젝트 ID(예: "PRJ-03"). 없거나 모르는 ID면 오류 대신 status="project_required" /
      "project_not_found"와 프로젝트 후보(available_projects)를 돌려준다 — 사용자에게 어느 사업인지 물어본 뒤 다시 부른다
    - min_level: 이 등급 이상만 돌려준다. "low"(기본, 전부)·"medium"·"high". 그 밖의 값은 ValueError
    - as_of: 기준일(YYYY-MM-DD). 생략하면 AS_OF_DATE. Risk의 미해결 여부에는 쓰지 않고, 연결 업무가 지연인지
      (related_task_delayed) 판단에 쓴다
    - limit: 돌려받을 최대 건수. 1~50(기본 20). total에 전체 건수, truncated에 잘림 여부가 있다

    반환: 머리(status·message·project_id·project_name·doc_ids·as_of_date·total·limit·truncated) + min_level +
    items(점수 내림차순). Risk마다 risk_id·title·description·status·probability·impact·score·level·담당자(owner)·
    식별일·기한·대응 방안(mitigation)·연결 업무(related_task)와 그 업무의 지연 여부·이 Risk에 연결된 Issue 목록(issues).

    search_documents와 함께 쓰는 예 — "가장 위험한 Risk가 RFP의 어느 요구사항과 관련 있어?":
    1) get_open_risks(project_id="PRJ-02", min_level="high")
    2) 1위 Risk의 제목·연결 업무 이름으로 search_documents(query="외부 시스템 연계 일정", doc_ids=결과의 doc_ids)를 불러
       관련 요구사항 원문과 출처를 붙인다

    오류: ValueError(min_level·limit이 잘못됨), PmsUnavailableError(PMS DB에 연결할 수 없음)
    """
    if min_level not in RISK_LEVEL_ORDER:
        raise ValueError(f"min_level은 {RISK_LEVEL_ORDER} 중 하나여야 한다: {min_level!r}")
    level = RISK_LEVEL_ORDER[RISK_LEVEL_ORDER.index(min_level)]
    resolved = _resolve_project(deps, project_id, as_of, limit)
    header = resolved.header
    as_of_date: date = header["as_of_date"]
    if resolved.project is None:
        return OpenRisksResult(**header, min_level=level, items=[])

    pid = resolved.project.project_id
    tasks = {w.wbs_id: w for w in deps.reader.wbs_items(pid)}
    names = deps.reader.user_names(pid)
    issues_by_risk: dict[str, list[LinkedIssue]] = defaultdict(list)
    for issue in sorted(deps.reader.issues(pid), key=lambda i: i.issue_id):
        if issue.related_risk_id:
            issues_by_risk[issue.related_risk_id].append(
                LinkedIssue(
                    issue_id=issue.issue_id,
                    title=issue.title,
                    status=issue.status,
                    severity=issue.severity,
                )
            )
    floor = RISK_LEVEL_ORDER.index(level)
    picked = [
        r
        for r in deps.reader.risks(pid)
        if is_open_risk(r.status)
        and RISK_LEVEL_ORDER.index(risk_level(r.probability, r.impact)) >= floor
    ]
    picked.sort(key=lambda r: (-risk_score(r.probability, r.impact), r.risk_id))
    rows = []
    for r in picked:
        task = tasks.get(r.related_wbs_id) if r.related_wbs_id else None
        rows.append(
            OpenRisk(
                risk_id=r.risk_id,
                title=r.title,
                description=r.description,
                status=r.status,
                probability=r.probability,
                impact=r.impact,
                score=risk_score(r.probability, r.impact),
                level=risk_level(r.probability, r.impact),
                owner=_person(r.owner_user_id, names),
                identified_date=r.identified_date,
                due_date=r.due_date,
                mitigation=r.mitigation,
                related_task=_task_ref(task) if task else None,
                related_task_delayed=bool(
                    task and is_delayed(task.status, task.planned_end, as_of_date)
                ),
                issues=issues_by_risk.get(r.risk_id, []),
            )
        )
    header.update(
        total=len(rows),
        truncated=len(rows) > limit,
        message=f"{resolved.project.name}: 미해결 Risk {len(rows)}건({level} 이상)",
    )
    return OpenRisksResult(**header, min_level=level, items=rows[:limit])


# --- get_deliverables ---


def get_deliverables(
    project_id: str | None = None,
    status_filter: DeliverableFilter = "all",
    wbs_id: str | None = None,
    requirement_id: str | None = None,
    as_of: date | str | None = None,
    limit: int = DEFAULT_LIMIT,
    *,
    deps: PmsDeps,
) -> DeliverablesResult:
    """프로젝트의 산출물을 상태·업무·요구사항으로 걸러 돌려준다. 기한 지난 미제출이 먼저 온다.

    판단 규칙(data-design.md 6절)
    - 기한 지난 미제출(overdue): status가 not_submitted이고 마감일(due_date)이 기준일보다 앞선 것
    - 관련 산출물(wbs_id를 줄 때): 그 업무의 산출물 + 그 업무와 같은 요구사항에 연결된 다른 업무의 산출물

    인자
    - project_id: 조회할 프로젝트 ID(예: "PRJ-03"). 없거나 모르는 ID면 오류 대신 status="project_required" /
      "project_not_found"와 프로젝트 후보(available_projects)를 돌려준다 — 사용자에게 어느 사업인지 물어본 뒤 다시 부른다
    - status_filter: "all"(기본)·"not_submitted"(미제출 전부)·"overdue"(기한 지난 미제출)·"submitted"(제출, 승인 전)·
      "approved"(승인). 그 밖의 값은 ValueError
    - wbs_id: 이 업무(예: "3.2")의 관련 산출물만. 프로젝트에 없는 업무면 ValueError
    - requirement_id: 이 요구사항(예: "SFR-004")에 연결된 업무의 산출물만. 연결된 업무가 없으면 0건이고 message에 적는다
      (대응 업무가 없는 요구사항 — 누락 후보일 수 있다). wbs_id와 함께 주면 둘 다 만족하는 것
    - as_of: 기준일(YYYY-MM-DD). 생략하면 AS_OF_DATE
    - limit: 돌려받을 최대 건수. 1~50(기본 20). total에 전체 건수, truncated에 잘림 여부가 있다

    반환: 머리(status·message·project_id·project_name·doc_ids·as_of_date·total·limit·truncated) + 적용한 필터 +
    items. 산출물마다 deliverable_id·name·phase·status·due_date·submitted_date·overdue·days_overdue·연결 업무(task)·
    그 업무의 요구사항(requirement_ids). 정렬: 기한 지난 미제출(지난 일수 내림차순) → 마감일 → ID

    search_documents와 함께 쓰는 예 — "SFR-004 관련 산출물이 다 나왔어? RFP에서 요구한 산출물은 뭐야?":
    1) get_deliverables(project_id="PRJ-03", requirement_id="SFR-004")로 제출 상태를 본다
    2) search_documents(query="SFR-004 산출정보", doc_ids=결과의 doc_ids)로 RFP가 요구한 산출물(산출정보)을 찾아 비교한다

    오류: ValueError(status_filter·limit·wbs_id가 잘못됨), PmsUnavailableError(PMS DB에 연결할 수 없음)
    """
    if status_filter not in DELIVERABLE_FILTERS:
        raise ValueError(
            f"status_filter는 {DELIVERABLE_FILTERS} 중 하나여야 한다: {status_filter!r}"
        )
    filt: DeliverableFilter = status_filter
    resolved = _resolve_project(deps, project_id, as_of, limit)
    header = resolved.header
    as_of_date: date = header["as_of_date"]
    common: dict[str, Any] = {
        "status_filter": filt,
        "wbs_id": wbs_id,
        "requirement_id": requirement_id,
    }
    if resolved.project is None:
        return DeliverablesResult(**header, **common, items=[])

    pid = resolved.project.project_id
    tasks = {w.wbs_id: w for w in deps.reader.wbs_items(pid)}
    links = deps.reader.wbs_requirements(pid)
    reqs = _requirements_by_task(links)
    allowed: set[str] | None = None  # 업무 조건. None이면 모든 업무
    if wbs_id is not None:
        if wbs_id not in tasks:
            raise ValueError(f"프로젝트 {pid}에 업무 wbs_id '{wbs_id}'가 없다")
        shared = set(reqs.get(wbs_id, []))
        allowed = {wbs_id} | {link.wbs_id for link in links if link.requirement_id in shared}
    unlinked_requirement = False  # 요구사항 단독으로 연결 업무가 없다(누락 후보일 수 있다)
    if requirement_id is not None:
        if requirement_id not in deps.reader.requirement_ids(pid):
            raise ValueError(f"프로젝트 {pid}에 요구사항 '{requirement_id}'가 없다")
        linked = {link.wbs_id for link in links if link.requirement_id == requirement_id}
        unlinked_requirement = not linked
        allowed = linked if allowed is None else allowed & linked

    def keep(status: str, due: date) -> bool:
        if filt == "all":
            return True
        if filt == "overdue":
            return is_overdue_deliverable(status, due, as_of_date)
        return status == filt

    rows = []
    for d in deps.reader.deliverables(pid):
        if (allowed is not None and d.wbs_id not in allowed) or not keep(d.status, d.due_date):
            continue
        overdue = is_overdue_deliverable(d.status, d.due_date, as_of_date)
        rows.append(
            DeliverableItem(
                deliverable_id=d.deliverable_id,
                name=d.name,
                phase=d.phase,
                status=d.status,
                due_date=d.due_date,
                submitted_date=d.submitted_date,
                overdue=overdue,
                days_overdue=delay_days(d.due_date, as_of_date) if overdue else 0,
                task=_task_ref(tasks[d.wbs_id]),
                requirement_ids=reqs.get(d.wbs_id, []),
            )
        )
    rows.sort(key=lambda d: (not d.overdue, -d.days_overdue, d.due_date, d.deliverable_id))
    message = f"{resolved.project.name}: 산출물 {len(rows)}건(필터 {filt})"
    if unlinked_requirement:
        message += f". 요구사항 {requirement_id}에 연결된 업무가 없다(대응 업무가 없는 요구사항 — 누락 후보일 수 있다)"
    header.update(total=len(rows), truncated=len(rows) > limit, message=message)
    return DeliverablesResult(**header, **common, items=rows[:limit])
