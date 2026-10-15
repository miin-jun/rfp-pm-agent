"""PMS 조회 툴의 결과 모델 (이슈 #21, docs/data-design.md 6절 "PMS 조회 툴").

세 툴(`get_delayed_tasks`·`get_open_risks`·`get_deliverables`)이 같은 머리 구조를 쓴다.

- status: `ok` / `project_required`(project_id 없음) / `project_not_found`(없는 project_id). ok가 아니면 items는 비고,
  message와 available_projects로 사용자에게 어느 사업인지 되묻는다 — 오류(예외)가 아니다
- total은 조건에 맞는 전체 건수, items는 그중 앞에서부터 limit개. truncated는 total > len(items)
- doc_ids는 그 프로젝트의 RFP doc_id 목록 — `search_documents(doc_ids=...)`에 그대로 넣는다(ADR-0006 결정 4)
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

from rfp_pm_agent.pms.rules import RiskLevel
from rfp_pm_agent.schemas.pms import Phase

ToolStatus = Literal["ok", "project_required", "project_not_found"]


class ProjectRef(BaseModel):
    project_id: str
    name: str
    doc_ids: list[str]


class PersonRef(BaseModel):
    user_id: str
    name: str | None  # 멤버 정보가 없으면 None


class TaskRef(BaseModel):
    wbs_id: str
    name: str
    status: str


class ToolResultBase(BaseModel):
    status: ToolStatus
    message: str  # ok면 결과 요약, 아니면 사용자에게 되물을 내용
    project_id: str | None
    project_name: str | None
    doc_ids: list[str]
    as_of_date: date
    total: int
    limit: int
    truncated: bool
    available_projects: list[ProjectRef]  # status가 ok가 아닐 때만 채운다


class DelayedTask(BaseModel):
    wbs_id: str
    name: str
    level: int  # WBS 깊이: 1 = 단계("2"), 2 = 그 아래 업무("2.3"), 3 = "2.3.1" … (v2 데이터는 2단계까지)
    phase: Phase
    status: str
    progress_pct: int
    planned_start: date
    planned_end: date
    delay_days: int
    this_week: bool
    assignee: PersonRef
    parent: TaskRef | None
    requirement_ids: list[str]


class DelayedTasksResult(ToolResultBase):
    scope: Literal["all", "this_week"]
    this_week_range: (
        tuple[date, date] | None
    )  # 이번 주 월요일 ~ 기준일 전날. 기준일이 월요일이면 None
    items: list[DelayedTask]


class LinkedIssue(BaseModel):
    issue_id: str
    title: str
    status: str
    severity: str


class OpenRisk(BaseModel):
    risk_id: str
    title: str
    description: str
    status: str
    probability: int
    impact: int
    score: int
    level: RiskLevel
    owner: PersonRef
    identified_date: date
    due_date: date
    mitigation: str
    related_task: TaskRef | None
    related_task_delayed: bool
    issues: list[LinkedIssue]


class OpenRisksResult(ToolResultBase):
    min_level: RiskLevel
    items: list[OpenRisk]


DeliverableFilter = Literal["all", "not_submitted", "overdue", "submitted", "approved"]


class DeliverableItem(BaseModel):
    deliverable_id: str
    name: str
    phase: Phase
    status: str
    due_date: date
    submitted_date: date | None
    overdue: bool
    days_overdue: int  # 기한 지난 미제출이면 기준일 − due_date, 아니면 0
    task: TaskRef
    requirement_ids: list[str]


class DeliverablesResult(ToolResultBase):
    status_filter: DeliverableFilter
    wbs_id: str | None
    requirement_id: str | None
    items: list[DeliverableItem]
