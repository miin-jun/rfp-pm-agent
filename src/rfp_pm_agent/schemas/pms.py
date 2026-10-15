"""합성 PMS 데이터의 행 스키마 (이슈 #20, docs/data-design.md 6절 테이블과 컬럼이 같다).

`PmsDataset`이 생성 결과 파일(`data/synth/pms_v1.json`)의 형식이고, 적재(`pms.load`)는 이 파일을 그대로
테이블에 넣는다 — 다시 적재할 때 LLM을 부르지 않는다.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Phase = Literal["분석", "설계", "구현", "시험", "이행"]
PHASE_ORDER: tuple[Phase, ...] = ("분석", "설계", "구현", "시험", "이행")


class ProjectRow(BaseModel):
    project_id: str
    name: str
    notice_no: str | None
    agency: str | None
    contractor: str
    start_date: date
    end_date: date
    status: Literal["planned", "in_progress", "done"]


class UserRow(BaseModel):
    user_id: str
    name: str
    email: str
    org: str
    org_type: Literal["발주", "수행", "협력"]


class ProjectMemberRow(BaseModel):
    project_id: str
    user_id: str
    role: Literal["pm", "developer", "client", "partner"]
    allocation_pct: int = Field(ge=0, le=100)
    start_date: date
    end_date: date


class RequirementRow(BaseModel):
    project_id: str
    requirement_id: str
    category: str | None
    title: str
    description: str
    source_doc_id: str
    source_chunk_id: str


class WbsItemRow(BaseModel):
    project_id: str
    wbs_id: str
    parent_wbs_id: str | None
    name: str
    phase: Phase
    assignee_user_id: str
    planned_start: date
    planned_end: date
    actual_start: date | None
    actual_end: date | None
    progress_pct: int = Field(ge=0, le=100)
    status: Literal["not_started", "in_progress", "done", "on_hold"]


class WbsRequirementRow(BaseModel):
    project_id: str
    wbs_id: str
    requirement_id: str


class DeliverableRow(BaseModel):
    deliverable_id: str
    project_id: str
    name: str
    phase: Phase
    wbs_id: str
    due_date: date
    submitted_date: date | None
    status: Literal["not_submitted", "submitted", "approved"]
    doc_id: str | None


class RiskRow(BaseModel):
    risk_id: str
    project_id: str
    title: str
    description: str
    probability: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    status: Literal["open", "mitigating", "closed"]
    owner_user_id: str
    identified_date: date
    due_date: date
    mitigation: str
    related_wbs_id: str | None


class IssueRow(BaseModel):
    issue_id: str
    project_id: str
    title: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    status: Literal["open", "in_progress", "resolved"]
    assignee_user_id: str
    raised_date: date
    resolved_date: date | None
    related_wbs_id: str | None
    related_risk_id: str | None


class DocumentRow(BaseModel):
    doc_id: str
    project_id: str
    title: str
    file_type: str
    sha256: str
    security_level: Literal["public", "internal", "confidential"]
    allowed_roles: list[str]
    parsed_at: datetime | None


class PmsDataset(BaseModel):
    """테이블 10개의 행 전부. 키 이름이 테이블 이름이다."""

    as_of_date: date
    seed: int
    projects: list[ProjectRow]
    users: list[UserRow]
    project_members: list[ProjectMemberRow]
    requirements: list[RequirementRow]
    documents: list[DocumentRow]
    wbs_items: list[WbsItemRow]
    wbs_requirements: list[WbsRequirementRow]
    deliverables: list[DeliverableRow]
    risks: list[RiskRow]
    issues: list[IssueRow]
