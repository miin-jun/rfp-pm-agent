"""PMS 테이블 (이슈 #20, docs/data-design.md 6절).

RFP 1건 = 프로젝트 1개다. WBS 번호("2.3")는 프로젝트 안에서만 유일하므로 업무를 가리키는 키는
`(project_id, wbs_id)`다. 상태 값·확률·진척률·날짜 순서는 CHECK 제약으로도 막는다 — 합성 코드의 검증
(`pms.synth.validate`)을 거치지 않은 행이 들어와도 DB가 거부한다.

마이그레이션: `migrations/versions/` (`uv run alembic upgrade head`).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PROJECT_STATUSES = ("planned", "in_progress", "done")
ORG_TYPES = ("발주", "수행", "협력")
MEMBER_ROLES = ("pm", "developer", "client", "partner")
PHASES = ("분석", "설계", "구현", "시험", "이행")
WBS_STATUSES = ("not_started", "in_progress", "done", "on_hold")
DELIVERABLE_STATUSES = ("not_submitted", "submitted", "approved")
RISK_STATUSES = ("open", "mitigating", "closed")
ISSUE_STATUSES = ("open", "in_progress", "resolved")
ISSUE_SEVERITIES = ("low", "medium", "high", "critical")
SECURITY_LEVELS = ("public", "internal", "confidential")


def _in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint(_in("status", PROJECT_STATUSES), name="ck_projects_status"),
        CheckConstraint("start_date <= end_date", name="ck_projects_dates"),
    )

    project_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    notice_no: Mapped[str | None] = mapped_column(String(32))
    agency: Mapped[str | None] = mapped_column(Text)
    contractor: Mapped[str] = mapped_column(Text)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16))


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("org_type", ORG_TYPES), name="ck_users_org_type"),)

    user_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text)
    org: Mapped[str] = mapped_column(Text)
    org_type: Mapped[str] = mapped_column(String(8))


class ProjectMember(Base):
    __tablename__ = "project_members"
    __table_args__ = (
        CheckConstraint(_in("role", MEMBER_ROLES), name="ck_project_members_role"),
        CheckConstraint("allocation_pct BETWEEN 0 AND 100", name="ck_project_members_alloc"),
        CheckConstraint("start_date <= end_date", name="ck_project_members_dates"),
    )

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(16))
    allocation_pct: Mapped[int] = mapped_column(Integer)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)


class Requirement(Base):
    __tablename__ = "requirements"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), primary_key=True)
    requirement_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    category: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    source_doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"))
    source_chunk_id: Mapped[str] = mapped_column(Text)


class WbsItem(Base):
    __tablename__ = "wbs_items"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "parent_wbs_id"], ["wbs_items.project_id", "wbs_items.wbs_id"]
        ),
        ForeignKeyConstraint(
            ["project_id", "assignee_user_id"],
            ["project_members.project_id", "project_members.user_id"],
        ),
        CheckConstraint(_in("phase", PHASES), name="ck_wbs_items_phase"),
        CheckConstraint(_in("status", WBS_STATUSES), name="ck_wbs_items_status"),
        CheckConstraint("progress_pct BETWEEN 0 AND 100", name="ck_wbs_items_progress"),
        CheckConstraint("planned_start <= planned_end", name="ck_wbs_items_planned"),
        CheckConstraint(
            "actual_end IS NULL OR (actual_start IS NOT NULL AND actual_start <= actual_end)",
            name="ck_wbs_items_actual",
        ),
        # done ⇔ progress 100 그리고 actual_end 있음 (data-design.md 7절 검증 규칙)
        CheckConstraint(
            "(status = 'done') = (progress_pct = 100 AND actual_end IS NOT NULL)",
            name="ck_wbs_items_done",
        ),
    )

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), primary_key=True)
    wbs_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    parent_wbs_id: Mapped[str | None] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(Text)
    phase: Mapped[str] = mapped_column(String(8))
    assignee_user_id: Mapped[str] = mapped_column(String(16))
    planned_start: Mapped[date] = mapped_column(Date)
    planned_end: Mapped[date] = mapped_column(Date)
    actual_start: Mapped[date | None] = mapped_column(Date)
    actual_end: Mapped[date | None] = mapped_column(Date)
    progress_pct: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))


class WbsRequirement(Base):
    """업무 ↔ 요구사항 연결 (PMS ↔ FMS 연결 고리). 합성 데이터에서는 RTM 정답이다."""

    __tablename__ = "wbs_requirements"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "wbs_id"], ["wbs_items.project_id", "wbs_items.wbs_id"]
        ),
        ForeignKeyConstraint(
            ["project_id", "requirement_id"],
            ["requirements.project_id", "requirements.requirement_id"],
        ),
    )

    project_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    wbs_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    requirement_id: Mapped[str] = mapped_column(String(32), primary_key=True)


class Deliverable(Base):
    __tablename__ = "deliverables"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "wbs_id"], ["wbs_items.project_id", "wbs_items.wbs_id"]
        ),
        CheckConstraint(_in("phase", PHASES), name="ck_deliverables_phase"),
        CheckConstraint(_in("status", DELIVERABLE_STATUSES), name="ck_deliverables_status"),
        # 제출일이 있으면 제출(또는 승인) 상태다
        CheckConstraint(
            "(submitted_date IS NULL) = (status = 'not_submitted')",
            name="ck_deliverables_submitted",
        ),
    )

    deliverable_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    name: Mapped[str] = mapped_column(Text)
    phase: Mapped[str] = mapped_column(String(8))
    wbs_id: Mapped[str] = mapped_column(String(16))
    due_date: Mapped[date] = mapped_column(Date)
    submitted_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16))
    doc_id: Mapped[str | None] = mapped_column(ForeignKey("documents.doc_id"))


class Risk(Base):
    __tablename__ = "risks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "related_wbs_id"], ["wbs_items.project_id", "wbs_items.wbs_id"]
        ),
        ForeignKeyConstraint(
            ["project_id", "owner_user_id"],
            ["project_members.project_id", "project_members.user_id"],
        ),
        CheckConstraint(_in("status", RISK_STATUSES), name="ck_risks_status"),
        CheckConstraint("probability BETWEEN 1 AND 5", name="ck_risks_probability"),
        CheckConstraint("impact BETWEEN 1 AND 5", name="ck_risks_impact"),
        CheckConstraint("identified_date <= due_date", name="ck_risks_dates"),
        # issues가 (project_id, risk_id)로 참조한다 — 다른 프로젝트의 Risk를 연결하지 못하게
        UniqueConstraint("project_id", "risk_id", name="uq_risks_project_risk"),
    )

    risk_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    probability: Mapped[int] = mapped_column(Integer)
    impact: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    owner_user_id: Mapped[str] = mapped_column(String(16))
    identified_date: Mapped[date] = mapped_column(Date)
    due_date: Mapped[date] = mapped_column(Date)
    mitigation: Mapped[str] = mapped_column(Text)
    related_wbs_id: Mapped[str | None] = mapped_column(String(16))


class Issue(Base):
    __tablename__ = "issues"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "related_wbs_id"], ["wbs_items.project_id", "wbs_items.wbs_id"]
        ),
        ForeignKeyConstraint(
            ["project_id", "assignee_user_id"],
            ["project_members.project_id", "project_members.user_id"],
        ),
        ForeignKeyConstraint(
            ["project_id", "related_risk_id"], ["risks.project_id", "risks.risk_id"]
        ),
        CheckConstraint(_in("status", ISSUE_STATUSES), name="ck_issues_status"),
        CheckConstraint(_in("severity", ISSUE_SEVERITIES), name="ck_issues_severity"),
        # 해결일은 resolved일 때만 있다
        CheckConstraint(
            "(resolved_date IS NOT NULL) = (status = 'resolved')", name="ck_issues_resolved"
        ),
        CheckConstraint(
            "resolved_date IS NULL OR raised_date <= resolved_date", name="ck_issues_dates"
        ),
    )

    issue_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    assignee_user_id: Mapped[str] = mapped_column(String(16))
    raised_date: Mapped[date] = mapped_column(Date)
    resolved_date: Mapped[date | None] = mapped_column(Date)
    related_wbs_id: Mapped[str | None] = mapped_column(String(16))
    related_risk_id: Mapped[str | None] = mapped_column(String(24))


class DocumentRecord(Base):
    """FMS 문서 메타·권한 (`documents`). 청크 권한 필드의 원본이다(data-design.md 8절)."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(_in("security_level", SECURITY_LEVELS), name="ck_documents_security"),
    )

    doc_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    title: Mapped[str] = mapped_column(Text)
    file_type: Mapped[str] = mapped_column(String(8))
    sha256: Mapped[str] = mapped_column(String(64))
    security_level: Mapped[str] = mapped_column(String(16))
    allowed_roles: Mapped[list[str]] = mapped_column(ARRAY(Text))
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# 적재 순서(FK 순서). 지울 때는 거꾸로
TABLES_IN_LOAD_ORDER = (
    Project,
    User,
    ProjectMember,
    DocumentRecord,
    Requirement,
    WbsItem,
    WbsRequirement,
    Deliverable,
    Risk,
    Issue,
)
