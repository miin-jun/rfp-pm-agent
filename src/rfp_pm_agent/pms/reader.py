"""PMS DB 읽기 (이슈 #21). 조회 툴(`tools/pms_tools.py`)이 쓰는 프로젝트 단위 조회.

- SQL은 프로젝트 행을 가져오기만 한다. 조건은 `project_id` 하나이고 SQLAlchemy가 바인딩 파라미터로 넘긴다(문자열 조립 없음)
- 지연·이번 주·미해결·기한 지남 판단은 SQL에 두지 않는다. 툴이 `pms.rules`(정답 파일과 같은 정의)로 판단한다 —
  규칙이 SQL과 Python 두 곳으로 갈라지지 않게 하고, 판단을 DB 없이 단위 테스트한다. 프로젝트당 업무는 50개 안팎이다
- 접속은 읽기 전용 계정(`POSTGRES_READONLY_DSN`, agent_ro)이고, 세션도 `default_transaction_read_only=on`으로 연다
- 연결할 수 없으면(`sqlalchemy.exc.OperationalError`) `PmsUnavailableError`
"""

from __future__ import annotations

from typing import Any, Protocol

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import OperationalError

from rfp_pm_agent.pms.errors import PmsUnavailableError
from rfp_pm_agent.pms.models import (
    Deliverable,
    DocumentRecord,
    Issue,
    Project,
    ProjectMember,
    Requirement,
    Risk,
    User,
    WbsItem,
    WbsRequirement,
)
from rfp_pm_agent.schemas.pms import (
    DeliverableRow,
    DocumentRow,
    IssueRow,
    ProjectRow,
    RequirementRow,
    RiskRow,
    WbsItemRow,
    WbsRequirementRow,
)


class PmsReader(Protocol):
    """프로젝트 단위 조회. 결과는 `schemas.pms`의 행 모델(테이블 열 그대로)이다."""

    def projects(self) -> list[ProjectRow]: ...
    def documents(self, project_id: str) -> list[DocumentRow]: ...
    def user_names(self, project_id: str) -> dict[str, str]: ...
    def requirement_ids(self, project_id: str) -> list[str]: ...
    def requirements(self, project_id: str) -> list[RequirementRow]: ...
    def wbs_items(self, project_id: str) -> list[WbsItemRow]: ...
    def wbs_requirements(self, project_id: str) -> list[WbsRequirementRow]: ...
    def deliverables(self, project_id: str) -> list[DeliverableRow]: ...
    def risks(self, project_id: str) -> list[RiskRow]: ...
    def issues(self, project_id: str) -> list[IssueRow]: ...


def readonly_engine(dsn: str) -> Engine:
    """읽기 전용 세션으로 여는 엔진. 계정 권한(SELECT만)에 더해 트랜잭션도 읽기 전용이다."""
    return create_engine(dsn, connect_args={"options": "-c default_transaction_read_only=on"})


class SqlPmsReader:
    """PostgreSQL 구현. 메서드마다 연결 1개를 열고 SELECT 1회."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def _rows(self, statement: Any) -> list[dict[str, Any]]:
        try:
            with self._engine.connect() as conn:
                return [dict(row._mapping) for row in conn.execute(statement)]
        except OperationalError as e:
            raise PmsUnavailableError(str(e.orig) if e.orig else str(e)) from e

    @staticmethod
    def _columns(model: Any) -> list[Any]:
        return [getattr(model, c.key) for c in model.__table__.columns]

    def projects(self) -> list[ProjectRow]:
        stmt = select(*self._columns(Project)).order_by(Project.project_id)
        return [ProjectRow.model_validate(r) for r in self._rows(stmt)]

    def documents(self, project_id: str) -> list[DocumentRow]:
        stmt = (
            select(*self._columns(DocumentRecord))
            .where(DocumentRecord.project_id == project_id)
            .order_by(DocumentRecord.doc_id)
        )
        return [DocumentRow.model_validate(r) for r in self._rows(stmt)]

    def user_names(self, project_id: str) -> dict[str, str]:
        stmt = (
            select(User.user_id, User.name)
            .join(ProjectMember, ProjectMember.user_id == User.user_id)
            .where(ProjectMember.project_id == project_id)
        )
        return {r["user_id"]: r["name"] for r in self._rows(stmt)}

    def requirement_ids(self, project_id: str) -> list[str]:
        stmt = (
            select(Requirement.requirement_id)
            .where(Requirement.project_id == project_id)
            .order_by(Requirement.requirement_id)
        )
        return [r["requirement_id"] for r in self._rows(stmt)]

    def requirements(self, project_id: str) -> list[RequirementRow]:
        stmt = (
            select(*self._columns(Requirement))
            .where(Requirement.project_id == project_id)
            .order_by(Requirement.requirement_id)
        )
        return [RequirementRow.model_validate(r) for r in self._rows(stmt)]

    def wbs_items(self, project_id: str) -> list[WbsItemRow]:
        stmt = select(*self._columns(WbsItem)).where(WbsItem.project_id == project_id)
        return [WbsItemRow.model_validate(r) for r in self._rows(stmt)]

    def wbs_requirements(self, project_id: str) -> list[WbsRequirementRow]:
        stmt = select(*self._columns(WbsRequirement)).where(WbsRequirement.project_id == project_id)
        return [WbsRequirementRow.model_validate(r) for r in self._rows(stmt)]

    def deliverables(self, project_id: str) -> list[DeliverableRow]:
        stmt = select(*self._columns(Deliverable)).where(Deliverable.project_id == project_id)
        return [DeliverableRow.model_validate(r) for r in self._rows(stmt)]

    def risks(self, project_id: str) -> list[RiskRow]:
        stmt = select(*self._columns(Risk)).where(Risk.project_id == project_id)
        return [RiskRow.model_validate(r) for r in self._rows(stmt)]

    def issues(self, project_id: str) -> list[IssueRow]:
        stmt = select(*self._columns(Issue)).where(Issue.project_id == project_id)
        return [IssueRow.model_validate(r) for r in self._rows(stmt)]
