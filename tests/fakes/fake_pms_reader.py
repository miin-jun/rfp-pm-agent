"""DB 없이 `PmsDataset`을 읽는 가짜 `PmsReader` (이슈 #21). 조회 조건은 SQL 구현과 같다(project_id 일치)."""

from __future__ import annotations

from rfp_pm_agent.schemas.pms import (
    DeliverableRow,
    DocumentRow,
    IssueRow,
    PmsDataset,
    ProjectRow,
    RequirementRow,
    RiskRow,
    WbsItemRow,
    WbsRequirementRow,
)


class FakePmsReader:
    def __init__(self, ds: PmsDataset) -> None:
        self.ds = ds
        self.calls: list[tuple[str, str | None]] = []

    def projects(self) -> list[ProjectRow]:
        self.calls.append(("projects", None))
        return sorted(self.ds.projects, key=lambda p: p.project_id)

    def documents(self, project_id: str) -> list[DocumentRow]:
        self.calls.append(("documents", project_id))
        return [d for d in self.ds.documents if d.project_id == project_id]

    def user_names(self, project_id: str) -> dict[str, str]:
        self.calls.append(("user_names", project_id))
        members = {m.user_id for m in self.ds.project_members if m.project_id == project_id}
        return {u.user_id: u.name for u in self.ds.users if u.user_id in members}

    def requirement_ids(self, project_id: str) -> list[str]:
        self.calls.append(("requirement_ids", project_id))
        return sorted(r.requirement_id for r in self.ds.requirements if r.project_id == project_id)

    def requirements(self, project_id: str) -> list[RequirementRow]:
        self.calls.append(("requirements", project_id))
        rows = [r for r in self.ds.requirements if r.project_id == project_id]
        return sorted(rows, key=lambda r: r.requirement_id)

    def wbs_items(self, project_id: str) -> list[WbsItemRow]:
        self.calls.append(("wbs_items", project_id))
        return [w for w in self.ds.wbs_items if w.project_id == project_id]

    def wbs_requirements(self, project_id: str) -> list[WbsRequirementRow]:
        self.calls.append(("wbs_requirements", project_id))
        return [x for x in self.ds.wbs_requirements if x.project_id == project_id]

    def deliverables(self, project_id: str) -> list[DeliverableRow]:
        self.calls.append(("deliverables", project_id))
        return [d for d in self.ds.deliverables if d.project_id == project_id]

    def risks(self, project_id: str) -> list[RiskRow]:
        self.calls.append(("risks", project_id))
        return [r for r in self.ds.risks if r.project_id == project_id]

    def issues(self, project_id: str) -> list[IssueRow]:
        self.calls.append(("issues", project_id))
        return [i for i in self.ds.issues if i.project_id == project_id]
