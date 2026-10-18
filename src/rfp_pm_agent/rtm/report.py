"""사업별 RTM(요구사항 추적 매트릭스) 생성 (이슈 #39, docs/adr/0012-rtm-reranker-mapping.md).

- 연결 판단은 `rtm.mapping.build_rtm`(기준값 0.5)이 한다. 점수는 미리 계산한 요구사항×업무 리랭커 점수표
  (`RTM_SCORES_PATH`, `scripts/build_rtm_scores.py`)를 읽는다 — 이 모듈은 리랭커를 부르지 않는다
- 업무 상태·날짜·산출물은 PMS DB(읽기 전용 계정)에서 읽고, 지연·기한 지남은 `pms.rules`(docs/data-design.md 6절,
  조회 툴 #21과 같은 함수)로 판단한다. 기준일은 `AS_OF_DATE`
- 근거: 연결마다 리랭커 점수와 요구사항의 근거 청크 ID(`<doc_id>:block_requirement:<요구사항 ID>`)
- 출력: `RTM_OUT_DIR`/rtm_<project_id>.md·.csv

CLI: `uv run python -m rfp_pm_agent.rtm.report`
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from rfp_pm_agent.config import PostgresConfig, RtmConfig, SynthConfig
from rfp_pm_agent.pms.reader import PmsReader, SqlPmsReader, readonly_engine
from rfp_pm_agent.pms.rules import delay_days, is_delayed, is_overdue_deliverable
from rfp_pm_agent.rtm.mapping import build_rtm
from rfp_pm_agent.schemas.pms import DeliverableRow, ProjectRow, RequirementRow, WbsItemRow

# 측정 전 고정한 기준값(ADR-0012). 요구사항별 최고 점수가 이 값 미만이면 누락 후보
THRESHOLD = 0.5

CSV_COLUMNS = [
    "section",
    "requirement_id",
    "requirement_title",
    "wbs_id",
    "wbs_name",
    "score",
    "wbs_status",
    "delayed",
    "delay_days",
    "deliverables",
    "chunk_id",
]


class ScoreTable(BaseModel):
    """사업 1개의 요구사항×업무 점수표. 모든 (요구사항, 업무) 쌍에 점수가 있다."""

    project_id: str
    requirement_ids: list[str]  # 점수표에 나온 순서
    wbs_ids: list[str]
    scores: dict[tuple[str, str], float]  # (requirement_id, wbs_id) → 점수
    chunk_ids: dict[str, str]  # requirement_id → 근거 청크 ID
    injected: set[str]  # 점수표의 is_injected=true 요구사항(평가 시험 문제)


class TaskView(BaseModel):
    wbs_id: str
    name: str
    status: str
    delayed: bool
    delay_days: int | None


class RtmLink(BaseModel):
    task: TaskView
    score: float
    deliverables: list[str]  # "<산출물 ID> <상태>[ 기한 지남]"


class RtmRequirement(BaseModel):
    requirement_id: str
    title: str
    chunk_id: str
    links: list[RtmLink]  # 점수 높은 순. 비어 있으면 누락 후보


class ProjectRtm(BaseModel):
    project_id: str
    project_name: str
    as_of_date: date
    threshold: float
    requirements: list[RtmRequirement]
    out_of_scope: list[TaskView]  # 어느 요구사항과도 연결되지 않은 업무

    @property
    def missing(self) -> list[RtmRequirement]:
        return [r for r in self.requirements if not r.links]


def load_scores(path: Path) -> dict[str, ScoreTable]:
    """점수표 CSV를 사업별로 읽는다. 빠진 쌍이 있거나 같은 쌍이 두 번 나오면 ValueError."""
    tables: dict[str, ScoreTable] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid, rid, wid = row["project_id"], row["requirement_id"], row["wbs_id"]
            table = tables.setdefault(
                pid,
                ScoreTable(
                    project_id=pid,
                    requirement_ids=[],
                    wbs_ids=[],
                    scores={},
                    chunk_ids={},
                    injected=set(),
                ),
            )
            if rid not in table.chunk_ids:
                table.requirement_ids.append(rid)
                table.chunk_ids[rid] = row["chunk_id"]
            elif table.chunk_ids[rid] != row["chunk_id"]:
                raise ValueError(f"{pid}/{rid}: chunk_id가 행마다 다르다")
            if wid not in table.wbs_ids:
                table.wbs_ids.append(wid)
            if (rid, wid) in table.scores:
                raise ValueError(f"{pid}/{rid}/{wid}: 같은 쌍이 두 번 나온다")
            table.scores[(rid, wid)] = float(row["score"])
            if row["is_injected"] == "true":
                table.injected.add(rid)
    for table in tables.values():
        expected = len(table.requirement_ids) * len(table.wbs_ids)
        if len(table.scores) != expected:
            raise ValueError(
                f"{table.project_id}: 점수 {len(table.scores)}쌍, "
                f"요구사항×업무 {expected}쌍 — 빠진 쌍이 있다"
            )
    return tables


def _task_view(item: WbsItemRow, as_of: date) -> TaskView:
    delayed = is_delayed(item.status, item.planned_end, as_of)
    return TaskView(
        wbs_id=item.wbs_id,
        name=item.name,
        status=item.status,
        delayed=delayed,
        delay_days=delay_days(item.planned_end, as_of) if delayed else None,
    )


def _deliverable_label(d: DeliverableRow, as_of: date) -> str:
    overdue = is_overdue_deliverable(d.status, d.due_date, as_of)
    return f"{d.deliverable_id} {d.status}" + (" 기한 지남" if overdue else "")


def build_project_rtm(
    table: ScoreTable,
    project: ProjectRow,
    requirements: list[RequirementRow],
    wbs_items: list[WbsItemRow],
    deliverables: list[DeliverableRow],
    as_of: date,
) -> ProjectRtm:
    """점수표 + PMS 행으로 사업 1개의 RTM을 만든다.

    점수표의 업무는 level 2 이상(상위 업무가 있는 것)이고 단계(level 1) 업무는 연결·범위 외 판단에서 뺀다.
    점수표와 DB의 요구사항·업무 목록이나 근거 청크 ID가 다르면 ValueError — 다른 데이터로 만든 점수표를 쓰지 않게 한다.
    """
    pid = table.project_id
    tasks = {w.wbs_id: w for w in wbs_items if w.parent_wbs_id is not None}
    if set(tasks) != set(table.wbs_ids):
        raise ValueError(f"{pid}: 점수표 업무와 DB 업무(단계 제외)가 다르다")
    reqs = {r.requirement_id: r for r in requirements}
    if set(reqs) != set(table.requirement_ids):
        raise ValueError(f"{pid}: 점수표 요구사항과 DB 요구사항이 다르다")
    for rid, chunk_id in table.chunk_ids.items():
        if reqs[rid].source_chunk_id != chunk_id:
            raise ValueError(f"{pid}/{rid}: 점수표 chunk_id와 DB source_chunk_id가 다르다")

    by_task: dict[str, list[str]] = {}
    for d in sorted(deliverables, key=lambda d: d.deliverable_id):
        by_task.setdefault(d.wbs_id, []).append(_deliverable_label(d, as_of))

    result = build_rtm(table.requirement_ids, table.wbs_ids, table.scores, threshold=THRESHOLD)
    rows = [
        RtmRequirement(
            requirement_id=rid,
            title=reqs[rid].title,
            chunk_id=table.chunk_ids[rid],
            links=[
                RtmLink(
                    task=_task_view(tasks[wid], as_of),
                    score=score,
                    deliverables=by_task.get(wid, []),
                )
                for wid, score in result.links[rid]
            ],
        )
        for rid in table.requirement_ids
    ]
    return ProjectRtm(
        project_id=pid,
        project_name=project.name,
        as_of_date=as_of,
        threshold=THRESHOLD,
        requirements=rows,
        out_of_scope=[_task_view(tasks[wid], as_of) for wid in result.out_of_scope],
    )


def _delay_label(task: TaskView) -> str:
    return f"지연 {task.delay_days}일" if task.delayed else ""


def render_csv(rtm: ProjectRtm) -> str:
    """한 행 = 연결 1개 / 누락 후보 요구사항 1개 / 범위 외 후보 업무 1개 (`section`으로 구분)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    empty = dict.fromkeys(CSV_COLUMNS, "")

    def task_fields(task: TaskView) -> dict[str, str]:
        return {
            "wbs_id": task.wbs_id,
            "wbs_name": task.name,
            "wbs_status": task.status,
            "delayed": str(task.delayed).lower(),
            "delay_days": "" if task.delay_days is None else str(task.delay_days),
        }

    for req in rtm.requirements:
        req_fields = {
            "requirement_id": req.requirement_id,
            "requirement_title": req.title,
            "chunk_id": req.chunk_id,
        }
        if not req.links:
            writer.writerow(empty | req_fields | {"section": "누락 후보"})
        for link in req.links:
            writer.writerow(
                empty
                | req_fields
                | task_fields(link.task)
                | {
                    "section": "연결",
                    "score": f"{link.score:.4f}",
                    "deliverables": "; ".join(link.deliverables),
                }
            )
    for task in rtm.out_of_scope:
        writer.writerow(empty | task_fields(task) | {"section": "범위 외 후보"})
    return buf.getvalue()


def _cell(text: str) -> str:
    """Markdown 표 칸 — 줄바꿈과 `|`가 표를 깨지 않게 바꾼다."""
    return " ".join(text.split()).replace("|", "\\|")


def _table(header: list[str], rows: Iterable[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def render_md(rtm: ProjectRtm) -> str:
    linked = [r for r in rtm.requirements if r.links]
    n_links = sum(len(r.links) for r in linked)
    lines = [
        f"# RTM — {rtm.project_id} {rtm.project_name}",
        "",
        f"- 기준일: {rtm.as_of_date.isoformat()} · 연결 기준: 리랭커 점수 ≥ {rtm.threshold}",
        (
            f"- 요구사항 {len(rtm.requirements)}건: 연결 {len(linked)}건(연결 {n_links}개) · "
            f"누락 후보 {len(rtm.missing)}건 · 범위 외 후보 업무 {len(rtm.out_of_scope)}건"
        ),
        "- 근거: 연결마다 리랭커 점수(요구사항 원문 ↔ 업무 이름)와 요구사항의 근거 청크 ID",
        "",
        "## 요구사항별 연결",
        "",
    ]
    lines += _table(
        [
            "요구사항 ID",
            "명칭",
            "연결 업무(점수)",
            "업무 상태",
            "지연",
            "산출물 상태",
            "근거 chunk_id",
        ],
        (
            [
                req.requirement_id,
                req.title,
                f"{link.task.wbs_id} {link.task.name} ({link.score:.4f})",
                link.task.status,
                _delay_label(link.task),
                ", ".join(link.deliverables),
                req.chunk_id,
            ]
            for req in linked
            for link in req.links
        ),
    )
    lines += ["", "## 누락 후보 (연결된 업무 없음)", ""]
    if rtm.missing:
        lines += _table(
            ["요구사항 ID", "명칭", "근거 chunk_id"],
            ([r.requirement_id, r.title, r.chunk_id] for r in rtm.missing),
        )
    else:
        lines.append("없음")
    lines += ["", "## 범위 외 후보 (연결된 요구사항 없는 업무)", ""]
    if rtm.out_of_scope:
        lines += _table(
            ["업무", "이름", "상태", "지연"],
            ([t.wbs_id, t.name, t.status, _delay_label(t)] for t in rtm.out_of_scope),
        )
    else:
        lines.append("없음")
    return "\n".join(lines) + "\n"


def build_all(reader: PmsReader, tables: dict[str, ScoreTable], as_of: date) -> list[ProjectRtm]:
    """DB의 사업과 점수표의 사업이 같아야 한다(다르면 ValueError)."""
    projects = {p.project_id: p for p in reader.projects()}
    if set(projects) != set(tables):
        raise ValueError(f"DB 사업 {sorted(projects)}와 점수표 사업 {sorted(tables)}이 다르다")
    return [
        build_project_rtm(
            tables[pid],
            projects[pid],
            reader.requirements(pid),
            reader.wbs_items(pid),
            reader.deliverables(pid),
            as_of,
        )
        for pid in sorted(projects)
    ]


def write_outputs(rtms: list[ProjectRtm], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rtm in rtms:
        md_path = out_dir / f"rtm_{rtm.project_id}.md"
        csv_path = out_dir / f"rtm_{rtm.project_id}.csv"
        md_path.write_text(render_md(rtm), encoding="utf-8")
        csv_path.write_text(render_csv(rtm), encoding="utf-8-sig")
        written += [md_path, csv_path]
    return written


def main() -> None:
    cfg = RtmConfig.from_env()
    as_of = SynthConfig.from_env().as_of_date
    reader = SqlPmsReader(readonly_engine(PostgresConfig.from_env().readonly_dsn))
    rtms = build_all(reader, load_scores(cfg.scores_path), as_of)
    write_outputs(rtms, cfg.out_dir)
    print(f"기준일 {as_of} · 점수표 {cfg.scores_path} → {cfg.out_dir}/rtm_<project_id>.md·.csv")
    for rtm in rtms:
        n_links = sum(len(r.links) for r in rtm.requirements)
        print(
            f"  {rtm.project_id}: 요구사항 {len(rtm.requirements)} · 연결 {n_links} · "
            f"누락 후보 {len(rtm.missing)} · 범위 외 후보 {len(rtm.out_of_scope)}"
        )


if __name__ == "__main__":
    main()
