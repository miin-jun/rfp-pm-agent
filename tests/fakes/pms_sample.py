"""합성 PMS 단위·통합 테스트가 함께 쓰는 가짜 원천 문서·초안과 샘플 데이터셋 (이슈 #20). LLM·DB 없음."""

from __future__ import annotations

from datetime import date

from rfp_pm_agent.pms.draft import TaskDraft, WbsDraft
from rfp_pm_agent.pms.synth import (
    Source,
    build_requirements,
    build_truth,
    generate,
    pick_missing,
    project_id_for,
)
from rfp_pm_agent.schemas.document import Document, Requirement
from rfp_pm_agent.schemas.pms import PHASE_ORDER, PmsDataset

AS_OF = date(2026, 9, 24)  # 목요일
SEED = 42


def requirement(rid: str, **fields: str) -> Requirement:
    return Requirement(
        requirement_id=rid,
        prefix=rid.split("-")[0],
        fields=fields,
        raw_fields={},
        text=f"{rid} 본문",
        source_order=0,
    )


def document(doc_id: str, n: int = 25) -> Document:
    reqs = [
        requirement(f"SFR-{i:03d}", name=f"기능 {i}", definition=f"정의 {i}", category="기능")
        for i in range(1, n + 1)
    ]
    return Document(
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        bid_title=f"{doc_id} 사업",
        format="pdf",
        parse_status="parsed",
        has_requirements=True,
        requirement_count=n,
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=[],
        requirements=reqs,
    )


def draft(requirement_ids: list[str], n_tasks: int = 35) -> WbsDraft:
    """단계마다 7개. 처음 두 업무는 사업관리(요구사항 없음), 나머지는 요구사항을 차례로 1개씩 연결."""
    tasks = []
    for i in range(n_tasks):
        phase = PHASE_ORDER[i * len(PHASE_ORDER) // n_tasks]
        rids = [] if i < 2 else [requirement_ids[(i - 2) % len(requirement_ids)]]
        tasks.append(
            TaskDraft(
                phase=phase, name=f"업무 {i}", requirement_ids=rids, deliverables=[f"산출물 {i}"]
            )
        )
    return WbsDraft(tasks=tasks)


def build_sample(as_of: date = AS_OF, seed: int = SEED) -> tuple[PmsDataset, dict[str, object]]:
    sources = [Source(document(d), f"R26-{d}", "0" * 64) for d in ("docA", "docB")]
    drafts: dict[str, WbsDraft] = {}
    excluded: dict[str, list[str]] = {}
    for index, source in enumerate(sources, start=1):
        reqs = build_requirements(project_id_for(index), source.document)
        missing = pick_missing(seed, source.document.doc_id, [r.requirement_id for r in reqs])
        excluded[project_id_for(index)] = missing
        drafts[source.document.doc_id] = draft(
            [r.requirement_id for r in reqs if r.requirement_id not in missing]
        )
    ds, injected = generate(sources, drafts, as_of=as_of, seed=seed)
    doc_ids = {project_id_for(i): s.document.doc_id for i, s in enumerate(sources, start=1)}
    truth = build_truth(ds, excluded=excluded, injected=injected, doc_ids=doc_ids)
    return ds, truth
