"""합성 PMS 생성 (이슈 #20, docs/data-design.md 7절 "정답을 알고 만든다").

단계
1. 원천 문서 선택(`select_sources`): 색인 청크 파일에 요구사항 청크가 있고 요구사항이 `MIN_REQUIREMENTS`건 이상인 문서. RFP 1건 = 프로젝트 1개
2. 요구사항 추출(`build_requirements`): 파싱 결과의 요구사항 그대로(실제 데이터). `source_chunk_id`는 그 요구사항의 검색 청크
3. 누락 후보(`pick_missing`): 사업관리(PMR·PSR)를 뺀 요구사항에서 전체의 약 `MISSING_RATIO`를 seed로 골라 LLM에 주지 않는다 → 대응 업무가 없는 요구사항(#39 정답)
4. WBS 초안(`pms.draft`): LLM이 업무 이름·단계·연결 요구사항·산출물 이름만 만든다
5. 일정·상태·사람·Risk·Issue(`generate_project`): 코드가 seed(`SYNTH_SEED` + doc_id)로 정한다. LLM을 부르지 않는다
   - 기준일 이전에 끝나는 업무는 done, 걸친 업무는 in_progress, 이후는 not_started
   - 지연 주입: 말단 업무의 약 `DELAY_RATIO`. 이번 주 지연 1건(기준일이 월요일이 아니면) + 이전에 끝났어야 할 업무
   - Risk·Issue 각 `RISK_COUNT`·`ISSUE_COUNT` 범위, open이 반드시 섞인다
6. 검증(`validate`): data-design.md 7절 "검증 규칙" + 참조 무결성. 오류가 하나라도 있으면 파일을 쓰지 않는다
7. 정답(`build_truth`): 지연·미해결 Risk·미제출 산출물·누락 후보·범위 외 업무·RTM을 `pms.rules`로 계산해 `data/eval/pms_truth_v2.json`

CLI (`uv run python -m rfp_pm_agent.pms.synth <명령>`)
- `draft [--limit N]`: 저장된 초안이 없는 문서만 LLM을 부른다(**비용**). `--limit 1`로 1건만 먼저 확인한다
- `generate`: 초안 + seed로 `data/synth/pms_v2.json`과 정답 파일을 쓰고 통계를 출력한다(LLM 없음)
- `load`: `data/synth/pms_v2.json`을 DB에 적재한다(기존 행은 지운다). 먼저 `uv run alembic upgrade head`
- `stats`: 저장된 데이터의 통계만 출력한다
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from rfp_pm_agent.pms.draft import (
    DRAFT_PROMPT_VERSION,
    CachedDraft,
    WbsDraft,
    draft_path,
    load_cached_draft,
    request_draft,
    save_draft,
)
from rfp_pm_agent.pms.rules import (
    delay_days,
    is_delayed,
    is_delayed_this_week,
    is_open_risk,
    this_week_delay_range,
)
from rfp_pm_agent.schemas.document import Document
from rfp_pm_agent.schemas.pms import (
    PHASE_ORDER,
    DeliverableRow,
    DocumentRow,
    IssueRow,
    PmsDataset,
    ProjectMemberRow,
    ProjectRow,
    RequirementRow,
    RiskRow,
    UserRow,
    WbsItemRow,
    WbsRequirementRow,
)

DEFAULT_PARSED_DIR = Path("data/parsed")
DEFAULT_MANIFEST = Path("data/raw/manifest.jsonl")
DEFAULT_CHUNKS = Path("data/chunks/block_requirement.jsonl")
DEFAULT_DRAFTS_DIR = Path("data/synth/wbs_drafts")
# v1(초안 프롬프트 wbs_draft_v1)의 정답 data/eval/pms_truth_v1.json은 보존한다(오류 기록, ADR-0008)
DEFAULT_DATASET = Path("data/synth/pms_v2.json")
DEFAULT_TRUTH = Path("data/eval/pms_truth_v2.json")
TRUTH_VERSION = "pms_truth_v2"

MIN_REQUIREMENTS = 20
MISSING_RATIO = 0.12  # 누락 후보 비율(요청 10~15%의 가운데). 프로젝트마다 최소 1건
MANAGEMENT_PREFIXES = ("PMR", "PSR")  # 누락 후보에서 빼는 사업관리 계열(v2)
DELAY_RATIO = 0.15  # 지연 주입 비율(data-design.md 7절 4단계). 프로젝트마다 최소 2건
RISK_COUNT = (5, 10)
ISSUE_COUNT = (5, 10)
# 프로젝트 기간: 기준일까지 지난 일수와 전체 일수. 전체가 항상 더 길어 기준일이 기간 안에 든다
ELAPSED_DAYS = (100, 200)
DURATION_DAYS = (240, 330)
# 단계별 기간 비율(분석·설계·구현·시험·이행)
PHASE_SHARE = (0.15, 0.20, 0.35, 0.20, 0.10)
ALL_ROLES = ["pm", "developer", "client", "partner"]

# 가짜 이름·조직. 성 + 두 음절을 seed로 조합한다. 회사 이름은 "(가상)"을 붙여 실존 조직과 구별한다
_SURNAMES = "김이박최정강조윤장임한오서신권황안송류홍"
_SYLLABLES = "민서준하도윤지우현예수진태은소재유성가온채린건율"
_CONTRACTORS = [
    "한빛정보기술(가상)",
    "누리시스템(가상)",
    "다온데이터(가상)",
    "가람소프트(가상)",
    "세움아이티(가상)",
]
_PARTNERS = ["바른연계(가상)", "새봄보안(가상)", "이음클라우드(가상)", "온새미디자인(가상)"]

_RISK_TEMPLATES = [
    (
        "핵심 개발 인력 이탈",
        "주요 개발자의 퇴사·타 사업 투입 가능성이 있다",
        "대체 인력 사전 확보, 산출물·코드 인수인계 문서화",
    ),
    (
        "요구사항 변경 요청 증가",
        "발주기관 검토 과정에서 범위 변경 요청이 늘고 있다",
        "변경관리 절차 적용, 변경 영향 분석 후 합의",
    ),
    (
        "외부 시스템 연계 일정 지연",
        "연계 대상 기관의 인터페이스 사양 확정이 늦어지고 있다",
        "연계 사양 조기 확정 회의, 모의 연계로 선행 시험",
    ),
    (
        "데이터 이관 품질 저하",
        "기존 데이터의 정합성 오류가 예상보다 많다",
        "이관 전 정제 기준 수립, 이관 리허설 2회",
    ),
    (
        "보안 취약점 조치 지연",
        "보안 점검 결과 조치 항목이 누적되고 있다",
        "취약점 우선순위 분류, 주간 조치 현황 점검",
    ),
    (
        "장비·SW 납품 지연",
        "도입 장비의 공급 일정이 불확실하다",
        "대체 공급처 확인, 설치 일정 재배치",
    ),
    (
        "발주기관 검토·승인 지연",
        "산출물 검토 회신이 늦어 후속 단계 착수가 밀릴 수 있다",
        "검토 일정 사전 합의, 주간 회의에서 회신 기한 확인",
    ),
    (
        "성능 목표 미달 가능성",
        "부하 시험 전 단계에서 응답 시간이 목표에 못 미친다",
        "병목 구간 분석, 캐시·인덱스 개선",
    ),
    (
        "클라우드 이용 비용 초과",
        "시험 환경 자원 사용량이 계획보다 많아 예산을 넘을 수 있다",
        "자원 사용량 주간 점검, 미사용 자원 정리",
    ),
    (
        "사용자 교육 참여 저조",
        "이행 단계 사용자 교육 일정에 현업 참여가 확정되지 않았다",
        "부서별 교육 일정 사전 공지, 온라인 교육 자료 병행",
    ),
]
_ISSUE_TEMPLATES = [
    ("시험 환경 서버 접속 불가", "시험 환경 방화벽 정책 미반영으로 접속이 되지 않는다"),
    ("요구사항 해석 차이", "발주기관과 수행사의 요구사항 해석이 달라 재협의가 필요하다"),
    ("연계 API 응답 오류", "연계 대상 API가 사양과 다른 응답 형식을 돌려준다"),
    ("산출물 양식 불일치", "발주기관 표준 양식과 제출 산출물 양식이 다르다"),
    ("개발 서버 디스크 부족", "개발 서버 디스크 사용량이 90%를 넘었다"),
    ("화면 설계 재작업", "사용자 검토 결과 주요 화면 구성을 다시 설계해야 한다"),
    ("시험 데이터 미확보", "통합시험에 필요한 시험 데이터가 제공되지 않았다"),
    ("라이선스 확인 지연", "도입 SW 라이선스 범위 확인이 끝나지 않았다"),
    ("운영 서버 접근 권한 미부여", "운영 서버 작업 계정 발급 요청이 처리되지 않았다"),
    ("회의록 승인 지연", "주간 회의록 승인 회신이 2주째 없다"),
]


@dataclass(frozen=True)
class Source:
    """프로젝트 1개의 원천 RFP."""

    document: Document
    notice_no: str | None
    sha256: str


# --- 1~3. 원천 문서·요구사항·누락 후보 ---


def requirement_chunk_id(doc_id: str, requirement_id: str) -> str:
    return f"{doc_id}:block_requirement:{requirement_id}"


def select_sources(
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    chunks_path: Path = DEFAULT_CHUNKS,
    min_requirements: int = MIN_REQUIREMENTS,
) -> list[Source]:
    """요구사항 청크가 색인 청크 파일에 있고 요구사항이 `min_requirements`건 이상인 문서(doc_id 순).

    요구사항은 `source_chunk_id`로 검색 청크와 이어져야 하므로, 청크 파일에 없는 문서(같은 사업의 다른 판본 등)는 뺀다.
    """
    chunk_ids: set[str] = set()
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk_ids.add(json.loads(line)["chunk_id"])
    manifest: dict[str, dict[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                manifest[row["doc_id"]] = row

    sources: list[Source] = []
    for path in sorted(parsed_dir.glob("*.json")):
        doc = Document.model_validate_json(path.read_text("utf-8"))
        if doc.parse_status != "parsed" or len(doc.requirements) < min_requirements:
            continue
        if not all(
            requirement_chunk_id(doc.doc_id, r.requirement_id) in chunk_ids
            for r in doc.requirements
        ):
            continue
        meta = manifest.get(doc.doc_id, {})
        sources.append(Source(doc, meta.get("notice_no"), meta.get("sha256", "")))
    return sources


def _raw_field(raw_fields: dict[str, str], *labels: str) -> str:
    """원문 라벨(공백 무시)이 labels 중 하나인 값. 없으면 빈 문자열."""
    wanted = {label.replace(" ", "") for label in labels}
    return next((v for k, v in raw_fields.items() if k.replace(" ", "") in wanted), "")


def build_requirements(project_id: str, doc: Document) -> list[RequirementRow]:
    """파싱 결과의 요구사항을 그대로 옮긴다. 같은 ID가 두 번 나오면 처음 것만 쓴다.

    제목: fields.name → 원문 "요구사항 명" → ID. 설명: fields.definition → detail → 원문 "요구사항 내용" → 본문 텍스트.
    """
    rows: list[RequirementRow] = []
    seen: set[str] = set()
    for req in doc.requirements:
        if req.requirement_id in seen:
            continue
        seen.add(req.requirement_id)
        f = req.fields
        title = f.get("name") or _raw_field(req.raw_fields, "요구사항명", "요구사항명칭")
        description = (
            f.get("definition")
            or f.get("detail")
            or _raw_field(req.raw_fields, "요구사항내용", "세부내용")
            or req.text
        )
        rows.append(
            RequirementRow(
                project_id=project_id,
                requirement_id=req.requirement_id,
                category=f.get("category") or req.prefix,
                title=(title or req.requirement_id).strip(),
                description=description.strip(),
                source_doc_id=doc.doc_id,
                source_chunk_id=requirement_chunk_id(doc.doc_id, req.requirement_id),
            )
        )
    return rows


def project_rng(seed: int, doc_id: str, purpose: str) -> random.Random:
    """프로젝트·용도별 난수. 문자열 seed는 Python 버전·실행과 관계없이 같은 수열을 낸다."""
    return random.Random(f"{seed}:{doc_id}:{purpose}")


def is_management_requirement(requirement_id: str) -> bool:
    """사업관리 계열(프로젝트관리 PMR·지원 PSR) 요구사항인지. 접두어로 판정한다(data-design.md 3절 표)."""
    return requirement_id.split("-")[0] in MANAGEMENT_PREFIXES


def pick_missing(seed: int, doc_id: str, requirement_ids: Sequence[str]) -> list[str]:
    """누락 후보로 남길 요구사항(원래 순서). 개수 = round(전체 × MISSING_RATIO), 최소 1.

    사업관리 요구사항(`MANAGEMENT_PREFIXES`)은 고르지 않는다 — 착수·주간·완료보고 같은 관리 업무는 어느 사업에나 있어,
    LLM에 주지 않아도 비슷한 업무가 생겨 "대응 업무가 없다"는 정답이 틀리게 된다(ADR-0008). 후보가 모자라면 있는 만큼만 고른다.
    """
    eligible = [rid for rid in requirement_ids if not is_management_requirement(rid)]
    count = min(len(eligible), max(1, round(len(requirement_ids) * MISSING_RATIO)))
    chosen = set(project_rng(seed, doc_id, "missing").sample(eligible, count))
    return [rid for rid in requirement_ids if rid in chosen]


def project_id_for(index: int) -> str:
    return f"PRJ-{index:02d}"


# --- 5. 일정·상태·사람·Risk·Issue ---


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _fake_names(rng: random.Random, count: int, taken: set[str]) -> list[str]:
    names: list[str] = []
    while len(names) < count:
        name = rng.choice(_SURNAMES) + rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES)
        if name not in taken:
            taken.add(name)
            names.append(name)
    return names


@dataclass
class _Team:
    members: list[ProjectMemberRow]
    users: list[UserRow]
    pm: str
    developers: list[str]


def _build_team(
    rng: random.Random,
    project: ProjectRow,
    first_user_no: int,
    taken_names: set[str],
    partner: str,
) -> _Team:
    roles = ["pm"] + ["developer"] * rng.randint(4, 6) + ["client"] * 2 + ["partner"]
    names = _fake_names(rng, len(roles), taken_names)
    users: list[UserRow] = []
    members: list[ProjectMemberRow] = []
    org_by_role = {
        "pm": (project.contractor, "수행"),
        "developer": (project.contractor, "수행"),
        "client": (f"{project.project_id} 발주기관(가상)", "발주"),
        "partner": (partner, "협력"),
    }
    alloc_by_role = {"pm": (100, 100), "developer": (5, 10), "client": (2, 2), "partner": (3, 5)}
    for offset, (role, name) in enumerate(zip(roles, names, strict=True)):
        user_id = f"U{first_user_no + offset:03d}"
        org, org_type = org_by_role[role]
        users.append(
            UserRow(
                user_id=user_id,
                name=name,
                email=f"{user_id.lower()}@example.com",
                org=org,
                org_type=org_type,  # type: ignore[arg-type]
            )
        )
        low, high = alloc_by_role[role]
        alloc = 100 if role == "pm" else rng.randint(low, high) * 10
        members.append(
            ProjectMemberRow(
                project_id=project.project_id,
                user_id=user_id,
                role=role,  # type: ignore[arg-type]
                allocation_pct=alloc,
                start_date=project.start_date,
                end_date=project.end_date,
            )
        )
    developers = [m.user_id for m in members if m.role == "developer"]
    pm = next(m.user_id for m in members if m.role == "pm")
    return _Team(members, users, pm, developers)


def _phase_spans(start: date, end: date) -> list[tuple[date, date]]:
    """단계 5개의 (시작, 끝). 이어 붙이고 마지막 단계는 프로젝트 끝에서 끝난다."""
    total = (end - start).days + 1
    spans: list[tuple[date, date]] = []
    cursor = start
    used = 0.0
    for i, share in enumerate(PHASE_SHARE):
        used += share
        phase_end = (
            end if i == len(PHASE_SHARE) - 1 else start + timedelta(days=round(total * used) - 1)
        )
        spans.append((cursor, phase_end))
        cursor = phase_end + timedelta(days=1)
    return spans


def _schedule_status(
    rng: random.Random, start: date, end: date, as_of: date
) -> tuple[str, int, date | None, date | None]:
    """주입 전 상태: 기준일 전에 끝나면 done, 걸치면 in_progress, 이후면 not_started."""
    yesterday = as_of - timedelta(days=1)
    if end < as_of:
        actual_start = min(start + timedelta(days=rng.randint(0, 2)), end)
        actual_end = end + timedelta(days=rng.randint(-2, 3))
        actual_end = min(max(actual_end, actual_start), yesterday)
        return "done", 100, actual_start, actual_end
    if start >= as_of:
        return "not_started", 0, None, None
    span = (end - start).days + 1
    progress = round(100 * (as_of - start).days / span)
    return "in_progress", min(95, max(5, progress)), start, None


@dataclass
class ProjectData:
    project: ProjectRow
    users: list[UserRow]
    members: list[ProjectMemberRow]
    requirements: list[RequirementRow]
    document: DocumentRow
    wbs_items: list[WbsItemRow]
    wbs_requirements: list[WbsRequirementRow]
    deliverables: list[DeliverableRow]
    risks: list[RiskRow]
    issues: list[IssueRow]
    injected_delays: list[str]


def generate_project(
    index: int,
    source: Source,
    draft: WbsDraft,
    *,
    as_of: date,
    seed: int,
    first_user_no: int,
    taken_names: set[str],
) -> ProjectData:
    """프로젝트 1개의 모든 행. 같은 입력(초안·seed·기준일)이면 항상 같은 결과다."""
    doc = source.document
    project_id = project_id_for(index)
    rng = project_rng(seed, doc.doc_id, "schedule")

    start = as_of - timedelta(days=rng.randint(*ELAPSED_DAYS))
    end = start + timedelta(days=rng.randint(*DURATION_DAYS))
    project = ProjectRow(
        project_id=project_id,
        name=doc.bid_title,
        notice_no=source.notice_no,
        agency=None,  # 수집 기록(manifest)에 발주기관이 없다. 실존 기관 담당자 정보는 만들지 않는다
        contractor=rng.choice(_CONTRACTORS),
        start_date=start,
        end_date=end,
        status="in_progress",
    )
    team = _build_team(rng, project, first_user_no, taken_names, rng.choice(_PARTNERS))
    requirements = build_requirements(project_id, doc)

    # WBS: 1단계(단계 5개) + 2단계(초안 업무)
    spans = _phase_spans(start, end)
    items: list[WbsItemRow] = []
    links: list[WbsRequirementRow] = []
    leaf_deliverable_names: dict[str, list[str]] = {}
    for phase_no, (phase, (ps, pe)) in enumerate(zip(PHASE_ORDER, spans, strict=True), start=1):
        tasks = [t for t in draft.tasks if t.phase == phase]
        length = (pe - ps).days + 1
        for task_no, task in enumerate(tasks, start=1):
            wbs_id = f"{phase_no}.{task_no}"
            duration = min(length, rng.randint(max(2, length // 8), max(3, length // 3)))
            t_start = ps + timedelta(days=rng.randint(0, length - duration))
            t_end = t_start + timedelta(days=duration - 1)
            status, progress, a_start, a_end = _schedule_status(rng, t_start, t_end, as_of)
            items.append(
                WbsItemRow(
                    project_id=project_id,
                    wbs_id=wbs_id,
                    parent_wbs_id=str(phase_no),
                    name=task.name,
                    phase=phase,
                    assignee_user_id=team.pm
                    if not task.requirement_ids
                    else rng.choice(team.developers),
                    planned_start=t_start,
                    planned_end=t_end,
                    actual_start=a_start,
                    actual_end=a_end,
                    progress_pct=progress,
                    status=status,  # type: ignore[arg-type]
                )
            )
            links += [
                WbsRequirementRow(project_id=project_id, wbs_id=wbs_id, requirement_id=rid)
                for rid in task.requirement_ids
            ]
            leaf_deliverable_names[wbs_id] = task.deliverables

    phase_spans = {str(n): span for n, span in enumerate(spans, start=1)}
    injected = _inject_delays(project_rng(seed, doc.doc_id, "delay"), items, phase_spans, as_of)
    items = _with_phase_items(items, spans, team.pm, project_id) + items

    deliverables = _build_deliverables(
        project_rng(seed, doc.doc_id, "deliverable"), items, leaf_deliverable_names, as_of
    )
    risks = _build_risks(project_rng(seed, doc.doc_id, "risk"), project, team, items, as_of)
    issues = _build_issues(
        project_rng(seed, doc.doc_id, "issue"), project, team, items, risks, as_of
    )
    document = DocumentRow(
        doc_id=doc.doc_id,
        project_id=project_id,
        title=doc.bid_title,
        file_type=doc.format,
        sha256=source.sha256,
        security_level="public",
        allowed_roles=list(ALL_ROLES),
        parsed_at=None,
    )
    return ProjectData(
        project,
        team.users,
        team.members,
        requirements,
        document,
        items,
        links,
        deliverables,
        risks,
        issues,
        injected,
    )


def _inject_delays(
    rng: random.Random,
    leaves: list[WbsItemRow],
    phase_spans: dict[str, tuple[date, date]],
    as_of: date,
) -> list[str]:
    """말단 업무를 지연시킨다(제자리 수정). 주입한 wbs_id 목록을 돌려준다.

    - 이번 주 지연 1건(기준일이 월요일이면 건너뛴다). 후보를 앞에서부터 찾는다:
      ① 기준일에 걸친 진행 중 업무 → planned_end를 이번 주 월요일 ~ 기준일 전날로 당긴다
      ② planned_end가 그 범위에 있는 done 업무 → 미완료로 되돌린다
      ③ 상위 단계가 그 범위까지 이어지는 done 업무 → planned_end를 범위 안으로 늦추고 미완료로 되돌린다
      ④ 범위가 들어 있는 단계의 시작 전 업무 → 단계 안에서 범위에 끝나도록 당겨 배치하고 진행 중으로 둔다
         (기준일이 단계 초반이고 그 단계 업무가 모두 기준일 뒤에 시작할 때. v2 PRJ-08에서 발견)
    - 나머지: 이번 주 월요일 전에 끝났어야 할(done) 업무를 미완료로 되돌린다. 3건 이상이면 1건은 on_hold
    phase_spans: 상위 업무(parent_wbs_id) → 단계 (시작, 끝). ③·④에서 상위 기간을 넘지 않게 한다
    """
    target = max(2, round(len(leaves) * DELAY_RATIO))
    injected: list[str] = []
    window = this_week_delay_range(as_of)
    if window is not None:
        monday, last = window
        spanning = [t for t in leaves if t.status == "in_progress" and t.planned_start <= last]
        ended_in_window = [
            t for t in leaves if t.status == "done" and monday <= t.planned_end <= last
        ]
        extendable = [
            t
            for t in leaves
            if t.status == "done"
            and t.planned_end < monday
            and t.parent_wbs_id is not None
            and phase_spans[t.parent_wbs_id][1] >= monday
        ]
        pullable = [
            t
            for t in leaves
            if t.status == "not_started"
            and t.parent_wbs_id is not None
            and phase_spans[t.parent_wbs_id][0] <= last
            and phase_spans[t.parent_wbs_id][1] >= monday
        ]
        if spanning:
            task = rng.choice(spanning)
            task.planned_end = _rand_date(rng, max(monday, task.planned_start), last)
        elif ended_in_window:
            task = rng.choice(ended_in_window)
        elif extendable:
            task = rng.choice(extendable)
            assert task.parent_wbs_id is not None
            task.planned_end = _rand_date(
                rng, monday, min(last, phase_spans[task.parent_wbs_id][1])
            )
        elif pullable:
            task = rng.choice(pullable)
            assert task.parent_wbs_id is not None
            phase_start, phase_end = phase_spans[task.parent_wbs_id]
            task.planned_end = _rand_date(rng, max(monday, phase_start), min(last, phase_end))
            task.planned_start = _rand_date(
                rng, max(phase_start, task.planned_end - timedelta(days=14)), task.planned_end
            )
            task.actual_start = task.planned_start
        else:
            task = None
        if task is not None:
            task.status = "in_progress"
            task.progress_pct = rng.randint(50, 90)
            task.actual_end = None
            injected.append(task.wbs_id)
    cutoff = window[0] if window else as_of
    past = [t for t in leaves if t.status == "done" and t.planned_end < cutoff]
    chosen = rng.sample(past, min(len(past), target - len(injected)))
    for n, task in enumerate(chosen):
        on_hold = n == 0 and len(chosen) >= 3
        task.status = "on_hold" if on_hold else "in_progress"
        task.progress_pct = rng.randint(10, 60) if on_hold else rng.randint(30, 90)
        task.actual_end = None
        injected.append(task.wbs_id)
    return injected


def _with_phase_items(
    leaves: list[WbsItemRow], spans: list[tuple[date, date]], pm: str, project_id: str
) -> list[WbsItemRow]:
    """1단계 업무(단계별). 날짜는 단계 기간, 상태·진척은 하위 업무에서 계산한다."""
    phases: list[WbsItemRow] = []
    for phase_no, (phase, (ps, pe)) in enumerate(zip(PHASE_ORDER, spans, strict=True), start=1):
        children = [t for t in leaves if t.parent_wbs_id == str(phase_no)]
        statuses = {c.status for c in children}
        starts = [c.actual_start for c in children if c.actual_start is not None]
        if statuses == {"done"}:
            status, progress = "done", 100
            actual_end: date | None = max(c.actual_end for c in children if c.actual_end)
        elif statuses == {"not_started"}:
            status, progress, actual_end = "not_started", 0, None
        else:
            status, actual_end = "in_progress", None
            progress = min(99, round(sum(c.progress_pct for c in children) / len(children)))
        phases.append(
            WbsItemRow(
                project_id=project_id,
                wbs_id=str(phase_no),
                parent_wbs_id=None,
                name=f"{phase} 단계",
                phase=phase,
                assignee_user_id=pm,
                planned_start=ps,
                planned_end=pe,
                actual_start=min(starts) if starts else None,
                actual_end=actual_end,
                progress_pct=progress,
                status=status,  # type: ignore[arg-type]
            )
        )
    return phases


def _build_deliverables(
    rng: random.Random,
    items: list[WbsItemRow],
    names_by_wbs: dict[str, list[str]],
    as_of: date,
) -> list[DeliverableRow]:
    """말단 업무의 산출물. 마감일 = 업무 planned_end. 업무가 done이면 제출(70% 승인), 아니면 미제출."""
    by_id = {i.wbs_id: i for i in items}
    project_id = items[0].project_id
    rows: list[DeliverableRow] = []
    yesterday = as_of - timedelta(days=1)
    for wbs_id, names in names_by_wbs.items():
        task = by_id[wbs_id]
        for name in names:
            submitted: date | None = None
            status = "not_submitted"
            if task.status == "done" and task.actual_end is not None:
                submitted = min(task.actual_end + timedelta(days=rng.randint(0, 3)), yesterday)
                status = "approved" if rng.random() < 0.7 else "submitted"
            rows.append(
                DeliverableRow(
                    deliverable_id=f"{project_id}-D{len(rows) + 1:03d}",
                    project_id=project_id,
                    name=name,
                    phase=task.phase,
                    wbs_id=wbs_id,
                    due_date=task.planned_end,
                    submitted_date=submitted,
                    status=status,  # type: ignore[arg-type]
                    doc_id=None,
                )
            )
    return rows


def _statuses(rng: random.Random, count: int, required: list[str], pool: list[str]) -> list[str]:
    statuses = required + [rng.choice(pool) for _ in range(count - len(required))]
    rng.shuffle(statuses)
    return statuses


def _build_risks(
    rng: random.Random, project: ProjectRow, team: _Team, items: list[WbsItemRow], as_of: date
) -> list[RiskRow]:
    """Risk `RISK_COUNT`개. open 2·mitigating 1·closed 1은 반드시 있다. 지연 업무가 있으면 1건은 그 업무의 지연 위험."""
    count = rng.randint(*RISK_COUNT)
    statuses = _statuses(
        rng, count, ["open", "open", "mitigating", "closed"], ["open", "mitigating", "closed"]
    )
    leaves = [i for i in items if i.parent_wbs_id is not None]
    delayed = [t for t in leaves if is_delayed(t.status, t.planned_end, as_of)]
    templates = rng.sample(_RISK_TEMPLATES, min(count, len(_RISK_TEMPLATES)))
    yesterday = as_of - timedelta(days=1)
    rows: list[RiskRow] = []
    for n, status in enumerate(statuses):
        related: str | None = None
        if n == 0 and delayed:
            task = rng.choice(delayed)
            title = f"'{task.name}' 지연에 따른 후속 일정 영향"
            description = f"{task.wbs_id} 업무가 계획 종료일({task.planned_end})을 넘겨 후속 업무 착수가 밀릴 수 있다"
            mitigation = "투입 인력 보강, 후속 업무 일정 재조정"
            related = task.wbs_id
        else:
            title, description, mitigation = templates[n % len(templates)]
            if rng.random() < 0.5:
                related = rng.choice(leaves).wbs_id
        identified = _rand_date(rng, project.start_date, yesterday)
        due = min(identified + timedelta(days=rng.randint(14, 60)), project.end_date)
        rows.append(
            RiskRow(
                risk_id=f"{project.project_id}-R{n + 1:02d}",
                project_id=project.project_id,
                title=title,
                description=description,
                probability=rng.randint(1, 5),
                impact=rng.randint(1, 5),
                status=status,  # type: ignore[arg-type]
                owner_user_id=rng.choice([team.pm, *team.developers]),
                identified_date=identified,
                due_date=due,
                mitigation=mitigation,
                related_wbs_id=related,
            )
        )
    return rows


def _build_issues(
    rng: random.Random,
    project: ProjectRow,
    team: _Team,
    items: list[WbsItemRow],
    risks: list[RiskRow],
    as_of: date,
) -> list[IssueRow]:
    """Issue `ISSUE_COUNT`개. open 2·in_progress 1·resolved 1은 반드시 있다. 약 40%는 Risk와 연결."""
    count = rng.randint(*ISSUE_COUNT)
    statuses = _statuses(
        rng, count, ["open", "open", "in_progress", "resolved"], ["open", "in_progress", "resolved"]
    )
    leaves = [i for i in items if i.parent_wbs_id is not None]
    templates = rng.sample(_ISSUE_TEMPLATES, min(count, len(_ISSUE_TEMPLATES)))
    yesterday = as_of - timedelta(days=1)
    rows: list[IssueRow] = []
    for n, status in enumerate(statuses):
        title, description = templates[n % len(templates)]
        raised = _rand_date(rng, project.start_date, yesterday)
        resolved = _rand_date(rng, raised, yesterday) if status == "resolved" else None
        rows.append(
            IssueRow(
                issue_id=f"{project.project_id}-I{n + 1:02d}",
                project_id=project.project_id,
                title=title,
                description=description,
                severity=rng.choice(["low", "medium", "high", "critical"]),
                status=status,  # type: ignore[arg-type]
                assignee_user_id=rng.choice([team.pm, *team.developers]),
                raised_date=raised,
                resolved_date=resolved,
                related_wbs_id=rng.choice(leaves).wbs_id if rng.random() < 0.6 else None,
                related_risk_id=rng.choice(risks).risk_id if rng.random() < 0.4 else None,
            )
        )
    return rows


def generate(
    sources: Sequence[Source], drafts: dict[str, WbsDraft], *, as_of: date, seed: int
) -> tuple[PmsDataset, dict[str, list[str]]]:
    """전체 데이터셋과 프로젝트별 주입 지연 wbs_id. drafts는 doc_id → 초안."""
    projects: list[ProjectData] = []
    taken_names: set[str] = set()
    next_user = 1
    for index, source in enumerate(sources, start=1):
        data = generate_project(
            index,
            source,
            drafts[source.document.doc_id],
            as_of=as_of,
            seed=seed,
            first_user_no=next_user,
            taken_names=taken_names,
        )
        next_user += len(data.users)
        projects.append(data)
    dataset = PmsDataset(
        as_of_date=as_of,
        seed=seed,
        projects=[p.project for p in projects],
        users=[u for p in projects for u in p.users],
        project_members=[m for p in projects for m in p.members],
        requirements=[r for p in projects for r in p.requirements],
        documents=[p.document for p in projects],
        wbs_items=[w for p in projects for w in p.wbs_items],
        wbs_requirements=[link for p in projects for link in p.wbs_requirements],
        deliverables=[d for p in projects for d in p.deliverables],
        risks=[r for p in projects for r in p.risks],
        issues=[i for p in projects for i in p.issues],
    )
    return dataset, {p.project.project_id: p.injected_delays for p in projects}


# --- 6. 검증 ---


def validate(ds: PmsDataset) -> list[str]:
    """data-design.md 7절 검증 규칙 + 참조 무결성. 위반 문장 목록(비어 있으면 통과)."""
    errors: list[str] = []
    as_of = ds.as_of_date
    projects = {p.project_id: p for p in ds.projects}
    users = {u.user_id for u in ds.users}
    members = {(m.project_id, m.user_id) for m in ds.project_members}
    reqs = {(r.project_id, r.requirement_id) for r in ds.requirements}
    wbs = {(w.project_id, w.wbs_id): w for w in ds.wbs_items}
    risk_projects = {r.risk_id: r.project_id for r in ds.risks}

    def dup(kind: str, keys: Iterable[object]) -> None:
        for key, n in Counter(keys).items():
            if n > 1:
                errors.append(f"{kind} 키 중복: {key}")

    dup("users", (u.user_id for u in ds.users))
    dup("projects", (p.project_id for p in ds.projects))
    dup("requirements", ((r.project_id, r.requirement_id) for r in ds.requirements))
    dup("wbs_items", ((w.project_id, w.wbs_id) for w in ds.wbs_items))
    dup("deliverables", (d.deliverable_id for d in ds.deliverables))
    dup("risks", (r.risk_id for r in ds.risks))
    dup("issues", (i.issue_id for i in ds.issues))
    dup("project_members", ((m.project_id, m.user_id) for m in ds.project_members))
    dup(
        "wbs_requirements",
        ((x.project_id, x.wbs_id, x.requirement_id) for x in ds.wbs_requirements),
    )
    dup("documents", (d.doc_id for d in ds.documents))
    doc_ids = {d.doc_id for d in ds.documents}

    for u in ds.users:
        if not u.email.endswith("@example.com"):
            errors.append(f"{u.user_id}: 가짜 이메일(@example.com)이 아니다")
    for p in ds.projects:
        if not p.start_date <= as_of <= p.end_date:
            errors.append(f"{p.project_id}: 기준일이 프로젝트 기간 밖")
    for m in ds.project_members:
        if m.user_id not in users:
            errors.append(f"project_members {m.project_id}/{m.user_id}: 없는 사용자")

    def in_project(pid: str, label: str, *days: date | None) -> None:
        p = projects.get(pid)
        if p is None:
            errors.append(f"{label}: 없는 프로젝트 {pid}")
            return
        for day in days:
            if day is not None and not p.start_date <= day <= p.end_date:
                errors.append(f"{label}: 날짜 {day}가 프로젝트 기간 밖")

    for m in ds.project_members:
        in_project(
            m.project_id, f"project_members {m.project_id}/{m.user_id}", m.start_date, m.end_date
        )
    for req in ds.requirements:
        in_project(req.project_id, f"requirement {req.project_id}/{req.requirement_id}")
        if req.source_doc_id not in doc_ids:
            errors.append(
                f"requirement {req.project_id}/{req.requirement_id}: 없는 문서 {req.source_doc_id}"
            )

    children: dict[tuple[str, str], list[str]] = defaultdict(list)
    for w in ds.wbs_items:
        if w.parent_wbs_id is not None:
            children[(w.project_id, w.parent_wbs_id)].append(w.status)
    for w in ds.wbs_items:
        label = f"wbs {w.project_id}/{w.wbs_id}"
        in_project(
            w.project_id, label, w.planned_start, w.planned_end, w.actual_start, w.actual_end
        )
        if w.planned_start > w.planned_end:
            errors.append(f"{label}: planned_start > planned_end")
        if w.actual_start and w.actual_end and w.actual_start > w.actual_end:
            errors.append(f"{label}: actual_start > actual_end")
        if w.actual_end is not None and w.actual_start is None:
            errors.append(f"{label}: actual_end만 있다")
        for day in (w.actual_start, w.actual_end):
            if day is not None and day >= as_of:
                errors.append(f"{label}: 실적 날짜 {day}가 기준일 당일 또는 이후")
        # 상태와 실적 날짜: 시작 전이면 실적 시작일이 없고, 시작했으면(진행·보류·완료) 있다
        if (w.status == "not_started") != (w.actual_start is None):
            errors.append(f"{label}: 상태 {w.status}와 실적 시작일이 맞지 않는다")
        if w.status == "done" and any(
            s != "done" for s in children.get((w.project_id, w.wbs_id), [])
        ):
            errors.append(f"{label}: done인데 하위 업무가 미완료")
        if (w.status == "done") != (w.progress_pct == 100 and w.actual_end is not None):
            errors.append(f"{label}: done ⇔ progress 100·actual_end 위반")
        if (w.project_id, w.assignee_user_id) not in members:
            errors.append(f"{label}: 담당자 {w.assignee_user_id}가 프로젝트 멤버가 아니다")
        if w.parent_wbs_id is not None:
            parent = wbs.get((w.project_id, w.parent_wbs_id))
            if parent is None:
                errors.append(f"{label}: 상위 업무 {w.parent_wbs_id}가 없다")
            elif not parent.planned_start <= w.planned_start <= w.planned_end <= parent.planned_end:
                errors.append(f"{label}: 기간이 상위 업무 기간 밖")
    for link in ds.wbs_requirements:
        if (link.project_id, link.wbs_id) not in wbs:
            errors.append(f"wbs_requirements: 없는 업무 {link.project_id}/{link.wbs_id}")
        if (link.project_id, link.requirement_id) not in reqs:
            errors.append(
                f"wbs_requirements: 없는 요구사항 {link.project_id}/{link.requirement_id}"
            )
    for d in ds.deliverables:
        label = f"deliverable {d.deliverable_id}"
        in_project(d.project_id, label, d.due_date, d.submitted_date)
        task = wbs.get((d.project_id, d.wbs_id))
        if task is None:
            errors.append(f"{label}: 없는 업무 {d.wbs_id}")
        elif task.phase != d.phase:
            errors.append(f"{label}: 단계 {d.phase}가 업무 단계 {task.phase}와 다르다")
        if d.doc_id is not None and d.doc_id not in doc_ids:
            errors.append(f"{label}: 없는 문서 {d.doc_id}")
        if (d.submitted_date is None) != (d.status == "not_submitted"):
            errors.append(f"{label}: 제출일과 상태가 맞지 않는다")
        if d.submitted_date is not None and d.submitted_date >= as_of:
            errors.append(f"{label}: 제출일이 기준일 당일 또는 이후")
    for r in ds.risks:
        label = f"risk {r.risk_id}"
        in_project(r.project_id, label, r.identified_date, r.due_date)
        if r.identified_date > r.due_date or r.identified_date >= as_of:
            errors.append(f"{label}: 식별일이 기한 이후이거나 기준일 당일 또는 이후")
        if (r.project_id, r.owner_user_id) not in members:
            errors.append(f"{label}: 담당자가 프로젝트 멤버가 아니다")
        if r.related_wbs_id and (r.project_id, r.related_wbs_id) not in wbs:
            errors.append(f"{label}: 없는 업무 {r.related_wbs_id}")
    for i in ds.issues:
        label = f"issue {i.issue_id}"
        in_project(i.project_id, label, i.raised_date, i.resolved_date)
        if i.raised_date >= as_of or (
            i.resolved_date and not i.raised_date <= i.resolved_date < as_of
        ):
            errors.append(f"{label}: 등록·해결일 순서 오류")
        if (i.resolved_date is not None) != (i.status == "resolved"):
            errors.append(f"{label}: 해결일과 상태가 맞지 않는다")
        if (i.project_id, i.assignee_user_id) not in members:
            errors.append(f"{label}: 담당자가 프로젝트 멤버가 아니다")
        if i.related_wbs_id and (i.project_id, i.related_wbs_id) not in wbs:
            errors.append(f"{label}: 없는 업무 {i.related_wbs_id}")
        if i.related_risk_id and risk_projects.get(i.related_risk_id) != i.project_id:
            errors.append(f"{label}: 같은 프로젝트에 없는 Risk {i.related_risk_id}")
    for doc in ds.documents:
        in_project(doc.project_id, f"document {doc.doc_id}", None)
    return errors


# --- 7. 정답·통계 ---


def unmapped_requirements(ds: PmsDataset) -> dict[str, list[str]]:
    """프로젝트별 업무와 연결되지 않은 요구사항(원래 순서)."""
    linked = {(link.project_id, link.requirement_id) for link in ds.wbs_requirements}
    out: dict[str, list[str]] = defaultdict(list)
    for r in ds.requirements:
        if (r.project_id, r.requirement_id) not in linked:
            out[r.project_id].append(r.requirement_id)
    return dict(out)


def build_truth(
    ds: PmsDataset,
    *,
    excluded: dict[str, list[str]],
    injected: dict[str, list[str]],
    doc_ids: dict[str, str],
    draft_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """정답 파일 내용. 모든 판단은 `pms.rules`의 정의로 데이터에서 계산한다.

    draft_sha256: 프로젝트별 LLM 초안 파일의 sha256 — 정답이 어느 초안에서 나왔는지 남긴다(재현 확인용).

    excluded: 프로젝트별 누락 후보로 LLM에 주지 않은 요구사항. 연결되지 않은 요구사항 중 이 목록에 없으면
    `llm_omitted`(LLM이 빠뜨림)로 표시한다.
    """
    as_of = ds.as_of_date
    window = this_week_delay_range(as_of)
    unmapped = unmapped_requirements(ds)
    projects: list[dict[str, Any]] = []
    for p in ds.projects:
        pid = p.project_id
        items = [w for w in ds.wbs_items if w.project_id == pid]
        links = [link for link in ds.wbs_requirements if link.project_id == pid]
        delivs = [d for d in ds.deliverables if d.project_id == pid]
        risks = [r for r in ds.risks if r.project_id == pid]
        linked_wbs = {link.wbs_id for link in links}
        delayed = [
            {
                "wbs_id": w.wbs_id,
                "name": w.name,
                "status": w.status,
                "planned_end": w.planned_end.isoformat(),
                "delay_days": delay_days(w.planned_end, as_of),
                "this_week": is_delayed_this_week(w.status, w.planned_end, as_of),
                "level": 1 if w.parent_wbs_id is None else 2,
                "injected": w.wbs_id in injected.get(pid, []),
            }
            for w in items
            if is_delayed(w.status, w.planned_end, as_of)
        ]
        open_risks = sorted(
            (r for r in risks if is_open_risk(r.status)),
            key=lambda r: (-(r.probability * r.impact), r.risk_id),
        )
        rtm: dict[str, dict[str, list[str]]] = {}
        for link in links:
            entry = rtm.setdefault(link.requirement_id, {"wbs_ids": [], "deliverable_ids": []})
            entry["wbs_ids"].append(link.wbs_id)
            entry["deliverable_ids"] += [
                d.deliverable_id for d in delivs if d.wbs_id == link.wbs_id
            ]
        projects.append(
            {
                "project_id": pid,
                "doc_id": doc_ids[pid],
                "draft_sha256": (draft_sha256 or {}).get(pid),
                "name": p.name,
                "delayed_tasks": delayed,
                "open_risks": [
                    {"risk_id": r.risk_id, "status": r.status, "score": r.probability * r.impact}
                    for r in open_risks
                ],
                "overdue_unsubmitted_deliverables": [
                    d.deliverable_id
                    for d in delivs
                    if d.status == "not_submitted" and d.due_date < as_of
                ],
                "missing_requirements": [
                    {
                        "requirement_id": rid,
                        "reason": "injected" if rid in excluded.get(pid, []) else "llm_omitted",
                    }
                    for rid in unmapped.get(pid, [])
                ],
                "out_of_scope_tasks": [
                    w.wbs_id
                    for w in items
                    if w.parent_wbs_id is not None and w.wbs_id not in linked_wbs
                ],
                "rtm": [{"requirement_id": rid, **v} for rid, v in sorted(rtm.items())],
            }
        )
    return {
        "version": TRUTH_VERSION,
        "as_of_date": as_of.isoformat(),
        "seed": ds.seed,
        "this_week_range": [d.isoformat() for d in window] if window else None,
        "draft_prompt_version": DRAFT_PROMPT_VERSION,
        "rules": "docs/data-design.md 6절 핵심 판단 규칙 (rfp_pm_agent.pms.rules)",
        "projects": projects,
    }


def stats(ds: PmsDataset, truth: dict[str, Any]) -> dict[str, Any]:
    """검증 수치: 테이블별 행 수, 지연·미해결 Risk·미제출 산출물·누락 후보 수, RTM 커버리지."""
    tables = {
        name: len(getattr(ds, name))
        for name in (
            "projects",
            "users",
            "project_members",
            "requirements",
            "documents",
            "wbs_items",
            "wbs_requirements",
            "deliverables",
            "risks",
            "issues",
        )
    }
    projects = truth["projects"]
    delayed = [t for p in projects for t in p["delayed_tasks"]]
    missing = [m for p in projects for m in p["missing_requirements"]]
    total_reqs = len(ds.requirements)
    injected_missing = sum(m["reason"] == "injected" for m in missing)
    mapped = total_reqs - len(missing)
    leaves = [w for w in ds.wbs_items if w.parent_wbs_id is not None]
    return {
        "as_of_date": ds.as_of_date.isoformat(),
        "tables": tables,
        "leaf_tasks": len(leaves),
        "delayed_tasks": len(delayed),
        "delayed_leaf_tasks": sum(t["level"] == 2 for t in delayed),
        "delayed_this_week": sum(t["this_week"] for t in delayed),
        "open_risks": sum(len(p["open_risks"]) for p in projects),
        "overdue_unsubmitted_deliverables": sum(
            len(p["overdue_unsubmitted_deliverables"]) for p in projects
        ),
        "not_submitted_deliverables": sum(d.status == "not_submitted" for d in ds.deliverables),
        "missing_requirements": len(missing),
        "missing_injected": injected_missing,
        "missing_llm_omitted": len(missing) - injected_missing,
        "out_of_scope_tasks": sum(len(p["out_of_scope_tasks"]) for p in projects),
        "rtm_coverage": f"{mapped}/{total_reqs} ({mapped / total_reqs:.1%})"
        if total_reqs
        else "0/0",
        "rtm_coverage_given_to_llm": (
            f"{mapped}/{total_reqs - injected_missing}" if total_reqs else "0/0"
        ),
    }


# --- CLI ---


def _sources_with_missing(seed: int) -> list[tuple[Source, list[RequirementRow], list[str]]]:
    out = []
    for index, source in enumerate(select_sources(), start=1):
        reqs = build_requirements(project_id_for(index), source.document)
        out.append(
            (
                source,
                reqs,
                pick_missing(seed, source.document.doc_id, [r.requirement_id for r in reqs]),
            )
        )
    return out


def cmd_draft(limit: int | None, drafts_dir: Path) -> int:
    from rfp_pm_agent.clients.cost import CostLogger
    from rfp_pm_agent.clients.llm import OpenAICompatLLMClient
    from rfp_pm_agent.clients.logged import LoggingLLMClient
    from rfp_pm_agent.config import ClientsConfig, SynthConfig

    synth = SynthConfig.from_env()
    config = ClientsConfig.from_env()
    llm = LoggingLLMClient(OpenAICompatLLMClient(config), config, CostLogger(config.cost_log_path))
    todo = [
        (s, reqs, missing)
        for s, reqs, missing in _sources_with_missing(synth.synth_seed)
        if load_cached_draft(drafts_dir, s.document.doc_id, missing) is None
    ]
    print(f"초안이 필요한 문서 {len(todo)}건, 이번 실행 {min(len(todo), limit or len(todo))}건")
    for source, reqs, missing in todo[:limit]:
        given = [r for r in reqs if r.requirement_id not in missing]
        cached = request_draft(
            llm, source.document.doc_id, source.document.bid_title, given, missing
        )
        path = save_draft(drafts_dir, cached)
        tokens = (
            sum(u.input_tokens for u in cached.usage),
            sum(u.output_tokens for u in cached.usage),
        )
        print(
            f"{path} 업무 {len(cached.draft.tasks)}개, 시도 {len(cached.usage)}회, "
            f"버린 ID {cached.dropped_requirement_ids}, 토큰 입력 {tokens[0]}·출력 {tokens[1]}"
        )
    return 0


def check_as_of(configured: date, recorded: date, what: str) -> None:
    """설정 기준일(`AS_OF_DATE`)과 파일에 기록된 기준일이 다르면 ValueError.

    `.env`의 기준일이 데이터와 다르면 조회 툴(#21)이 다른 날 기준으로 판단하고, 다시 생성하면 정답 파일을 다른 기준일로 덮는다.
    """
    if configured != recorded:
        raise ValueError(
            f"AS_OF_DATE({configured})가 {what}의 기준일({recorded})과 다르다 — "
            f".env의 AS_OF_DATE를 확인한다. 기준일을 바꾸려면 {what}을 지우고 다시 만든다"
        )


def cmd_generate(drafts_dir: Path, dataset_path: Path, truth_path: Path) -> int:
    import hashlib

    from rfp_pm_agent.config import SynthConfig

    synth = SynthConfig.from_env()
    if truth_path.exists():
        recorded = date.fromisoformat(json.loads(truth_path.read_text("utf-8"))["as_of_date"])
        check_as_of(synth.as_of_date, recorded, str(truth_path))
    items = _sources_with_missing(synth.synth_seed)
    drafts: dict[str, CachedDraft] = {}
    for source, _, missing in items:
        cached = load_cached_draft(drafts_dir, source.document.doc_id, missing)
        if cached is None:
            raise SystemExit(f"{source.document.doc_id} 초안이 없다 — 먼저 `draft`를 실행한다")
        drafts[source.document.doc_id] = cached
    sources = [s for s, _, _ in items]
    ds, injected = generate(
        sources,
        {k: v.draft for k, v in drafts.items()},
        as_of=synth.as_of_date,
        seed=synth.synth_seed,
    )
    errors = validate(ds)
    if errors:
        print("\n".join(errors[:50]))
        raise SystemExit(f"검증 실패 {len(errors)}건 — 파일을 쓰지 않았다")
    excluded = {project_id_for(i): m for i, (_, _, m) in enumerate(items, start=1)}
    doc_ids = {project_id_for(i): s.document.doc_id for i, s in enumerate(sources, start=1)}
    hashes = {
        project_id_for(i): hashlib.sha256(
            draft_path(drafts_dir, s.document.doc_id).read_bytes()
        ).hexdigest()
        for i, s in enumerate(sources, start=1)
    }
    truth = build_truth(
        ds, excluded=excluded, injected=injected, doc_ids=doc_ids, draft_sha256=hashes
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(ds.model_dump_json(indent=1) + "\n", "utf-8")
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(dataset_path)
    print(truth_path)
    print(json.dumps(stats(ds, truth), ensure_ascii=False, indent=1))
    return 0


def load_dataset_file(path: Path) -> PmsDataset:
    return PmsDataset.model_validate_json(path.read_text("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="합성 PMS 생성·적재 (이슈 #20)")
    parser.add_argument("command", choices=["draft", "generate", "load", "stats"])
    parser.add_argument("--limit", type=int, help="draft: 이번에 LLM을 부를 문서 수")
    parser.add_argument("--drafts-dir", type=Path, default=DEFAULT_DRAFTS_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    args = parser.parse_args(argv)
    if args.command == "draft":
        return cmd_draft(args.limit, args.drafts_dir)
    if args.command == "generate":
        return cmd_generate(args.drafts_dir, args.dataset, args.truth)
    from rfp_pm_agent.config import SynthConfig

    ds = load_dataset_file(args.dataset)
    check_as_of(SynthConfig.from_env().as_of_date, ds.as_of_date, str(args.dataset))
    if args.command == "load":
        from rfp_pm_agent.config import PostgresConfig
        from rfp_pm_agent.pms.load import load_into_db

        counts = load_into_db(PostgresConfig.from_env().dsn, ds)
        print(json.dumps(counts, ensure_ascii=False, indent=1))
        return 0
    truth = json.loads(args.truth.read_text("utf-8"))
    print(json.dumps(stats(ds, truth), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
