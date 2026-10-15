"""WBS 초안 — LLM이 요구사항 목록으로 업무·연결 요구사항·산출물 이름을 만든다 (이슈 #20, data-design.md 7절 2단계).

- 프로젝트당 LLM 1회(형식이 틀리면 1회 더, 최대 `MAX_ATTEMPTS`). 응답은 JSON 객체 하나이고 `WbsDraft`로 검증한다
- 목록에 없는 요구사항 ID는 버리고 버린 ID를 기록한다. 날짜·상태·담당자는 LLM에 맡기지 않는다(`pms.synth`가 코드로 정한다)
- 결과는 `data/synth/wbs_drafts/<doc_id>.json`에 저장한다. 같은 문서·같은 제외 목록·같은 프롬프트 버전이면 다시 부르지 않는다
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from rfp_pm_agent.clients.llm import ChatMessage, LLMClient
from rfp_pm_agent.schemas.pms import PHASE_ORDER, Phase, RequirementRow

DRAFT_PROMPT_VERSION = "wbs_draft_v1"
MIN_TASKS = 30
MAX_TASKS = 60
MIN_TASKS_PER_PHASE = 1  # 프롬프트는 3개 이상을 요청한다. 검증은 "빈 단계가 없다"까지만 본다
MAX_ATTEMPTS = 2
# 요구사항 제목을 프롬프트에 넣을 때의 최대 글자 수(입력 토큰 절약)
TITLE_MAX_CHARS = 60

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


class DraftError(ValueError):
    """LLM 응답이 형식·개수 규칙을 지키지 않았다."""


class TaskDraft(BaseModel):
    phase: Phase
    name: str = Field(min_length=1)
    requirement_ids: list[str] = []
    deliverables: list[str] = []


class WbsDraft(BaseModel):
    tasks: list[TaskDraft]


class DraftUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


class CachedDraft(BaseModel):
    """`data/synth/wbs_drafts/<doc_id>.json` 형식."""

    doc_id: str
    prompt_version: str
    excluded_requirement_ids: list[str]
    draft: WbsDraft
    dropped_requirement_ids: list[str]
    usage: list[DraftUsage]  # 시도마다 1개(실패한 시도 포함)


def load_draft_prompt() -> str:
    path = resources.files("rfp_pm_agent.pms").joinpath("prompts", f"{DRAFT_PROMPT_VERSION}.md")
    return path.read_text("utf-8")


def build_draft_messages(
    project_name: str, requirements: Sequence[RequirementRow]
) -> list[ChatMessage]:
    """system(프롬프트 파일) + user(사업명과 요구사항 `- ID [분류] 제목` 목록)."""
    lines = []
    for req in requirements:
        title = (
            req.title if len(req.title) <= TITLE_MAX_CHARS else req.title[:TITLE_MAX_CHARS] + "…"
        )
        category = f" [{req.category}]" if req.category else ""
        lines.append(f"- {req.requirement_id}{category} {title}")
    user = f"사업명: {project_name}\n요구사항 {len(requirements)}건:\n" + "\n".join(lines)
    return [
        ChatMessage(role="system", content=load_draft_prompt()),
        ChatMessage(role="user", content=user),
    ]


def parse_draft(content: str, allowed_ids: Sequence[str]) -> tuple[WbsDraft, list[str]]:
    """응답을 `WbsDraft`로 읽고, 목록에 없는 요구사항 ID를 버린다. (초안, 버린 ID 목록)을 돌려준다.

    DraftError: JSON이 아니거나, 스키마가 다르거나, 업무 수가 30~60 밖이거나, 업무가 없는 단계가 있을 때.
    업무 안에서 같은 ID가 두 번 나오면 한 번만 남긴다.
    """
    text = _FENCE_RE.sub("", content.strip())
    try:
        draft = WbsDraft.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise DraftError(f"응답 형식 오류: {type(e).__name__}: {str(e)[:200]}") from e
    if not MIN_TASKS <= len(draft.tasks) <= MAX_TASKS:
        raise DraftError(f"업무 수 {len(draft.tasks)}개가 {MIN_TASKS}~{MAX_TASKS} 밖이다")
    for phase in PHASE_ORDER:
        if sum(t.phase == phase for t in draft.tasks) < MIN_TASKS_PER_PHASE:
            raise DraftError(f"'{phase}' 단계에 업무가 없다")

    allowed = set(allowed_ids)
    dropped: list[str] = []
    for task in draft.tasks:
        kept: list[str] = []
        for rid in task.requirement_ids:
            if rid not in allowed:
                if rid not in dropped:
                    dropped.append(rid)
            elif rid not in kept:
                kept.append(rid)
        task.requirement_ids = kept
        task.deliverables = [d.strip() for d in task.deliverables if d.strip()][:2]
        task.name = task.name.strip()
    return draft, dropped


def request_draft(
    llm: LLMClient,
    doc_id: str,
    project_name: str,
    requirements: Sequence[RequirementRow],
    excluded_requirement_ids: Sequence[str],
) -> CachedDraft:
    """LLM을 불러 초안을 만든다. 형식 오류면 다시 부른다(최대 `MAX_ATTEMPTS`회). 끝까지 실패하면 DraftError."""
    messages = build_draft_messages(project_name, requirements)
    allowed = [r.requirement_id for r in requirements]
    usage: list[DraftUsage] = []
    last_error: DraftError | None = None
    for _ in range(MAX_ATTEMPTS):
        response = llm.chat(messages)
        usage.append(
            DraftUsage(
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                reasoning_tokens=response.reasoning_tokens,
            )
        )
        try:
            draft, dropped = parse_draft(response.content, allowed)
        except DraftError as e:
            last_error = e
            continue
        return CachedDraft(
            doc_id=doc_id,
            prompt_version=DRAFT_PROMPT_VERSION,
            excluded_requirement_ids=list(excluded_requirement_ids),
            draft=draft,
            dropped_requirement_ids=dropped,
            usage=usage,
        )
    raise DraftError(f"{doc_id}: {MAX_ATTEMPTS}회 모두 실패 — 마지막 오류: {last_error}")


def draft_path(drafts_dir: Path, doc_id: str) -> Path:
    return drafts_dir / f"{doc_id}.json"


def load_cached_draft(
    drafts_dir: Path, doc_id: str, excluded_requirement_ids: Sequence[str]
) -> CachedDraft | None:
    """저장된 초안. 없거나, 프롬프트 버전·제외 목록이 지금과 다르면 None(다시 만들어야 한다)."""
    path = draft_path(drafts_dir, doc_id)
    if not path.exists():
        return None
    cached = CachedDraft.model_validate_json(path.read_text("utf-8"))
    if cached.prompt_version != DRAFT_PROMPT_VERSION:
        return None
    if cached.excluded_requirement_ids != list(excluded_requirement_ids):
        return None
    return cached


def save_draft(drafts_dir: Path, cached: CachedDraft) -> Path:
    drafts_dir.mkdir(parents=True, exist_ok=True)
    path = draft_path(drafts_dir, cached.doc_id)
    path.write_text(cached.model_dump_json(indent=2) + "\n", "utf-8")
    return path
