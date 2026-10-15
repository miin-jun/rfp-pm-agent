"""WBS 초안 파싱·요청·캐시 (이슈 #20). LLM은 `ScriptedLLMClient`다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rfp_pm_agent.pms.draft import (
    DRAFT_PROMPT_VERSION,
    MAX_ATTEMPTS,
    DraftError,
    build_draft_messages,
    load_cached_draft,
    parse_draft,
    request_draft,
    save_draft,
)
from rfp_pm_agent.schemas.pms import PHASE_ORDER, RequirementRow
from tests.fakes.fake_llm import ScriptedLLMClient

IDS = ["SFR-001", "SFR-002", "PER-001"]


def _req(rid: str, title: str = "제목") -> RequirementRow:
    return RequirementRow(
        project_id="PRJ-01",
        requirement_id=rid,
        category="기능",
        title=title,
        description="설명",
        source_doc_id="doc",
        source_chunk_id=f"doc:block_requirement:{rid}",
    )


def _tasks(n: int = 30, **first: Any) -> list[dict[str, Any]]:
    tasks = [
        {
            "phase": PHASE_ORDER[i * len(PHASE_ORDER) // n],
            "name": f"업무 {i}",
            "requirement_ids": [IDS[i % len(IDS)]],
            "deliverables": ["산출물"],
        }
        for i in range(n)
    ]
    tasks[0].update(first)
    return tasks


def _response(tasks: list[dict[str, Any]]) -> str:
    return json.dumps({"tasks": tasks}, ensure_ascii=False)


def test_parse_draft_reads_json_and_keeps_order() -> None:
    draft, dropped = parse_draft(_response(_tasks()), IDS)

    assert len(draft.tasks) == 30
    assert draft.tasks[0].name == "업무 0"
    assert dropped == []


def test_parse_draft_accepts_code_fence() -> None:
    draft, _ = parse_draft(f"```json\n{_response(_tasks())}\n```", IDS)

    assert len(draft.tasks) == 30


def test_parse_draft_drops_unknown_ids_and_dedupes() -> None:
    tasks = _tasks(requirement_ids=["SFR-001", "XXX-999", "SFR-001", "YYY-001"])

    draft, dropped = parse_draft(_response(tasks), IDS)

    assert draft.tasks[0].requirement_ids == ["SFR-001"]
    assert dropped == ["XXX-999", "YYY-001"]


def test_parse_draft_keeps_at_most_two_nonblank_deliverables() -> None:
    draft, _ = parse_draft(_response(_tasks(deliverables=[" ", "A", "B", "C"])), IDS)

    assert draft.tasks[0].deliverables == ["A", "B"]


@pytest.mark.parametrize(
    "content",
    [
        "업무를 만들었습니다",
        json.dumps({"items": []}),
        _response(_tasks(phase="운영")),  # 없는 단계
        _response(_tasks(29)),
        _response(_tasks(61)),
    ],
)
def test_parse_draft_rejects_bad_responses(content: str) -> None:
    with pytest.raises(DraftError):
        parse_draft(content, IDS)


def test_parse_draft_rejects_empty_phase() -> None:
    tasks = [{**t, "phase": "분석"} if t["phase"] == "이행" else t for t in _tasks()]

    with pytest.raises(DraftError, match="이행"):
        parse_draft(_response(tasks), IDS)


def test_build_draft_messages_lists_given_requirements_only() -> None:
    long_title = "가" * 100
    messages = build_draft_messages("A 사업", [_req("SFR-001", long_title), _req("SFR-002")])

    assert [m.role for m in messages] == ["system", "user"]
    user = messages[1].content
    assert "사업명: A 사업" in user
    assert "- SFR-001 [기능] " + "가" * 60 + "…" in user
    assert "SFR-002" in user and "PER-001" not in user


def test_request_draft_retries_once_after_format_error() -> None:
    llm = ScriptedLLMClient(["형식 오류", _response(_tasks())])

    cached = request_draft(llm, "doc", "A 사업", [_req(r) for r in IDS], ["DAR-001"])

    assert len(llm.calls) == 2
    assert len(cached.usage) == 2
    assert cached.excluded_requirement_ids == ["DAR-001"]
    assert cached.prompt_version == DRAFT_PROMPT_VERSION


def test_request_draft_gives_up_after_max_attempts() -> None:
    llm = ScriptedLLMClient(["형식 오류"] * MAX_ATTEMPTS)

    with pytest.raises(DraftError, match="모두 실패"):
        request_draft(llm, "doc", "A 사업", [_req(r) for r in IDS], [])
    assert len(llm.calls) == MAX_ATTEMPTS


def test_cache_is_reused_only_for_same_excluded_list(tmp_path: Path) -> None:
    llm = ScriptedLLMClient([_response(_tasks())])
    cached = request_draft(llm, "doc", "A 사업", [_req(r) for r in IDS], ["DAR-001"])
    save_draft(tmp_path, cached)

    assert load_cached_draft(tmp_path, "doc", ["DAR-001"]) == cached
    assert (
        load_cached_draft(tmp_path, "doc", ["DAR-002"]) is None
    )  # seed가 바뀌어 제외 목록이 다르면 다시 만든다
    assert load_cached_draft(tmp_path, "other", ["DAR-001"]) is None


def test_draft_prompt_v2_links_management_requirements() -> None:
    """v1 규칙 5는 관리 업무를 항상 빈 목록으로 두게 해 보고 요구사항이 연결될 업무가 사라졌다 (ADR-0008)."""
    assert DRAFT_PROMPT_VERSION == "wbs_draft_v2"
    prompt = build_draft_messages("A 사업", [_req("PMR-001")])[0].content
    assert "사업관리 요구사항(PMR 등)이 목록에 있으면" in prompt


def test_drafts_are_saved_per_prompt_version(tmp_path: Path) -> None:
    llm = ScriptedLLMClient([_response(_tasks())])
    cached = request_draft(llm, "doc", "A 사업", [_req(r) for r in IDS], [])

    path = save_draft(tmp_path, cached)

    assert path == tmp_path / DRAFT_PROMPT_VERSION / "doc.json"
