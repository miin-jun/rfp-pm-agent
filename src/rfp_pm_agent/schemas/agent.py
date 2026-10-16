"""에이전트 답변 결과 스키마 (이슈 #22, docs/adr/0010-langgraph-agent.md).

질문(사용자 턴) 하나에 `Agent.ask`가 돌려준다. 출처 관련 필드는 #19 `GroundedAnswer`와 같은 뜻이다
(data-design.md 5절 "근거 답변 answer_question" 반환 표). 다른 점은 청크 번호가 대화(thread) 전체에서 이어진다는 것이다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rfp_pm_agent.schemas.answer import LLMUsage

# ok: 툴이 결과를 돌려줌 / project_required·project_not_found: PMS 툴의 되묻기 상태(ADR-0009 결정 2)
# invalid_args: 인자 오류(ValueError·검증 실패, LLM이 고쳐 다시 부를 수 있다) / unavailable: 서버 연결 불가
ToolCallStatus = Literal[
    "ok", "project_required", "project_not_found", "invalid_args", "unavailable"
]


class ToolCallRecord(BaseModel):
    """이번 턴에 실행한 툴 호출 1건. args는 LLM이 만든 인자 그대로다(검증 전)."""

    name: str
    args: dict[str, Any]
    status: ToolCallStatus
    result_count: int = Field(ge=0)  # 검색 결과 수 또는 PMS 결과의 total. 오류면 0


class AgentAnswer(BaseModel):
    """질문 하나의 답. answer는 LLM 답 그대로(청크 번호 [n] 포함)이고, 출처 문자열은 citations·citation_map에 있다."""

    answer: str
    refused: bool  # answer가 거절 문구("문서에서 찾을 수 없음")인지 — `is_refusal`과 같은 판정
    citations: list[str]  # 답에 처음 나온 순서, 같은 문자열은 한 번만
    citation_map: dict[int, str]  # 답에 나온 유효 청크 번호 → 출처 문자열
    cited_chunk_ids: list[str]
    invalid_citations: list[int]  # 이 대화에서 툴 결과로 준 적 없는 번호
    tool_calls: list[ToolCallRecord]
    llm_calls: int = Field(ge=0)
    reached_max_steps: bool  # LLM 호출이 AGENT_MAX_STEPS에 닿아 마지막 호출을 툴 없이 했는지
    project_id: (
        str | None
    )  # 이번 턴이 끝났을 때 대화의 대상 프로젝트(PMS 툴이 ok를 돌려줬을 때 정해진다)
    doc_ids: list[str]  # 이번 턴이 끝났을 때 대화의 대상 문서
    usage: LLMUsage  # 이번 턴 LLM 호출의 토큰 합계
