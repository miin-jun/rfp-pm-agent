"""시연 화면 API 스키마 (이슈 #40, docs/adr/0015-demo-ui-fastapi-single-html.md).

`POST /api/ask`의 요청·응답이다. 답·거절 판정·출처 문자열은 `AgentAnswer`(schemas/agent.py) 그대로이고,
출처마다 청크 원문을 붙인다(docs/data-design.md 5절 "시연 화면 API").
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    question: Question


class SourceChunk(BaseModel):
    """답에 나온 출처 번호 하나. 문서 필드·원문은 OpenSearch에서 chunk_id로 읽는다 — 읽지 못하면 None."""

    number: int = Field(ge=1)  # 답의 [n]
    chunk_id: str
    citation: str  # `format_citation` 문자열(AgentAnswer.citation_map)
    bid_title: str | None
    format: Literal["pdf", "hwp", "hwpx"] | None
    printed_page: int | None  # 인쇄 쪽번호
    page: int | None  # 0부터 세는 PDF 페이지. HWP·HWPX는 None
    requirement_id: str | None
    text: str | None  # 청크 원문


class AskResponse(BaseModel):
    answer: str  # LLM 답 그대로([n] 포함)
    refused: bool  # `is_refusal` 판정 — 근거를 찾지 못한 답
    sources: list[SourceChunk]  # 답에 처음 나온 순서
    tools_used: list[str]  # 실행한 툴 이름, 호출 순서대로(같은 툴을 두 번 부르면 두 번)
    elapsed_ms: int = Field(ge=0)  # 에이전트 실행 + 원문 조회
