"""근거 답변 결과 스키마 (이슈 #19, docs/data-design.md 5절 "근거 답변 answer_question").

`answer_question`이 돌려주는 결과다. 필드 의미와 거절일 때의 값은 data-design.md 5절 반환 표가 기준이다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """LLM 호출 1회의 토큰 수. output_tokens는 추론 토큰을 포함한 출력 토큰 전체다."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)


class GroundedAnswer(BaseModel):
    """질문 하나의 답변. 거절이면 answer는 거절 문구이고 출처 관련 목록은 모두 비어 있다."""

    answer: str
    refused: bool
    # 출처 문자열(data-design.md 2절 5가지). 답에 처음 나온 순서, 같은 문자열은 한 번만
    citations: list[str]
    # 답에 나온 유효 인용 번호(1~hit_count) → 그 청크의 출처 문자열. 답에 처음 나온 순서. 거절이면 빈 dict
    citation_map: dict[int, str]
    # 인용된 청크의 chunk_id. 답에 처음 나온 순서, 같은 청크는 한 번만
    cited_chunk_ids: list[str]
    # 검색 결과 범위(1~hit_count) 밖의 인용 번호. 처음 나온 순서, 같은 번호는 한 번만
    invalid_citations: list[int]
    hit_count: int = Field(ge=0)  # 검색 결과 수(LLM에 넣은 청크 수)
    llm_usage: LLMUsage | None  # LLM을 부르지 않았으면 None(검색 결과 0건)
