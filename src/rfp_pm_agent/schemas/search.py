"""검색 툴 결과 스키마 (이슈 #18 PR ③, docs/data-design.md 5절 "검색 툴 search_documents").

`search_documents`가 돌려주는 결과 한 건이다. 평가용 `schemas/eval.py`의 `SearchHit`(chunk_id·doc_id·
score·text만)과 따로 둔다 — 서비스 결과에는 출처 표기(#19)와 에이전트 평가(#25)에 쓰는 필드가 더 있다.
필드 의미와 null이 되는 경우는 data-design.md 5절 반환 표가 기준이다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentHit(BaseModel):
    """검색 결과 한 건. 목록은 score 내림차순이고 rank는 그 순서(1부터)다."""

    rank: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    bid_title: str
    format: Literal["pdf", "hwp", "hwpx"]
    requirement_id: str | None
    page: int | None  # 0부터 세는 PDF 페이지. HWP·HWPX는 None
    printed_page: int | None  # 인쇄 쪽번호. 블록 청크와 못 읽은 요구사항은 None
    block_type: Literal["paragraph", "table"] | None  # 요구사항 청크는 None
    text: str
    # search_documents에서는 리랭커 점수(TEI /rerank 기본 응답, 0~1). 한 번의 호출 안에서만 비교한다
    score: float
