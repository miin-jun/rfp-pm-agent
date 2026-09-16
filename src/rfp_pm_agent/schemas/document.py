"""문서 공통 스키마 (이슈 #11) — docs/data-design.md 2절과 일치해야 한다.

Document→Section→Block 계층(원래 2절)을 Section 없는 평면 구조로 교체했다.
근거: docs/parsing-exploration.md "요구사항 정의표 구조" 절 실측 — 요구사항
1개는 표(또는 페이지에 걸친 표 조각들) 1개이지 섹션 단위가 아니었고, HWP·HWPX는
페이지 개념 자체가 없어 heading_path보다 requirement_id·페이지 기준 출처 표기가
더 실제 구조에 맞았다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Block(BaseModel):
    """요구사항으로 소비되지 않은 콘텐츠(문단·표) 한 조각."""

    block_id: str
    type: Literal["paragraph", "table"]
    text: str
    table: list[list[str | None]] | None = None
    source_order: int
    pdf_page: int | None = None  # 0-based, PDF만


class Requirement(BaseModel):
    """요구사항 정의표 1개(또는 페이지에 걸쳐 이어붙인 조각들)에서 만든 레코드."""

    requirement_id: str  # "SFR-001"
    prefix: str  # "SFR" — 코드 자체에서 분리, 하드코딩된 접두어 목록 없음
    fields: dict[
        str, str
    ]  # 별칭 매핑된 표준 키만 (category/id/name/definition/detail/output/related)
    raw_fields: dict[str, str]  # 원문 라벨 그대로, 별칭에 없는 라벨도 보존
    text: str  # fields를 사람이 읽기 좋게 이어붙인 텍스트
    source_order: int
    pdf_page_start: int | None = None  # 0-based
    pdf_page_end: int | None = None  # 0-based
    printed_page_start: int | None = None  # 문서 하단 인쇄 쪽번호, 못 읽으면 null
    printed_page_end: int | None = None


class Document(BaseModel):
    doc_id: str
    source_file: str
    bid_title: str
    format: Literal["pdf", "hwp", "hwpx"]
    parse_status: Literal["parsed", "unsupported_format", "failed"]
    has_requirements: bool
    requirement_count: int
    declared_total: int | None  # 문서 자체 "합계" 숫자, 없거나 후보가 여럿이면 null
    summary_ids: list[str]  # 목록(요약)표에서 모은 코드 (검증용)
    validation_warnings: list[str]
    blocks: list[Block]
    requirements: list[Requirement]
