"""PDF → Document 파서 (PyMuPDF). 이슈 #11.

docs/parsing-exploration.md 실측 근거:
- pymupdf `find_tables()`가 요구사항 정의표를 표 구조로 정상 인식한다(텍스트만
  나오지 않는다) — 그래서 pdfplumber가 아니라 pymupdf를 최종 채택했다
  (docs/tech-stack.md 8절).
- 페이지 하단(높이의 85% 아래)에 `- N -` 형식의 인쇄 쪽번호가 있다. 고정
  오프셋으로 계산하지 않고 실제 위치·텍스트를 읽는다. 못 찾으면 null.
- 세부내용이 길면 정의표 하나가 여러 페이지의 별개 표 객체로 쪼개진다. 이어붙임은
  "코드 없는 표가 (a) 그 페이지의 첫 콘텐츠 블록이고 (b) 직전 조각 이후 문단
  블록이 끼어들지 않았고 (c) 열 수가 직전 조각과 같을 때"만 한다. 머리글·꼬리말
  영역(페이지 상단 8%·하단 15%)은 이 판정에서 제외한다 — 인쇄 쪽번호 추출과
  같은 영역 상수를 쓴다.

`_build_document_parts`는 pymupdf 객체와 분리된 순수 함수다 — 실제 PDF 없이도
실물에서 추출한 표 matrix(fixture)만으로 이어붙임·종료 로직을 단위 테스트할 수
있게 하기 위함이다.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf

from rfp_pm_agent.ingest.parsers.common import (
    RequirementBuilder,
    collect_loose_codes,
    find_declared_total,
    table_role,
    validate_requirement_ids,
)
from rfp_pm_agent.schemas.document import Block, Document, Requirement

HEADER_ZONE_RATIO = 0.08  # 페이지 높이의 위 8% — 머리글 영역
FOOTER_ZONE_RATIO = 0.85  # 페이지 높이의 85% 아래 — 꼬리말(인쇄 쪽번호) 영역
PRINTED_PAGE_PATTERN = re.compile(r"^-\s*(\d{1,4})\s*-$")


def _in_footer_zone(y0: float, height: float) -> bool:
    return y0 >= height * FOOTER_ZONE_RATIO


def _in_header_zone(y1: float, height: float) -> bool:
    return y1 <= height * HEADER_ZONE_RATIO


def extract_printed_page_number(page: Any) -> int | None:
    """페이지 하단(꼬리말 영역)에서 "- N -" 형식의 인쇄 쪽번호를 읽는다.
    고정 오프셋 계산 없이 실제 텍스트를 읽고, 못 찾으면 None."""
    height = page.rect.height
    found: int | None = None
    for block in page.get_text("blocks"):
        y0, text = block[1], block[4]
        if not _in_footer_zone(y0, height):
            continue
        for line in text.strip().splitlines():
            m = PRINTED_PAGE_PATTERN.match(line.strip())
            if m:
                found = int(m.group(1))
    return found


def _bbox_contains_center(
    bbox: tuple[float, float, float, float], point: tuple[float, float]
) -> bool:
    x0, y0, x1, y1 = bbox
    px, py = point
    return x0 <= px <= x1 and y0 <= py <= y1


class TableItem:
    def __init__(
        self, bbox: tuple[float, float, float, float], matrix: list[list[str | None]]
    ) -> None:
        self.bbox = bbox
        self.matrix = matrix


class ParagraphItem:
    def __init__(self, bbox: tuple[float, float, float, float], text: str) -> None:
        self.bbox = bbox
        self.text = text


PageItem = TableItem | ParagraphItem


def _page_items(page: Any) -> list[PageItem]:
    """페이지 콘텐츠를 표·문단 조각으로 나눠 위→아래 순서로 반환한다.
    머리글·꼬리말 영역은 제외한다."""
    height = page.rect.height
    finder = page.find_tables()
    tables = sorted(finder.tables, key=lambda t: t.bbox[1])
    table_items = [TableItem(t.bbox, t.extract()) for t in tables]

    raw_blocks = page.get_text("blocks")
    paragraph_items: list[ParagraphItem] = []
    for b in raw_blocks:
        x0, y0, x1, y1, text, _block_no, block_type = b[:7]
        if block_type != 0:
            continue
        if _in_header_zone(y1, height) or _in_footer_zone(y0, height):
            continue
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        if any(_bbox_contains_center(t.bbox, center) for t in table_items):
            continue
        if text.strip():
            paragraph_items.append(ParagraphItem((x0, y0, x1, y1), text))

    items: list[PageItem] = [*table_items, *paragraph_items]
    items.sort(key=lambda it: it.bbox[1])
    return items


def build_document_parts(
    pages: list[list[PageItem]],
    printed_page: Callable[[int], int | None],
) -> tuple[list[Block], list[Requirement], set[str]]:
    """페이지별 항목 시퀀스에서 blocks/requirements/summary_ids를 만든다.
    pymupdf와 무관한 순수 함수 — 실물 matrix(fixture)만으로 단위 테스트 가능."""
    blocks: list[Block] = []
    requirements: list[Requirement] = []
    summary_ids: set[str] = set()

    source_order = 0
    current: RequirementBuilder | None = None
    current_code: str | None = None
    current_source_order: int | None = None
    current_start_page: int | None = None
    current_last_page: int | None = None
    current_col_count: int | None = None
    paragraph_seen_since_attach = False

    def finalize_current() -> None:
        nonlocal current, current_code, current_start_page, current_last_page
        nonlocal current_col_count, current_source_order, paragraph_seen_since_attach
        if current is not None and current_code is not None:
            requirements.append(
                Requirement(
                    requirement_id=current_code,
                    prefix=current_code[:3],
                    fields=current.fields,
                    raw_fields=current.raw_fields,
                    text=current.build_text(),
                    source_order=current_source_order or 0,
                    pdf_page_start=current_start_page,
                    pdf_page_end=current_last_page,
                    printed_page_start=(
                        printed_page(current_start_page) if current_start_page is not None else None
                    ),
                    printed_page_end=(
                        printed_page(current_last_page) if current_last_page is not None else None
                    ),
                )
            )
        current = None
        current_code = None
        current_start_page = None
        current_last_page = None
        current_col_count = None
        current_source_order = None
        paragraph_seen_since_attach = False

    for pno, items in enumerate(pages):
        for local_idx, item in enumerate(items):
            source_order += 1
            if isinstance(item, TableItem):
                matrix = item.matrix
                cols = len(matrix[0]) if matrix else 0
                role, code = table_role(matrix)

                if role == "definition":
                    finalize_current()
                    current = RequirementBuilder()
                    for row in matrix:
                        current.add_row(row)
                    current_code = code
                    current_source_order = source_order
                    current_start_page = pno
                    current_last_page = pno
                    current_col_count = cols
                    paragraph_seen_since_attach = False
                elif role == "summary":
                    summary_ids |= collect_loose_codes(matrix)
                    blocks.append(
                        Block(
                            block_id=f"b{source_order:04d}",
                            type="table",
                            text="\n".join(" | ".join(c or "" for c in row) for row in matrix),
                            table=matrix,
                            source_order=source_order,
                            pdf_page=pno,
                        )
                    )
                else:  # "other" — 이어짐 후보이거나 무관한 표
                    attach = (
                        current is not None
                        and local_idx == 0
                        and not paragraph_seen_since_attach
                        and cols == current_col_count
                        and current_last_page is not None
                        and pno <= current_last_page + 1
                    )
                    if attach and current is not None:
                        for row in matrix:
                            current.add_row(row)
                        current_last_page = pno
                        paragraph_seen_since_attach = False
                    else:
                        finalize_current()
                        blocks.append(
                            Block(
                                block_id=f"b{source_order:04d}",
                                type="table",
                                text="\n".join(" | ".join(c or "" for c in row) for row in matrix),
                                table=matrix,
                                source_order=source_order,
                                pdf_page=pno,
                            )
                        )
            else:  # paragraph
                if current is not None:
                    paragraph_seen_since_attach = True
                blocks.append(
                    Block(
                        block_id=f"b{source_order:04d}",
                        type="paragraph",
                        text=item.text,
                        source_order=source_order,
                        pdf_page=pno,
                    )
                )

    finalize_current()
    return blocks, requirements, summary_ids


def parse_pdf(path: str | Path, *, doc_id: str, bid_title: str) -> Document:
    doc = pymupdf.open(path)  # type: ignore[no-untyped-call]

    pages = [_page_items(doc[pno]) for pno in range(doc.page_count)]

    printed_page_cache: dict[int, int | None] = {}

    def printed_page(pno: int) -> int | None:
        if pno not in printed_page_cache:
            printed_page_cache[pno] = extract_printed_page_number(doc[pno])
        return printed_page_cache[pno]

    blocks, requirements, summary_ids = build_document_parts(pages, printed_page)

    declared_total: int | None = None
    total_warnings: list[str] = []
    if requirements:
        # 요구사항이 하나도 없으면 declared_total은 찾지 않는다 — 검증 목적 자체가
        # "요구사항 개수 vs 문서 자체 합계"인데, 문서에 요구사항 코드가 아예 없으면
        # 다른 표(예: 평가 배점표의 "합계 100")를 잘못 대응시킬 위험이 있다(실측).
        all_tables = [item.matrix for page in pages for item in page if isinstance(item, TableItem)]
        declared_total, total_warnings = find_declared_total(all_tables)

    requirement_ids = [r.requirement_id for r in requirements]
    validation_warnings = total_warnings + validate_requirement_ids(
        requirement_ids, summary_ids, declared_total
    )

    doc.close()  # type: ignore[no-untyped-call]

    return Document(
        doc_id=doc_id,
        source_file=str(path),
        bid_title=bid_title,
        format="pdf",
        parse_status="parsed",
        has_requirements=len(requirements) > 0,
        requirement_count=len(requirements),
        declared_total=declared_total,
        summary_ids=sorted(summary_ids),
        validation_warnings=validation_warnings,
        blocks=blocks,
        requirements=requirements,
    )
