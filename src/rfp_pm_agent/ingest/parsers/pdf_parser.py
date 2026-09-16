"""PDF → Document 파서 (PyMuPDF). 이슈 #11.

docs/parsing-exploration.md 실측 근거:
- pymupdf `find_tables()`가 요구사항 정의표를 표 구조로 정상 인식한다(텍스트만
  나오지 않는다) — 그래서 pdfplumber가 아니라 pymupdf를 최종 채택했다
  (docs/tech-stack.md 8절).
- 페이지 하단(높이의 85% 아래)에 `- N -` 형식의 인쇄 쪽번호가 있다. 고정
  오프셋으로 계산하지 않고 실제 위치·텍스트를 읽는다. 못 찾으면 null.
- 세부내용이 길면 정의표 하나가 여러 페이지의 별개 표 객체로 쪼개진다. 이어붙임은
  "코드 없는 표가 (a) 그 페이지의 첫 콘텐츠 블록이고 (b) 직전 조각 이후 문단
  블록이 끼어들지 않았고 (c) 열 수가 직전 조각과 같거나 더 적을 때"만 한다.
  (c)가 "이하"인 이유: **처음엔 "산출정보·관련요구사항 값이 둘 다 비어 있어서
  pymupdf가 그 열을 못 잡는다"고 봤는데, 사용자가 PDF 원본과 직접 대조해 이 설명이
  틀렸음을 확인했다** — ECR-005 등 5건은 값이 실제로 있었는데(예: "납품확인서,
  설치계획서, 설치결과서, 기술지원확약서") `find_tables()`가 그 값 열 전체를
  bbox 밖으로 놓쳐(라벨 셀만 포함하는 아주 좁은 bbox를 반환) 표 밖 자유
  텍스트(문단)로 떨어뜨린 것이었다 — "값이 비어 있다"가 아니라 "표가 값 열
  자체를 못 찾아 내용이 표 밖으로 유실된다"였다. 그래서 열 수 완화만으로는
  부족하고, `absorb_trailing_paragraphs`로 표 밖에 떨어진 문단을 되찾아야
  했다(아래 참고). 열이 늘어나는 쪽은 구조가 다른 무관한 표라는 신호로 보고
  그대로 거부한다. 머리글·꼬리말 영역(페이지 상단 8%·하단 15%)은 이 판정에서
  제외한다 — 인쇄 쪽번호 추출과 같은 영역 상수를 쓴다.
- 열 수가 문서의 정상 정의표 열 수(최빈값)보다 적은 표를 만나면, 그 직후 같은
  페이지에서 이어지는 문단들을 `absorb_trailing_paragraphs`로 되찾는다 — 위
  bbox 유실 문제의 실제 복구 장치. 문단의 첫 줄이 알려진 라벨이면 그 필드를
  채우고, 아니면 직전에 열려 있던 라벨(보통 세부내용)에 이어붙인다. 단, 그
  직전 라벨이 "세부내용"이 아니면(정의표 시작 자체가 이미 조각나 있던 극단
  사례 QUR-003에서 실제로 "요구사항 고유번호"가 되어 버린 적이 있다) 이어붙이지
  않고 버린다 — 짧은 구조적 필드(고유번호 등)가 자유 텍스트로 오염되는 것보다
  그 필드를 비워 두는 게 낫다(`common.py` `RequirementBuilder.add_paragraph`).

`_build_document_parts`는 pymupdf 객체와 분리된 순수 함수다 — 실제 PDF 없이도
실물에서 추출한 표 matrix(fixture)만으로 이어붙임·종료 로직을 단위 테스트할 수
있게 하기 위함이다.
"""

from __future__ import annotations

import re
from collections import Counter
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


def _modal_definition_col_count(pages: list[list[PageItem]]) -> int:
    """문서 전체에서 "제대로 잡힌" 정의표의 열 수(최빈값)를 구한다. 이 문서에서
    정의표는 거의 다 3열이지만, 문서마다 다를 수 있어 하드코딩하지 않고 실제
    데이터에서 뽑는다 — 아래 열 수 부족분 복구 로직의 기준값으로 쓴다."""
    counts = [
        len(item.matrix[0]) if item.matrix else 0
        for page in pages
        for item in page
        if isinstance(item, TableItem) and table_role(item.matrix)[0] == "definition"
    ]
    if not counts:
        return 0
    return Counter(counts).most_common(1)[0][0]


def build_document_parts(
    pages: list[list[PageItem]],
    printed_page: Callable[[int], int | None],
) -> tuple[list[Block], list[Requirement], set[str]]:
    """페이지별 항목 시퀀스에서 blocks/requirements/summary_ids를 만든다.
    pymupdf와 무관한 순수 함수 — 실물 matrix(fixture)만으로 단위 테스트 가능."""
    blocks: list[Block] = []
    requirements: list[Requirement] = []
    summary_ids: set[str] = set()
    modal_col_count = _modal_definition_col_count(pages)

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

    def absorb_trailing_paragraphs(
        items: list[PageItem], start_idx: int, fallback_label: str | None
    ) -> int:
        """실측(ECR-005 외 5건): pymupdf가 산출정보·관련요구사항 값 열의
        존재 자체를 못 잡으면(그 열에 렌더링된 텍스트가 하나도 없어서), 표는
        라벨만 남긴 채 열 수가 줄어들고 실제 값은 표 밖 자유 텍스트(문단)로
        떨어진다 — "값이 비어 있다"가 아니라 "값이 표 밖으로 유실됐다"였다.
        열 수가 문서의 정상 정의표 열 수(modal_col_count)보다 적은 표를
        처리한 직후에는, 같은 페이지에서 바로 이어지는 문단들을 이 요구사항에
        되돌려 붙인다. 각 문단의 첫 줄이 알려진 라벨이면 그 필드를 채우고
        (표 자체가 비워 둔 값을 덮어씀), 아니면 fallback_label(이 표를 만나기
        직전까지 열려 있던 라벨 — 보통 세부내용)에 이어붙인다."""
        nonlocal source_order
        j = start_idx
        while j < len(items):
            next_item = items[j]
            if not isinstance(next_item, ParagraphItem):
                break
            assert current is not None
            current.add_paragraph(next_item.text, fallback_label=fallback_label)
            source_order += 1
            j += 1
        return j

    for pno, items in enumerate(pages):
        idx = 0
        while idx < len(items):
            item = items[idx]
            local_idx = idx
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
                    if cols < modal_col_count:
                        idx = absorb_trailing_paragraphs(items, idx + 1, current.current_label)
                        continue
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
                    # cols<=current_col_count인 이유: pymupdf가 열 경계를 셀
                    # 텍스트의 실제 위치로 추론하는데, 산출정보·관련요구사항 값이
                    # 둘 다 빈 문자열이면 그 열엔 렌더링된 텍스트가 하나도 없어
                    # pymupdf가 열 자체의 존재를 못 잡는다 — 그래서 이어지는
                    # 조각만 3열→2열로 줄어든다(실측: ECR-005/ECR-007/SER-006/
                    # SFR-003/SFR-005). 열이 늘어나는 쪽은 반대로 "이 조각에 값이
                    # 있는 열이 원본보다 많다"는 뜻이라 구조가 다른 무관한 표라는
                    # 신호로 보고 그대로 거부한다.
                    #
                    # 이 완화만으로는 흡수 위험이 커지지 않는다 — local_idx==0
                    # 조건이 같은 페이지의 두 번째 이후 블록을 걸러 준다. 실측
                    # 사례: 37쪽(0-based 36)은 `SFR-006` 정의표(3열, 이미 완결)
                    # 바로 아래 같은 페이지에 무관한 2열 표(`세부 항목|개발 내용`)
                    # 가 있는데, 열 수 조건은 통과해도(2<=3) 그 표가 페이지의
                    # "두 번째" 블록이라 흡수되지 않는다
                    # (tests/unit/test_parsers_pdf.py
                    #  test_unrelated_smaller_table_on_same_page_is_not_absorbed).
                    attach = (
                        current is not None
                        and local_idx == 0
                        and not paragraph_seen_since_attach
                        and current_col_count is not None
                        and cols <= current_col_count
                        and current_last_page is not None
                        and pno <= current_last_page + 1
                    )
                    if attach and current is not None:
                        pre_label = current.current_label
                        for row in matrix:
                            current.add_row(row)
                        current_last_page = pno
                        paragraph_seen_since_attach = False
                        if cols < modal_col_count:
                            idx = absorb_trailing_paragraphs(items, idx + 1, pre_label)
                            continue
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
            idx += 1

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
