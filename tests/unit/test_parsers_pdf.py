import json
from pathlib import Path
from typing import Any

from rfp_pm_agent.ingest.parsers.pdf_parser import (
    PageItem,
    ParagraphItem,
    TableItem,
    build_document_parts,
)

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "pdf_definition_tables.json"
BBOX = (0.0, 100.0, 500.0, 700.0)


def _load_fixtures() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return result


def _no_printed_page(_pno: int) -> int | None:
    return None


def test_sfr_002_page_fragments_join_into_one_requirement() -> None:
    """실물(천안시 PDF) SFR-002의 26~30쪽 표 조각 5개(fixture, 이번 세션에
    pymupdf로 그대로 뽑음)를 그대로 넣으면 하나의 Requirement로 합쳐져야 한다."""
    fixtures = _load_fixtures()
    fragments = fixtures["sfr_002_page_fragments"]  # pages 26,27,28,29,30(첫 표)

    pages: list[list[PageItem]] = [[TableItem(BBOX, m)] for m in fragments]

    blocks, requirements, _summary_ids = build_document_parts(pages, _no_printed_page)

    assert len(requirements) == 1
    req = requirements[0]
    assert req.requirement_id == "SFR-002"
    assert req.pdf_page_start == 0
    assert req.pdf_page_end == 4  # 5개 페이지(0-based 0~4)에 걸쳐 이어붙음
    assert req.fields["name"] == "스마트 폴 기능"
    assert "산출정보" in req.raw_fields
    assert blocks == []  # 전부 요구사항으로 소비됨, 남는 블록 없음


def test_unrelated_table_after_gap_is_not_absorbed() -> None:
    """실물(천안시 PDF) 마지막 요구사항(86쪽, PSR, 3열)과 완전히 무관한 표
    (90쪽, 8열 도서 목록) — 페이지가 안 이어지고 열 수도 달라 흡수되면 안 된다."""
    fixtures = _load_fixtures()
    last_req_tables = fixtures["psr_last_page86"]  # 86쪽 표 2개(둘 다 정의표 시작)
    unrelated = fixtures["unrelated_page90"]  # 90쪽 8열 표

    pages: list[list[PageItem]] = [[] for _ in range(5)]  # 0=86쪽 ... 4=90쪽(87~89쪽 표 없음)
    pages[0] = [TableItem(BBOX, m) for m in last_req_tables]
    pages[4] = [TableItem(BBOX, unrelated)]

    blocks, requirements, _summary_ids = build_document_parts(pages, _no_printed_page)

    requirement_ids = {r.requirement_id for r in requirements}
    assert "PSR-005" in requirement_ids or "PSR-006" in requirement_ids
    for r in requirements:
        assert r.pdf_page_end == 0  # 86쪽 안에서 끝남, 90쪽까지 안 늘어남

    # 90쪽 표는 일반 table 블록으로만 남아야 한다
    unrelated_blocks = [b for b in blocks if b.pdf_page == 4]
    assert len(unrelated_blocks) == 1
    assert unrelated_blocks[0].type == "table"
    table = unrelated_blocks[0].table
    assert table is not None
    assert len(table[0]) == 8


def test_paragraph_between_fragments_breaks_the_chain() -> None:
    """로직 검증용 합성 입력 — 코드 없는 표 조각이라도 직전 조각 사이에 문단
    블록이 끼면 이어붙이면 안 된다(조건 B)."""
    fixtures = _load_fixtures()
    fragments = fixtures["sfr_002_page_fragments"]

    pages: list[list[PageItem]] = [
        [TableItem(BBOX, fragments[0])],  # SFR-002 시작
        [ParagraphItem(BBOX, "무관한 문단")],
        [TableItem(BBOX, fragments[1])],  # 원래는 이어짐 조각
    ]

    blocks, requirements, _summary_ids = build_document_parts(pages, _no_printed_page)

    assert len(requirements) == 1
    assert requirements[0].pdf_page_end == 0  # 문단 때문에 1쪽에서 못 이어붙음
    table_blocks = [b for b in blocks if b.type == "table"]
    assert len(table_blocks) == 1  # 조각이 일반 블록으로 남음


def test_column_count_mismatch_breaks_the_chain() -> None:
    """로직 검증용 합성 입력 — 열 수가 다른 코드 없는 표는 이어붙이면 안 된다."""
    fixtures = _load_fixtures()
    fragments = fixtures["sfr_002_page_fragments"]
    mismatched_cols_table: list[list[str | None]] = [["a", "b"]]  # 2열 (원래 3열)

    pages: list[list[PageItem]] = [
        [TableItem(BBOX, fragments[0])],
        [TableItem(BBOX, mismatched_cols_table)],
    ]

    blocks, requirements, _summary_ids = build_document_parts(pages, _no_printed_page)

    assert len(requirements) == 1
    assert requirements[0].pdf_page_end == 0
    assert any(b.type == "table" and b.pdf_page == 1 for b in blocks)
