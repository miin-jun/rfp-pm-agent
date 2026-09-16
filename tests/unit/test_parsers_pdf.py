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


def test_more_columns_than_current_breaks_the_chain() -> None:
    """로직 검증용 합성 입력 — 열이 더 많은 코드 없는 표는 구조가 다른 무관한
    표라는 신호이므로 이어붙이면 안 된다(열이 더 적은 쪽은 실측으로 확인된
    "값 열 자체를 pymupdf가 bbox 밖으로 놓쳐 표 밖 문단으로 떨어진" 정상
    이어짐 케이스라 허용한다 — 아래
    test_fewer_columns_due_to_empty_trailing_values_is_still_joined 참고)."""
    fixtures = _load_fixtures()
    fragments = fixtures["sfr_002_page_fragments"]
    more_cols_table: list[list[str | None]] = [["a", "b", "c", "d"]]  # 4열 (원래 3열)

    pages: list[list[PageItem]] = [
        [TableItem(BBOX, fragments[0])],
        [TableItem(BBOX, more_cols_table)],
    ]

    blocks, requirements, _summary_ids = build_document_parts(pages, _no_printed_page)

    assert len(requirements) == 1
    assert requirements[0].pdf_page_end == 0
    assert any(b.type == "table" and b.pdf_page == 1 for b in blocks)


def _build_page_items(raw_items: list[dict[str, Any]]) -> list[PageItem]:
    result: list[PageItem] = []
    for it in raw_items:
        if it["kind"] == "table":
            result.append(TableItem(BBOX, it["matrix"]))
        else:
            result.append(ParagraphItem(BBOX, it["text"]))
    return result


def test_fewer_columns_due_to_empty_trailing_values_is_still_joined() -> None:
    """실물(천안시 PDF) ECR-005 — 45쪽(0-based 44)에서 시작해 46쪽(0-based 45)
    으로 이어진다. 처음엔 "산출정보·관련요구사항 값이 둘 다 비어 있어서
    pymupdf가 그 열 자체를 못 잡는다"고 설명했는데, 사용자가 PDF 원본과
    직접 대조해 실제로는 값이 있는데(산출정보="납품확인서, 설치계획서,
    설치결과서, 기술지원확약서") pymupdf가 그 값 열의 존재 자체를 못 잡아
    표 밖 자유 텍스트(문단)로 떨어뜨린 것임을 확인했다 — "값이 비어 있다"가
    아니라 "값이 표 밖으로 유실됐다"였다. fixture는 표 matrix만이 아니라
    46쪽의 실제 문단(자유 텍스트)까지 실물 그대로 포함한다(pdf_parser.py의
    `_page_items`로 직접 뽑음)."""
    fixtures = _load_fixtures()
    raw_pages = fixtures["ecr_005_page_items"]  # 45,46쪽(0-based 44,45)

    pages: list[list[PageItem]] = [_build_page_items(p) for p in raw_pages]

    blocks, requirements, _summary_ids = build_document_parts(pages, _no_printed_page)

    assert len(requirements) == 1
    req = requirements[0]
    assert req.requirement_id == "ECR-005"
    assert req.pdf_page_end == 1
    assert req.fields["output"] == "납품확인서, 설치계획서, 설치결과서, 기술지원확약서"
    assert "17) 한전불입금" in req.text
    assert blocks == []


def test_unrelated_smaller_table_on_same_page_is_not_absorbed() -> None:
    """실물(천안시 PDF, 37쪽/0-based 36) — `SFR-006` 정의표(3열, 7행 전부
    완결) 바로 아래 같은 페이지에 완전히 무관한 표(`['세부 항목',
    '개발 내용']`, 2열)가 있다. 열 수 조건을 "같거나 더 적을 때"로 완화한
    뒤에도, 이 표는 **같은 페이지의 두 번째 블록**이라 "그 페이지의 첫
    콘텐츠 블록" 조건에서 걸러진다 — 열 수 완화가 이 조건까지 무력화하지
    않는다는 걸 확인하는 회귀 테스트."""
    fixtures = _load_fixtures()
    same_page_tables = fixtures["sfr_006_and_unrelated_orphan_same_page"]

    pages: list[list[PageItem]] = [[TableItem(BBOX, m) for m in same_page_tables]]

    blocks, requirements, _summary_ids = build_document_parts(pages, _no_printed_page)

    assert len(requirements) == 1
    assert requirements[0].requirement_id == "SFR-006"
    assert requirements[0].pdf_page_end == 0

    table_blocks = [b for b in blocks if b.type == "table"]
    assert len(table_blocks) == 1
    assert table_blocks[0].table == [["세부 항목", "개발 내용"]]
