"""실제 수집 파일 9건 중 요구사항 코드가 있는 두 문서를 실제로 파싱한다.

이슈 #11 사전 조사(docs/parsing-exploration.md)에서 확인한 실제 문서 두 건을
그대로 쓴다 — HWP는 요구사항 코드 37개(연구행정 데이터 기반 AI 플랫폼 구축),
PDF는 71개(2024년 천안시 거점형 스마트도시 조성사업)로 문서 자체가 "합계 71"을
밝히고 있다. `data/raw/api/`에 원본 파일이 없는 환경(CI 등)에서는 skip한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfp_pm_agent.ingest.parse import parse_document

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "manifest.jsonl"
RAW_API_DIR = REPO_ROOT / "data" / "raw" / "api"


def _find_entry(keywords: list[str]) -> dict[str, str] | None:
    if not MANIFEST_PATH.exists():
        return None
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry: dict[str, str] = json.loads(line)
        title = entry.get("notice_title") or ""
        if all(k in title for k in keywords):
            return entry
    return None


def _resolve_source_file(doc_id: str) -> Path | None:
    if not RAW_API_DIR.exists():
        return None
    for p in RAW_API_DIR.iterdir():
        if p.name.startswith(doc_id):
            return p
    return None


def _skip_reason(keywords: list[str]) -> tuple[dict[str, str] | None, Path | None]:
    entry = _find_entry(keywords)
    if entry is None:
        return None, None
    path = _resolve_source_file(entry["doc_id"])
    return entry, path


@pytest.fixture()
def hwp_target() -> tuple[dict[str, str], Path]:
    entry, path = _skip_reason(["연구행정", "AI", "플랫폼"])
    if entry is None or path is None:
        pytest.skip("원본 HWP 샘플(연구행정 데이터 기반 AI 플랫폼 구축)이 이 환경에 없음")
    return entry, path


@pytest.fixture()
def pdf_target() -> tuple[dict[str, str], Path]:
    entry, path = _skip_reason(["천안시", "스마트"])
    if entry is None or path is None:
        pytest.skip("원본 PDF 샘플(2024년 천안시 거점형 스마트도시 조성사업)이 이 환경에 없음")
    return entry, path


def test_hwp_extracts_37_requirements_and_no_declared_total(
    hwp_target: tuple[dict[str, str], Path],
) -> None:
    entry, path = hwp_target
    doc = parse_document(path, doc_id=entry["doc_id"], bid_title=entry["notice_title"] or "")

    assert doc.parse_status == "parsed"
    assert doc.has_requirements is True
    assert doc.requirement_count == 37
    # 실측: HWP 요약표의 "합계" 셀 값이 비어 있음 — 문서 자체가 총량을 밝히지 않는다.
    assert doc.declared_total is None
    assert doc.validation_warnings == []


def test_pdf_extracts_71_requirements_matching_declared_total(
    pdf_target: tuple[dict[str, str], Path],
) -> None:
    entry, path = pdf_target
    doc = parse_document(path, doc_id=entry["doc_id"], bid_title=entry["notice_title"] or "")

    assert doc.parse_status == "parsed"
    assert doc.has_requirements is True
    assert doc.requirement_count == 71
    # 실측: 문서 22쪽 분류 요약표의 "합 계" 셀이 71을 명시한다.
    assert doc.declared_total == 71

    requirement_ids = [r.requirement_id for r in doc.requirements]
    assert len(requirement_ids) == len(set(requirement_ids))  # 중복 0건
    assert set(requirement_ids) == set(doc.summary_ids)  # 요약표·정의표 코드 집합 일치

    # 값 기반 필드 검사(validate_required_field_values)의 실측 결과 — 그대로
    # 못 박는다. QUR-003의 detail은 알려진 한계(PR #46 "알려진 한계" 참고)이고,
    # PSR-003은 문서 자체가 전부 공란인 템플릿 행이다(원본 대조로 확인, 파서
    # 결함 아님). 이 목록이 늘어나면 새로운 유실이거나 새로운 원본 공란이다 —
    # 둘 다 원본과 대조 없이 이 값에 맞춰 늘리지 않는다.
    assert doc.validation_warnings == [
        "QUR-003: detail 필드 값이 비어 있음",
        "PSR-003: category 필드 값이 비어 있음",
        "PSR-003: definition 필드 값이 비어 있음",
        "PSR-003: detail 필드 값이 비어 있음",
        "PSR-003: name 필드 값이 비어 있음",
    ]


def test_pdf_multi_page_requirement_count(pdf_target: tuple[dict[str, str], Path]) -> None:
    """docs/parsing-exploration.md는 "68개 중 15개(약 22%)가 페이지 경계로
    조각남"이라고 적었는데, 그건 행 수(<7)를 기준으로 한 근사치였다. 실제로
    페이지를 이어붙이는 로직을 구현해 측정해 보니(사용자가 독립적으로
    합계·누락 필드·고아 블록을 대조해 검증) 71개 중 15개로 — 우연히 같은
    숫자지만 기준 코드 집합이 다르다(SIR 포함 71 vs 제외 68).

    처음 구현했을 때는 11개로 나왔다. 그 5건(ECR-005, ECR-007, SER-006,
    SFR-003, SFR-005)을 "이어지는 조각의 산출정보·관련요구사항 값이 둘 다
    비어 있다"고 설명했는데, **이 설명은 원본과 대조해 보니 틀렸다** — 사용자가
    PDF 원본을 직접 확인해, ECR-005의 46쪽(1-based) 원문에 "출력전압 : 27V"부터
    "17) 한전불입금..."까지의 세부내용 이어짐과 산출정보 값("납품확인서,
    설치계획서, 설치결과서, 기술지원확약서")이 실제로 존재함을 확인했다. 진짜
    원인은 pymupdf가 그 페이지에서 표를 찾긴 하지만(`find_tables()`가 라벨
    셀만 포함하는 아주 좁은 bbox를 반환), 값이 있는 열 자체를 표 bbox 밖으로
    완전히 놓쳐 표 밖 자유 텍스트(문단)로 떨어뜨린 것이었다 — "값이 비어
    있다"가 아니라 "표가 값 열을 못 찾아 내용이 표 밖으로 유실된다"였다.
    `pdf_parser.py`의 `absorb_trailing_paragraphs`가 이 유실분을 되찾는다.

    QUR-003도 같은 원인으로 name/definition/output이 비어 있었다(6번째
    사례, 표준 필드 "존재"만 확인하는 검사로는 못 잡았다 — 값이 채워졌는지도
    확인해야 드러난다)."""
    entry, path = pdf_target
    doc = parse_document(path, doc_id=entry["doc_id"], bid_title=entry["notice_title"] or "")

    multi_page = [r for r in doc.requirements if r.pdf_page_start != r.pdf_page_end]
    assert len(multi_page) == 15

    for r in multi_page:
        assert r.pdf_page_start is not None
        assert r.pdf_page_end is not None
        assert r.pdf_page_start < r.pdf_page_end  # pdf_page_*: 0-based 장 번호

    standard_fields = {"category", "id", "name", "definition", "detail", "output", "related"}
    for r in doc.requirements:
        assert standard_fields <= set(r.fields.keys()), (
            f"{r.requirement_id}: 표준 필드 누락 {standard_fields - set(r.fields.keys())}"
        )

    by_id = {r.requirement_id: r for r in doc.requirements}

    # 값 존재까지 확인 — 키만 있고 값이 빈 채로 남는 회귀를 잡기 위함
    # (실측: ECR-005 등은 키는 있었지만 값이 원본과 다르게 비어 있었다).
    assert by_id["ECR-005"].fields["output"] == "납품확인서, 설치계획서, 설치결과서, 기술지원확약서"
    assert "17) 한전불입금" in by_id["ECR-005"].text
    assert "• 2010년형 경찰청 규격 사용" in by_id["ECR-007"].text
    assert "능한 Tool을 사용하여 정적코드 분석 수행" in by_id["SER-006"].text
    assert "9) 사이니지" in by_id["SFR-003"].text


def test_pdf_printed_page_numbers_differ_from_pdf_page_when_cover_has_no_number(
    pdf_target: tuple[dict[str, str], Path],
) -> None:
    """실측: 이 문서는 1쪽(0-based 0)이 표지라 인쇄 쪽번호가 없고, 그 뒤로는
    "인쇄 쪽번호 == pdf_page(0-based)"가 우연히 성립한다(1-based 장 번호에서
    표지 1장을 뺀 값과 같기 때문). 두 값을 같은 뜻으로 혼동하기 쉬우므로 각각
    따로 assert하고 기준을 주석으로 남긴다."""
    entry, path = pdf_target
    doc = parse_document(path, doc_id=entry["doc_id"], bid_title=entry["notice_title"] or "")

    sfr_002 = next(r for r in doc.requirements if r.requirement_id == "SFR-002")
    # pdf_page_start/end: 0-based 장 번호 (fitz/pymupdf 인덱스 그대로)
    assert sfr_002.pdf_page_start == 25
    assert sfr_002.pdf_page_end == 29
    # printed_page_start/end: 페이지 하단에 실제 인쇄된 "- N -" 쪽번호
    # (이 문서에서는 우연히 pdf_page와 같은 값이지만, 별도로 읽은 값이다)
    assert sfr_002.printed_page_start == 25
    assert sfr_002.printed_page_end == 29
