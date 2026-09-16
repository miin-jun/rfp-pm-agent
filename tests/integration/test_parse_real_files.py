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
    assert doc.validation_warnings == []


def test_pdf_multi_page_requirement_count(pdf_target: tuple[dict[str, str], Path]) -> None:
    """docs/parsing-exploration.md는 "68개 중 15개(약 22%)가 페이지 경계로
    조각남"이라고 적었지만, 그건 행 수(<7)를 기준으로 한 근사치였다. 실제로
    페이지를 이어붙이는 로직을 구현해 측정한 정확한 값은 71개 중 11개다 —
    행 수 기준 추정치와 실측 사이의 차이는 이 테스트 자체가 기록으로 남긴다."""
    entry, path = pdf_target
    doc = parse_document(path, doc_id=entry["doc_id"], bid_title=entry["notice_title"] or "")

    multi_page = [r for r in doc.requirements if r.pdf_page_start != r.pdf_page_end]
    assert len(multi_page) == 11

    for r in multi_page:
        assert r.pdf_page_start is not None
        assert r.pdf_page_end is not None
        assert r.pdf_page_start < r.pdf_page_end  # pdf_page_*: 0-based 장 번호


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
