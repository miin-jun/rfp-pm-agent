from pathlib import Path

from rfp_pm_agent.ingest.parsers.common import (
    RequirementBuilder,
    find_declared_total,
    is_code,
    parse_row,
    table_role,
)
from rfp_pm_agent.ingest.parsers.hwp_parser import _parse_html

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def test_typo_label_still_extracts_id() -> None:
    """실물(연구행정 AI 플랫폼 HWP를 hwp5html로 변환한 HTML)에서 SER-004 표를
    그대로 잘라낸 fixture. 라벨이 "요구사항 고유번호"가 아니라 "요구사항
    교유번호"(오타)다. 코드는 라벨 문구가 아니라 셀 자체로 판정하므로 문제없이
    뽑혀야 한다."""
    raw = (FIXTURE_DIR / "hwp_html_ser004_table.html").read_text(encoding="utf-8")
    assert "요구사항 교유번호" in raw  # fixture에 실제 오타가 있는지 확인

    _blocks, requirements, _summary_ids, _all_tables = _parse_html(raw)

    assert len(requirements) == 1
    assert requirements[0].requirement_id == "SER-004"
    assert requirements[0].fields["id"] == "SER-004"


def test_sir_prefix_recognized_without_hardcoded_list() -> None:
    """실물(천안시 PDF, 53~55쪽)에서 pymupdf가 뽑은 SIR-001 정의표 matrix.
    접두어를 하드코딩하지 않았으므로 SIR도 코드로 인식돼야 한다."""
    import json

    fixtures = json.loads((FIXTURE_DIR / "pdf_definition_tables.json").read_text(encoding="utf-8"))
    matrix = fixtures["sir_001_single_page"]

    role, code = table_role(matrix)
    assert role == "definition"
    assert code == "SIR-001"

    builder = RequirementBuilder()
    for row in matrix:
        builder.add_row(row)
    assert builder.fields["id"] == "SIR-001"


def test_standard_citation_not_recognized_as_code() -> None:
    """실물(연구행정 AI 플랫폼 HWP)에 있는 표준 인용구
    "KICS.KO-10.0307"(모바일 웹 콘텐츠 저작 지침 번호) — 코드로 오탐되면 안 됨."""
    assert not is_code("(KICS.KO-10.0307)")
    assert not is_code("KICS.KO-10.0307")


def test_material_spec_not_recognized_as_code() -> None:
    """실물(천안시 PDF, ECR-007 세부내용)에 있는 재질 규격 "SUS-316L" — 코드
    패턴과 우연히 겹치지만 코드로 오탐되면 안 됨."""
    assert not is_code("SUS-316L")
    assert not is_code("함체 : SUS-316L 분체도장, 잠금장치, 방수밴드, 먼지필터")


def test_parse_row_three_cells_uses_last_two_as_label_value() -> None:
    """실측(HWP 220행·PDF 443행 전수 확인, docs/parsing-exploration.md)으로
    non-null 3개 행은 늘 (그룹라벨, 라벨, 값) 형태였다."""
    assert parse_row(["요구사항 상세설명", "정의", "값"]) == ("정의", "값")
    assert parse_row([None, "세부\n내용", "값"]) == ("세부\n내용", "값")


def test_find_declared_total_ignores_budget_currency_strings() -> None:
    """ "합계" 행이라도 값이 "금 41,914,000원" 같은 서식 문자열이면 순수 정수가
    아니므로 declared_total 후보에서 제외돼야 한다(예산 합계 오탐 방지)."""
    budget_table = [["합계", "금 41,914,000원 (부가가치세 별도)"]]
    value, warnings = find_declared_total(budget_table)  # type: ignore[arg-type]
    assert value is None
    assert warnings == []


def test_find_declared_total_two_candidates_yields_null_and_warning() -> None:
    """로직 검증용 합성 입력 — "합계" 행이 둘 이상이면 어느 쪽이 맞는지 애매하므로
    null + 경고를 남겨야 한다."""
    tables: list[list[list[str | None]]] = [[["합계", "10"]], [["합계", "20"]]]
    value, warnings = find_declared_total(tables)
    assert value is None
    assert len(warnings) == 1
