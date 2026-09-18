from rfp_pm_agent.ingest.parsers.common import (
    validate_required_field_values,
    validate_requirement_ids,
)


def test_summary_and_definition_mismatch_is_warned() -> None:
    """로직 검증용 합성 입력 — 요약표 코드와 정의표 코드가 다르면 경고를
    남겨야 한다(예외를 던지지 않는다)."""
    warnings = validate_requirement_ids(
        requirement_ids=["SFR-001", "SFR-002"],
        summary_ids={"SFR-001", "SFR-002", "SFR-003"},
        declared_total=None,
    )
    assert len(warnings) == 1
    assert "SFR-003" in warnings[0]


def test_matching_ids_produce_no_warning() -> None:
    """로직 검증용 합성 입력 — 정상 케이스에서는 경고가 없어야 한다."""
    warnings = validate_requirement_ids(
        requirement_ids=["SFR-001", "SFR-002"],
        summary_ids={"SFR-001", "SFR-002"},
        declared_total=2,
    )
    assert warnings == []


def test_duplicate_requirement_id_is_warned() -> None:
    """로직 검증용 합성 입력 — 같은 코드로 정의표가 2개 만들어지면(안전장치)
    경고에 기록돼야 한다."""
    warnings = validate_requirement_ids(
        requirement_ids=["SFR-001", "SFR-001"],
        summary_ids={"SFR-001"},
        declared_total=None,
    )
    assert any("중복" in w and "SFR-001" in w for w in warnings)


def test_declared_total_mismatch_is_warned() -> None:
    """로직 검증용 합성 입력 — 문서 자체 합계와 파싱 개수가 다르면 경고를
    남겨야 한다."""
    warnings = validate_requirement_ids(
        requirement_ids=["SFR-001"],
        summary_ids={"SFR-001"},
        declared_total=5,
    )
    assert any("declared_total" in w for w in warnings)


def test_empty_required_field_value_is_warned() -> None:
    """로직 검증용 합성 입력 — 실물(천안시 PDF) QUR-003·PSR-003에서 detail 등이
    키는 있지만 값이 비어 있던 실패를 재현한다. 키 존재만으로는 못 잡는다."""
    warnings = validate_required_field_values(
        [("QUR-003", {"category": "품질 요구사항", "id": "QUR-003", "detail": ""})],
        required_fields={"category", "id", "detail"},
    )
    assert warnings == ["QUR-003: detail 필드 값이 비어 있음"]


def test_whitespace_only_value_is_also_warned() -> None:
    """로직 검증용 합성 입력 — 공백만 있는 값도 빈 값으로 본다."""
    warnings = validate_required_field_values(
        [("SFR-001", {"detail": "   \n  "})],
        required_fields={"detail"},
    )
    assert warnings == ["SFR-001: detail 필드 값이 비어 있음"]


def test_filled_required_fields_produce_no_warning() -> None:
    """로직 검증용 합성 입력 — 정상 케이스에서는 경고가 없어야 한다."""
    warnings = validate_required_field_values(
        [("SFR-001", {"category": "기능 요구사항", "id": "SFR-001", "detail": "내용"})],
        required_fields={"category", "id", "detail"},
    )
    assert warnings == []


def test_related_and_output_excluded_by_default_field_set() -> None:
    """실물(천안시 PDF) 근거: related는 71개 전부, output은 24/71이 원본
    자체가 빈 문자열이었다 — 파서가 REQUIRED_FIELD_VALUES에서 둘 다 뺀
    이유를 코드로도 확인한다(호출부 계약 — 상수 자체가 관용구)."""
    from rfp_pm_agent.ingest.parsers.common import REQUIRED_FIELD_VALUES

    assert "related" not in REQUIRED_FIELD_VALUES
    assert "output" not in REQUIRED_FIELD_VALUES
    assert REQUIRED_FIELD_VALUES == {"category", "id", "name", "definition", "detail"}
