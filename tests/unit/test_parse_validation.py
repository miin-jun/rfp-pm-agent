from rfp_pm_agent.ingest.parsers.common import validate_requirement_ids


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
