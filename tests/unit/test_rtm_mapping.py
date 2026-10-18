"""`rtm.mapping.build_rtm` (이슈 #39, ADR-0012) — 기준값 경계, 정렬, 빈 입력, 누락·범위 외 검출."""

from __future__ import annotations

from rfp_pm_agent.rtm.mapping import build_rtm


def _scores(table: dict[str, dict[str, float]]) -> dict[tuple[str, str], float]:
    return {(rid, wid): s for rid, row in table.items() for wid, s in row.items()}


def test_score_exactly_at_threshold_is_linked() -> None:
    result = build_rtm(["R1"], ["W1"], {("R1", "W1"): 0.5})

    assert result.links == {"R1": [("W1", 0.5)]}
    assert result.missing == []


def test_score_just_below_threshold_is_not_linked() -> None:
    result = build_rtm(["R1"], ["W1"], {("R1", "W1"): 0.4999})

    assert result.links == {"R1": []}
    assert result.missing == ["R1"]
    assert result.out_of_scope == ["W1"]


def test_default_threshold_is_half() -> None:
    scores = {("R1", "W1"): 0.5, ("R1", "W2"): 0.4999}

    assert build_rtm(["R1"], ["W1", "W2"], scores).links == {"R1": [("W1", 0.5)]}


def test_links_are_sorted_by_score_descending() -> None:
    scores = _scores({"R1": {"W1": 0.6, "W2": 0.9, "W3": 0.7, "W4": 0.1}})

    result = build_rtm(["R1"], ["W1", "W2", "W3", "W4"], scores)

    assert result.links["R1"] == [("W2", 0.9), ("W3", 0.7), ("W1", 0.6)]


def test_empty_inputs() -> None:
    result = build_rtm([], [], {})

    assert result.links == {}
    assert result.missing == []
    assert result.out_of_scope == []


def test_requirements_without_tasks_are_all_missing() -> None:
    result = build_rtm(["R1", "R2"], [], {})

    assert result.links == {"R1": [], "R2": []}
    assert result.missing == ["R1", "R2"]


def test_tasks_without_requirements_are_all_out_of_scope() -> None:
    result = build_rtm([], ["W1", "W2"], {})

    assert result.out_of_scope == ["W1", "W2"]


def test_planted_unrelated_task_is_out_of_scope() -> None:
    """실데이터(합성 PMS v2)에는 범위 외 업무가 0건이라 가짜 데이터에 무관한 업무를 심어 확인한다."""
    scores = _scores(
        {
            "SFR-001": {"2.1": 0.92, "2.2": 0.31, "9.9": 0.03},
            "SFR-002": {"2.1": 0.12, "2.2": 0.81, "9.9": 0.07},
            "SER-001": {"2.1": 0.55, "2.2": 0.48, "9.9": 0.11},
        }
    )

    result = build_rtm(["SFR-001", "SFR-002", "SER-001"], ["2.1", "2.2", "9.9"], scores)

    assert result.out_of_scope == ["9.9"]
    assert result.missing == []


def test_missing_and_out_of_scope_keep_input_order() -> None:
    scores = _scores({"R2": {"W2": 0.1, "W1": 0.2}, "R1": {"W2": 0.3, "W1": 0.4}})

    result = build_rtm(["R2", "R1"], ["W2", "W1"], scores)

    assert result.missing == ["R2", "R1"]
    assert result.out_of_scope == ["W2", "W1"]
