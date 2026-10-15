"""PMS 판단 규칙 (이슈 #20, docs/data-design.md 6절 "핵심 판단 규칙")."""

from __future__ import annotations

from datetime import date

import pytest

from rfp_pm_agent.pms.rules import (
    delay_days,
    is_delayed,
    is_delayed_this_week,
    is_open_risk,
    this_week_delay_range,
)

THURSDAY = date(2026, 9, 24)
MONDAY = date(2026, 9, 21)


def test_this_week_range_is_monday_to_day_before_as_of() -> None:
    assert this_week_delay_range(THURSDAY) == (MONDAY, date(2026, 9, 23))


def test_this_week_range_is_empty_when_as_of_is_monday() -> None:
    assert this_week_delay_range(MONDAY) is None


def test_this_week_range_on_sunday_covers_monday_to_saturday() -> None:
    assert this_week_delay_range(date(2026, 9, 27)) == (MONDAY, date(2026, 9, 26))


@pytest.mark.parametrize(
    ("status", "planned_end", "expected"),
    [
        ("in_progress", date(2026, 9, 23), True),
        ("on_hold", date(2026, 9, 1), True),
        ("not_started", date(2026, 9, 23), True),
        ("done", date(2026, 9, 1), False),
        ("in_progress", THURSDAY, False),  # 기준일 당일 마감은 아직 지연이 아니다
        ("in_progress", date(2026, 9, 30), False),
    ],
)
def test_is_delayed(status: str, planned_end: date, expected: bool) -> None:
    assert is_delayed(status, planned_end, THURSDAY) is expected


def test_delay_days_counts_from_planned_end() -> None:
    assert delay_days(date(2026, 9, 20), THURSDAY) == 4


@pytest.mark.parametrize(
    ("planned_end", "expected"),
    [
        (MONDAY, True),
        (date(2026, 9, 23), True),
        (date(2026, 9, 20), False),  # 지난주에 지연된 것은 "이번 주 지연"이 아니다
        (THURSDAY, False),
    ],
)
def test_is_delayed_this_week(planned_end: date, expected: bool) -> None:
    assert is_delayed_this_week("in_progress", planned_end, THURSDAY) is expected


def test_is_delayed_this_week_never_true_on_monday() -> None:
    assert not is_delayed_this_week("in_progress", date(2026, 9, 20), MONDAY)


@pytest.mark.parametrize(
    ("status", "expected"), [("open", True), ("mitigating", True), ("closed", False)]
)
def test_is_open_risk(status: str, expected: bool) -> None:
    assert is_open_risk(status) is expected
