"""PMS 판단 규칙 — docs/data-design.md 6절 "핵심 판단 규칙"을 코드 한 곳에 둔다 (이슈 #20).

합성 데이터의 정답(`data/eval/pms_truth_v2.json`)과 조회 툴(#21, `tools/pms_tools.py`)이 모두 이 함수들로 판단한다 —
툴의 SQL은 project_id로 행을 가져오기만 하고 판단하지 않는다.
기준일은 항상 인자로 받는다(`AS_OF_DATE`) — 시스템 날짜를 쓰지 않는다.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

OPEN_RISK_STATUSES = ("open", "mitigating")


def this_week_delay_range(as_of: date) -> tuple[date, date] | None:
    """ "이번 주 지연된 업무"의 `planned_end` 범위: 이번 주 월요일 ~ 기준일 전날(양끝 포함).

    기준일이 월요일이면 범위가 비어 None이다.
    """
    monday = as_of - timedelta(days=as_of.weekday())
    last = as_of - timedelta(days=1)
    return None if last < monday else (monday, last)


def is_delayed(status: str, planned_end: date, as_of: date) -> bool:
    """지연 업무: `status != 'done'` AND `planned_end < 기준일`."""
    return status != "done" and planned_end < as_of


def delay_days(planned_end: date, as_of: date) -> int:
    """지연 일수: 기준일 − planned_end (일). 지연 업무에만 쓴다."""
    return (as_of - planned_end).days


def is_delayed_this_week(status: str, planned_end: date, as_of: date) -> bool:
    """이번 주 지연된 업무: 지연 업무 중 `planned_end`가 이번 주 월요일 ~ 기준일 전날인 것."""
    window = this_week_delay_range(as_of)
    return (
        window is not None
        and is_delayed(status, planned_end, as_of)
        and window[0] <= planned_end <= window[1]
    )


def is_open_risk(status: str) -> bool:
    """미해결 Risk: `status IN ('open', 'mitigating')`. 정렬은 probability × impact 내림차순."""
    return status in OPEN_RISK_STATUSES


# Risk 등급: probability × impact 점수 구간 (#21). data-design.md 6절 "Risk 등급"
RISK_LEVEL_HIGH_MIN = 15
RISK_LEVEL_MEDIUM_MIN = 8
RiskLevel = Literal["low", "medium", "high"]
RISK_LEVEL_ORDER: tuple[RiskLevel, ...] = ("low", "medium", "high")


def risk_score(probability: int, impact: int) -> int:
    """Risk 점수: probability × impact (1~25). 미해결 Risk 정렬 기준."""
    return probability * impact


def risk_level(probability: int, impact: int) -> RiskLevel:
    """Risk 등급: 점수 15 이상 high, 8~14 medium, 7 이하 low."""
    score = risk_score(probability, impact)
    if score >= RISK_LEVEL_HIGH_MIN:
        return "high"
    if score >= RISK_LEVEL_MEDIUM_MIN:
        return "medium"
    return "low"


def is_overdue_deliverable(status: str, due_date: date, as_of: date) -> bool:
    """기한 지난 미제출 산출물: `status = 'not_submitted'` AND `due_date < 기준일`."""
    return status == "not_submitted" and due_date < as_of
