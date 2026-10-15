"""PMS 조회 툴을 실제 DB와 정답 파일로 확인한다 (이슈 #21, `-m integration`).

실행: `docker compose up -d postgres`, 합성 PMS v2 적재(`uv run python -m rfp_pm_agent.pms.synth load`) 후
`uv run pytest tests/integration -m integration -q`.

- 접속은 `PmsDeps.from_env()` — 읽기 전용 계정(`POSTGRES_READONLY_DSN`, agent_ro)과 `AS_OF_DATE`
- 정답은 `data/eval/pms_truth_v2.json`. 9개 프로젝트 전부에서 지연·이번 주 지연·미해결 Risk(순서까지)·기한 지난 미제출 산출물을 비교한다
- DB가 꺼져 있거나 다른 데이터가 적재돼 있으면 skip하지 않고 실패한다
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from rfp_pm_agent.config import PostgresConfig
from rfp_pm_agent.pms.errors import PmsUnavailableError
from rfp_pm_agent.pms.reader import SqlPmsReader, readonly_engine
from rfp_pm_agent.tools.pms_tools import (
    MAX_LIMIT,
    PmsDeps,
    get_delayed_tasks,
    get_deliverables,
    get_open_risks,
)

pytestmark = pytest.mark.integration

TRUTH_PATH = Path(__file__).resolve().parents[2] / "data/eval/pms_truth_v2.json"
TRUTH: dict[str, Any] = json.loads(TRUTH_PATH.read_text("utf-8"))
PROJECT_IDS = [p["project_id"] for p in TRUTH["projects"]]


def _truth(pid: str) -> dict[str, Any]:
    project: dict[str, Any] = next(p for p in TRUTH["projects"] if p["project_id"] == pid)
    return project


@pytest.fixture(scope="module")
def deps() -> PmsDeps:
    d = PmsDeps.from_env()
    assert d.as_of_date == date.fromisoformat(TRUTH["as_of_date"]), (
        "AS_OF_DATE가 정답 파일 기준일과 다르다 — .env를 확인한다"
    )
    return d


def test_db_has_the_truth_projects(deps: PmsDeps) -> None:
    result = get_delayed_tasks(None, deps=deps)
    assert result.status == "project_required"
    assert [p.project_id for p in result.available_projects] == PROJECT_IDS
    assert [p.doc_ids for p in result.available_projects] == [
        [p["doc_id"]] for p in TRUTH["projects"]
    ]


@pytest.mark.parametrize("pid", PROJECT_IDS)
def test_delayed_tasks_match_truth(pid: str, deps: PmsDeps) -> None:
    truth = _truth(pid)["delayed_tasks"]

    result = get_delayed_tasks(pid, limit=MAX_LIMIT, deps=deps)

    assert result.total == len(truth) and not result.truncated
    assert {(t.wbs_id, t.delay_days, t.status) for t in result.items} == {
        (t["wbs_id"], t["delay_days"], t["status"]) for t in truth
    }


@pytest.mark.parametrize("pid", PROJECT_IDS)
def test_this_week_delayed_tasks_match_truth(pid: str, deps: PmsDeps) -> None:
    truth = sorted(t["wbs_id"] for t in _truth(pid)["delayed_tasks"] if t["this_week"])

    result = get_delayed_tasks(pid, scope="this_week", limit=MAX_LIMIT, deps=deps)

    assert sorted(t.wbs_id for t in result.items) == truth
    assert len(truth) >= 1  # v2는 9개 프로젝트 모두 이번 주 지연이 있다


@pytest.mark.parametrize("pid", PROJECT_IDS)
def test_open_risks_match_truth_in_order(pid: str, deps: PmsDeps) -> None:
    truth = _truth(pid)["open_risks"]

    result = get_open_risks(pid, limit=MAX_LIMIT, deps=deps)

    assert [(r.risk_id, r.score, r.status) for r in result.items] == [
        (r["risk_id"], r["score"], r["status"]) for r in truth
    ]


@pytest.mark.parametrize("pid", PROJECT_IDS)
def test_overdue_deliverables_match_truth(pid: str, deps: PmsDeps) -> None:
    truth = sorted(_truth(pid)["overdue_unsubmitted_deliverables"])

    result = get_deliverables(pid, status_filter="overdue", limit=MAX_LIMIT, deps=deps)

    assert sorted(d.deliverable_id for d in result.items) == truth


def test_injection_like_project_id_is_just_not_found(deps: PmsDeps) -> None:
    """project_id는 바인딩 파라미터로만 비교된다. SQL 조각을 넣어도 다른 프로젝트 행이 나오지 않는다."""
    result = get_delayed_tasks("PRJ-01' OR '1'='1", deps=deps)
    assert result.status == "project_not_found" and result.items == []


def test_readonly_session_rejects_writes() -> None:
    engine = readonly_engine(PostgresConfig.from_env().readonly_dsn)
    try:
        with pytest.raises(DBAPIError, match="read-only"), engine.begin() as conn:
            conn.execute(text("UPDATE wbs_items SET progress_pct = 0 WHERE wbs_id = '1'"))
    finally:
        engine.dispose()


def test_agent_ro_account_has_no_write_permission_even_in_read_write_transaction() -> None:
    """세션 읽기 전용과 별개로 계정(agent_ro) 권한도 막는지 본다. 위 테스트는 read-only 검사가 먼저 걸려 권한을 확인하지 못한다."""
    engine = readonly_engine(PostgresConfig.from_env().readonly_dsn)
    try:
        with pytest.raises(DBAPIError, match="permission denied"), engine.begin() as conn:
            conn.execute(text("SET TRANSACTION READ WRITE"))
            conn.execute(text("UPDATE wbs_items SET progress_pct = 0 WHERE wbs_id = '1'"))
    finally:
        engine.dispose()


def test_unreachable_db_is_pms_unavailable() -> None:
    reader = SqlPmsReader(readonly_engine("postgresql+psycopg://agent_ro:x@127.0.0.1:1/si"))
    deps = PmsDeps(reader=reader, as_of_date=date(2026, 9, 24))
    with pytest.raises(PmsUnavailableError):
        get_delayed_tasks("PRJ-01", deps=deps)
