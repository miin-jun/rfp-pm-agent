"""PMS 스키마·적재를 실제 PostgreSQL로 확인한다 (이슈 #20, `-m integration`).

실행: `docker compose up -d postgres` 후 `uv run pytest tests/integration -m integration -q`.

- 운영 DB(`POSTGRES_DSN`의 DB)를 건드리지 않는다. 같은 서버에 임시 DB를 만들어 Alembic으로 스키마를 올리고,
  끝나면 지운다
- PostgreSQL이 꺼져 있으면 skip하지 않고 실패한다 — test_opensearch_search.py와 같은 이유
- 데이터는 단위 테스트와 같은 가짜 문서·초안으로 만든다(LLM 없음)
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from rfp_pm_agent.config import PostgresConfig
from rfp_pm_agent.pms.load import load_into_db
from rfp_pm_agent.pms.models import Issue, Risk, WbsItem, WbsRequirement
from rfp_pm_agent.schemas.pms import PmsDataset
from tests.fakes.pms_sample import build_sample

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _temp_database(*, migrate: bool) -> Iterator[str]:
    """같은 서버에 임시 DB를 만들고(migrate=True면 Alembic head까지 올리고) 끝나면 지운다."""
    base = make_url(PostgresConfig.from_env().dsn)
    name = f"pms_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    dsn = base.set(database=name).render_as_string(hide_password=False)
    try:
        if migrate:
            cfg = Config(str(REPO_ROOT / "alembic.ini"))
            cfg.set_main_option("sqlalchemy.url", dsn.replace("%", "%%"))
            cfg.attributes["configure_logger"] = False
            command.upgrade(cfg, "head")
        yield dsn
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="module")
def temp_dsn() -> Iterator[str]:
    with _temp_database(migrate=True) as dsn:
        yield dsn


@pytest.fixture(scope="module")
def dataset() -> PmsDataset:
    ds, _ = build_sample()
    return ds


def test_load_inserts_every_row_and_reload_replaces(temp_dsn: str, dataset: PmsDataset) -> None:
    first = load_into_db(temp_dsn, dataset)
    second = load_into_db(temp_dsn, dataset)  # 다시 적재해도 중복되지 않는다

    assert first == second
    assert first["wbs_items"] == len(dataset.wbs_items)
    engine = create_engine(temp_dsn)
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(WbsItem)) == len(dataset.wbs_items)
    engine.dispose()


def test_delayed_sql_matches_python_rule(temp_dsn: str, dataset: PmsDataset) -> None:
    """data-design.md 6절 정의를 SQL로 옮긴 결과가 생성 코드의 판단과 같다."""
    load_into_db(temp_dsn, dataset)
    expected = sum(
        w.status != "done" and w.planned_end < dataset.as_of_date for w in dataset.wbs_items
    )
    engine = create_engine(temp_dsn)
    with engine.connect() as conn:
        got = conn.scalar(
            select(func.count())
            .select_from(WbsItem)
            .where(WbsItem.status != "done", WbsItem.planned_end < dataset.as_of_date)
        )
    engine.dispose()
    assert got == expected > 0


def _insert_fails(dsn: str, model: type, row: dict[str, object], constraint: str) -> None:
    """insert가 IntegrityError로 실패하고, 오류가 `constraint`(제약 이름의 정규식)를 가리키는지 본다."""
    engine = create_engine(dsn)
    try:
        with pytest.raises(IntegrityError, match=constraint), engine.begin() as conn:
            conn.execute(insert(model), [row])
    finally:
        engine.dispose()


def test_db_rejects_link_to_unknown_requirement(temp_dsn: str, dataset: PmsDataset) -> None:
    load_into_db(temp_dsn, dataset)
    link = dataset.wbs_requirements[0]
    _insert_fails(
        temp_dsn,
        WbsRequirement,
        {"project_id": link.project_id, "wbs_id": link.wbs_id, "requirement_id": "XXX-999"},
        "wbs_requirements_project_id_requirement_id_fkey",
    )


def test_db_rejects_done_task_without_actual_end(temp_dsn: str, dataset: PmsDataset) -> None:
    load_into_db(temp_dsn, dataset)
    row = next(w for w in dataset.wbs_items if w.status == "done").model_dump()
    row.update(wbs_id="9.99", actual_end=None)
    _insert_fails(temp_dsn, WbsItem, row, "ck_wbs_items_done")


def test_db_rejects_reversed_dates_and_bad_status(temp_dsn: str, dataset: PmsDataset) -> None:
    load_into_db(temp_dsn, dataset)
    leaf = next(
        w for w in dataset.wbs_items if w.parent_wbs_id and w.planned_start < w.planned_end
    ).model_dump()
    leaf.update(wbs_id="9.98", planned_start=leaf["planned_end"], planned_end=leaf["planned_start"])
    _insert_fails(temp_dsn, WbsItem, leaf, "ck_wbs_items_planned")
    risk = dataset.risks[0].model_dump()
    risk.update(risk_id="X-R99", status="unknown")
    _insert_fails(temp_dsn, Risk, risk, "ck_risks_status")


def test_db_rejects_issue_linked_to_other_projects_risk(temp_dsn: str, dataset: PmsDataset) -> None:
    load_into_db(temp_dsn, dataset)
    issue = next(i for i in dataset.issues if i.project_id == "PRJ-01").model_dump()
    other = next(r for r in dataset.risks if r.project_id == "PRJ-02")
    issue.update(issue_id="X-I99", related_risk_id=other.risk_id)
    _insert_fails(temp_dsn, Issue, issue, "issues_project_id_related_risk_id_fkey")


def test_load_stops_when_schema_is_missing(dataset: PmsDataset) -> None:
    with (
        _temp_database(migrate=False) as dsn,
        pytest.raises(RuntimeError, match="alembic upgrade head"),
    ):
        load_into_db(dsn, dataset)
