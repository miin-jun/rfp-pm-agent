"""합성 PMS 적재 (이슈 #20). `PmsDataset`을 테이블에 넣는다.

한 트랜잭션에서 기존 행을 FK 역순으로 지우고 적재 순서대로 넣는다 — 중간에 실패하면 이전 상태로 돌아간다.
스키마는 Alembic이 만든다(`uv run alembic upgrade head`). 테이블이 없으면 그 안내와 함께 멈춘다.
"""

from __future__ import annotations

from sqlalchemy import create_engine, delete, insert, inspect

from rfp_pm_agent.pms.models import TABLES_IN_LOAD_ORDER
from rfp_pm_agent.schemas.pms import PmsDataset


def load_into_db(dsn: str, ds: PmsDataset) -> dict[str, int]:
    """기존 PMS 행을 지우고 ds를 넣는다. 테이블별로 넣은 행 수를 돌려준다."""
    engine = create_engine(dsn)
    try:
        existing = set(inspect(engine).get_table_names())
        missing = [m.__tablename__ for m in TABLES_IN_LOAD_ORDER if m.__tablename__ not in existing]
        if missing:
            raise RuntimeError(f"테이블이 없다 {missing} — 먼저 `uv run alembic upgrade head`")
        counts: dict[str, int] = {}
        with engine.begin() as conn:
            for model in reversed(TABLES_IN_LOAD_ORDER):
                conn.execute(delete(model))
            for model in TABLES_IN_LOAD_ORDER:
                rows = [row.model_dump() for row in getattr(ds, model.__tablename__)]
                if rows:
                    conn.execute(insert(model), rows)
                counts[model.__tablename__] = len(rows)
        return counts
    finally:
        engine.dispose()
