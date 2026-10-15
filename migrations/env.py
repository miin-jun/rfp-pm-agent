"""Alembic 실행 환경 (이슈 #20).

접속 주소는 `config.PostgresConfig`(`POSTGRES_DSN`)에서 읽는다. 코드가 `sqlalchemy.url`을 넘기면(통합 테스트가
임시 DB를 쓸 때) 그 값을 쓴다. 테이블 정의는 `rfp_pm_agent.pms.models`가 기준이다.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from rfp_pm_agent.config import PostgresConfig
from rfp_pm_agent.pms.models import Base

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or PostgresConfig.from_env().dsn


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
