"""RTM을 실제 DB와 점수표로 만든다 (이슈 #39, `-m integration`).

실행: `docker compose up -d postgres`, 합성 PMS v2 적재 후 `uv run pytest tests/integration -m integration -q`.
점수표 `data/eval/rtm_scores_v1.csv`는 git에 없다 — 없으면 `scripts/build_rtm_scores.py`로 먼저 만든다(ADR-0012 결정 1).
점수표(`RTM_SCORES_PATH`)와 DB의 요구사항·업무(단계 제외)·근거 청크 ID가 9개 사업 모두 맞는지 본다(다르면
`build_project_rtm`이 ValueError). 누락 정답(시험 50건) 채점은 여기서 하지 않는다(`eval.run_rtm_eval`).
"""

from __future__ import annotations

import pytest

from rfp_pm_agent.config import PostgresConfig, RtmConfig, SynthConfig
from rfp_pm_agent.pms.reader import SqlPmsReader, readonly_engine
from rfp_pm_agent.rtm.report import build_all, load_scores

pytestmark = pytest.mark.integration


def test_rtm_builds_for_all_projects_from_db_and_scores() -> None:
    reader = SqlPmsReader(readonly_engine(PostgresConfig.from_env().readonly_dsn))
    tables = load_scores(RtmConfig.from_env().scores_path)

    rtms = build_all(reader, tables, SynthConfig.from_env().as_of_date)

    assert len(rtms) == 9
    assert sum(len(r.requirements) for r in rtms) == 416


def test_reader_requirements_match_requirement_ids() -> None:
    reader = SqlPmsReader(readonly_engine(PostgresConfig.from_env().readonly_dsn))

    for project in reader.projects():
        rows = reader.requirements(project.project_id)
        assert [r.requirement_id for r in rows] == reader.requirement_ids(project.project_id)
        assert all(r.project_id == project.project_id for r in rows)
