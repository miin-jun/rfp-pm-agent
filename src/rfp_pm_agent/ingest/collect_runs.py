"""수집 실행 기록 (`data/raw/collect_runs.jsonl`, 이슈 #50).

한 줄 = 한 실행의 조회 구간 하나. manifest는 "통과해 저장된 공고"의 기록이고
이 파일은 "실행"의 기록이다 — `--limit` 도달로 API를 부르지 않은 구간도 남겨야
"공고가 없었다"와 "조회하지 않았다"를 구별할 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

SkipReason = Literal["limit_reached"]


class CollectRun(BaseModel):
    run_id: str  # 같은 실행의 구간들은 같은 값(실행 시작 시각 ISO)
    recorded_at: str
    inqry_bgn_dt: str  # YYYYMMDDHHMM (nara_client.INQRY_DATETIME_FORMAT)
    inqry_end_dt: str
    called: bool  # 이 구간에 검색 API를 호출했는가
    skip_reason: SkipReason | None = None  # called=False일 때 이유
    total_count: int | None = None  # 응답의 totalCount. 호출 안 했으면 None
    pages_fetched: int = 0
    response_count: int = 0  # 실제로 받은 공고 건수 (total_count보다 적으면 누락)
    filter_passed: int = 0  # 첨부 있음 + is_sw_related 통과
    new_saved: int = 0  # manifest에 새로 저장한 건수 (doc_id 중복 제외)


def append_run(path: str | Path, run: CollectRun) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(run.model_dump_json() + "\n")


def read_runs(path: str | Path) -> list[CollectRun]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [CollectRun.model_validate_json(line) for line in f if line.strip()]
