"""호출별 비용 로그.

모든 모델 호출은 `CostLogger`를 통해 모델명·입력/출력 토큰 수·예상 비용·지연(ms)을
JSONL로 남긴다. 로컬 TEI/vLLM처럼 자체 서빙이라 과금이 없는 호출은 항상
`cost_usd=0.0`으로 기록한다 (임베딩·리랭커가 그 경우).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rfp_pm_agent.config import ClientsConfig


@dataclass(frozen=True)
class CostLogEntry:
    ts: str
    client: str  # "llm" | "embedding" | "reranker"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class CostLogger:
    """`log_path`에 `CostLogEntry`를 한 줄씩 append하는 JSONL 로거."""

    def __init__(self, log_path: str | Path) -> None:
        self._log_path = Path(log_path)

    def log(self, entry: CostLogEntry) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")


def compute_llm_cost(config: ClientsConfig, *, input_tokens: int, output_tokens: int) -> float:
    """OpenAI 계열 단가로 비용을 계산한다. 단가는 변동하므로 config(.env)의
    `LLM_PRICE_INPUT_PER_1M`/`LLM_PRICE_OUTPUT_PER_1M`을 주기적으로 확인해 갱신해야
    한다. OpenAI 요금 페이지가 100만 토큰(1M) 단위로 표기하므로 계산도 1M
    기준으로 맞춘다. 기본값은 0이라, 값을 채우기 전에는 항상 비용 0으로 기록된다."""
    return (input_tokens / 1_000_000) * config.llm_price_input_per_1m + (
        output_tokens / 1_000_000
    ) * config.llm_price_output_per_1m


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
