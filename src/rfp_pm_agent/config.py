"""Project configuration, read from environment variables only.

CLAUDE.md 코드 규칙: 설정은 이 파일 한 곳에서 환경변수로 읽는다. 코드에
URL·키·모델명·타임아웃을 하드코딩하지 않는다. 이 파일은 현재 이슈(#9,
모델 클라이언트)에 필요한 값만 담고 있고, 이후 이슈(OpenSearch·Postgres 등)가
같은 파일에 자기 설정을 추가한다.
"""

from __future__ import annotations

import os

from pydantic import BaseModel


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


class ClientsConfig(BaseModel):
    """LLM·임베딩·리랭커 클라이언트 설정. base_url만 바꾸면 로컬↔원격 전환된다."""

    llm_base_url: str
    llm_api_key: str | None
    llm_model_dev: str
    llm_model_eval: str
    llm_timeout_s: float
    # 단가는 OpenAI가 수시로 바꾼다 — 값을 추측해서 채우지 않는다.
    # .env.example 기본값은 0이고, 실제 단가는 착수 시 공식 가격표를 보고 채운다.
    llm_price_input_per_1k: float
    llm_price_output_per_1k: float

    embed_base_url: str
    embed_api_key: str | None
    embed_model_id: str
    embed_timeout_s: float

    rerank_base_url: str
    rerank_api_key: str | None
    rerank_model_id: str
    rerank_timeout_s: float

    cost_log_path: str

    @classmethod
    def from_env(cls) -> ClientsConfig:
        return cls(
            llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.environ.get("OPENAI_API_KEY") or None,
            llm_model_dev=os.environ.get("LLM_MODEL_DEV", ""),
            llm_model_eval=os.environ.get("LLM_MODEL_EVAL", ""),
            llm_timeout_s=_get_float("LLM_TIMEOUT_S", 60.0),
            llm_price_input_per_1k=_get_float("LLM_PRICE_INPUT_PER_1K", 0.0),
            llm_price_output_per_1k=_get_float("LLM_PRICE_OUTPUT_PER_1K", 0.0),
            embed_base_url=os.environ.get("EMBED_BASE_URL", "http://localhost:8080"),
            embed_api_key=os.environ.get("TEI_API_KEY") or None,
            embed_model_id=os.environ.get("EMBED_MODEL_ID", ""),
            embed_timeout_s=_get_float("EMBED_TIMEOUT_S", 30.0),
            rerank_base_url=os.environ.get("RERANK_BASE_URL", "http://localhost:8081"),
            rerank_api_key=os.environ.get("TEI_API_KEY") or None,
            rerank_model_id=os.environ.get("RERANK_MODEL_ID", ""),
            rerank_timeout_s=_get_float("RERANK_TIMEOUT_S", 30.0),
            cost_log_path=os.environ.get("COST_LOG_PATH", "data/cost_log.jsonl"),
        )
