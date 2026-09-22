"""Project configuration, read from environment variables only.

CLAUDE.md 코드 규칙: 설정은 이 파일 한 곳에서 환경변수로 읽는다. 코드에
URL·키·모델명·타임아웃을 하드코딩하지 않는다. 이 파일은 현재 이슈(#9,
모델 클라이언트)에 필요한 값만 담고 있고, 이후 이슈(OpenSearch·Postgres 등)가
같은 파일에 자기 설정을 추가한다.

이슈 #10 검증 중 발견: `.env`를 읽는 코드가 레포 어디에도 없어서, 지금까지
모든 설정이 (실제로는 `.env`에 값이 있어도) 하드코딩된 기본값으로만 동작하고
있었다 (docs/learning-log.md 네 번째 항목). 아래 `load_dotenv(override=False)`
한 줄로 이 모듈이 처음 import될 때 `.env`를 한 번 읽는다. 이미 설정된 환경변수
(쉘 export, CI secrets 등)는 덮지 않고, `.env` 파일이 없어도 예외 없이 넘어간다
(python-dotenv 기본 동작 — 예외를 던지지 않음).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)

load_dotenv(override=False)


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
    # 단가는 OpenAI가 수시로 바꾼다 — 값을 추측해서 채우지 않는다. OpenAI 요금
    # 페이지가 100만 토큰(1M) 단위로 표기하므로 단위를 맞춰 둔다 — 1K로 두면
    # 값을 옮겨 적을 때 1000배 오차가 난다. .env.example 기본값은 0이고,
    # 실제 단가는 착수 시 공식 가격표를 보고 채운다.
    llm_price_input_per_1m: float
    llm_price_output_per_1m: float

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
            llm_price_input_per_1m=_get_float("LLM_PRICE_INPUT_PER_1M", 0.0),
            llm_price_output_per_1m=_get_float("LLM_PRICE_OUTPUT_PER_1M", 0.0),
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


class NaraApiConfig(BaseModel):
    """나라장터 입찰공고정보서비스(공공데이터포털) 클라이언트 설정."""

    api_key: str
    base_url: str
    timeout_s: float

    @classmethod
    def from_env(cls) -> NaraApiConfig:
        api_key = os.environ.get("NARA_API_KEY", "")
        if "%" in api_key:
            # 공공데이터포털이 주는 "인코딩된" 키를 그대로 넣으면 httpx가 쿼리
            # 파라미터로 보낼 때 '%'를 다시 인코딩(%→%25)해 이중 인코딩이 되고,
            # 서버가 403을 낸다 — 원인이 기록에 남지 않는 403보다 경고가 낫다
            # (docs/learning-log.md 다섯 번째 항목).
            logger.warning(
                "NARA_API_KEY에 '%%' 문자가 있습니다 — 포털이 주는 인코딩된 값을 그대로 "
                "넣은 것으로 보입니다. httpx가 요청 시 한 번 더 인코딩해 403이 날 수 "
                "있으니 .env.example 안내대로 디코딩된 값(예: %%2B → +)으로 바꿔 넣으세요."
            )
        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "NARA_BASE_URL", "https://apis.data.go.kr/1230000/ad/BidPublicInfoService"
            ),
            timeout_s=_get_float("NARA_TIMEOUT_S", 30.0),
        )
