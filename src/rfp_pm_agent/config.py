"""Project configuration, read from environment variables only.

CLAUDE.md 코드 규칙: 설정은 이 파일 한 곳에서 환경변수로 읽는다. 코드에
URL·키·모델명·타임아웃을 하드코딩하지 않는다. 이슈마다 필요한 설정을 이 파일에
추가한다 (모델 클라이언트 #9, 나라장터 #10, OpenSearch #17).

이슈 #10 검증 중 발견: `.env`를 읽는 코드가 레포 어디에도 없어서, 지금까지
모든 설정이 (실제로는 `.env`에 값이 있어도) 하드코딩된 기본값으로만 동작하고
있었다 (docs/learning-log.md 네 번째 항목). 아래 `load_dotenv(override=False)`
한 줄로 이 모듈이 처음 import될 때 `.env`를 한 번 읽는다. 이미 설정된 환경변수
(쉘 export, CI secrets 등)는 덮지 않고, `.env` 파일이 없어도 예외 없이 넘어간다
(python-dotenv 기본 동작 — 예외를 던지지 않음).

.env 경로는 이 파일 위치로 정한다(#18). 인자 없는 `load_dotenv()`는 `__main__`에
`__file__`이 없으면 현재 디렉터리부터 찾는데, `python -m rfp_pm_agent.ingest.index_chunks`는
부모 패키지 `__init__`이 config를 불러오는 시점이 바로 그 상태라 레포 밖에서 실행하면
.env를 못 읽고 기본값으로 동작했다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 이 파일은 src/rfp_pm_agent/ 아래에 있으므로 parents[2]가 레포 루트다(uv sync의 편집 설치 기준)
REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPO_ROOT / ".env"

load_dotenv(DOTENV_PATH, override=False)


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw.strip().lower() in {"1", "true", "yes"}:
        return True
    if raw.strip().lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name}은 true/false 중 하나여야 합니다: {raw!r}")


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


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
    # 모델마다 학습 때 쓴 입력 접두어가 다르다. multilingual-e5는 "query: "/"passage: "가
    # 필요하고(모델 카드: 없으면 성능 저하), KURE-v1·bge-m3는 필요 없다(모델 카드 확인,
    # 2026-09-26). 끝의 공백까지 값이다 — .env에서는 따옴표로 감싼다 (EMBED_QUERY_PREFIX="query: ")
    embed_query_prefix: str = ""
    embed_passage_prefix: str = ""
    # 모델 최대 길이를 넘는 입력을 잘라서 임베딩할지. 기본은 False(서버가 413으로 거절).
    # #16에서 e5(최대 512토큰)만 True로 켜고, 자른 입력 수를 기록한다
    embed_truncate: bool = False

    rerank_base_url: str
    rerank_api_key: str | None
    rerank_model_id: str
    rerank_timeout_s: float
    # TEI는 요청 1건에 넣을 수 있는 입력 수를 서버 옵션 `--max-client-batch-size`
    # (기본 32)로 제한하고, 넘으면 422를 낸다. 클라이언트는 이 값씩 나눠 보낸다.
    # docker-compose.yml도 같은 환경변수로 서버의 `MAX_CLIENT_BATCH_SIZE`를 정하므로
    # `.env` 한 곳만 바꾸면 서버·클라이언트가 함께 바뀐다. 0 이하는 거부한다 — 0이면
    # range()가 에러를 내고, 음수면 요청을 하나도 보내지 않고 빈 결과를 돌려준다.
    tei_max_client_batch_size: int = Field(gt=0)
    # TEI 모델 파일이 있는 Docker 볼륨 이름(docker-compose.yml의 tei_models, compose 프로젝트
    # 이름이 앞에 붙는다). 실행 기록에 모델 revision(스냅샷 해시)을 남길 때 읽는다 — TEI 1.9.4의
    # /info는 revision을 주지 않고 띄우면 model_sha가 null이다 (이슈 #16)
    tei_models_volume: str = "rfp-pm-agent_tei_models"

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
            embed_query_prefix=os.environ.get("EMBED_QUERY_PREFIX", ""),
            embed_passage_prefix=os.environ.get("EMBED_PASSAGE_PREFIX", ""),
            embed_truncate=_get_bool("EMBED_TRUNCATE", False),
            rerank_base_url=os.environ.get("RERANK_BASE_URL", "http://localhost:8081"),
            rerank_api_key=os.environ.get("TEI_API_KEY") or None,
            rerank_model_id=os.environ.get("RERANK_MODEL_ID", ""),
            rerank_timeout_s=_get_float("RERANK_TIMEOUT_S", 30.0),
            tei_max_client_batch_size=_get_int("TEI_MAX_CLIENT_BATCH_SIZE", 32),
            tei_models_volume=os.environ.get("TEI_MODELS_VOLUME", "rfp-pm-agent_tei_models"),
            cost_log_path=os.environ.get("COST_LOG_PATH", "data/cost_log.jsonl"),
        )


class OpenSearchConfig(BaseModel):
    """OpenSearch 연결과 인덱스 이름 (이슈 #17).

    index_alias는 검색 코드가 부르는 이름이고, index_name은 색인 모듈이 쓰는 실제 인덱스다.
    인덱스는 임베딩 모델별로 나누므로(docs/data-design.md 5절) index_name에 모델 이름이
    들어간다 — 모델명을 코드에 두지 않으려고 기본값을 비워 두고, 비어 있으면 색인 CLI가 멈춘다.
    """

    url: str
    index_alias: str
    index_name: str

    @classmethod
    def from_env(cls) -> OpenSearchConfig:
        return cls(
            url=os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
            index_alias=os.environ.get("OPENSEARCH_INDEX_ALIAS", "rfp_chunks"),
            index_name=os.environ.get("OPENSEARCH_INDEX_NAME", ""),
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
