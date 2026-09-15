"""나라장터 입찰공고정보서비스(공공데이터포털 조달청) API 클라이언트.

확인 필요: `SEARCH_OPERATION`과 `_parse_item`의 필드명(`bidNtceNo` 등)은
공공데이터포털 문서를 보고 정한 최선의 추정치다. 실제 응답과 다르면
`_parse_item` 하나만 고치면 되도록 파싱을 이 함수에 모아 뒀다. 응답 봉투
구조(`response.header`/`response.body`)는 데이터포털 전체 공통 규격이라
이 부분의 신뢰도는 높다.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel

from rfp_pm_agent.config import NaraApiConfig

logger = logging.getLogger(__name__)

# httpx는 기본적으로 자기 로거("httpx")에 INFO 레벨로 요청 URL 전체를 찍는다
# ("HTTP Request: GET https://...?serviceKey=...") — 인증키가 쿼리 파라미터로
# 실려 있어서 그대로 두면 애플리케이션 로그 레벨(INFO)만으로도 키가 새 나간다.
# 이슈 #10 검증 중 실측으로 발견 (docs/learning-log.md 네 번째 항목).
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0

# 용역(소프트웨어 개발/구축) 공고 목록 조회 오퍼레이션.
SEARCH_OPERATION = "getBidPblancListInfoServcPPSSrch"

SW_KEYWORDS = ("소프트웨어", "정보시스템", "정보화", "SW")

_SERVICE_KEY_PATTERN = re.compile(r"(serviceKey=)[^&\s'\"]+", re.IGNORECASE)


def _redact(text: str) -> str:
    """로그·예외 메시지에 들어갈 문자열에서 `serviceKey` 값을 `***`로 가린다."""
    return _SERVICE_KEY_PATTERN.sub(r"\1***", text)


class NaraApiError(RuntimeError):
    """나라장터 API가 오류 응답(header.resultCode != "00")을 반환했을 때."""


class NaraRequestError(RuntimeError):
    """나라장터 요청이 재시도 후에도 실패했을 때.

    원본 예외(httpx.HTTPStatusError 등)의 메시지에는 요청 URL 전체(인증키
    쿼리 파라미터 포함)가 들어 있다. 그 예외를 그대로 올리지 않고, 가린
    메시지만 담은 이 예외로 바꿔 올린다. `from None`으로 원본 예외를
    체이닝하지 않는다 — 체이닝하면 트레이스백에 원본의 가려지지 않은
    메시지가 그대로 다시 찍힌다.
    """


class NaraBidItem(BaseModel):
    notice_no: str
    notice_title: str
    agency: str
    spec_doc_urls: list[str]


def _parse_item(raw: dict[str, Any]) -> NaraBidItem:
    urls = [url for i in range(1, 11) if (url := raw.get(f"specDocUrl{i}"))]
    return NaraBidItem(
        notice_no=str(raw.get("bidNtceNo", "")),
        notice_title=str(raw.get("bidNtceNm", "")),
        agency=str(raw.get("ntceInsttNm", "")),
        spec_doc_urls=urls,
    )


def parse_search_response(payload: dict[str, Any]) -> list[NaraBidItem]:
    """공공데이터포털 표준 응답 봉투를 파싱한다. resultCode가 "00"이 아니면
    `NaraApiError`를 낸다."""
    response = payload.get("response", {})
    header = response.get("header", {})
    result_code = header.get("resultCode")
    if result_code != "00":
        raise NaraApiError(f"나라장터 API 오류: {header.get('resultMsg')} (code={result_code})")

    body = response.get("body", {})
    items_container = body.get("items") or []
    # XML→JSON 변환 흔적으로 {"item": [...]} 또는 {"item": {...}} 형태로 올 수 있음
    if isinstance(items_container, dict):
        items_container = items_container.get("item", [])
    if isinstance(items_container, dict):
        items_container = [items_container]
    return [_parse_item(item) for item in items_container]


def has_attachment(item: NaraBidItem) -> bool:
    return len(item.spec_doc_urls) > 0


def is_sw_related(item: NaraBidItem) -> bool:
    return any(keyword in item.notice_title for keyword in SW_KEYWORDS)


class NaraApiClient:
    """`http_client`를 주입하면(예: `httpx.MockTransport`) 실제 네트워크 없이
    동작을 검증할 수 있다."""

    def __init__(self, config: NaraApiConfig, *, http_client: httpx.Client | None = None) -> None:
        self._config = config
        self._http = http_client or httpx.Client(base_url=config.base_url, timeout=config.timeout_s)

    def search_service_bids(
        self, *, begin_date: str, end_date: str, num_of_rows: int = 100, page_no: int = 1
    ) -> list[NaraBidItem]:
        params: dict[str, str | int] = {
            "serviceKey": self._config.api_key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            "inqryDiv": 1,
            "inqryBgnDt": begin_date,
            "inqryEndDt": end_date,
            "type": "json",
        }
        context = f"search(begin={begin_date}, end={end_date}, page={page_no})"
        response = self._get_with_retry(f"/{SEARCH_OPERATION}", params=params, context=context)
        return parse_search_response(response.json())

    def download_attachment(self, url: str, *, context: str) -> bytes:
        response = self._get_with_retry(url, params=None, context=context)
        return response.content

    def _get_with_retry(
        self, url: str, *, params: dict[str, str | int] | None, context: str
    ) -> httpx.Response:
        safe_context = _redact(context)
        last_message = "알 수 없는 오류"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._http.get(url, params=params)
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_message = _redact(str(exc))
                logger.warning(
                    "나라장터 요청 실패 (%s), 시도 %d/%d: %s",
                    safe_context,
                    attempt,
                    MAX_RETRIES,
                    last_message,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
        logger.error("나라장터 요청 최종 실패 (%s): %s", safe_context, last_message)
        raise NaraRequestError(f"{safe_context}: {last_message}") from None
