"""나라장터 입찰공고정보서비스(공공데이터포털 조달청) API 클라이언트.

확인 필요: `SEARCH_OPERATION`과 `_parse_item`의 필드명(`bidNtceNo` 등)은
공공데이터포털 문서를 보고 정한 최선의 추정치다. 실제 응답과 다르면
`_parse_item` 하나만 고치면 되도록 파싱을 이 함수에 모아 뒀다. 응답 봉투
구조(`response.header`/`response.body`)는 데이터포털 전체 공통 규격이라
이 부분의 신뢰도는 높다.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel

from rfp_pm_agent.config import NaraApiConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0

# 용역(소프트웨어 개발/구축) 공고 목록 조회 오퍼레이션.
SEARCH_OPERATION = "getBidPblancListInfoServcPPSSrch"

SW_KEYWORDS = ("소프트웨어", "정보시스템", "정보화", "SW")


class NaraApiError(RuntimeError):
    """나라장터 API가 오류 응답(header.resultCode != "00")을 반환했을 때."""


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
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._http.get(url, params=params)
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_exc = exc
                logger.warning(
                    "나라장터 요청 실패 (%s), 시도 %d/%d: %s", context, attempt, MAX_RETRIES, exc
                )
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
        logger.error("나라장터 요청 최종 실패 (%s): %s", context, last_exc)
        assert last_exc is not None
        raise last_exc
