"""나라장터 입찰공고정보서비스(공공데이터포털 조달청) API 클라이언트.

필드명은 2026-09-15 실제 API 응답(curl로 확보)을 보고 확정했다 —
추측치가 아니다. 이전에 문서만 보고 추정했던 필드명은 대부분 틀렸었다
(docs/learning-log.md 다섯 번째 항목).
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

# 첨부파일 중 "제안요청서"만 고른다. 여러 확장자가 있으면 이 우선순위로
# 하나만 선택한다 (docs/tech-stack.md 8절: HWPX가 파싱이 가장 안정적이고,
# PDF가 그다음, HWP는 별도 변환이 필요해 가장 낮은 우선순위).
PROPOSAL_KEYWORD = "제안요청서"
EXTENSION_PRIORITY = ("hwpx", "pdf", "hwp")

# is_sw_related 2차(보조) 기준 — 공고명 키워드. "소프트웨어"·"SW"는 1차
# 기준(정보화사업 여부·분류명)이 이미 커버해서 여기 넣지 않는다.
SW_TITLE_KEYWORDS = ("구축", "고도화", "시스템", "정보화")

# 1차 기준에 쓰는 분류명 키워드 — 발주기관이 사업 등록 시 직접 붙인 값이라
# 공고명 표현 차이(구축/고도화/개발 등 제각각)보다 신뢰도가 높다.
SW_CLASSIFICATION_KEYWORDS = ("정보통신", "소프트웨어", "정보화")

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


class NaraAttachment(BaseModel):
    url: str
    file_name: str


class NaraBidItem(BaseModel):
    notice_no: str
    notice_title: str
    agency: str  # 공고기관 (ntceInsttNm)
    demand_agency: str  # 수요기관 (dminsttNm)
    detail_url: str  # 공고 상세 URL (bidNtceDtlUrl)
    attachments: list[NaraAttachment]  # ntceSpecDocUrl{n} ↔ ntceSpecFileNm{n} 짝
    service_division: str  # 용역 구분명 (srvceDivNm)
    large_category: str  # 물품분류 대분류명 (pubPrcrmntLrgClsfcNm)
    mid_category: str  # 물품분류 중분류명 (pubPrcrmntMidClsfcNm)
    info_biz_yn: str  # 정보화사업 여부 "Y"/"N" (infoBizYn)


def _parse_item(raw: dict[str, Any]) -> NaraBidItem:
    attachments = [
        NaraAttachment(url=url, file_name=file_name)
        for i in range(1, 11)
        if (url := raw.get(f"ntceSpecDocUrl{i}")) and (file_name := raw.get(f"ntceSpecFileNm{i}"))
    ]
    return NaraBidItem(
        notice_no=str(raw.get("bidNtceNo", "")),
        notice_title=str(raw.get("bidNtceNm", "")),
        agency=str(raw.get("ntceInsttNm", "")),
        demand_agency=str(raw.get("dminsttNm", "")),
        detail_url=str(raw.get("bidNtceDtlUrl", "")),
        attachments=attachments,
        service_division=str(raw.get("srvceDivNm", "")),
        large_category=str(raw.get("pubPrcrmntLrgClsfcNm", "")),
        mid_category=str(raw.get("pubPrcrmntMidClsfcNm", "")),
        info_biz_yn=str(raw.get("infoBizYn", "")),
    )


def parse_search_response(payload: dict[str, Any]) -> list[NaraBidItem]:
    """공공데이터포털 표준 응답 봉투를 파싱한다. resultCode가 "00"이 아니면
    `NaraApiError`를 낸다 (실제 응답에서 확인: resultCode/resultMsg,
    body.items/totalCount/numOfRows/pageNo)."""
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


def select_proposal_attachment(item: NaraBidItem) -> NaraAttachment | None:
    """첨부파일 중 파일명에 "제안요청서"가 들어간 것만 후보로 삼고, 여러 개면
    `EXTENSION_PRIORITY`(hwpx > pdf > hwp) 순으로 하나만 고른다. 우선순위
    목록에 없는 확장자는 가장 낮은 우선순위로 취급한다. 후보가 없으면
    `None`."""
    candidates = [a for a in item.attachments if PROPOSAL_KEYWORD in a.file_name]
    if not candidates:
        return None

    def _ext_rank(attachment: NaraAttachment) -> int:
        name = attachment.file_name
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        return (
            EXTENSION_PRIORITY.index(ext) if ext in EXTENSION_PRIORITY else len(EXTENSION_PRIORITY)
        )

    return min(candidates, key=_ext_rank)


def has_attachment(item: NaraBidItem) -> bool:
    return select_proposal_attachment(item) is not None


def is_sw_related(item: NaraBidItem) -> bool:
    """소프트웨어 개발/구축 용역인지 판단한다.

    1차 기준(신뢰도 높음): `infoBizYn`(정보화사업 여부)이 "Y"이거나,
    발주기관이 직접 붙인 분류명(대/중분류)에 정보통신·소프트웨어 계열
    키워드가 있는가. 이 값들은 발주기관이 사업 등록 시 스스로 분류한
    것이라 공고명 표현(구축/고도화/개발 등 제각각)보다 신뢰도가 높다.

    2차 기준(보조): 1차로 못 정하면 공고명에 "구축·고도화·시스템·정보화"
    키워드가 있는지 본다. 실제 응답 예시의 "의상 제작 및 운영 용역"처럼
    이 키워드가 전혀 없으면 걸러진다.
    """
    if item.info_biz_yn.strip().upper() == "Y":
        return True
    classification = f"{item.large_category} {item.mid_category}"
    if any(keyword in classification for keyword in SW_CLASSIFICATION_KEYWORDS):
        return True
    return any(keyword in item.notice_title for keyword in SW_TITLE_KEYWORDS)


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
