"""나라장터 입찰공고정보서비스(공공데이터포털 조달청) API 클라이언트.

필드명은 2026-09-15 실제 API 응답(curl로 확보)을 보고 확정했다 —
추측치가 아니다. 이전에 문서만 보고 추정했던 필드명은 대부분 틀렸었다
(docs/learning-log.md 다섯 번째 항목).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, Literal

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

# is_sw_related가 통과시킨 조건을 나타내는 문자열. 함수 시그니처의 반환
# 타입을 `Literal[...] | None`으로 직접 쓰지 않고 별도 별칭으로 뺀 이유:
# 이 파일은 `from __future__ import annotations`로 어노테이션을 문자열
# 지연 평가하는데, ruff의 따옴표 정리 자동수정이 함수 시그니처 안의
# `Literal["a", "b"]` 리터럴 값 따옴표까지 지워버려(`Literal[a, b]`) 이름이
# 안 정의됐다는 오류가 났다 — 별칭으로 빼서 시그니처에는 Literal 문자열
# 리터럴이 직접 나타나지 않게 했다.
SwMatchReason = Literal["info_biz_yn", "classification", "title_keyword"]

_SERVICE_KEY_PATTERN = re.compile(r"(serviceKey=)[^&\s'\"]+", re.IGNORECASE)


def _redact(text: str) -> str:
    """로그·예외 메시지에 들어갈 문자열에서 `serviceKey` 값을 `***`로 가린다."""
    return _SERVICE_KEY_PATTERN.sub(r"\1***", text)


# 알려진 두 번째 응답 구조 — 정상 응답과 같은 header.resultCode/resultMsg
# 모양을 다른 최상위 키로 감싼다. 이슈 #47 실측: 조회 범위 초과(resultCode=07)
# 때 이 구조로 온다. 실제 응답 원문:
#   {"nkoneps.com.response.ResponseError": {"header": {"resultCode": "07",
#     "resultMsg": "입력범위값 초과 에러"}}}
NKONEPS_ERROR_KEY = "nkoneps.com.response.ResponseError"

# 알려진 두 응답 구조(response, NKONEPS_ERROR_KEY) 어느 쪽에도 안 맞으면
# payload 앞부분을 이만큼 잘라 예외 메시지에 담는다 — 새 응답 구조가 와도
# `code=None`으로 정보 없이 사라지지 않게 하는 게 이슈 #47의 핵심이다.
PAYLOAD_TRUNCATE_LEN = 500

# 나라장터 참고문서: inqryDiv=1(등록일시) 조회는 조회 범위가 최대 1개월이다
# (이슈 #47). 정확한 "1개월" 경계(윤년·월별 일수)를 계산하는 대신, 어떤
# 달이든 최대 31일이므로 31일을 상한으로 쓴다 — 이보다 좁게 잡을 위험은
# 있어도(예: 2월처럼 짧은 달 기준으로는 더 엄격해야 할 수도 있음), 넓게
# 잡아 서버가 거부할 범위를 통과시키는 일은 없다.
MAX_SEARCH_RANGE_DAYS = 31

# inqryBgnDt/inqryEndDt 형식. 참고문서 규격은 YYYYMMDDHHMM(12자리)인데,
# 기존 코드는 YYYYMMDD(8자리)만 보내고 있었다 — 실측(2026-09-20, 같은
# 조건에서 시각만 0000/2359/8자리로 바꿔 요청)으로 서버가 부족한 자리를
# 0000(자정)으로 채운다는 게 확인됐다(세 결과 모두 동일 totalCount). 즉
# 8자리만 보내면 조회 종료일 당일 등록된 공고가 조회 범위에서 빠진다 —
# 이 파일은 형식만 검사하고, 실제로 0000/2359를 채워 보내는 건
# collect.py의 책임이다.
INQRY_DATETIME_FORMAT = "%Y%m%d%H%M"

# search_all_service_bids의 기본 페이지 크기. 한 달 조회 응답이 2925건이었다
# (PR #48 실측) — 100건씩이면 30번, 이 값이면 3번이다. 일일 호출 한도(1000건)를
# 아끼기 위해 크게 잡는다. 서버가 허용하는 numOfRows 상한은 참고문서로
# 확인하지 못했다 — 서버가 에러 없이 더 적게 보내더라도 search_all_service_bids는
# 빈 페이지에서 멈추므로 무한 호출은 없고, 받은 건수를 total_count와 비교하면
# 누락이 드러난다.
SEARCH_PAGE_SIZE = 999


def _truncate_payload(payload: Any) -> str:
    """예외 메시지에 담을 payload 조각. json.dumps 후 자르고 `_redact`로
    인증키 패턴을 가린다 — payload는 서버 응답이라 인증키가 없을 것으로
    예상되지만, 확인 없이 단정하지 않고 방어적으로 가린다(이슈 #47)."""
    try:
        dumped = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        dumped = str(payload)
    return _redact(dumped[:PAYLOAD_TRUNCATE_LEN])


class NaraApiError(RuntimeError):
    """나라장터 API가 오류 응답(resultCode != "00", 알려진 구조 또는 알 수
    없는 구조 둘 다 포함)을 반환했을 때."""


class NaraSearchRangeError(ValueError):
    """조회 기간이 허용 범위(최대 1개월)를 넘었거나 형식이 틀렸을 때. API를
    호출하기 전에 거절한다(이슈 #47) — 서버 왕복과 일일 호출 한도를
    낭비하지 않기 위함."""


def _validate_search_range(begin_date: str, end_date: str) -> None:
    """`begin_date`/`end_date`가 `INQRY_DATETIME_FORMAT`(12자리)인지, 조회
    범위가 `MAX_SEARCH_RANGE_DAYS`를 넘지 않는지 확인한다. 형식이 틀리거나
    범위를 넘으면 `NaraSearchRangeError`를 낸다 — API를 부르기 전에."""
    try:
        # 이 문자열은 나라장터 API의 자체 조회 파라미터 형식일 뿐 실제 타임존
        # 정보가 없다 — 여기서는 두 값의 일수 차이만 계산하므로 tzinfo를
        # 무엇으로 붙이든 결과는 같다(DTZ007 회피용 UTC 부착).
        begin = datetime.strptime(begin_date, INQRY_DATETIME_FORMAT).replace(tzinfo=UTC)
        end = datetime.strptime(end_date, INQRY_DATETIME_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise NaraSearchRangeError(
            f"begin_date/end_date는 {INQRY_DATETIME_FORMAT} 형식(12자리)이어야 함: "
            f"begin={begin_date!r}, end={end_date!r}"
        ) from exc

    if (end - begin).days > MAX_SEARCH_RANGE_DAYS:
        raise NaraSearchRangeError(
            f"조회 범위가 허용 범위(최대 {MAX_SEARCH_RANGE_DAYS}일)를 넘음: "
            f"begin={begin_date}, end={end_date}"
        )


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


class NaraSearchPage(BaseModel):
    """검색 응답 한 페이지. `total_count`는 응답 body의 `totalCount` — 이
    페이지 건수가 아니라 조회 조건 전체의 건수다."""

    items: list[NaraBidItem]
    total_count: int


class NaraSearchResult(BaseModel):
    """`search_all_service_bids`의 결과. `pages_fetched`는 실제로 호출한
    페이지 수(빈 페이지 포함)."""

    items: list[NaraBidItem]
    total_count: int
    pages_fetched: int


def parse_search_response(payload: dict[str, Any]) -> list[NaraBidItem]:
    """`parse_search_page`의 items만 돌려준다 (기존 호출부 호환)."""
    return parse_search_page(payload).items


def parse_search_page(payload: dict[str, Any]) -> NaraSearchPage:
    """나라장터 API 응답을 파싱한다. 알려진 응답 구조는 두 가지다:
    1) 정상/표준 오류 응답 — 최상위 "response" 키, header.resultCode/resultMsg,
       body.items/totalCount/numOfRows/pageNo (실제 응답에서 확인)
    2) 알려진 오류 응답 구조 — 최상위 `NKONEPS_ERROR_KEY` 키 아래 같은 모양의
       header.resultCode/resultMsg (이슈 #47 실측: 조회 범위 초과 시 이 구조로 옴)

    resultCode가 "00"이 아니면 두 구조 모두 `NaraApiError`를 낸다. 이 두 구조
    어디에도 맞지 않는 payload가 오면(새로운/미확인 응답 구조), 원인을 알 수
    없다고 예외 없이 빈 결과로 넘기지 않고 payload 앞부분을 잘라 예외 메시지에 담아
    `NaraApiError`를 낸다 — 인증키는 `_redact`로 가린다."""
    if "response" in payload:
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
        return NaraSearchPage(
            items=[_parse_item(item) for item in items_container],
            total_count=int(body.get("totalCount") or 0),
        )

    if NKONEPS_ERROR_KEY in payload:
        header = payload.get(NKONEPS_ERROR_KEY, {}).get("header", {})
        result_code = header.get("resultCode")
        result_msg = header.get("resultMsg")
        raise NaraApiError(
            f"나라장터 API 오류({NKONEPS_ERROR_KEY}): {result_msg} (code={result_code})"
        )

    raise NaraApiError(f"나라장터 API 응답 구조를 알 수 없음: {_truncate_payload(payload)}")


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


def is_sw_related(item: NaraBidItem) -> SwMatchReason | None:
    """소프트웨어 개발/구축 용역인지 판단하고, 통과시킨 조건을 반환한다.

    1차 기준(신뢰도 높음): `infoBizYn`(정보화사업 여부)이 "Y"이거나,
    발주기관이 직접 붙인 분류명(대/중분류)에 정보통신·소프트웨어 계열
    키워드가 있는가. 이 값들은 발주기관이 사업 등록 시 스스로 분류한
    것이라 공고명 표현(구축/고도화/개발 등 제각각)보다 신뢰도가 높다.

    2차 기준(보조): 1차로 못 정하면 공고명에 "구축·고도화·시스템·정보화"
    키워드가 있는지 본다. 실제 응답 예시의 "의상 제작 및 운영 용역"처럼
    이 키워드가 전혀 없으면 걸러진다.

    반환값은 조기 반환 순서(① infoBizYn → ② 분류명 → ③ 공고명)대로 어느
    조건이 통과시켰는지를 나타낸다: `"info_biz_yn"` / `"classification"` /
    `"title_keyword"`. 아무 조건도 만족하지 않으면 `None`이다 — 재즈
    페스티벌 공고(R26BK01684655)가 실제로는 ③(공고명의 "시스템")만으로
    통과했는데도 이슈 #47에서 ①(infoBizYn)이 원인으로 의심됐던 사례가 있어
    (docs/learning-log.md), 호출하는 쪽이 어느 조건이 통과시켰는지 사후에
    알 수 있도록 bool 대신 문자열을 반환한다.
    """
    if item.info_biz_yn.strip().upper() == "Y":
        return "info_biz_yn"
    classification = f"{item.large_category} {item.mid_category}"
    if any(keyword in classification for keyword in SW_CLASSIFICATION_KEYWORDS):
        return "classification"
    if any(keyword in item.notice_title for keyword in SW_TITLE_KEYWORDS):
        return "title_keyword"
    return None


class NaraApiClient:
    """`http_client`를 주입하면(예: `httpx.MockTransport`) 실제 네트워크 없이
    동작을 검증할 수 있다."""

    def __init__(self, config: NaraApiConfig, *, http_client: httpx.Client | None = None) -> None:
        self._config = config
        self._http = http_client or httpx.Client(base_url=config.base_url, timeout=config.timeout_s)

    def search_service_bids(
        self, *, begin_date: str, end_date: str, num_of_rows: int = 100, page_no: int = 1
    ) -> list[NaraBidItem]:
        """용역 입찰공고를 조회한다. `begin_date`/`end_date`는 `INQRY_DATETIME_FORMAT`
        형식(`YYYYMMDDHHMM`, 12자리)이어야 한다 — 8자리만 보내면 서버가 부족한
        HHMM 자리를 0000(자정)으로 채워, 조회 종료일 당일 등록된 공고가 조회
        범위에서 빠진다(이슈 #47 실측 확인). 조회 범위(end - begin)가
        `MAX_SEARCH_RANGE_DAYS`(31일, inqryDiv=1의 참고문서상 최대 조회 범위인
        1개월의 상한)를 넘으면 API를 호출하지 않고 `NaraSearchRangeError`를
        낸다 — 서버 왕복과 일일 호출 한도를 낭비하지 않기 위함."""
        return self._search_page(
            begin_date=begin_date, end_date=end_date, num_of_rows=num_of_rows, page_no=page_no
        ).items

    def search_all_service_bids(
        self, *, begin_date: str, end_date: str, num_of_rows: int = SEARCH_PAGE_SIZE
    ) -> NaraSearchResult:
        """`search_service_bids`와 같은 조회를 1페이지부터 반복해 전체 결과를
        받는다. 받은 누적 건수가 1페이지 응답의 `totalCount` 이상이 되거나 빈
        페이지가 오면 멈춘다 — 빈 페이지 조건은 서버가 `numOfRows`를 에러 없이
        줄이거나 `totalCount`가 실제보다 클 때 무한 호출을 막는다. 이때 받은
        건수가 `total_count`보다 적으면 누락이 있다는 뜻이므로 호출부가 기록해
        비교해야 한다. 조회 범위 검증은 첫 호출 전에 한다."""
        _validate_search_range(begin_date, end_date)
        items: list[NaraBidItem] = []
        total_count = 0
        page_no = 0
        while True:
            page_no += 1
            page = self._search_page(
                begin_date=begin_date, end_date=end_date, num_of_rows=num_of_rows, page_no=page_no
            )
            if page_no == 1:
                total_count = page.total_count
            items.extend(page.items)
            if not page.items or len(items) >= total_count:
                break
        return NaraSearchResult(items=items, total_count=total_count, pages_fetched=page_no)

    def _search_page(
        self, *, begin_date: str, end_date: str, num_of_rows: int, page_no: int
    ) -> NaraSearchPage:
        _validate_search_range(begin_date, end_date)
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
        return parse_search_page(response.json())

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
