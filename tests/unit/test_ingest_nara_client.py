import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from rfp_pm_agent.config import NaraApiConfig
from rfp_pm_agent.ingest.nara_client import (
    SEARCH_PAGE_SIZE,
    NaraApiClient,
    NaraApiError,
    NaraBidItem,
    NaraRequestError,
    NaraSearchRangeError,
    has_attachment,
    is_sw_related,
    parse_search_page,
    parse_search_response,
    select_proposal_attachment,
)

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "nara_search_response.json"


def _load_fixture() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return result


def _config() -> NaraApiConfig:
    return NaraApiConfig(api_key="test-key", base_url="https://nara.test", timeout_s=5.0)


def test_parse_search_response_extracts_items() -> None:
    items = parse_search_response(_load_fixture())

    assert len(items) == 3
    first = items[0]
    assert first.notice_no == "20260901001"
    assert first.agency == "테스트기관"
    assert first.demand_agency == "테스트수요기관"
    assert first.detail_url == "https://www.g2b.go.kr/detail?bidNtceNo=20260901001"
    # ntceSpecDocUrl{n} ↔ ntceSpecFileNm{n} 같은 번호끼리 짝지어졌는지 확인
    assert len(first.attachments) == 3
    assert first.attachments[0].url.endswith("fileId=1")
    assert first.attachments[0].file_name == "제안요청서.pdf"
    assert first.attachments[1].url.endswith("fileId=2")
    assert first.attachments[1].file_name == "제안요청서.hwpx"


def test_parse_search_response_raises_on_error_code() -> None:
    payload = {"response": {"header": {"resultCode": "99", "resultMsg": "APPLICATION ERROR"}}}

    with pytest.raises(NaraApiError):
        parse_search_response(payload)


def test_parse_search_response_handles_nkoneps_error_structure() -> None:
    """이슈 #47 실측: 조회 범위 초과 시 응답이 "response" 키가 아니라
    "nkoneps.com.response.ResponseError" 키 아래 같은 모양(header.resultCode/
    resultMsg)으로 온다. 실제 원문 그대로 재현한다."""
    payload = {
        "nkoneps.com.response.ResponseError": {
            "header": {"resultCode": "07", "resultMsg": "입력범위값 초과 에러"}
        }
    }

    with pytest.raises(NaraApiError) as exc_info:
        parse_search_response(payload)

    assert "07" in str(exc_info.value)
    assert "입력범위값 초과 에러" in str(exc_info.value)


def test_parse_search_response_unknown_structure_includes_payload_fragment_and_redacts_key() -> (
    None
):
    """알려진 두 응답 구조(response, nkoneps 에러) 어디에도 맞지 않으면 원인
    불명으로 빈 결과로 처리되어 원인이 사라지지 않고 payload 조각이 예외 메시지에 남아야 한다.
    payload 안에 인증키처럼 보이는 문자열이 섞여 있어도(서버가 절대 이런 값을
    돌려주지 않을 것으로 예상되지만 확인 없이 단정하지 않는다) 가려져야
    한다."""
    payload = {
        "unexpectedTopLevelKey": {
            "message": "이런 구조는 처음 봄",
            "leaked": "serviceKey=SHOULD_BE_REDACTED_VALUE",
        }
    }

    with pytest.raises(NaraApiError) as exc_info:
        parse_search_response(payload)

    message = str(exc_info.value)
    assert "unexpectedTopLevelKey" in message
    assert "이런 구조는 처음 봄" in message
    assert "SHOULD_BE_REDACTED_VALUE" not in message
    assert "serviceKey=***" in message


def test_select_proposal_attachment_prefers_hwpx_over_pdf() -> None:
    items = parse_search_response(_load_fixture())

    attachment = select_proposal_attachment(items[0])

    assert attachment is not None
    assert attachment.file_name == "제안요청서.hwpx"  # pdf·hwp도 있지만 hwpx 우선
    # 저장 파일명은 URL이 아니라 file_name에서 가져와야 함(전부 downloadFile.do라 URL로는 구분 불가)
    assert "downloadFile.do" in attachment.url


def test_select_proposal_attachment_returns_none_when_no_proposal_named_file() -> None:
    items = parse_search_response(_load_fixture())

    # 세 번째 공고는 "규격서.pdf"만 있고 "제안요청서"가 들어간 파일이 없음
    assert select_proposal_attachment(items[2]) is None


def test_is_sw_related_filters_non_it_service_by_real_example() -> None:
    """실제 응답 예시("의상 제작 및 운영 용역")가 걸러지는지 확인 — 지어낸
    입력이 아니라 이슈에서 받은 실제 반례로 검증한다."""
    items = parse_search_response(_load_fixture())

    assert is_sw_related(items[0]) == "info_biz_yn"  # infoBizYn=Y + 정보통신/소프트웨어개발 분류
    assert is_sw_related(items[1]) is None  # 의상 제작 및 운영 용역
    assert is_sw_related(items[2]) is None  # 시설물 유지보수


def test_has_attachment_and_is_sw_related_filter_correctly() -> None:
    items = parse_search_response(_load_fixture())
    sw_with_attachment = [item for item in items if has_attachment(item) and is_sw_related(item)]

    assert len(sw_with_attachment) == 1
    assert sw_with_attachment[0].notice_no == "20260901001"


def _bid_item(
    *,
    info_biz_yn: str = "N",
    large_category: str = "기타",
    mid_category: str = "기타",
    notice_title: str = "일반 용역",
) -> NaraBidItem:
    return NaraBidItem(
        notice_no="TEST",
        notice_title=notice_title,
        agency="테스트기관",
        demand_agency="테스트수요기관",
        detail_url="https://example.test",
        attachments=[],
        service_division="일반용역",
        large_category=large_category,
        mid_category=mid_category,
        info_biz_yn=info_biz_yn,
    )


def test_is_sw_related_returns_info_biz_yn_when_only_that_condition_true() -> None:
    """이슈 #47 문제 2 재발 방지 — 재즈 페스티벌 공고(R26BK01684655)가
    실제로는 title_keyword로 통과했는데도 infoBizYn이 원인으로 의심됐다.
    조건을 하나씩만 참으로 만들어 어느 조건이 통과시켰는지 구별한다."""
    item = _bid_item(
        info_biz_yn="Y", large_category="기타", mid_category="기타", notice_title="일반 용역"
    )

    assert is_sw_related(item) == "info_biz_yn"


def test_is_sw_related_returns_classification_when_only_that_condition_true() -> None:
    item = _bid_item(
        info_biz_yn="N", large_category="정보통신", mid_category="기타", notice_title="일반 용역"
    )

    assert is_sw_related(item) == "classification"


def test_is_sw_related_returns_title_keyword_when_only_that_condition_true() -> None:
    item = _bid_item(
        info_biz_yn="N",
        large_category="기타",
        mid_category="기타",
        notice_title="정보시스템 구축 용역",
    )

    assert is_sw_related(item) == "title_keyword"


def test_is_sw_related_returns_none_when_no_condition_true() -> None:
    item = _bid_item(
        info_biz_yn="N", large_category="기타", mid_category="기타", notice_title="일반 용역"
    )

    assert is_sw_related(item) is None


def test_search_service_bids_uses_configured_base_url_no_real_network() -> None:
    captured_urls: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(request.url)
        return httpx.Response(200, json=_load_fixture())

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    client = NaraApiClient(_config(), http_client=http_client)

    items = client.search_service_bids(begin_date="202608010000", end_date="202609010000")

    assert len(items) == 3
    assert str(captured_urls[0]).startswith("https://nara.test/")


def test_search_service_bids_rejects_range_over_one_month_without_http_call() -> None:
    """조회 범위가 MAX_SEARCH_RANGE_DAYS(31일)를 넘으면 API를 호출하지 않고
    거절해야 한다 — 서버 왕복·일일 호출 한도 낭비를 막기 위함(이슈 #47)."""
    call_count = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["count"] += 1
        return httpx.Response(200, json=_load_fixture())

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    client = NaraApiClient(_config(), http_client=http_client)

    with pytest.raises(NaraSearchRangeError):
        client.search_service_bids(begin_date="202601010000", end_date="202603010000")

    assert call_count["count"] == 0


def test_download_attachment_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rfp_pm_agent.ingest.nara_client.BACKOFF_BASE_S", 0.0)
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise httpx.TimeoutException("simulated timeout", request=request)
        return httpx.Response(200, content=b"fake pdf bytes")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    client = NaraApiClient(_config(), http_client=http_client)

    data = client.download_attachment("https://nara.test/files/a.pdf", context="test")

    assert data == b"fake pdf bytes"
    assert attempts["count"] == 2


def test_download_attachment_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rfp_pm_agent.ingest.nara_client.BACKOFF_BASE_S", 0.0)
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.TimeoutException("always fails", request=request)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    client = NaraApiClient(_config(), http_client=http_client)

    with pytest.raises(NaraRequestError):
        client.download_attachment("https://nara.test/files/a.pdf", context="notice=X")

    assert attempts["count"] == 3


def test_401_response_does_not_leak_service_key_in_logs_or_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "SECRET_KEY_VALUE_123"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    monkeypatch.setattr("rfp_pm_agent.ingest.nara_client.BACKOFF_BASE_S", 0.0)
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    config = NaraApiConfig(api_key=secret, base_url="https://nara.test", timeout_s=5.0)
    client = NaraApiClient(config, http_client=http_client)

    with caplog.at_level(logging.WARNING), pytest.raises(NaraRequestError) as exc_info:
        client.search_service_bids(begin_date="202608010000", end_date="202609010000")

    assert secret not in str(exc_info.value)
    assert secret not in caplog.text
    assert "***" in str(exc_info.value)


# --- 페이지네이션 (이슈 #50) ---


def _page_payload(page_no: int, page_items: int, total_count: int) -> dict[str, Any]:
    items = [
        {"bidNtceNo": f"P{page_no:03d}{i:04d}", "bidNtceNm": "테스트 공고"}
        for i in range(page_items)
    ]
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "정상"},
            "body": {
                "items": items,
                "numOfRows": page_items,
                "pageNo": page_no,
                "totalCount": total_count,
            },
        }
    }


def _paged_client(
    total_count: int, page_sizes: list[int], seen_params: list[dict[str, str]]
) -> NaraApiClient:
    """page_sizes[i] = (i+1)페이지 응답 건수. 범위를 넘는 페이지는 0건."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen_params.append(params)
        page_no = int(params["pageNo"])
        size = page_sizes[page_no - 1] if page_no <= len(page_sizes) else 0
        return httpx.Response(200, json=_page_payload(page_no, size, total_count))

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    return NaraApiClient(_config(), http_client=http_client)


def test_parse_search_page_reads_total_count() -> None:
    page = parse_search_page(_load_fixture())

    assert page.total_count == 3
    assert len(page.items) == 3


def test_search_all_service_bids_fetches_until_total_count() -> None:
    seen: list[dict[str, str]] = []
    client = _paged_client(total_count=250, page_sizes=[100, 100, 50], seen_params=seen)

    result = client.search_all_service_bids(
        begin_date="202608010000", end_date="202608302359", num_of_rows=100
    )

    assert [p["pageNo"] for p in seen] == ["1", "2", "3"]
    assert len(result.items) == 250
    assert result.total_count == 250
    assert result.pages_fetched == 3


def test_search_all_service_bids_stops_on_empty_page() -> None:
    """서버가 numOfRows를 에러 없이 줄이거나 totalCount가 실제보다 크면, 빈
    페이지에서 멈춰야 한다 — 무한 호출로 일일 한도를 태우지 않기 위함."""
    seen: list[dict[str, str]] = []
    client = _paged_client(total_count=500, page_sizes=[100], seen_params=seen)

    result = client.search_all_service_bids(
        begin_date="202608010000", end_date="202608302359", num_of_rows=100
    )

    assert len(seen) == 2  # 1페이지 100건, 2페이지 0건에서 멈춤
    assert len(result.items) == 100
    assert result.total_count == 500
    assert result.pages_fetched == 2


def test_search_all_service_bids_zero_results_calls_once() -> None:
    seen: list[dict[str, str]] = []
    client = _paged_client(total_count=0, page_sizes=[], seen_params=seen)

    result = client.search_all_service_bids(begin_date="202608010000", end_date="202608302359")

    assert len(seen) == 1
    assert result.items == []
    assert result.total_count == 0
    assert result.pages_fetched == 1


def test_search_all_service_bids_uses_large_page_size_by_default() -> None:
    seen: list[dict[str, str]] = []
    client = _paged_client(total_count=1, page_sizes=[1], seen_params=seen)

    client.search_all_service_bids(begin_date="202608010000", end_date="202608302359")

    assert int(seen[0]["numOfRows"]) == SEARCH_PAGE_SIZE
    assert SEARCH_PAGE_SIZE > 100


def test_search_all_service_bids_rejects_range_before_any_call() -> None:
    seen: list[dict[str, str]] = []
    client = _paged_client(total_count=1, page_sizes=[1], seen_params=seen)

    with pytest.raises(NaraSearchRangeError):
        client.search_all_service_bids(begin_date="202601010000", end_date="202603010000")

    assert seen == []
