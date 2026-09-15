import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from rfp_pm_agent.config import NaraApiConfig
from rfp_pm_agent.ingest.nara_client import (
    NaraApiClient,
    NaraApiError,
    NaraRequestError,
    has_attachment,
    is_sw_related,
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

    assert is_sw_related(items[0]) is True  # infoBizYn=Y + 정보통신/소프트웨어개발 분류
    assert is_sw_related(items[1]) is False  # 의상 제작 및 운영 용역
    assert is_sw_related(items[2]) is False  # 시설물 유지보수


def test_has_attachment_and_is_sw_related_filter_correctly() -> None:
    items = parse_search_response(_load_fixture())
    sw_with_attachment = [item for item in items if has_attachment(item) and is_sw_related(item)]

    assert len(sw_with_attachment) == 1
    assert sw_with_attachment[0].notice_no == "20260901001"


def test_search_service_bids_uses_configured_base_url_no_real_network() -> None:
    captured_urls: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(request.url)
        return httpx.Response(200, json=_load_fixture())

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    client = NaraApiClient(_config(), http_client=http_client)

    items = client.search_service_bids(begin_date="20260801", end_date="20260901")

    assert len(items) == 3
    assert str(captured_urls[0]).startswith("https://nara.test/")


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
        client.search_service_bids(begin_date="20260801", end_date="20260901")

    assert secret not in str(exc_info.value)
    assert secret not in caplog.text
    assert "***" in str(exc_info.value)
