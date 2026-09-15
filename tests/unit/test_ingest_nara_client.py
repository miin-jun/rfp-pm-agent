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
    assert items[0].notice_no == "20260901001"
    assert items[0].spec_doc_urls == ["https://example.test/files/rfp-001.hwpx"]


def test_parse_search_response_raises_on_error_code() -> None:
    payload = {"response": {"header": {"resultCode": "99", "resultMsg": "APPLICATION ERROR"}}}

    with pytest.raises(NaraApiError):
        parse_search_response(payload)


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
