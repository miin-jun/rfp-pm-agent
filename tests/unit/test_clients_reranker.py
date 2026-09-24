import json
from pathlib import Path

import httpx
import pytest

from rfp_pm_agent.clients.cost import CostLogger
from rfp_pm_agent.clients.logged import LoggingRerankerClient
from rfp_pm_agent.clients.reranker import TEIRerankerClient
from tests.fakes.fake_reranker import FakeRerankerClient
from tests.unit.conftest import make_clients_config


def test_fake_reranker_client_is_deterministic() -> None:
    fake = FakeRerankerClient()
    query = "데이터 이관 요구사항"
    documents = ["데이터 이관 계획서", "전혀 관련 없는 문서"]

    first = fake.rerank(query, documents)
    second = fake.rerank(query, documents)

    assert first == second
    assert len(first) == 2


def test_fake_reranker_client_ranks_overlapping_document_higher() -> None:
    fake = FakeRerankerClient()
    query = "데이터 이관 요구사항"
    documents = ["전혀 관련 없는 문서", "데이터 이관 요구사항 정의서"]

    scores = fake.rerank(query, documents)

    assert scores[1] > scores[0]


def test_tei_reranker_client_uses_configured_base_url_without_network() -> None:
    config = make_clients_config(rerank_base_url="http://runpod.example:8081")

    client = TEIRerankerClient(config)

    assert client.base_url.rstrip("/") == "http://runpod.example:8081"


def test_logging_reranker_client_records_zero_cost(tmp_path: Path) -> None:
    config = make_clients_config(rerank_model_id="bge-reranker-test")
    log_path = tmp_path / "cost.jsonl"
    wrapped = LoggingRerankerClient(FakeRerankerClient(), config, CostLogger(log_path))

    scores = wrapped.rerank("query", ["doc a", "doc b"])

    assert len(scores) == 2
    logged = log_path.read_text(encoding="utf-8")
    assert '"client": "reranker"' in logged
    assert '"model": "bge-reranker-test"' in logged
    assert '"cost_usd": 0.0' in logged


def _rerank_transport(sent: list[list[str]]) -> httpx.MockTransport:
    """실제 TEI처럼 묶음 안 위치(0부터)를 index로, 점수 내림차순으로 돌려준다.

    점수는 문서 문자열 "doc-<원래 위치>"의 숫자를 1000으로 나눈 값이라 원래
    위치를 점수로 역추적할 수 있다.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        texts: list[str] = json.loads(request.content)["texts"]
        sent.append(texts)
        results = [
            {"index": i, "score": int(t.removeprefix("doc-")) / 1000} for i, t in enumerate(texts)
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return httpx.Response(200, json=results)

    return httpx.MockTransport(handle)


def test_tei_reranker_client_splits_40_documents_and_restores_original_positions() -> None:
    config = make_clients_config(tei_max_client_batch_size=32)
    sent: list[list[str]] = []
    client = TEIRerankerClient(config, transport=_rerank_transport(sent))
    documents = [f"doc-{i}" for i in range(40)]

    scores = client.rerank("질의", documents)

    assert [len(batch) for batch in sent] == [32, 8]
    # 두 번째 묶음의 서버 index는 0~7이지만, 원래 위치 32~39로 되돌아가야 한다
    assert scores[32:] == [i / 1000 for i in range(32, 40)]
    assert scores == [i / 1000 for i in range(40)]


def test_tei_reranker_client_raises_when_response_index_mismatches_batch() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[{"index": 0, "score": 0.5}])
    )
    client = TEIRerankerClient(make_clients_config(), transport=transport)

    with pytest.raises(ValueError, match="index"):
        client.rerank("질의", ["문서 A", "문서 B"])


def test_tei_reranker_client_empty_documents_sends_no_request() -> None:
    sent: list[list[str]] = []
    client = TEIRerankerClient(make_clients_config(), transport=_rerank_transport(sent))

    assert client.rerank("질의", []) == []
    assert sent == []


@pytest.mark.parametrize(("status", "expected"), [(200, True), (503, False)])
def test_tei_reranker_client_health_reflects_status(status: int, expected: bool) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status))
    client = TEIRerankerClient(make_clients_config(), transport=transport)

    assert client.health() is expected


def test_tei_reranker_client_health_false_when_connection_fails() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = TEIRerankerClient(make_clients_config(), transport=httpx.MockTransport(handle))

    assert client.health() is False


@pytest.mark.parametrize(("n", "expected_batches"), [(32, [32]), (33, [32, 1])])
def test_tei_reranker_client_batch_boundaries(n: int, expected_batches: list[int]) -> None:
    sent: list[list[str]] = []
    client = TEIRerankerClient(make_clients_config(), transport=_rerank_transport(sent))

    scores = client.rerank("질의", [f"doc-{i}" for i in range(n)])

    assert [len(batch) for batch in sent] == expected_batches
    assert scores == [i / 1000 for i in range(n)]


def test_tei_reranker_client_sends_bearer_token_when_api_key_set() -> None:
    seen: list[str | None] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

    config = make_clients_config(rerank_api_key="secret-token")
    TEIRerankerClient(config, transport=httpx.MockTransport(handle)).rerank("질의", ["문서"])

    assert seen == ["Bearer secret-token"]
