import json
from pathlib import Path

import httpx
import pytest

from rfp_pm_agent.clients.cost import CostLogger
from rfp_pm_agent.clients.embedding import EMBEDDING_DIM, TEIEmbeddingClient
from rfp_pm_agent.clients.logged import LoggingEmbeddingClient
from tests.fakes.fake_embedding import FakeEmbeddingClient
from tests.unit.conftest import make_clients_config


def test_fake_embedding_client_is_deterministic_and_1024_dim() -> None:
    fake = FakeEmbeddingClient()

    first = fake.embed(["안녕하세요", "데이터 이관"])
    second = fake.embed(["안녕하세요", "데이터 이관"])

    assert first == second
    assert len(first) == 2
    assert all(len(vector) == EMBEDDING_DIM for vector in first)


def test_fake_embedding_client_different_text_different_vector() -> None:
    fake = FakeEmbeddingClient()

    a, b = fake.embed(["문서 A", "문서 B"])

    assert a != b


def test_tei_embedding_client_uses_configured_base_url_without_network() -> None:
    config = make_clients_config(embed_base_url="http://runpod.example:8080")

    client = TEIEmbeddingClient(config)

    assert client.base_url.rstrip("/") == "http://runpod.example:8080"


def test_logging_embedding_client_records_zero_cost(tmp_path: Path) -> None:
    config = make_clients_config(embed_model_id="kure-v1-test")
    log_path = tmp_path / "cost.jsonl"
    wrapped = LoggingEmbeddingClient(FakeEmbeddingClient(), config, CostLogger(log_path))

    vectors = wrapped.embed(["hello"])

    assert len(vectors[0]) == EMBEDDING_DIM
    logged = log_path.read_text(encoding="utf-8")
    assert '"client": "embedding"' in logged
    assert '"model": "kure-v1-test"' in logged
    assert '"cost_usd": 0.0' in logged


def _embed_handler(requests: list[list[str]]) -> httpx.MockTransport:
    """/embed 요청의 입력을 기록하고, 입력마다 [문장 길이, 0, ...] 벡터를 돌려준다."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        inputs: list[str] = json.loads(request.content)["inputs"]
        requests.append(inputs)
        return httpx.Response(
            200, json=[[float(len(t))] + [0.0] * (EMBEDDING_DIM - 1) for t in inputs]
        )

    return httpx.MockTransport(handle)


def test_tei_embedding_client_splits_requests_by_max_client_batch_size() -> None:
    config = make_clients_config(tei_max_client_batch_size=32)
    sent: list[list[str]] = []
    client = TEIEmbeddingClient(config, transport=_embed_handler(sent))
    texts = ["가" * (i + 1) for i in range(70)]

    vectors = client.embed(texts)

    assert [len(batch) for batch in sent] == [32, 32, 6]
    assert len(vectors) == 70
    # 묶음을 이어 붙인 결과가 입력 순서와 같아야 한다
    assert [v[0] for v in vectors] == [float(i + 1) for i in range(70)]


def test_tei_embedding_client_empty_input_sends_no_request() -> None:
    sent: list[list[str]] = []
    client = TEIEmbeddingClient(make_clients_config(), transport=_embed_handler(sent))

    assert client.embed([]) == []
    assert sent == []


def test_tei_embedding_client_raises_on_http_error() -> None:
    # TEI 1.9.4는 토큰 한도를 넘는 입력을 422로 거절한다(2026-10-08 실측)
    too_long = {
        "error": "Input validation error: `inputs` must have less than 8192 tokens. Given: 30003",
        "error_type": "Validation",
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(422, json=too_long))
    client = TEIEmbeddingClient(make_clients_config(), transport=transport)

    with pytest.raises(httpx.HTTPStatusError):
        client.embed(["너무 긴 문장"])


def test_tei_embedding_client_sends_bearer_token_when_api_key_set() -> None:
    seen: list[str | None] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json=[[0.0] * EMBEDDING_DIM])

    config = make_clients_config(embed_api_key="secret-token")
    TEIEmbeddingClient(config, transport=httpx.MockTransport(handle)).embed(["a"])

    assert seen == ["Bearer secret-token"]


@pytest.mark.parametrize(("status", "expected"), [(200, True), (503, False)])
def test_tei_embedding_client_health_reflects_status(status: int, expected: bool) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status))
    client = TEIEmbeddingClient(make_clients_config(), transport=transport)

    assert client.health() is expected


def test_tei_embedding_client_health_false_when_connection_fails() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = TEIEmbeddingClient(make_clients_config(), transport=httpx.MockTransport(handle))

    assert client.health() is False


@pytest.mark.parametrize(("n", "expected_batches"), [(32, [32]), (33, [32, 1])])
def test_tei_embedding_client_batch_boundaries(n: int, expected_batches: list[int]) -> None:
    sent: list[list[str]] = []
    client = TEIEmbeddingClient(make_clients_config(), transport=_embed_handler(sent))

    vectors = client.embed(["가" * (i + 1) for i in range(n)])

    assert [len(batch) for batch in sent] == expected_batches
    assert [v[0] for v in vectors] == [float(i + 1) for i in range(n)]


def test_tei_embedding_client_raises_when_vector_count_mismatches_batch() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[[0.0] * EMBEDDING_DIM])
    )
    client = TEIEmbeddingClient(make_clients_config(), transport=transport)

    with pytest.raises(ValueError, match="벡터 수"):
        client.embed(["문장 A", "문장 B"])


# --- 입력 종류(query/passage) 접두어와 자르기 (이슈 #16) ---


def _recording_transport(
    bodies: list[tuple[str, dict[str, object]]],
    *,
    max_input_length: int = 5,
) -> httpx.MockTransport:
    """경로별 요청 본문을 기록한다. /tokenize는 글자 1개를 토큰 1개로, /info는 최대 길이를 준다."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            bodies.append(("/info", {}))
            return httpx.Response(200, json={"model_id": "m", "max_input_length": max_input_length})
        body = json.loads(request.content)
        bodies.append((request.url.path, body))
        inputs: list[str] = body["inputs"]
        if request.url.path == "/tokenize":
            return httpx.Response(200, json=[[{"id": 0}] * len(t) for t in inputs])
        return httpx.Response(200, json=[[0.0] * EMBEDDING_DIM for _ in inputs])

    return httpx.MockTransport(handle)


def test_input_type에_맞는_접두어를_붙여_보낸다() -> None:
    config = make_clients_config(embed_query_prefix="query: ", embed_passage_prefix="passage: ")
    bodies: list[tuple[str, dict[str, object]]] = []
    client = TEIEmbeddingClient(config, transport=_recording_transport(bodies))

    client.embed(["사업 기간"], input_type="query")
    client.embed(["사업기간: 3개월"])  # 기본은 passage

    embeds = [b for path, b in bodies if path == "/embed"]
    assert embeds[0]["inputs"] == ["query: 사업 기간"]
    assert embeds[1]["inputs"] == ["passage: 사업기간: 3개월"]


def test_접두어가_비어_있으면_문장을_그대로_보낸다() -> None:
    bodies: list[tuple[str, dict[str, object]]] = []
    client = TEIEmbeddingClient(make_clients_config(), transport=_recording_transport(bodies))

    client.embed(["사업 기간"], input_type="query")

    assert bodies == [("/embed", {"inputs": ["사업 기간"], "normalize": True, "truncate": False})]


def test_자르기가_꺼져_있으면_길이를_세지_않고_truncate_false로_보낸다() -> None:
    bodies: list[tuple[str, dict[str, object]]] = []
    client = TEIEmbeddingClient(make_clients_config(), transport=_recording_transport(bodies))

    report = client.embed_with_report(["아주 긴 문장입니다"])

    assert [path for path, _ in bodies] == ["/embed"]
    assert report.truncated == []


def test_자르기가_켜져_있으면_최대_길이를_넘는_입력의_위치를_묶음을_넘어_기록한다() -> None:
    # 최대 5토큰, 묶음 크기 2. 글자 수 = 토큰 수 (가짜 /tokenize)
    config = make_clients_config(embed_truncate=True, tei_max_client_batch_size=2)
    bodies: list[tuple[str, dict[str, object]]] = []
    client = TEIEmbeddingClient(config, transport=_recording_transport(bodies))
    texts = ["가나다", "가나다라마바", "가나다라마", "가나다라마바사"]  # 3, 6, 5, 7 토큰

    report = client.embed_with_report(texts)

    assert report.truncated == [1, 3]  # 5는 넘지 않음, 두 번째 묶음의 위치는 이어서 센다
    assert len(report.vectors) == 4
    assert all(b["truncate"] is True for path, b in bodies if path == "/embed")


def test_자르기_판정은_접두어를_붙인_길이로_센다() -> None:
    config = make_clients_config(embed_truncate=True, embed_passage_prefix="passage: ")
    bodies: list[tuple[str, dict[str, object]]] = []
    client = TEIEmbeddingClient(config, transport=_recording_transport(bodies, max_input_length=10))

    report = client.embed_with_report(["가나"])  # "passage: 가나" = 11글자

    assert report.truncated == [0]


def test_자르기가_켜져_있어도_빈_입력이면_요청을_보내지_않는다() -> None:
    bodies: list[tuple[str, dict[str, object]]] = []
    client = TEIEmbeddingClient(
        make_clients_config(embed_truncate=True), transport=_recording_transport(bodies)
    )

    assert client.embed_with_report([]).vectors == []
    assert bodies == []
