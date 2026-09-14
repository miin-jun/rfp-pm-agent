from pathlib import Path

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
