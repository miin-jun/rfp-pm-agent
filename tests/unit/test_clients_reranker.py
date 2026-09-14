from pathlib import Path

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
