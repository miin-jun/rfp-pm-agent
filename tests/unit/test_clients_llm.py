from pathlib import Path

import pytest

from rfp_pm_agent.clients.cost import CostLogger, compute_llm_cost
from rfp_pm_agent.clients.llm import ChatMessage, OpenAICompatLLMClient
from rfp_pm_agent.clients.logged import LoggingLLMClient
from rfp_pm_agent.config import ClientsConfig
from tests.fakes.fake_llm import FakeLLMClient
from tests.unit.conftest import make_clients_config


def test_fake_llm_client_is_deterministic() -> None:
    fake = FakeLLMClient()
    messages = [ChatMessage(role="user", content="hello there")]

    first = fake.chat(messages)
    second = fake.chat(messages)

    assert first == second
    assert first.content == "echo: hello there"
    assert first.input_tokens == 2
    assert first.output_tokens == 3


def test_openai_compat_client_uses_configured_base_url_without_network() -> None:
    config = make_clients_config(llm_base_url="http://vllm.local:9000/v1")

    client = OpenAICompatLLMClient(config)

    # base_url만 확인한다 — .chat()은 부르지 않으므로 네트워크 호출이 없다.
    assert client.base_url == "http://vllm.local:9000/v1/"


def test_env_var_change_flows_through_to_client_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://runpod-vllm.example:8000/v1")

    config = ClientsConfig.from_env()
    client = OpenAICompatLLMClient(config)

    assert client.base_url == "http://runpod-vllm.example:8000/v1/"


def test_logging_llm_client_records_cost_and_latency(tmp_path: Path) -> None:
    config = make_clients_config(llm_price_input_per_1m=1000.0, llm_price_output_per_1m=2000.0)
    log_path = tmp_path / "cost.jsonl"
    logger = CostLogger(log_path)
    wrapped = LoggingLLMClient(FakeLLMClient(), config, logger)

    response = wrapped.chat([ChatMessage(role="user", content="a b")])

    assert response.content == "echo: a b"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"client": "llm"' in lines[0]
    assert '"input_tokens": 2' in lines[0]
    # cost = 2/1_000_000*1000.0(입력) + 3/1_000_000*2000.0(출력) = 0.002 + 0.006 = 0.008
    assert '"cost_usd": 0.008' in lines[0]


def test_llm_price_defaults_to_zero_cost(tmp_path: Path) -> None:
    config: ClientsConfig = make_clients_config()
    log_path = tmp_path / "cost.jsonl"
    wrapped = LoggingLLMClient(FakeLLMClient(), config, CostLogger(log_path))

    wrapped.chat([ChatMessage(role="user", content="hi")])

    assert '"cost_usd": 0.0' in log_path.read_text(encoding="utf-8")


def test_compute_llm_cost_uses_per_1m_not_per_1k() -> None:
    # OpenAI 요금 페이지 표기 그대로: 입력 $3/1M, 출력 $15/1M 토큰 100/200개.
    # 1K로 착각해 나누면 1000배 큰(잘못된) 값이 나온다.
    config = make_clients_config(llm_price_input_per_1m=3.0, llm_price_output_per_1m=15.0)

    cost = compute_llm_cost(config, input_tokens=100, output_tokens=200)

    assert cost == pytest.approx(100 / 1_000_000 * 3.0 + 200 / 1_000_000 * 15.0)
    assert cost == pytest.approx(0.0033)
