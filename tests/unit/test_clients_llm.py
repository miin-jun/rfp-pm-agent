from pathlib import Path

import pytest

from rfp_pm_agent.clients.cost import CostLogger
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
    config = make_clients_config(llm_price_input_per_1k=1.0, llm_price_output_per_1k=2.0)
    log_path = tmp_path / "cost.jsonl"
    logger = CostLogger(log_path)
    wrapped = LoggingLLMClient(FakeLLMClient(), config, logger)

    response = wrapped.chat([ChatMessage(role="user", content="a b")])

    assert response.content == "echo: a b"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"client": "llm"' in lines[0]
    assert '"input_tokens": 2' in lines[0]
    # cost = 2/1000*1.0(입력) + 3/1000*2.0(출력) = 0.002 + 0.006 = 0.008
    assert '"cost_usd": 0.008' in lines[0]


def test_llm_price_defaults_to_zero_cost(tmp_path: Path) -> None:
    config: ClientsConfig = make_clients_config()
    log_path = tmp_path / "cost.jsonl"
    wrapped = LoggingLLMClient(FakeLLMClient(), config, CostLogger(log_path))

    wrapped.chat([ChatMessage(role="user", content="hi")])

    assert '"cost_usd": 0.0' in log_path.read_text(encoding="utf-8")
