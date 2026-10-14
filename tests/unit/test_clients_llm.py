import json
from pathlib import Path
from typing import Any

import pytest
from openai.types.chat import ChatCompletion

from rfp_pm_agent.clients.cost import CostLogger, compute_llm_cost
from rfp_pm_agent.clients.llm import ChatMessage, OpenAICompatLLMClient
from rfp_pm_agent.clients.logged import LoggingLLMClient
from rfp_pm_agent.config import ClientsConfig
from tests.fakes.fake_llm import FakeLLMClient, ScriptedLLMClient
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


def test_scripted_llm_client_returns_responses_in_order_and_records_calls() -> None:
    fake = ScriptedLLMClient(
        ["첫 답", "둘째 답"], input_tokens=7, output_tokens=5, reasoning_tokens=2
    )
    first_messages = [ChatMessage(role="user", content="질문 1")]
    second_messages = [ChatMessage(role="user", content="질문 2")]

    first = fake.chat(first_messages)
    second = fake.chat(second_messages, model="other-model")

    assert (first.content, first.model) == ("첫 답", "fake-llm")
    assert (second.content, second.model) == ("둘째 답", "other-model")
    assert (first.input_tokens, first.output_tokens, first.reasoning_tokens) == (7, 5, 2)
    assert fake.calls == [(first_messages, None), (second_messages, "other-model")]


def test_scripted_llm_client_fails_when_called_more_than_scripted() -> None:
    fake = ScriptedLLMClient([])

    with pytest.raises(AssertionError, match="정해 둔 응답 0개"):
        fake.chat([ChatMessage(role="user", content="질문")])
    assert len(fake.calls) == 1


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


def _completion_json(usage: dict[str, Any]) -> dict[str, Any]:
    """OpenAI `/chat/completions` 응답 본문(필수 필드만)."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-llm-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "답"},
            }
        ],
        "usage": usage,
    }


def _client_with_fake_create(
    monkeypatch: pytest.MonkeyPatch, usage: dict[str, Any], **config_overrides: Any
) -> tuple[OpenAICompatLLMClient, list[dict[str, Any]]]:
    """OpenAI SDK의 `chat.completions.create`를 가짜로 바꾼 클라이언트와, 받은 인자 목록.

    네트워크를 쓰지 않는다. 가짜는 받은 키워드 인자를 남기고 정해 둔 usage의 응답을 돌려준다.
    """
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return ChatCompletion.model_validate(_completion_json(usage))

    client = OpenAICompatLLMClient(make_clients_config(**config_overrides))
    monkeypatch.setattr(client._openai.chat.completions, "create", fake_create)
    return client, calls


def test_chat_sends_configured_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client_with_fake_create(
        monkeypatch,
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        llm_reasoning_effort="low",
    )

    client.chat([ChatMessage(role="user", content="hi")])

    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["model"] == "test-llm-model"


def test_chat_sends_reasoning_effort_changed_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client_with_fake_create(
        monkeypatch,
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        llm_reasoning_effort="none",
    )

    client.chat([ChatMessage(role="user", content="hi")])

    assert calls[0]["reasoning_effort"] == "none"


def test_chat_reads_reasoning_tokens_and_keeps_them_in_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenAI는 추론 토큰을 completion_tokens 안에 넣어 준다(답 30 + 추론 64 = 94)
    client, _ = _client_with_fake_create(
        monkeypatch,
        {
            "prompt_tokens": 20,
            "completion_tokens": 94,
            "total_tokens": 114,
            "completion_tokens_details": {"reasoning_tokens": 64},
        },
    )

    response = client.chat([ChatMessage(role="user", content="hi")])

    assert response.input_tokens == 20
    assert response.output_tokens == 94
    assert response.reasoning_tokens == 64


def test_chat_reasoning_tokens_default_to_zero_when_server_omits_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # vLLM·Ollama처럼 completion_tokens_details를 주지 않는 서버
    client, _ = _client_with_fake_create(
        monkeypatch, {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    )

    assert client.chat([ChatMessage(role="user", content="hi")]).reasoning_tokens == 0


def test_logging_llm_client_bills_reasoning_tokens_as_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_clients_config(llm_price_input_per_1m=0.10, llm_price_output_per_1m=0.50)
    client, _ = _client_with_fake_create(
        monkeypatch,
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "total_tokens": 2_000_000,
            "completion_tokens_details": {"reasoning_tokens": 900_000},
        },
    )
    log_path = tmp_path / "cost.jsonl"

    LoggingLLMClient(client, config, CostLogger(log_path)).chat(
        [ChatMessage(role="user", content="hi")]
    )

    [entry] = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    # 출력 100만 토큰 전부(추론 90만 포함)에 출력 단가를 매긴다: 0.10 + 0.50
    assert entry["cost_usd"] == pytest.approx(0.60)
    assert entry["output_tokens"] == 1_000_000
    assert entry["reasoning_tokens"] == 900_000
