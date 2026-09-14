import pytest

from rfp_pm_agent.config import ClientsConfig


def test_from_env_reads_base_urls_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm.local:8000/v1")
    monkeypatch.setenv("LLM_MODEL_DEV", "gpt-test-dev")
    monkeypatch.setenv("EMBED_BASE_URL", "http://tei-embed.local:8080")
    monkeypatch.setenv("RERANK_BASE_URL", "http://tei-rerank.local:8081")

    config = ClientsConfig.from_env()

    assert config.llm_base_url == "http://vllm.local:8000/v1"
    assert config.llm_model_dev == "gpt-test-dev"
    assert config.embed_base_url == "http://tei-embed.local:8080"
    assert config.rerank_base_url == "http://tei-rerank.local:8081"


def test_from_env_defaults_prices_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PRICE_INPUT_PER_1M", raising=False)
    monkeypatch.delenv("LLM_PRICE_OUTPUT_PER_1M", raising=False)

    config = ClientsConfig.from_env()

    assert config.llm_price_input_per_1m == 0.0
    assert config.llm_price_output_per_1m == 0.0


def test_from_env_reads_per_1m_price(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenAI 요금 페이지 표기(100만 토큰당)를 그대로 옮겨 적는 값이라는 것을
    # 확인 — 1K로 착각해 1000배 오차가 나지 않는지가 핵심.
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_1M", "3.0")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_1M", "15.0")

    config = ClientsConfig.from_env()

    assert config.llm_price_input_per_1m == 3.0
    assert config.llm_price_output_per_1m == 15.0


def test_from_env_missing_api_key_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = ClientsConfig.from_env()

    assert config.llm_api_key is None
