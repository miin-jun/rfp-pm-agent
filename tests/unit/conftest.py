from typing import Any

import pytest

from rfp_pm_agent.config import ClientsConfig


def make_clients_config(**overrides: Any) -> ClientsConfig:
    defaults: dict[str, Any] = {
        "llm_base_url": "http://llm.test/v1",
        "llm_api_key": "test-key",
        "llm_model_dev": "test-llm-model",
        "llm_model_eval": "test-llm-model-eval",
        "llm_timeout_s": 5.0,
        "llm_price_input_per_1m": 0.0,
        "llm_price_output_per_1m": 0.0,
        "embed_base_url": "http://embed.test",
        "embed_api_key": None,
        "embed_model_id": "test-embed-model",
        "embed_timeout_s": 5.0,
        "rerank_base_url": "http://rerank.test",
        "rerank_api_key": None,
        "rerank_model_id": "test-rerank-model",
        "rerank_timeout_s": 5.0,
        "cost_log_path": "unused.jsonl",
    }
    defaults.update(overrides)
    return ClientsConfig(**defaults)


@pytest.fixture
def clients_config() -> ClientsConfig:
    return make_clients_config()
