"""에이전트용 채팅 모델 생성과 비용 로그 콜백 (이슈 #22). 네트워크 없음 — 요청 본문만 만들어 본다."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from rfp_pm_agent.clients.chat_model import CostLogCallback, make_chat_model
from rfp_pm_agent.clients.cost import CostLogger
from tests.fakes.fake_chat_model import ScriptedChatModel, answer
from tests.unit.conftest import make_clients_config


def test_make_chat_model_uses_responses_api_with_configured_effort(tmp_path: Path) -> None:
    config = make_clients_config(llm_reasoning_effort="low")

    model = make_chat_model(config, CostLogger(tmp_path / "cost.jsonl"))
    payload = model._get_request_payload([HumanMessage("q")])

    assert model.model_name == "test-llm-model"
    assert model.openai_api_base == "http://llm.test/v1"
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    # Responses API 요청 본문은 messages가 아니라 input이다
    assert "input" in payload and "messages" not in payload


def test_make_chat_model_model_override(tmp_path: Path) -> None:
    model = make_chat_model(make_clients_config(), CostLogger(tmp_path / "c.jsonl"), model="other")

    assert model.model_name == "other"


def test_cost_log_callback_writes_one_line_per_call(tmp_path: Path) -> None:
    config = make_clients_config(llm_price_input_per_1m=0.10, llm_price_output_per_1m=0.50)
    log_path = tmp_path / "cost.jsonl"
    model = ScriptedChatModel(
        responses=[answer("a", input_tokens=1000, output_tokens=200)],
        callbacks=[CostLogCallback(config, CostLogger(log_path))],
    )

    model.invoke([HumanMessage("q")])

    [line] = log_path.read_text("utf-8").splitlines()
    entry = json.loads(line)
    assert entry["client"] == "llm"
    assert entry["model"] == "fake-chat"
    assert (entry["input_tokens"], entry["output_tokens"], entry["reasoning_tokens"]) == (
        1000,
        200,
        3,
    )
    assert entry["cost_usd"] == 1000 / 1e6 * 0.10 + 200 / 1e6 * 0.50
    assert entry["latency_ms"] >= 0
