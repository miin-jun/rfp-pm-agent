"""에이전트(#22)용 LangChain 채팅 모델과 비용 로그 콜백.

CLAUDE.md 구조: "모델 호출은 clients/에서만". 에이전트는 이 모듈의 `make_chat_model`로만 `ChatOpenAI`를 만든다.
호출마다 `CostLogCallback`이 `COST_LOG_PATH`에 한 줄 남긴다 — `LoggingLLMClient`(#9)와 같은 형식이다.

Responses API를 쓰는 이유(ADR-0010 결정 1): gpt-6-luna는 Chat Completions에서 함수 호출을 추론 강도 `none`일 때만
받는다(모델 페이지, 2026-10-16 확인). Responses API는 그런 제한이 없어 #19와 같은 추론 강도(`LLM_REASONING_EFFORT`,
기본 low)로 툴을 부를 수 있다.

- `store=False`: OpenAI 서버에 응답을 저장하지 않는다. 그래서 다음 호출에 이전 응답 ID를 쓸 수 없어 대화 전체를 매번 보낸다
- `include=["reasoning.encrypted_content"]`: 저장하지 않는 대신 추론 항목을 암호화해 받아 다음 호출에 돌려준다.
  빠지면 langchain-openai가 추론 항목을 버리고 보낸다(`_construct_responses_api_input`의 store=False 처리)
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rfp_pm_agent.clients.cost import CostLogEntry, CostLogger, compute_llm_cost, now_iso
from rfp_pm_agent.config import ClientsConfig


class CostLogCallback(BaseCallbackHandler):
    """채팅 모델 호출 1회마다 `CostLogEntry`를 남긴다. 토큰은 `AIMessage.usage_metadata`에서 읽는다.

    usage_metadata가 없으면(서버가 사용량을 주지 않음) 토큰 0·비용 0으로 남긴다 — 호출이 있었다는 기록은 남긴다.
    """

    def __init__(self, config: ClientsConfig, cost_logger: CostLogger) -> None:
        self._config = config
        self._cost_logger = cost_logger
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[run_id] = time.monotonic()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._started.pop(run_id, None)
        latency_ms = (time.monotonic() - start) * 1000 if start is not None else 0.0
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration) and isinstance(
                    generation.message, AIMessage
                ):
                    self._log(generation.message, latency_ms)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        # 실패한 호출은 `LoggingLLMClient`처럼 기록하지 않는다(사용량을 모른다). 시작 시각만 지운다
        self._started.pop(run_id, None)

    def _log(self, message: AIMessage, latency_ms: float) -> None:
        usage = message.usage_metadata
        input_tokens = usage["input_tokens"] if usage else 0
        output_tokens = usage["output_tokens"] if usage else 0
        details = usage.get("output_token_details", {}) if usage else {}
        self._cost_logger.log(
            CostLogEntry(
                ts=now_iso(),
                client="llm",
                model=str(message.response_metadata.get("model_name", "")),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=compute_llm_cost(
                    self._config, input_tokens=input_tokens, output_tokens=output_tokens
                ),
                latency_ms=latency_ms,
                reasoning_tokens=int(details.get("reasoning", 0) or 0),
            )
        )


def make_chat_model(
    config: ClientsConfig, cost_logger: CostLogger, *, model: str | None = None
) -> ChatOpenAI:
    """Responses API·추론 강도 `LLM_REASONING_EFFORT`로 부르는 `ChatOpenAI`. 호출마다 비용 로그를 남긴다.

    model을 주지 않으면 `LLM_MODEL_DEV`. base_url·키·타임아웃은 `ClientsConfig`(환경변수)에서 읽는다.
    서버에 연결하지는 않는다(첫 호출 때 연결).
    """
    return ChatOpenAI(
        model=model or config.llm_model_dev,
        base_url=config.llm_base_url,
        api_key=SecretStr(config.llm_api_key or "not-needed"),
        timeout=config.llm_timeout_s,
        use_responses_api=True,
        reasoning={"effort": config.llm_reasoning_effort},
        store=False,
        include=["reasoning.encrypted_content"],
        callbacks=[CostLogCallback(config, cost_logger)],
    )
