"""LLM/임베딩/리랭커 클라이언트를 감싸 호출마다 비용 로그를 남기는 래퍼.

CLAUDE.md 구조: "모델 호출은 clients/에서만" — 실제 클라이언트를 직접 쓰지 않고
항상 이 래퍼로 감싸 써야 모든 호출이 빠짐없이 로그로 남는다.
"""

from __future__ import annotations

import time

from rfp_pm_agent.clients.cost import CostLogEntry, CostLogger, compute_llm_cost, now_iso
from rfp_pm_agent.clients.embedding import EmbeddingClient, InputType
from rfp_pm_agent.clients.llm import ChatMessage, LLMClient, LLMResponse
from rfp_pm_agent.clients.reranker import RerankerClient
from rfp_pm_agent.config import ClientsConfig


class LoggingLLMClient:
    def __init__(self, inner: LLMClient, config: ClientsConfig, cost_logger: CostLogger) -> None:
        self._inner = inner
        self._config = config
        self._cost_logger = cost_logger

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        start = time.monotonic()
        response = self._inner.chat(messages, model=model)
        latency_ms = (time.monotonic() - start) * 1000
        cost_usd = compute_llm_cost(
            self._config,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        self._cost_logger.log(
            CostLogEntry(
                ts=now_iso(),
                client="llm",
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )
        )
        return response


class LoggingEmbeddingClient:
    """임베딩은 자체 서빙(TEI)이라 과금이 없다 — cost_usd는 항상 0.0으로 기록한다."""

    def __init__(
        self, inner: EmbeddingClient, config: ClientsConfig, cost_logger: CostLogger
    ) -> None:
        self._inner = inner
        self._config = config
        self._cost_logger = cost_logger

    def embed(self, texts: list[str], *, input_type: InputType = "passage") -> list[list[float]]:
        start = time.monotonic()
        vectors = self._inner.embed(texts, input_type=input_type)
        latency_ms = (time.monotonic() - start) * 1000
        self._cost_logger.log(
            CostLogEntry(
                ts=now_iso(),
                client="embedding",
                model=self._config.embed_model_id,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
            )
        )
        return vectors


class LoggingRerankerClient:
    """리랭커도 자체 서빙(TEI)이라 과금이 없다 — cost_usd는 항상 0.0으로 기록한다."""

    def __init__(
        self, inner: RerankerClient, config: ClientsConfig, cost_logger: CostLogger
    ) -> None:
        self._inner = inner
        self._config = config
        self._cost_logger = cost_logger

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        start = time.monotonic()
        scores = self._inner.rerank(query, documents)
        latency_ms = (time.monotonic() - start) * 1000
        self._cost_logger.log(
            CostLogEntry(
                ts=now_iso(),
                client="reranker",
                model=self._config.rerank_model_id,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
            )
        )
        return scores
