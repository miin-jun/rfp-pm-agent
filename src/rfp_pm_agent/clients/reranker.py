"""Reranker client — 질의 + 문서 목록을 점수 목록으로 바꾼다.

TEI의 `/rerank` HTTP API를 부른다. `config.rerank_base_url`만 바꾸면 RunPod ↔
로컬 전환이 된다 (docs/tech-stack.md 4절).
"""

from __future__ import annotations

from typing import Protocol

import httpx

from rfp_pm_agent.config import ClientsConfig


class RerankerClient(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class TEIRerankerClient:
    """TEI `/rerank` 엔드포인트를 호출하는 구현체.

    실제 네트워크 호출은 `rerank()`에서만 일어난다. 단위 테스트는 `base_url`
    프로퍼티만 확인해 네트워크 없이 배선을 검증한다.
    """

    def __init__(self, config: ClientsConfig) -> None:
        self._config = config
        headers = (
            {"Authorization": f"Bearer {config.rerank_api_key}"} if config.rerank_api_key else {}
        )
        self._http = httpx.Client(
            base_url=config.rerank_base_url,
            timeout=config.rerank_timeout_s,
            headers=headers,
        )

    @property
    def base_url(self) -> str:
        return str(self._http.base_url)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        response = self._http.post("/rerank", json={"query": query, "texts": documents})
        response.raise_for_status()
        results: list[dict[str, float]] = response.json()
        by_index = {int(r["index"]): float(r["score"]) for r in results}
        return [by_index[i] for i in range(len(documents))]
