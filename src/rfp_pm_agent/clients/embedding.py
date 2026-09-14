"""Embedding client — 문장 목록을 벡터 목록으로 바꾼다.

TEI(Text Embeddings Inference)의 `/embed` HTTP API를 부른다. `config.embed_base_url`만
바꾸면 RunPod ↔ 로컬(4050/CPU) 전환이 된다 (docs/tech-stack.md 4절).
"""

from __future__ import annotations

from typing import Protocol

import httpx

from rfp_pm_agent.config import ClientsConfig

# KURE-v1(현재 기본 임베딩 모델) 기준. docs/data-design.md의 knn_vector 차원과
# 일치해야 한다. 이슈 #16에서 다른 모델로 바뀌면 이 값도 같이 바뀔 수 있다.
EMBEDDING_DIM = 1024


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class TEIEmbeddingClient:
    """TEI `/embed` 엔드포인트를 호출하는 구현체.

    `chat()`과 마찬가지로 실제 네트워크 호출은 `embed()`에서만 일어난다.
    단위 테스트는 `base_url` 프로퍼티만 확인해 네트워크 없이 배선을 검증한다.
    """

    def __init__(self, config: ClientsConfig) -> None:
        self._config = config
        headers = (
            {"Authorization": f"Bearer {config.embed_api_key}"} if config.embed_api_key else {}
        )
        self._http = httpx.Client(
            base_url=config.embed_base_url,
            timeout=config.embed_timeout_s,
            headers=headers,
        )

    @property
    def base_url(self) -> str:
        return str(self._http.base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._http.post("/embed", json={"inputs": texts})
        response.raise_for_status()
        vectors: list[list[float]] = response.json()
        return vectors
