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

    `chat()`과 마찬가지로 실제 네트워크 호출은 `embed()`·`health()`에서만 일어난다.
    단위 테스트는 `transport`에 `httpx.MockTransport`를 넣어 네트워크 없이 검증한다.
    """

    def __init__(
        self, config: ClientsConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._config = config
        headers = (
            {"Authorization": f"Bearer {config.embed_api_key}"} if config.embed_api_key else {}
        )
        self._http = httpx.Client(
            base_url=config.embed_base_url,
            timeout=config.embed_timeout_s,
            headers=headers,
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return str(self._http.base_url)

    def health(self) -> bool:
        """서버가 요청을 받을 준비가 됐으면 True.

        TEI `/health`는 모델 로드가 끝나면 200, 로드 중이거나 장애면 503을 준다.
        연결 자체가 안 되면(컨테이너가 꺼져 있음) 예외 대신 False를 돌려준다.
        """
        try:
            response = self._http.get("/health")
        except httpx.TransportError:
            return False
        return response.status_code == 200

    def embed(self, texts: list[str]) -> list[list[float]]:
        """문장 목록을 같은 순서의 벡터 목록으로 바꾼다.

        TEI는 요청 1건의 입력 수를 제한하므로(`tei_max_client_batch_size`, 기본 32)
        그 크기씩 나눠 순서대로 보내고 결과를 이어 붙인다. 모델 최대 길이를 넘는
        입력은 서버가 413으로 거절하고 `httpx.HTTPStatusError`가 난다 — 서버의
        `--auto-truncate`를 꺼 두었으므로 잘린 채 임베딩되는 일은 없다. 서버가 돌려준
        벡터 수가 보낸 문장 수와 다르면 문장과 벡터의 대응이 어긋나므로 `ValueError`를 낸다.
        """
        batch_size = self._config.tei_max_client_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._http.post("/embed", json={"inputs": batch})
            response.raise_for_status()
            batch_vectors: list[list[float]] = response.json()
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"TEI /embed 응답의 벡터 수({len(batch_vectors)})가 "
                    f"보낸 문장 수({len(batch)})와 다릅니다"
                )
            vectors.extend(batch_vectors)
        return vectors
