"""Embedding client — 문장 목록을 벡터 목록으로 바꾼다.

TEI(Text Embeddings Inference)의 `/embed` HTTP API를 부른다. `config.embed_base_url`만
바꾸면 RunPod ↔ 로컬(4050/CPU) 전환이 된다 (docs/tech-stack.md 4절).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from rfp_pm_agent.config import ClientsConfig

# KURE-v1(#16에서 채택, ADR-0001)의 벡터 차원. OpenSearch 매핑
# (ingest/index_mapping.json의 knn_vector dimension)과 같아야 한다 — 단위 테스트가 대조한다.
EMBEDDING_DIM = 1024


# 검색 질의("query")와 색인할 문서("passage")를 구분한다. 접두어가 필요한 모델
# (multilingual-e5)은 종류마다 다른 접두어를 붙인다 — config의 embed_*_prefix
InputType = Literal["query", "passage"]


class EmbeddingClient(Protocol):
    def embed(
        self, texts: list[str], *, input_type: InputType = "passage"
    ) -> list[list[float]]: ...


@dataclass
class EmbedReport:
    """`embed_with_report`의 결과. truncated는 모델 최대 길이를 넘어 잘린 입력의 위치(0부터)다.

    `embed_truncate`가 꺼져 있으면 길이를 세지 않으므로 truncated는 항상 비어 있다
    (그때는 긴 입력이 잘리지 않고 서버가 422로 거절한다 — TEI 1.9.4 실측, docs/data-design.md 5절).
    """

    vectors: list[list[float]]
    truncated: list[int] = field(default_factory=list)


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

    def info(self) -> dict[str, Any]:
        """TEI `/info` — model_id, model_sha(revision), max_input_length, version 등."""
        response = self._http.get("/info")
        response.raise_for_status()
        info: dict[str, Any] = response.json()
        return info

    def embed(self, texts: list[str], *, input_type: InputType = "passage") -> list[list[float]]:
        """문장 목록을 같은 순서의 벡터 목록으로 바꾼다. 자세한 동작은 `embed_with_report`."""
        return self.embed_with_report(texts, input_type=input_type).vectors

    def embed_with_report(
        self, texts: list[str], *, input_type: InputType = "passage"
    ) -> EmbedReport:
        """문장 목록을 같은 순서의 벡터 목록으로 바꾸고, 잘린 입력의 위치를 함께 돌려준다.

        - 접두어: input_type에 맞는 config 접두어(`embed_query_prefix`·`embed_passage_prefix`)를
          각 문장 앞에 붙여 보낸다. 빈 문자열이면 그대로 보낸다.
        - 나눠 보내기: TEI는 요청 1건의 입력 수를 제한하므로(`tei_max_client_batch_size`,
          기본 32) 그 크기씩 나눠 순서대로 보내고 결과를 이어 붙인다.
        - 자르기: `embed_truncate`가 False면 `truncate: false`로 보낸다. 모델 최대 길이를
          넘는 입력은 서버가 422(`Input validation error: ... must have less than N tokens`)로
          거절하고 `httpx.HTTPStatusError`가 난다(2026-10-08 실측. 413은 요청 본문 크기 한도를 넘을 때다). True면
          먼저 `/tokenize`로 입력별 토큰 수(특수 토큰 포함)를 세어 `/info`의
          `max_input_length`를 넘는 입력의 위치를 기록한 뒤, `truncate: true`로 보내
          서버가 뒤쪽을 잘라 임베딩하게 한다.
        - 정규화: `normalize: true`를 명시해 보낸다(TEI 스펙의 기본값도 true지만 버전마다
          같은지 확인하지 않았으므로 명시한다).

        서버가 돌려준 벡터 수가 보낸 문장 수와 다르면 문장과 벡터의 대응이 어긋나므로
        `ValueError`를 낸다.
        """
        prefix = (
            self._config.embed_query_prefix
            if input_type == "query"
            else self._config.embed_passage_prefix
        )
        truncate = self._config.embed_truncate
        max_len = int(self.info()["max_input_length"]) if truncate and texts else 0
        batch_size = self._config.tei_max_client_batch_size
        vectors: list[list[float]] = []
        truncated: list[int] = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + text for text in texts[start : start + batch_size]]
            if truncate:
                truncated.extend(
                    start + i
                    for i, n_tokens in enumerate(self._count_tokens(batch))
                    if n_tokens > max_len
                )
            response = self._http.post(
                "/embed", json={"inputs": batch, "normalize": True, "truncate": truncate}
            )
            response.raise_for_status()
            batch_vectors: list[list[float]] = response.json()
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"TEI /embed 응답의 벡터 수({len(batch_vectors)})가 "
                    f"보낸 문장 수({len(batch)})와 다릅니다"
                )
            vectors.extend(batch_vectors)
        return EmbedReport(vectors=vectors, truncated=truncated)

    def _count_tokens(self, batch: list[str]) -> list[int]:
        """TEI `/tokenize`로 입력별 토큰 수(특수 토큰 포함)를 센다."""
        response = self._http.post("/tokenize", json={"inputs": batch, "add_special_tokens": True})
        response.raise_for_status()
        tokens: list[list[dict[str, Any]]] = response.json()
        if len(tokens) != len(batch):
            raise ValueError(
                f"TEI /tokenize 응답의 입력 수({len(tokens)})가 보낸 문장 수({len(batch)})와 다릅니다"
            )
        return [len(t) for t in tokens]
