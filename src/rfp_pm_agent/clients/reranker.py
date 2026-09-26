"""Reranker client — 질의 + 문서 목록을 점수 목록으로 바꾼다.

TEI의 `/rerank` HTTP API를 부른다. `config.rerank_base_url`만 바꾸면 RunPod ↔
로컬 전환이 된다 (docs/tech-stack.md 4절).
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from rfp_pm_agent.config import ClientsConfig


class RerankerClient(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class TEIRerankerClient:
    """TEI `/rerank` 엔드포인트를 호출하는 구현체.

    실제 네트워크 호출은 `rerank()`·`health()`에서만 일어난다. 단위 테스트는
    `transport`에 `httpx.MockTransport`를 넣어 네트워크 없이 검증한다.
    """

    def __init__(
        self, config: ClientsConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._config = config
        headers = (
            {"Authorization": f"Bearer {config.rerank_api_key}"} if config.rerank_api_key else {}
        )
        self._http = httpx.Client(
            base_url=config.rerank_base_url,
            timeout=config.rerank_timeout_s,
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
        """TEI `/info` — model_id, model_sha(revision), version 등."""
        response = self._http.get("/info")
        response.raise_for_status()
        info: dict[str, Any] = response.json()
        return info

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """문서마다 질의와의 관련도 점수를 매겨 입력과 같은 순서로 돌려준다.

        TEI는 요청 1건의 문서 수를 제한하므로(`tei_max_client_batch_size`, 기본 32)
        그 크기씩 나눠 보낸다. 크로스인코더 점수는 문서마다 따로 계산되므로 나눠
        보내도 점수는 같다. TEI는 묶음마다 점수 내림차순으로 정렬한
        `[{"index", "score"}]`를 주고, 이 index는 **그 묶음 안에서의 위치**(0부터)다.
        그래서 묶음 시작 위치를 더해 원래 문서 순서로 되돌린다.
        """
        batch_size = self._config.tei_max_client_batch_size
        scores: list[float] = [0.0] * len(documents)
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            response = self._http.post("/rerank", json={"query": query, "texts": batch})
            response.raise_for_status()
            results: list[dict[str, float]] = response.json()
            if sorted(int(r["index"]) for r in results) != list(range(len(batch))):
                raise ValueError(
                    f"TEI /rerank 응답의 index가 묶음 크기({len(batch)})와 맞지 않습니다: "
                    f"{sorted(int(r['index']) for r in results)}"
                )
            for r in results:
                scores[start + int(r["index"])] = float(r["score"])
        return scores
