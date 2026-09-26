"""네트워크를 쓰지 않는 결정적 가짜 임베딩 클라이언트.

문자열 해시로 시드를 만들어 1024차원 벡터를 생성한다 — 같은 문장이면 항상 같은
벡터가 나온다. 응답 모양(문장 목록 → 벡터 목록)은 실제 TEI 클라이언트와 같다.
"""

from __future__ import annotations

import hashlib
import random

from rfp_pm_agent.clients.embedding import EMBEDDING_DIM, InputType


class FakeEmbeddingClient:
    """input_type은 받기만 하고 벡터에 반영하지 않는다(접두어 없는 모델처럼 동작)."""

    def embed(self, texts: list[str], *, input_type: InputType = "passage") -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]
