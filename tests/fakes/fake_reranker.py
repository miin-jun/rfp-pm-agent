"""네트워크를 쓰지 않는 결정적 가짜 리랭커 클라이언트.

질의와 문서의 단어 겹침 비율로 점수를 매긴다 — 같은 입력이면 항상 같은 점수가
나오고, 겹치는 단어가 없어도 0으로 동률이 되지 않도록 문서 해시로 아주 작은
차이를 더한다. 응답 모양(질의+문서 목록 → 점수 목록)은 실제 TEI 클라이언트와 같다.
"""

from __future__ import annotations

import hashlib


class FakeRerankerClient:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        query_words = set(query.split())
        return [self._score(query_words, doc) for doc in documents]

    @staticmethod
    def _score(query_words: set[str], document: str) -> float:
        doc_words = set(document.split())
        overlap = len(query_words & doc_words) / max(len(query_words), 1)
        tie_breaker = (
            int(hashlib.sha256(document.encode("utf-8")).hexdigest(), 16) % 1000 / 1_000_000
        )
        return overlap + tie_breaker
