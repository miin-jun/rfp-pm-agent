"""검색 서비스 오류 (이슈 #18 PR ③, docs/data-design.md 5절 "검색 툴 search_documents").

검색 서버(OpenSearch·TEI 임베딩·TEI 리랭커)에 연결할 수 없을 때 `search_documents`가 내는 예외다.
Agent(#22)·MCP 서버(#24)는 이 예외만 "검색 서버에 연결할 수 없다"는 툴 오류로 바꿔 LLM에 돌려주고,
그 밖의 예외는 버그로 보고 그대로 올린다. 원래 예외는 `raise ... from 원래예외`로 `__cause__`에 남긴다.
"""

from __future__ import annotations

from typing import Literal, get_args

SearchService = Literal["opensearch", "tei-embed", "tei-rerank"]


class SearchUnavailableError(Exception):
    """검색 서버 하나에 연결할 수 없다(연결 거절·응답 없음·5xx).

    - service: 연결할 수 없는 서버 — "opensearch" / "tei-embed" / "tei-rerank"
    - detail: 원래 오류 요약(사람이 읽는 문구)
    """

    def __init__(self, service: SearchService, detail: str) -> None:
        if service not in get_args(SearchService):
            raise ValueError(f"알 수 없는 검색 서버: {service}")
        self.service: SearchService = service
        self.detail = detail
        super().__init__(f"검색 서버({service})에 연결할 수 없다: {detail}")
