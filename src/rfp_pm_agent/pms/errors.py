"""PMS 조회 오류 (이슈 #21).

PMS DB(PostgreSQL)에 연결할 수 없을 때 조회 툴이 내는 예외다. `search.errors.SearchUnavailableError`와 같은 역할로,
Agent(#22)·MCP 서버(#24)는 이 예외만 "PMS DB에 연결할 수 없다"는 툴 오류로 바꿔 LLM에 돌려준다.
원래 예외는 `raise ... from 원래예외`로 `__cause__`에 남긴다.
"""

from __future__ import annotations


class PmsUnavailableError(Exception):
    """PMS DB에 연결할 수 없다(연결 거절·인증 실패·응답 없음). detail은 원래 오류 요약."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"PMS DB에 연결할 수 없다: {detail}")
