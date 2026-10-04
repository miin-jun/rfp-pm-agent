"""검색 요청을 기록하고 정해 둔 응답을 돌려주는 가짜 클라이언트 (이슈 #18).

`search/opensearch.py`의 검색 함수가 "어떤 body를 어느 이름으로 보냈는가"를 확인하는 용도다.
점수 계산·분석기·k-NN은 흉내 내지 않는다 — 그것은 실제 서버로만 확인한다
(tests/integration/test_opensearch_search.py).
"""

from __future__ import annotations

import copy
from typing import Any


class RecordingSearchClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response if response is not None else {"hits": {"hits": []}}
        self.calls: list[dict[str, Any]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"index": index, "body": copy.deepcopy(body)})
        return copy.deepcopy(self.response)
