"""청크 스키마 (이슈 #13) — 청킹 방식 비교 실험용.

방식마다 파일을 따로 쓰지만 레코드 모양은 하나로 맞춘다. 같은 검색·평가 코드가
세 방식을 그대로 받아 비교할 수 있어야 하기 때문이다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ChunkMethod = Literal["block", "requirement", "block_requirement"]


class Chunk(BaseModel):
    """색인 단위 한 조각.

    source_ids는 이 청크가 어느 원본 조각에서 나왔는지를 가리킨다
    (Block.block_id 또는 Requirement.requirement_id). 지금 세 방식은 모두
    원본 조각 1개당 청크 1개라 항상 길이 1이지만, 나중에 여러 블록을 합치는
    방식을 추가하면 그대로 늘어난다.
    """

    chunk_id: str  # "{doc_id}:{method}:{source_id}"
    doc_id: str
    method: ChunkMethod
    source_ids: list[str]
    text: str
