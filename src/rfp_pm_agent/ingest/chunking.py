"""청킹 (이슈 #13) — 파싱 결과를 세 가지 방식으로 잘라 jsonl로 저장한다.

검색 품질이 청킹 방식에 얼마나 좌우되는지 보려고 세 방식을 모두 만들어 둔다.
- block: 파서가 만든 블록 하나가 청크 하나. 짧은 블록도 합치지 않는다
- requirement: 요구사항 하나의 text가 청크 하나
- block_requirement: 위 두 방식의 청크를 모두 넣는다. 중복은 걸러내지 않는다

표 블록의 텍스트는 파서가 이미 행마다 `" | "`로 이어 붙여 Block.text에 넣어
두었다(hwp_parser.py·pdf_parser.py). 여기서 다시 만들지 않고 그 값을 그대로 쓴다.
평가 세트 qa_v1의 evidence도 그 형식이라 형식을 바꾸면 대조가 어긋난다.

CLI: `uv run python -m rfp_pm_agent.ingest.chunking`
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from rfp_pm_agent.schemas.chunk import Chunk, ChunkMethod
from rfp_pm_agent.schemas.document import Document

DEFAULT_PARSED_DIR = Path("data/parsed")
DEFAULT_CHUNKS_DIR = Path("data/chunks")

# c6d225f1669cfadb와 같은 사업(2028 WDC 부산)의 문서가 중복 수집됨.
# 평가 세트(qa_v1)가 c6d225f1만 쓰므로 대상 문서를 맞추기 위해 제외
EXCLUDED_DOC_IDS = frozenset({"d8f1005fed9e9a2b"})

logger = logging.getLogger(__name__)


def _chunk_id(doc_id: str, method: ChunkMethod, source_id: str) -> str:
    return f"{doc_id}:{method}:{source_id}"


def has_meaningful_char(text: str) -> bool:
    """유니코드 L(문자)·N(숫자) 범주 글자가 하나라도 있으면 True. 청킹(#62)과 dense 사전 검사에서 쓴다."""
    return any(unicodedata.category(c)[0] in "LN" for c in text)


def chunk_by_block(doc: Document) -> list[Chunk]:
    """블록 하나를 청크 하나로 만든다. 의미 글자(L·N)가 없는 블록은 버린다(#62)."""
    return [
        Chunk(
            chunk_id=_chunk_id(doc.doc_id, "block", block.block_id),
            doc_id=doc.doc_id,
            method="block",
            source_ids=[block.block_id],
            text=block.text,
        )
        for block in doc.blocks
        if has_meaningful_char(block.text)
    ]


def chunk_by_requirement(doc: Document) -> list[Chunk]:
    """요구사항 하나의 text를 청크 하나로 만든다."""
    return [
        Chunk(
            chunk_id=_chunk_id(doc.doc_id, "requirement", req.requirement_id),
            doc_id=doc.doc_id,
            method="requirement",
            source_ids=[req.requirement_id],
            text=req.text,
        )
        for req in doc.requirements
    ]


def chunk_by_block_and_requirement(doc: Document) -> list[Chunk]:
    """블록 청크와 요구사항 청크를 모두 넣는다. 내용이 겹쳐도 걸러내지 않는다."""
    chunks: list[Chunk] = []
    for chunk in chunk_by_block(doc) + chunk_by_requirement(doc):
        chunks.append(
            chunk.model_copy(
                update={
                    "method": "block_requirement",
                    "chunk_id": _chunk_id(doc.doc_id, "block_requirement", chunk.source_ids[0]),
                }
            )
        )
    return chunks


CHUNKERS = {
    "block": chunk_by_block,
    "requirement": chunk_by_requirement,
    "block_requirement": chunk_by_block_and_requirement,
}


def chunk_document(doc: Document, method: ChunkMethod) -> list[Chunk]:
    """방식 이름으로 청킹 함수를 고른다."""
    return CHUNKERS[method](doc)


def load_target_documents(
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    *,
    exclude: frozenset[str] = EXCLUDED_DOC_IDS,
) -> list[Document]:
    """has_requirements=true이고 제외 목록에 없는 문서를 doc_id 순으로 읽는다."""
    docs = [
        Document.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(parsed_dir.glob("*.json"))
    ]
    return sorted(
        (d for d in docs if d.has_requirements and d.doc_id not in exclude),
        key=lambda d: d.doc_id,
    )


def write_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    """jsonl로 저장하고 쓴 개수를 돌려준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
            written += 1
    return written


def load_chunks(path: Path) -> list[Chunk]:
    """`write_chunks`가 쓴 청크 jsonl을 읽어 검증한다."""
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def count_dropped_blocks(docs: Iterable[Document]) -> int:
    """의미 글자(L·N)가 없어 청크에서 버려지는 블록 수 (#62)."""
    return sum(1 for doc in docs for block in doc.blocks if not has_meaningful_char(block.text))


def run(
    *,
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    out_dir: Path = DEFAULT_CHUNKS_DIR,
    exclude: frozenset[str] = EXCLUDED_DOC_IDS,
) -> dict[str, int]:
    """세 방식을 모두 실행해 파일 3개를 쓰고 방식별 청크 수를 돌려준다."""
    docs = load_target_documents(parsed_dir, exclude=exclude)
    counts: dict[str, int] = {}
    for method in CHUNKERS:
        chunks = [c for doc in docs for c in chunk_document(doc, method)]  # type: ignore[arg-type]
        counts[method] = write_chunks(chunks, out_dir / f"{method}.jsonl")
    return counts


def main() -> None:
    """`uv run python -m rfp_pm_agent.ingest.chunking` 진입점."""
    logging.basicConfig(level=logging.INFO)
    docs = load_target_documents()
    logger.info("대상 문서: %d건 (제외: %s)", len(docs), ", ".join(sorted(EXCLUDED_DOC_IDS)))
    logger.info(
        "원본 조각: 블록 %d개, 요구사항 %d개",
        sum(len(d.blocks) for d in docs),
        sum(len(d.requirements) for d in docs),
    )
    logger.info("의미 글자가 없어 버린 블록: %d개", count_dropped_blocks(docs))
    for method, n in run().items():
        logger.info("%s: %d청크 → %s", method, n, DEFAULT_CHUNKS_DIR / f"{method}.jsonl")


if __name__ == "__main__":
    main()
