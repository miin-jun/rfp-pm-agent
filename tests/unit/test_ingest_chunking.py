"""청킹(이슈 #13) 단위 테스트 — 손으로 만든 Document로 돈다. 네트워크 없음.

세 방식의 청크 수가 원본 조각 수와 맞는지, 표 블록 텍스트 형식이 유지되는지,
모든 청크에 source_ids가 있는지를 고정한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfp_pm_agent.ingest import chunking
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Block, Document, Requirement

TABLE_ROWS: list[list[str | None]] = [
    ["구분", "배 점", "비    고"],
    ["기술평가", "정량적 평가", "20", "수요기관"],
    ["가격 평가", "10", "조달청"],
]
TABLE_TEXT = (
    "구분 | 배 점 | 비    고\n기술평가 | 정량적 평가 | 20 | 수요기관\n가격 평가 | 10 | 조달청"
)


def _block(block_id: str, text: str, order: int) -> Block:
    return Block(block_id=block_id, type="paragraph", text=text, source_order=order)


def _requirement(req_id: str, text: str, order: int) -> Requirement:
    return Requirement(
        requirement_id=req_id,
        prefix=req_id[:3],
        fields={"name": text},
        raw_fields={"요구사항명": text},
        text=text,
        source_order=order,
    )


def _document(
    doc_id: str = "aaaa111122223333",
    *,
    has_requirements: bool = True,
    blocks: list[Block] | None = None,
    requirements: list[Requirement] | None = None,
) -> Document:
    blocks = (
        blocks
        if blocks is not None
        else [
            _block("b0001", "사업 개요 문단이다.", 1),
            _block("b0002", "가", 2),  # 한 글자 블록
            Block(
                block_id="b0003",
                type="table",
                text=TABLE_TEXT,
                table=TABLE_ROWS,
                source_order=3,
            ),
        ]
    )
    requirements = (
        requirements
        if requirements is not None
        else [
            _requirement("SFR-001", "요구사항명: 로그인 기능", 4),
            _requirement("SFR-002", "요구사항명: 권한 관리", 5),
        ]
    )
    return Document(
        doc_id=doc_id,
        source_file=f"data/raw/api/{doc_id}.pdf",
        bid_title="테스트 사업",
        format="pdf",
        parse_status="parsed",
        has_requirements=has_requirements,
        requirement_count=len(requirements),
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=blocks,
        requirements=requirements,
    )


def test_block_chunk_count_matches_blocks() -> None:
    doc = _document()
    assert len(chunking.chunk_by_block(doc)) == len(doc.blocks)


def test_requirement_chunk_count_matches_requirements() -> None:
    doc = _document()
    assert len(chunking.chunk_by_requirement(doc)) == len(doc.requirements)


def test_block_requirement_is_sum_of_both() -> None:
    doc = _document()
    both = chunking.chunk_by_block_and_requirement(doc)
    assert len(both) == len(doc.blocks) + len(doc.requirements)
    texts = [c.text for c in both]
    assert texts == [b.text for b in doc.blocks] + [r.text for r in doc.requirements]
    assert {c.method for c in both} == {"block_requirement"}


def test_table_block_text_is_pipe_separated() -> None:
    """표 블록은 파서가 만든 `" | "` 형식을 글자까지 그대로 가져온다.

    평가 세트 qa_v1의 evidence가 이 형식이라, 여기서 다시 만들면 대조가 어긋난다.
    """
    doc = _document()
    table_block = next(b for b in doc.blocks if b.type == "table")
    chunk = next(c for c in chunking.chunk_by_block(doc) if c.source_ids == [table_block.block_id])
    assert chunk.text == table_block.text
    assert chunk.text.splitlines()[1] == "기술평가 | 정량적 평가 | 20 | 수요기관"


def test_every_chunk_has_source_ids() -> None:
    doc = _document()
    known = {b.block_id for b in doc.blocks} | {r.requirement_id for r in doc.requirements}
    for method in chunking.CHUNKERS:
        for chunk in chunking.chunk_document(doc, method):  # type: ignore[arg-type]
            assert chunk.source_ids
            assert set(chunk.source_ids) <= known


def test_short_and_empty_blocks_are_kept_as_their_own_chunks() -> None:
    """짧은 블록도 합치지 않고, text가 빈 블록도 버리지 않는다."""
    doc = _document(blocks=[_block("b0001", "가", 1), _block("b0002", "   ", 2)])
    chunks = chunking.chunk_by_block(doc)
    assert [c.text for c in chunks] == ["가", "   "]
    assert chunking.count_empty_blocks([doc]) == 1


def test_chunk_id_is_unique_within_method() -> None:
    doc = _document()
    for method in chunking.CHUNKERS:
        chunks = chunking.chunk_document(doc, method)  # type: ignore[arg-type]
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)
        assert all(c.chunk_id.startswith(f"{doc.doc_id}:{method}:") for c in chunks)


def test_load_target_documents_filters(tmp_path: Path) -> None:
    """has_requirements=false인 문서와 제외 doc_id를 걸러낸다."""
    for doc in (
        _document("0000aaaabbbbcccc"),
        _document("1111aaaabbbbcccc", has_requirements=False),
        _document("d8f1005fed9e9a2b"),
    ):
        (tmp_path / f"{doc.doc_id}.json").write_text(doc.model_dump_json(), encoding="utf-8")
    docs = chunking.load_target_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["0000aaaabbbbcccc"]


def test_write_chunks_roundtrip(tmp_path: Path) -> None:
    doc = _document()
    chunks = chunking.chunk_by_block(doc)
    path = tmp_path / "block.jsonl"
    assert chunking.write_chunks(chunks, path) == len(chunks)
    read = [
        Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert read == chunks


def test_run_writes_three_files(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    doc = _document()
    (parsed_dir / f"{doc.doc_id}.json").write_text(doc.model_dump_json(), encoding="utf-8")
    out_dir = tmp_path / "chunks"
    counts = chunking.run(parsed_dir=parsed_dir, out_dir=out_dir)
    assert counts == {
        "block": len(doc.blocks),
        "requirement": len(doc.requirements),
        "block_requirement": len(doc.blocks) + len(doc.requirements),
    }
    for method, n in counts.items():
        assert len((out_dir / f"{method}.jsonl").read_text(encoding="utf-8").splitlines()) == n


def test_chunk_document_rejects_unknown_method() -> None:
    with pytest.raises(KeyError):
        chunking.chunk_document(_document(), "sentence")  # type: ignore[arg-type]
