"""OpenSearch 증분 색인 단위 테스트 (이슈 #17).

OpenSearch는 `tests/fakes/fake_opensearch.py`, TEI는 `tests/fakes/fake_tei.py`(실제
`TEIEmbeddingClient` + MockTransport)로 대신한다. "TEI 호출 건수"는 TEI `/embed`가 받은
입력 수다(청크 5개는 요청 1건에 담겨 가지만 입력 5건으로 센다).

기대값은 이슈 #17 "결정 (2026-10-01)" 기준으로 고정했다. 구현 결과에 맞춰 바꾸지 않는다.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from opensearchpy import OpenSearch

from rfp_pm_agent.clients.embedding import EMBEDDING_DIM
from rfp_pm_agent.config import ClientsConfig
from rfp_pm_agent.ingest.index_chunks import (
    IndexingError,
    IndexReport,
    MassDeleteError,
    build_arg_parser,
    content_hash,
    embed_input,
    format_report,
    index_chunks,
    load_mapping,
)
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Block, Document, Requirement
from tests.fakes.fake_opensearch import FakeOpenSearch
from tests.fakes.fake_tei import FakeTEIServer
from tests.unit.conftest import make_clients_config

INDEX = "rfp_chunks_v1_test"
ALIAS = "rfp_chunks"
DOC_ID = "d0000000000000001"
FIXED_NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _document() -> Document:
    return Document(
        doc_id=DOC_ID,
        source_file="sample.pdf",
        bid_title="샘플 사업",
        format="pdf",
        parse_status="parsed",
        has_requirements=True,
        requirement_count=2,
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=[
            Block(block_id="b0001", type="paragraph", text="사업 개요", source_order=0, pdf_page=0),
            Block(block_id="b0002", type="table", text="구분 | 내용", source_order=1, pdf_page=1),
            Block(block_id="b0003", type="paragraph", text="사업 기간", source_order=2, pdf_page=2),
        ],
        requirements=[
            Requirement(
                requirement_id="SFR-001",
                prefix="SFR",
                fields={},
                raw_fields={},
                text="요구사항 고유번호 SFR-001 로그인",
                source_order=3,
                pdf_page_start=3,
            ),
            Requirement(
                requirement_id="SFR-002",
                prefix="SFR",
                fields={},
                raw_fields={},
                text="요구사항 고유번호 SFR-002 검색",
                source_order=4,
                pdf_page_start=4,
            ),
        ],
    )


def _chunk(source_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{DOC_ID}:block_requirement:{source_id}",
        doc_id=DOC_ID,
        method="block_requirement",
        source_ids=[source_id],
        text=text,
    )


def _chunks() -> list[Chunk]:
    doc = _document()
    return [_chunk(b.block_id, b.text) for b in doc.blocks] + [
        _chunk(r.requirement_id, r.text) for r in doc.requirements
    ]


class Env:
    """가짜 OpenSearch·TEI와 설정을 묶어, 같은 인덱스에 여러 번 실행할 수 있게 한다."""

    def __init__(self, config: ClientsConfig | None = None) -> None:
        self.os = FakeOpenSearch()
        self.tei = FakeTEIServer()
        self.config = config or make_clients_config()
        self.documents = {DOC_ID: _document()}

    def run(self, chunks: list[Chunk], **kwargs: Any) -> IndexReport:
        self.tei.reset_counts()
        return index_chunks(
            chunks,
            self.documents,
            client=cast(OpenSearch, self.os),
            embedder=self.tei.client(self.config),
            config=self.config,
            index_name=INDEX,
            alias=ALIAS,
            revision_reader=_no_volume_read,
            now=lambda: FIXED_NOW,
            **kwargs,
        )


def _no_volume_read(model_id: str) -> str:
    raise AssertionError("가짜 TEI는 model_sha를 주므로 볼륨을 읽으면 안 된다")


@pytest.fixture
def env() -> Env:
    return Env()


# --- 1. 빈 인덱스에 청크 5개 → create 5, TEI 호출 5건, _count 5 ---


def test_empty_index_creates_all(env: Env) -> None:
    report = env.run(_chunks())

    assert report.created == 5
    assert report.updated_content == report.updated_model == report.skipped == 0
    assert report.deleted == 0
    assert len(env.tei.embed_inputs) == 5
    assert report.embed_inputs == 5
    assert report.final_count == 5
    assert report.index_created is True
    assert env.os.aliases[ALIAS] == {INDEX}


def test_created_documents_have_decided_fields(env: Env) -> None:
    env.run(_chunks())
    docs = env.os.docs(ALIAS)

    block = docs[f"{DOC_ID}:block_requirement:b0002"]
    assert set(block) == {
        "chunk_id",
        "doc_id",
        "method",
        "source_ids",
        "text",
        "requirement_id",
        "block_type",
        "page",
        "embedding",
        "embedding_model",
        "content_hash",
        "indexed_at",
    }
    assert block["block_type"] == "table"
    assert block["requirement_id"] is None
    assert block["page"] == 1
    assert len(block["embedding"]) == EMBEDDING_DIM
    assert block["embedding_model"] == "test-embed-model@rev-1"
    assert block["indexed_at"] == FIXED_NOW.isoformat()

    req = docs[f"{DOC_ID}:block_requirement:SFR-001"]
    assert req["requirement_id"] == "SFR-001"
    assert req["block_type"] is None
    assert req["page"] == 3


def test_content_hash_is_sha256_of_string_sent_to_tei() -> None:
    env = Env(make_clients_config(embed_passage_prefix="passage: "))
    env.run(_chunks())

    sent = set(env.tei.embed_inputs)
    for doc in env.os.docs(ALIAS).values():
        expected = "passage: " + doc["text"]
        assert expected in sent
        assert doc["content_hash"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
        assert len(doc["content_hash"]) == 64


# --- 2. 같은 입력으로 재실행 → skip 5, TEI 호출 0건 ---


def test_rerun_with_same_input_skips_all(env: Env) -> None:
    env.run(_chunks())
    report = env.run(_chunks())

    assert report.skipped == 5
    assert report.created == report.updated_content == report.updated_model == 0
    assert report.deleted == 0
    assert len(env.tei.embed_inputs) == 0
    assert report.final_count == 5


# --- 3. 청크 1개 text 변경 → update 1(content), skip 4, TEI 호출 1건 ---


def test_changed_text_updates_one_for_content(env: Env) -> None:
    env.run(_chunks())
    chunks = _chunks()
    chunks[0] = chunks[0].model_copy(update={"text": "사업 개요 (수정)"})

    report = env.run(chunks)

    assert report.updated_content == 1
    assert report.updated_model == 0
    assert report.skipped == 4
    assert len(env.tei.embed_inputs) == 1
    assert env.os.docs(ALIAS)[chunks[0].chunk_id]["text"] == "사업 개요 (수정)"


# --- 4. embedding_model만 변경 → update 5(model), TEI 호출 5건 ---


def test_changed_embedding_model_updates_all_for_model(env: Env) -> None:
    env.run(_chunks())
    env.tei.model_sha = "rev-2"

    report = env.run(_chunks())

    assert report.updated_model == 5
    assert report.updated_content == 0
    assert report.skipped == 0
    assert len(env.tei.embed_inputs) == 5
    assert {d["embedding_model"] for d in env.os.docs(ALIAS).values()} == {"test-embed-model@rev-2"}


# --- 5. 파일에서 청크 1개 제거 → delete 1, 인덱스 4개 ---


def test_removed_chunk_is_deleted(env: Env) -> None:
    env.run(_chunks())
    chunks = _chunks()
    removed = chunks.pop()

    # 문서 5개 중 1개(20%)라 5% 안전장치에 걸린다. 삭제 동작 자체를 보는 테스트라
    # 허용 플래그를 켠다 — 안전장치는 테스트 7·8이 검증한다 (소유자 결정, 2026-10-01)
    report = env.run(chunks, allow_mass_delete=True)

    assert report.deleted == 1
    assert report.final_count == 4
    assert removed.chunk_id not in env.os.docs(ALIAS)
    assert len(env.tei.embed_inputs) == 0


# --- 6. --dry-run → 개수만 출력, 인덱스 변경 0, TEI 호출 0 ---


def test_dry_run_reports_counts_without_changes(env: Env) -> None:
    env.run(_chunks())
    chunks = _chunks()
    chunks[0] = chunks[0].model_copy(update={"text": "사업 개요 (수정)"})
    removed = chunks.pop()
    before = env.os.snapshot()
    bulk_before = env.os.bulk_calls

    report = env.run(chunks, dry_run=True)

    assert report.dry_run is True
    assert report.updated_content == 1
    assert report.skipped == 3
    assert report.deleted == 1
    assert report.delete_ids == [removed.chunk_id]
    assert env.os.snapshot() == before
    assert env.os.bulk_calls == bulk_before
    assert len(env.tei.embed_inputs) == 0
    summary = format_report(report)
    assert "dry-run" in summary
    assert removed.chunk_id in summary


def test_dry_run_on_missing_index_does_not_create_it(env: Env) -> None:
    report = env.run(_chunks(), dry_run=True)

    assert report.created == 5
    assert env.os.indexes == {}
    assert env.os.aliases == {}
    assert len(env.tei.embed_inputs) == 0


# --- 7. 삭제 대상이 5% 초과 → 오류로 중단, 인덱스 변경 0 ---


def test_mass_delete_over_threshold_aborts_without_changes(env: Env) -> None:
    env.run(_chunks())
    chunks = _chunks()
    chunks[0] = chunks[0].model_copy(update={"text": "사업 개요 (수정)"})
    chunks.pop()
    before = env.os.snapshot()

    with pytest.raises(MassDeleteError):
        env.run(chunks)

    assert env.os.snapshot() == before
    assert len(env.tei.embed_inputs) == 0


# --- 8. 7과 같은 조건에 --allow-mass-delete → 삭제 수행 ---


def test_mass_delete_allowed_with_flag(env: Env) -> None:
    env.run(_chunks())
    chunks = _chunks()
    chunks[0] = chunks[0].model_copy(update={"text": "사업 개요 (수정)"})
    chunks.pop()

    report = env.run(chunks, allow_mass_delete=True)

    assert report.deleted == 1
    assert report.updated_content == 1
    assert report.final_count == 4


# --- 9. 의미 글자 없는 청크 포함 → ValueError, 메시지에 chunk_id 포함 ---


def test_chunk_without_meaningful_char_raises_before_embedding(env: Env) -> None:
    chunks = _chunks()
    bad = chunks[1].model_copy(update={"text": " | "})
    chunks[1] = bad

    with pytest.raises(ValueError, match=bad.chunk_id):
        env.run(chunks)

    assert len(env.tei.embed_inputs) == 0
    assert env.os.indexes == {}


# --- 10. bulk에서 create가 "updated"로 응답 → 오류 ---


def test_create_answered_as_updated_is_error(env: Env) -> None:
    env.run(_chunks())
    hidden = _chunks()[0].chunk_id
    env.os.hidden_ids.add(hidden)  # 저장돼 있지만 검색에 안 보임 → create로 판정됨

    with pytest.raises(IndexingError, match=hidden):
        env.run(_chunks())


# --- 11. 같은 text라도 접두어가 다르면 content_hash가 다르다 ---


def test_content_hash_differs_by_prefix() -> None:
    text = "사업 개요"

    plain = content_hash(embed_input(text, ""))
    prefixed = content_hash(embed_input(text, "passage: "))

    assert plain != prefixed
    assert plain == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert prefixed == hashlib.sha256(f"passage: {text}".encode()).hexdigest()


# --- 보조: 매핑·인덱스 준비·revision ---


def test_mapping_matches_decision() -> None:
    mapping = load_mapping()
    props = mapping["mappings"]["properties"]

    assert set(props) == {
        "chunk_id",
        "doc_id",
        "method",
        "source_ids",
        "text",
        "requirement_id",
        "block_type",
        "page",
        "embedding",
        "embedding_model",
        "content_hash",
        "indexed_at",
    }
    assert props["text"] == {"type": "text", "analyzer": "korean"}
    embedding = props["embedding"]
    assert embedding["dimension"] == EMBEDDING_DIM
    assert embedding["method"] == {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil",
        "parameters": {"m": 16, "ef_construction": 100},
    }
    assert mapping["settings"]["index"]["knn"] is True
    tokenizer = mapping["settings"]["analysis"]["tokenizer"]["ko_tokenizer"]
    assert tokenizer == {"type": "nori_tokenizer", "decompound_mode": "mixed"}


def test_existing_index_mapping_is_not_changed(env: Env) -> None:
    env.os.indices.create(index=INDEX, body={"mappings": {"properties": {"marker": {}}}})

    env.run(_chunks())

    assert env.os.indexes[INDEX]["mappings"] == {"properties": {"marker": {}}}
    assert env.os.create_calls == 1
    assert env.os.aliases[ALIAS] == {INDEX}


def test_alias_pointing_to_other_index_is_left_alone(env: Env) -> None:
    env.os.indices.create(index="rfp_chunks_v1_other", body={"aliases": {ALIAS: {}}})

    report = env.run(_chunks())

    assert report.index_created is True
    assert env.os.aliases[ALIAS] == {"rfp_chunks_v1_other"}
    assert len(env.os.docs(INDEX)) == 5


def test_revision_read_from_volume_when_info_has_no_sha() -> None:
    env = Env()
    env.tei.model_sha = None
    seen: list[str] = []

    def reader(model_id: str) -> str:
        seen.append(model_id)
        return "snap-1"

    env.tei.reset_counts()
    index_chunks(
        _chunks(),
        env.documents,
        client=cast(OpenSearch, env.os),
        embedder=env.tei.client(env.config),
        config=env.config,
        index_name=INDEX,
        alias=ALIAS,
        revision_reader=reader,
        now=lambda: FIXED_NOW,
    )

    assert seen == ["test-embed-model"]
    assert {d["embedding_model"] for d in env.os.docs(INDEX).values()} == {
        "test-embed-model@snap-1"
    }


def test_tei_model_different_from_config_is_rejected(env: Env) -> None:
    env.tei.model_id = "other/model"

    with pytest.raises(ValueError, match="other/model"):
        env.run(_chunks())

    assert env.os.indexes == {}


def test_cli_help_renders() -> None:
    # help 문자열의 "5%"를 argparse가 서식 문자로 읽어 --help가 ValueError로 죽던 버그
    help_text = build_arg_parser().format_help()

    assert "--allow-mass-delete" in help_text
    assert "5%" in help_text
