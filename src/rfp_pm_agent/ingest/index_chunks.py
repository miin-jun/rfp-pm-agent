"""OpenSearch 증분 색인 (이슈 #17) — 청크 파일을 모델별 인덱스에 넣고, 바뀐 것만 다시 임베딩한다.

처리 순서
1. 인덱스 이름 검사: 비어 있거나, 별칭과 같거나, 이미 별칭으로 쓰이는 이름이면 ValueError.
   OpenSearch는 별칭으로도 읽고 쓸 수 있어서, 막지 않으면 다른 모델 인덱스를 덮어쓴다 (#17 리뷰 2번)
2. 청크 검사: 의미 글자(L·N)가 없는 청크가 있으면 chunk_id를 담아 ValueError (#62, dense.py와
   같은 기준). chunk_id 중복도 ValueError — 문서 `_id`로 쓰기 때문이다(#65)
3. 청크마다 content_hash 계산 = sha256(TEI `/embed`에 실제로 보내는 문자열). 그 문자열은
   `EMBED_PASSAGE_PREFIX + text`다 — `TEIEmbeddingClient`가 붙이는 접두어와 같은 설정값을 쓴다
4. 청크마다 metadata_hash 계산 = 임베딩하지 않는 필드(`METADATA_FIELDS`)의 sha256.
   직렬화 규칙은 `metadata_hash` 설명 참고
5. embedding_model = "{모델ID}@{revision}". TEI `/info`의 model_sha가 있으면 그 값, null이면
   (TEI 1.9.4 기본) #16과 같은 방법으로 TEI 모델 볼륨의 snapshots/를 읽는다
6. 인덱스가 있으면 chunk_id·content_hash·embedding_model·metadata_hash만 읽어 판정한다. 위에서부터
   처음 맞는 것 하나로 정한다(content > model > metadata)
   - 인덱스에 없음 → create
   - content_hash 다름 → update(content): 다시 임베딩하고 문서 전체를 다시 쓴다
   - embedding_model 다름 → update(model): 위와 같음
   - metadata_hash만 다름 → update(metadata): 벡터는 그대로 두고 메타데이터 필드와
     metadata_hash만 부분 update한다. TEI를 부르지 않는다. indexed_at도 그대로 둔다
   - 셋 다 같음 → skip / 인덱스에만 있음 → delete
7. 삭제 대상이 인덱스 문서 수의 5%를 넘으면 MassDeleteError (`--allow-mass-delete`로만 허용).
   인덱스 생성·별칭 연결·임베딩·적재보다 먼저 하므로, 걸리면 인덱스와 별칭은 바뀌지 않는다
8. 인덱스가 없으면 만들고(매핑은 `index_mapping.json`), 별칭이 아직 없을 때만 연결한다.
   이미 있는 인덱스의 매핑은 바꾸지 않는다
9. create·update(content·model) 대상만 `TEI_MAX_CLIENT_BATCH_SIZE`(기본 32)개씩 임베딩해 bulk로
   적재하고, update(metadata)를 부분 update로 보낸 뒤, stale 문서를 지운다.
   create 대상은 bulk `create`로 보낸다 — 같은 _id가 이미 있으면(검색에는 안 보였던 문서)
   서버가 409로 거절하고 기존 문서는 그대로 남는다. update(content·model)는 bulk `index`다
10. bulk 응답은 배치마다 항목별로 확인한다. 오류(409 포함)나 예상과 다른 결과가 하나라도 있으면
    IndexingError로 멈춘다 — 다음 배치의 임베딩·쓰기와 이후 단계(메타데이터 갱신·삭제)는 하지 않는다

읽기·쓰기는 별칭이 아니라 실제 인덱스 이름(`OPENSEARCH_INDEX_NAME`)으로 한다. 이 모듈은 모델별
인덱스를 만드는 쪽이고, 별칭이 다른 모델 인덱스를 가리키는 동안 새 인덱스를 채울 수 있어야
하기 때문이다. 검색 코드는 별칭만 쓴다(docs/data-design.md 5절).

`indexed_at`은 실제 색인 시각(UTC)이다. 합성 세계의 기준일(`AS_OF_DATE`)과는 관계없다.

CLI: `uv run python -m rfp_pm_agent.ingest.index_chunks --dry-run`
     `uv run python -m rfp_pm_agent.ingest.index_chunks [--allow-mass-delete]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import resources
from itertools import batched
from pathlib import Path
from typing import Any, Protocol

from opensearchpy import OpenSearch
from pydantic import BaseModel, Field

from rfp_pm_agent.clients.embedding import InputType, TEIEmbeddingClient
from rfp_pm_agent.config import ClientsConfig, OpenSearchConfig
from rfp_pm_agent.eval.run_chunk_eval import load_chunks
from rfp_pm_agent.eval.run_retrieval import (
    DEFAULT_COMPOSE_FILE,
    read_snapshot_revision,
    tei_image_tag,
)
from rfp_pm_agent.ingest.chunking import DEFAULT_PARSED_DIR, has_meaningful_char
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Document

DEFAULT_CHUNKS_FILE = Path("data/chunks/block_requirement.jsonl")
# 삭제 대상이 인덱스 문서 수의 이 비율을 넘으면 멈춘다 (이슈 #17 결정 4)
MASS_DELETE_RATIO = 0.05
FETCH_PAGE_SIZE = 1000
# 삭제·메타데이터 부분 update는 임베딩이 없어 TEI 배치 크기와 상관없이 이만큼씩 보낸다
WRITE_BATCH_SIZE = 500

logger = logging.getLogger(__name__)


class IndexingError(RuntimeError):
    """bulk 응답에 실패 항목이나 예상과 다른 결과가 있을 때."""


class MassDeleteError(IndexingError):
    """삭제 대상이 MASS_DELETE_RATIO를 넘었고 허용 플래그가 없을 때. 인덱스는 바뀌지 않았다."""


class Embedder(Protocol):
    def info(self) -> dict[str, Any]: ...

    def embed(
        self, texts: list[str], *, input_type: InputType = "passage"
    ) -> list[list[float]]: ...


def embed_input(text: str, passage_prefix: str) -> str:
    """TEI `/embed`에 실제로 보내는 문자열. `TEIEmbeddingClient`와 같은 규칙(접두어 + text)."""
    return passage_prefix + text


def content_hash(embed_text: str) -> str:
    """sha256 16진수 64자 전체. 입력은 `embed_input`의 결과(접두어 포함)여야 한다."""
    return hashlib.sha256(embed_text.encode("utf-8")).hexdigest()


# metadata_hash의 입력 필드. 문서에 저장하지만 임베딩하지 않는 필드다. chunk_id는 문서 _id(판정의
# 열쇠)라서, text는 content_hash가 덮어서, embedding·embedding_model·content_hash·indexed_at은
# 색인 과정이 만든 값이라서 넣지 않는다. 매핑에 임베딩하지 않는 필드를 추가하면 여기에도 넣는다.
METADATA_FIELDS = ("doc_id", "method", "source_ids", "requirement_id", "block_type", "page")


def metadata_hash(fields: Mapping[str, Any]) -> str:
    """`METADATA_FIELDS`만 골라 JSON으로 직렬화한 문자열의 sha256 16진수 64자.

    직렬화: `json.dumps(..., sort_keys=True, ensure_ascii=False, separators=(",", ":"))`를
    UTF-8로 인코딩한다. 키를 정렬하므로 dict 순서와 무관하고, 없는 필드는 null로 넣는다
    (필드가 빠진 것과 null을 구별하지 않는다). source_ids는 목록 순서 그대로 비교한다.
    """
    selected = {name: fields.get(name) for name in METADATA_FIELDS}
    serialized = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_mapping() -> dict[str, Any]:
    """인덱스 settings·mappings (docs/data-design.md 5절을 옮긴 JSON)."""
    raw = resources.files("rfp_pm_agent.ingest").joinpath("index_mapping.json").read_text("utf-8")
    mapping: dict[str, Any] = json.loads(raw)
    return mapping


class IndexedMeta(BaseModel):
    """인덱스에 이미 있는 문서에서 판정에 쓰는 세 값. 없는 필드는 빈 문자열로 읽는다."""

    content_hash: str
    embedding_model: str
    metadata_hash: str = ""


class ChunkHashes(BaseModel):
    """청크 파일 쪽에서 계산한 두 해시."""

    content_hash: str
    metadata_hash: str


class IndexPlan(BaseModel):
    """판정 결과. 목록은 chunk_id이고, 청크 파일 순서(delete는 chunk_id 순서)다."""

    create: list[str] = Field(default_factory=list)
    update_content: list[str] = Field(default_factory=list)
    update_model: list[str] = Field(default_factory=list)
    update_metadata: list[str] = Field(default_factory=list)
    skip: list[str] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)


class IndexReport(BaseModel):
    """실행 결과 요약.

    dry_run이면 create~delete 개수는 "실행했다면"의 값이고 인덱스는 바뀌지 않았다. 이때
    final_count는 실행 후 예상값이 아니라 지금 인덱스의 문서 수다.
    """

    index_name: str
    alias: str
    alias_targets: list[str]
    embedding_model: str
    dry_run: bool
    index_created: bool
    created: int
    updated_content: int
    updated_model: int
    updated_metadata: int
    skipped: int
    deleted: int
    delete_ids: list[str]
    mass_delete_blocked: bool
    embed_inputs: int
    embed_requests: int
    seconds: float
    final_count: int


def plan_index(
    hashes: Mapping[str, ChunkHashes], embedding_model: str, indexed: Mapping[str, IndexedMeta]
) -> IndexPlan:
    """청크별 해시와 인덱스 상태를 비교해 create/update/skip/delete를 정한다.

    우선순위는 content > model > metadata다 (모듈 설명 6번).
    """
    plan = IndexPlan()
    for chunk_id, digest in hashes.items():
        meta = indexed.get(chunk_id)
        if meta is None:
            plan.create.append(chunk_id)
        elif meta.content_hash != digest.content_hash:
            plan.update_content.append(chunk_id)
        elif meta.embedding_model != embedding_model:
            plan.update_model.append(chunk_id)
        elif meta.metadata_hash != digest.metadata_hash:
            plan.update_metadata.append(chunk_id)
        else:
            plan.skip.append(chunk_id)
    plan.delete = sorted(set(indexed) - set(hashes))
    return plan


def is_mass_delete(n_delete: int, n_indexed: int) -> bool:
    return n_delete > MASS_DELETE_RATIO * n_indexed


def resolve_embedding_model(
    info: Mapping[str, Any], config: ClientsConfig, revision_reader: Callable[[str], str]
) -> str:
    """TEI `/info`로 "{모델ID}@{revision}"을 만든다. 서버 모델이 EMBED_MODEL_ID와 다르면 ValueError."""
    model_id = str(info.get("model_id") or "")
    if not model_id:
        raise ValueError(f"TEI /info에 model_id가 없다: {dict(info)}")
    if config.embed_model_id and model_id != config.embed_model_id:
        raise ValueError(
            f"TEI 서버 모델({model_id})이 EMBED_MODEL_ID({config.embed_model_id})와 다르다"
        )
    revision = info.get("model_sha") or revision_reader(model_id)
    return f"{model_id}@{revision}"


def volume_revision_reader(config: ClientsConfig) -> Callable[[str], str]:
    """TEI 모델 볼륨의 snapshots/에서 revision을 읽는 함수 (#16 run_retrieval과 같은 방법).

    `docker run --rm`으로 `ls`만 하는 일회용 컨테이너를 띄운다(볼륨은 읽기 전용으로 붙인다).
    """

    def read(model_id: str) -> str:
        image = tei_image_tag(DEFAULT_COMPOSE_FILE)
        if image is None:
            raise ValueError(
                f"{DEFAULT_COMPOSE_FILE}에서 TEI 이미지를 찾지 못해 revision을 읽을 수 없다"
            )
        return read_snapshot_revision(model_id, volume=config.tei_models_volume, image=image)

    return read


def check_chunks(chunks: Sequence[Chunk]) -> None:
    """의미 글자 없는 청크(#62)와 chunk_id 중복(#65, `_id`로 씀)을 ValueError로 막는다."""
    no_letter_ids = [c.chunk_id for c in chunks if not has_meaningful_char(c.text)]
    if no_letter_ids:
        raise ValueError(
            f"의미 글자가 없는 청크 {len(no_letter_ids)}개는 임베딩할 수 없다 (#62): {no_letter_ids}"
        )
    counts = Counter(c.chunk_id for c in chunks)
    duplicated = sorted(cid for cid, n in counts.items() if n > 1)
    if duplicated:
        raise ValueError(f"chunk_id가 중복된다 — 문서 _id로 쓸 수 없다 (#65): {duplicated}")


def source_fields(chunk: Chunk, documents: Mapping[str, Document]) -> dict[str, Any]:
    """파싱 결과에서 requirement_id·block_type·page를 찾는다.

    - 블록 청크: block_type = Block.type, page = Block.pdf_page, requirement_id = None
    - 요구사항 청크: requirement_id = Requirement.requirement_id, page = pdf_page_start,
      block_type = None
    page는 파싱 결과와 같은 0부터 세는 PDF 페이지 번호이고, PDF가 아니면 None이다.
    """
    doc = documents.get(chunk.doc_id)
    if doc is None:
        raise ValueError(f"{chunk.chunk_id}의 파싱 결과(doc_id {chunk.doc_id})가 없다")
    if len(chunk.source_ids) != 1:
        raise ValueError(f"{chunk.chunk_id}의 source_ids가 1개가 아니다: {chunk.source_ids}")
    (source_id,) = chunk.source_ids
    for block in doc.blocks:
        if block.block_id == source_id:
            return {"requirement_id": None, "block_type": block.type, "page": block.pdf_page}
    for req in doc.requirements:
        if req.requirement_id == source_id:
            return {
                "requirement_id": req.requirement_id,
                "block_type": None,
                "page": req.pdf_page_start,
            }
    raise ValueError(f"{chunk.chunk_id}의 원본 조각 {source_id}를 파싱 결과에서 찾지 못했다")


def metadata_fields(chunk: Chunk, documents: Mapping[str, Document]) -> dict[str, Any]:
    """문서에 저장하는 메타데이터 필드 전부 (`METADATA_FIELDS`와 같은 키)."""
    return {
        "doc_id": chunk.doc_id,
        "method": chunk.method,
        "source_ids": chunk.source_ids,
        **source_fields(chunk, documents),
    }


def check_index_name(client: OpenSearch, index_name: str, alias: str) -> None:
    """인덱스 이름 자리에 별칭이 들어왔으면 ValueError (#17 리뷰 2번).

    OpenSearch는 `indices.exists`에 별칭 이름을 넣어도 True를 주고, 별칭으로 읽고 쓸 수 있다.
    그래서 막지 않으면 별칭이 가리키는 다른 모델의 인덱스를 이 모델로 전부 update(model)해
    덮어쓴다. 삭제가 아니라서 5% 안전장치에도 걸리지 않는다.
    """
    if index_name == alias:
        raise ValueError(
            f"인덱스 이름({index_name})이 별칭과 같다 — OPENSEARCH_INDEX_NAME에는 모델별 실제 "
            "인덱스 이름(예: rfp_chunks_v1_kure)을 넣는다"
        )
    if client.indices.exists_alias(name=index_name):
        targets = sorted(client.indices.get_alias(name=index_name))
        raise ValueError(
            f"인덱스 이름({index_name})이 이미 별칭이다(→ {', '.join(targets)}) — "
            "OPENSEARCH_INDEX_NAME에는 별칭이 아니라 실제 인덱스 이름을 넣는다"
        )


def ensure_index(client: OpenSearch, index_name: str, alias: str) -> bool:
    """인덱스가 없으면 만들고 True. 이미 있으면 매핑을 건드리지 않고 False.

    별칭은 아직 없을 때만 이 인덱스에 연결한다. 다른 인덱스를 가리키고 있으면 그대로 둔다 —
    모델 교체 때 새 인덱스를 다 채운 뒤 별칭을 옮기는 것은 사람이 따로 한다.
    """
    alias_exists = bool(client.indices.exists_alias(name=alias))
    if client.indices.exists(index=index_name):
        if not alias_exists:
            client.indices.put_alias(index=index_name, name=alias)
        return False
    body = load_mapping()
    if alias_exists:
        logger.warning(
            "별칭 %s가 이미 다른 인덱스를 가리켜 %s에 연결하지 않는다", alias, index_name
        )
    else:
        body["aliases"] = {alias: {}}
    client.indices.create(index=index_name, body=body)
    return True


def alias_targets(client: OpenSearch, alias: str) -> list[str]:
    if not client.indices.exists_alias(name=alias):
        return []
    return sorted(client.indices.get_alias(name=alias))


def fetch_indexed(client: OpenSearch, index_name: str) -> dict[str, IndexedMeta]:
    """인덱스의 모든 문서에서 content_hash·embedding_model·metadata_hash만 읽는다.

    chunk_id(keyword) 오름차순으로 정렬해 search_after로 FETCH_PAGE_SIZE씩 넘긴다.
    """
    body: dict[str, Any] = {
        "size": FETCH_PAGE_SIZE,
        "_source": ["content_hash", "embedding_model", "metadata_hash"],
        "sort": [{"chunk_id": "asc"}],
        "query": {"match_all": {}},
    }
    indexed: dict[str, IndexedMeta] = {}
    while True:
        hits: list[dict[str, Any]] = client.search(index=index_name, body=body)["hits"]["hits"]
        for hit in hits:
            source = hit.get("_source", {})
            indexed[hit["_id"]] = IndexedMeta(
                content_hash=str(source.get("content_hash", "")),
                embedding_model=str(source.get("embedding_model", "")),
                metadata_hash=str(source.get("metadata_hash", "")),
            )
        if len(hits) < FETCH_PAGE_SIZE:
            return indexed
        body["search_after"] = hits[-1]["sort"]


class BulkTally:
    """bulk 응답 항목을 기대 결과와 대조해 센다."""

    def __init__(self) -> None:
        self.counts = {"created": 0, "updated": 0, "deleted": 0}
        self.errors: list[str] = []

    def add(self, response: Mapping[str, Any], expected: Mapping[str, str]) -> None:
        items: list[dict[str, Any]] = response.get("items", [])
        if len(items) != len(expected):
            self.errors.append(
                f"bulk 응답 항목 수({len(items)})가 보낸 수({len(expected)})와 다르다"
            )
        for item in items:
            ((op, result),) = item.items()
            doc_id = str(result.get("_id"))
            got = result.get("result")
            want = expected.get(doc_id)
            if want is None or "error" in result or got != want:
                duplicated = op == "create" and result.get("status") == 409
                reason = " — 409 ID 중복(검색에 없던 _id가 이미 있음)" if duplicated else ""
                self.errors.append(
                    f"{op} {doc_id}: 기대 {want}, 응답 {got} {result.get('error', '')}{reason}"
                )
            else:
                self.counts[want] += 1

    def raise_if_errors(self, stage: str) -> None:
        if self.errors:
            shown = "; ".join(self.errors[:10])
            raise IndexingError(f"{stage} 중 bulk 오류 {len(self.errors)}건: {shown}")


def _document_body(
    chunk: Chunk,
    metadata: Mapping[str, Any],
    vector: list[float],
    *,
    embedding_model: str,
    digest: ChunkHashes,
    indexed_at: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        **metadata,
        "embedding": vector,
        "embedding_model": embedding_model,
        "content_hash": digest.content_hash,
        "metadata_hash": digest.metadata_hash,
        "indexed_at": indexed_at,
    }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def index_chunks(
    chunks: Sequence[Chunk],
    documents: Mapping[str, Document],
    *,
    client: OpenSearch,
    embedder: Embedder,
    config: ClientsConfig,
    index_name: str,
    alias: str,
    revision_reader: Callable[[str], str] | None = None,
    dry_run: bool = False,
    allow_mass_delete: bool = False,
    now: Callable[[], datetime] = _utc_now,
) -> IndexReport:
    """청크를 index_name에 증분 색인한다. 처리 순서와 판정 규칙은 모듈 설명을 본다.

    dry_run이면 인덱스를 만들지도 쓰지도 않고, TEI는 `/info`만 부른다(임베딩 0건).
    """
    start = time.perf_counter()
    if not index_name:
        raise ValueError("인덱스 이름이 비어 있다 (OPENSEARCH_INDEX_NAME)")
    check_index_name(client, index_name, alias)
    check_chunks(chunks)
    metadata_by_id = {c.chunk_id: metadata_fields(c, documents) for c in chunks}
    hashes = {
        c.chunk_id: ChunkHashes(
            content_hash=content_hash(embed_input(c.text, config.embed_passage_prefix)),
            metadata_hash=metadata_hash(metadata_by_id[c.chunk_id]),
        )
        for c in chunks
    }
    embedding_model = resolve_embedding_model(
        embedder.info(), config, revision_reader or volume_revision_reader(config)
    )

    # 판정과 5% 검사를 인덱스·별칭을 만들기 전에 한다 — 걸리면 아무것도 바뀌지 않게 (#17 리뷰 1번)
    indexed: dict[str, IndexedMeta] = {}
    if client.indices.exists(index=index_name):
        if not dry_run:
            client.indices.refresh(index=index_name)  # 직전 쓰기를 검색에 보이게 한다
        indexed = fetch_indexed(client, index_name)
    plan = plan_index(hashes, embedding_model, indexed)
    blocked = is_mass_delete(len(plan.delete), len(indexed)) and not allow_mass_delete
    index_created = False

    def report(*, embed_inputs: int, embed_requests: int, final_count: int) -> IndexReport:
        return IndexReport(
            index_name=index_name,
            alias=alias,
            alias_targets=alias_targets(client, alias),
            embedding_model=embedding_model,
            dry_run=dry_run,
            index_created=index_created,
            created=len(plan.create),
            updated_content=len(plan.update_content),
            updated_model=len(plan.update_model),
            updated_metadata=len(plan.update_metadata),
            skipped=len(plan.skip),
            deleted=len(plan.delete),
            delete_ids=plan.delete,
            mass_delete_blocked=blocked,
            embed_inputs=embed_inputs,
            embed_requests=embed_requests,
            seconds=time.perf_counter() - start,
            final_count=final_count,
        )

    if dry_run:
        return report(embed_inputs=0, embed_requests=0, final_count=len(indexed))
    if blocked:
        raise MassDeleteError(
            f"삭제 대상 {len(plan.delete)}개가 인덱스 문서 {len(indexed)}개의 "
            f"{MASS_DELETE_RATIO:.0%}를 넘는다 — --dry-run으로 대상을 확인하고 "
            f"--allow-mass-delete로 다시 실행한다. 인덱스는 바뀌지 않았다"
        )
    index_created = ensure_index(client, index_name, alias)

    by_id = {c.chunk_id: c for c in chunks}
    # create 대상은 bulk create로 보낸다 — 같은 _id가 있으면 서버가 409로 거절하고 문서는 그대로다.
    # update 대상은 bulk index(문서 전체 교체)로 보낸다 (이슈 #17 결정 3)
    op_by_id = dict.fromkeys(plan.create, "create")
    op_by_id.update(dict.fromkeys(plan.update_content + plan.update_model, "index"))
    expected_result = {
        cid: "created" if op == "create" else "updated" for cid, op in op_by_id.items()
    }
    targets = [cid for cid in hashes if cid in op_by_id]
    indexed_at = now().isoformat()
    tally = BulkTally()
    embed_inputs = embed_requests = 0
    for batch_ids in batched(targets, config.tei_max_client_batch_size):
        batch = [by_id[cid] for cid in batch_ids]
        vectors = embedder.embed([c.text for c in batch], input_type="passage")
        embed_inputs += len(batch)
        embed_requests += 1
        operations: list[dict[str, Any]] = []
        for chunk, vector in zip(batch, vectors, strict=True):
            op = op_by_id[chunk.chunk_id]
            operations.append({op: {"_index": index_name, "_id": chunk.chunk_id}})
            operations.append(
                _document_body(
                    chunk,
                    metadata_by_id[chunk.chunk_id],
                    vector,
                    embedding_model=embedding_model,
                    digest=hashes[chunk.chunk_id],
                    indexed_at=indexed_at,
                )
            )
        tally.add(client.bulk(body=operations), {cid: expected_result[cid] for cid in batch_ids})
        tally.raise_if_errors("적재")  # 배치마다 확인해 다음 배치의 임베딩 전에 멈춘다

    # 메타데이터만 바뀐 문서: 벡터·content_hash·indexed_at은 두고 메타데이터 필드만 덮는다
    for update_ids in batched(plan.update_metadata, WRITE_BATCH_SIZE):
        operations = []
        for cid in update_ids:
            operations.append({"update": {"_index": index_name, "_id": cid}})
            partial = {**metadata_by_id[cid], "metadata_hash": hashes[cid].metadata_hash}
            operations.append({"doc": partial})
        tally.add(client.bulk(body=operations), dict.fromkeys(update_ids, "updated"))
        tally.raise_if_errors("메타데이터 갱신")

    for delete_ids in batched(plan.delete, WRITE_BATCH_SIZE):
        operations = [{"delete": {"_index": index_name, "_id": cid}} for cid in delete_ids]
        tally.add(client.bulk(body=operations), dict.fromkeys(delete_ids, "deleted"))
        tally.raise_if_errors("삭제")

    client.indices.refresh(index=index_name)
    final_count = int(client.count(index=index_name)["count"])
    return report(embed_inputs=embed_inputs, embed_requests=embed_requests, final_count=final_count)


def format_report(report: IndexReport) -> str:
    """사람이 읽는 요약. dry-run이면 삭제 대상 ID를 모두 적는다."""
    head = "[dry-run] 색인 계획 (인덱스는 바뀌지 않았다)" if report.dry_run else "색인 결과"
    targets = ", ".join(report.alias_targets) or "(없음)"
    n_updated = report.updated_content + report.updated_model + report.updated_metadata
    count_label = "현재 _count" if report.dry_run else "최종 _count"
    lines = [
        f"{head} — {report.index_name} (별칭 {report.alias} → {targets})",
        f"embedding_model: {report.embedding_model}",
        (
            f"create {report.created} / update {n_updated} "
            f"(content {report.updated_content}, model {report.updated_model}, "
            f"metadata {report.updated_metadata}) / "
            f"skip {report.skipped} / delete {report.deleted}"
        ),
        (
            f"TEI 임베딩 입력 {report.embed_inputs}건 (요청 {report.embed_requests}건) / "
            f"소요 {report.seconds:.1f}s / {count_label} {report.final_count}"
        ),
    ]
    if report.index_created:
        lines.append(f"인덱스 {report.index_name}를 새로 만들었다")
    if report.dry_run and report.delete_ids:
        lines.append(f"삭제 대상 {len(report.delete_ids)}개: {', '.join(report.delete_ids)}")
    if report.mass_delete_blocked:
        lines.append(
            f"삭제 대상이 인덱스 문서 수의 {MASS_DELETE_RATIO:.0%}를 넘는다 — "
            "실제 실행은 --allow-mass-delete 없이는 중단된다"
        )
    return "\n".join(lines)


def load_documents(
    doc_ids: Iterable[str], parsed_dir: Path = DEFAULT_PARSED_DIR
) -> dict[str, Document]:
    """청크가 가리키는 문서의 파싱 결과(`{doc_id}.json`)만 읽는다."""
    return {
        doc_id: Document.model_validate_json(
            (parsed_dir / f"{doc_id}.json").read_text(encoding="utf-8")
        )
        for doc_id in sorted(set(doc_ids))
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="청크 파일을 OpenSearch 인덱스에 증분 색인한다.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_FILE, help="청크 jsonl")
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="인덱스를 읽기만 하고 create/update/skip/delete 개수와 삭제 대상 ID를 출력한다",
    )
    parser.add_argument(
        "--allow-mass-delete",
        action="store_true",
        # argparse는 help를 %-서식으로 처리하므로 "%"를 "%%"로 적는다
        help=f"삭제 대상이 인덱스 문서 수의 {MASS_DELETE_RATIO:.0%}를 넘어도 삭제한다".replace(
            "%", "%%"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """`uv run python -m rfp_pm_agent.ingest.index_chunks` 진입점."""
    logging.basicConfig(level=logging.INFO)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    clients_config = ClientsConfig.from_env()
    os_config = OpenSearchConfig.from_env()
    if not os_config.index_name:
        parser.error("OPENSEARCH_INDEX_NAME이 비어 있다 (.env.example 참고)")
    chunks = load_chunks(args.chunks)
    documents = load_documents((c.doc_id for c in chunks), args.parsed_dir)
    logger.info("청크 %d개 (%s), 문서 %d건", len(chunks), args.chunks, len(documents))
    report = index_chunks(
        chunks,
        documents,
        client=OpenSearch(hosts=[os_config.url]),
        embedder=TEIEmbeddingClient(clients_config),
        config=clients_config,
        index_name=os_config.index_name,
        alias=os_config.index_alias,
        dry_run=args.dry_run,
        allow_mass_delete=args.allow_mass_delete,
    )
    logger.info("%s", format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
