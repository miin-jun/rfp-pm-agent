"""실제 OpenSearch로 증분 색인의 매핑 검사를 확인한다 (이슈 #82, `-m integration`).

실행: `docker compose up -d opensearch` 후 `uv run pytest tests/integration -m integration -q`.

- #81 이전(v1) 필드 구성의 임시 인덱스를 만들어 색인을 시도하고, 쓰기 전에 MappingMismatchError로
  멈추는지, 임시 인덱스의 문서·매핑·별칭이 그대로인지 본다. 끝나면 임시 인덱스를 지운다
- 운영 인덱스·별칭(`OPENSEARCH_INDEX_ALIAS`)은 건드리지 않는다. 별칭 인자도 임시 이름을 넘기고,
  운영 별칭이 가리키는 인덱스가 전후 같은지 확인한다
- 실제 TEI는 쓰지 않는다(가짜 TEI). 매핑 검사는 TEI `/info`·`/embed`보다 먼저라 TEI가 필요 없다.
  가짜 TEI의 임베딩 입력 0건으로 쓰기 전에 멈췄음을 확인한다
- OpenSearch가 꺼져 있으면 skip하지 않고 실패한다 — test_opensearch_search.py와 같은 이유
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from opensearchpy import OpenSearch

from rfp_pm_agent.config import OpenSearchConfig
from rfp_pm_agent.ingest.index_chunks import MappingMismatchError, index_chunks, load_mapping
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Block, Document
from tests.fakes.fake_tei import FakeTEIServer
from tests.unit.conftest import make_clients_config

pytestmark = pytest.mark.integration

DOC_ID = "d0000000000000082"
# #81에서 매핑에 추가한 출처 필드 — v1 인덱스에는 없다
V1_MISSING = ["bid_title", "format", "printed_page"]


@pytest.fixture(scope="module")
def client() -> OpenSearch:
    config = OpenSearchConfig.from_env()
    os_client = OpenSearch(hosts=[config.url])
    assert os_client.ping(), f"OpenSearch 응답 없음: {config.url}"
    return os_client


@pytest.fixture
def v1_index(client: OpenSearch) -> Iterator[str]:
    """v1 필드 구성 + 문서 1건의 임시 인덱스. 테스트가 끝나면 지운다."""
    name = f"test82_mapping_check_{uuid.uuid4().hex[:8]}"
    body = load_mapping()
    for field in V1_MISSING:
        del body["mappings"]["properties"][field]
    client.indices.create(index=name, body=body)
    try:
        client.index(
            index=name,
            id="existing",
            body={"chunk_id": "existing", "doc_id": DOC_ID, "text": "기존 문서"},
            refresh=True,
        )
        yield name
    finally:
        client.indices.delete(index=name)


def _document() -> Document:
    return Document(
        doc_id=DOC_ID,
        source_file="sample.pdf",
        bid_title="샘플 사업",
        format="pdf",
        parse_status="parsed",
        has_requirements=False,
        requirement_count=0,
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=[Block(block_id="b0001", type="paragraph", text="사업 개요", source_order=0)],
        requirements=[],
    )


def _state(client: OpenSearch, index: str) -> dict[str, Any]:
    docs = client.search(index=index, body={"query": {"match_all": {}}, "size": 100})
    return {
        "docs": {hit["_id"]: hit["_source"] for hit in docs["hits"]["hits"]},
        "mapping": client.indices.get_mapping(index=index),
        "aliases": client.indices.get_alias(index=index),
    }


def _alias_targets(client: OpenSearch, alias: str) -> list[str]:
    if not client.indices.exists_alias(name=alias):
        return []
    return sorted(client.indices.get_alias(name=alias))


def test_v1_mapping_index_stops_before_write(client: OpenSearch, v1_index: str) -> None:
    prod_alias = OpenSearchConfig.from_env().index_alias
    tmp_alias = f"{v1_index}_alias"
    prod_before = _alias_targets(client, prod_alias)
    before = _state(client, v1_index)
    tei = FakeTEIServer()
    config = make_clients_config()
    chunk = Chunk(
        chunk_id=f"{DOC_ID}:block_requirement:b0001",
        doc_id=DOC_ID,
        method="block_requirement",
        source_ids=["b0001"],
        text="사업 개요",
    )

    with pytest.raises(MappingMismatchError) as caught:
        index_chunks(
            [chunk],
            {DOC_ID: _document()},
            client=client,
            embedder=tei.client(config),
            config=config,
            index_name=v1_index,
            alias=tmp_alias,
        )

    message = str(caught.value)
    assert v1_index in message
    assert "빠진 필드 3개: bid_title, format, printed_page" in message
    assert "type 다른 필드" not in message  # 실제 서버의 get_mapping 응답에서 type은 같게 나온다
    client.indices.refresh(index=v1_index)
    assert _state(client, v1_index) == before
    assert not client.indices.exists_alias(name=tmp_alias)
    assert _alias_targets(client, prod_alias) == prod_before
    assert len(tei.embed_inputs) == 0
