"""메모리 내 벡터 검색·벡터 캐시·리랭크 단위 테스트 (이슈 #16, 임시 구현).

TEI는 httpx.MockTransport로 대신한다 — 네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from rfp_pm_agent.clients.embedding import EMBEDDING_DIM, TEIEmbeddingClient
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import SearchHit, ServerInfo
from rfp_pm_agent.search.dense import (
    DenseIndex,
    build_or_load_vectors,
    cache_key,
    rerank_hits,
)
from tests.fakes.fake_reranker import FakeRerankerClient
from tests.unit.conftest import make_clients_config


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"d:m:b{i}", doc_id="d", method="block", source_ids=[f"b{i}"], text=f"t{i}")
        for i in range(n)
    ]


def test_코사인_유사도_높은_순서로_돌려주고_벡터_길이에_영향받지_않는다() -> None:
    vectors = np.array([[1.0, 0.0], [10.0, 10.0], [0.0, 5.0]], dtype=np.float32)
    index = DenseIndex(_chunks(3), vectors)

    hits = index.search_vector([0.0, 1.0], top_k=3)

    assert [h.chunk_id for h in hits] == ["d:m:b2", "d:m:b1", "d:m:b0"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(1 / np.sqrt(2))


def test_동점이면_청크_순서를_유지한다() -> None:
    vectors = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hits = DenseIndex(_chunks(3), vectors).search_vector([1.0, 0.0], top_k=2)
    assert [h.chunk_id for h in hits] == ["d:m:b0", "d:m:b1"]


def test_청크_수와_벡터_수가_다르면_거부한다() -> None:
    with pytest.raises(ValueError):
        DenseIndex(_chunks(2), np.zeros((3, 2), dtype=np.float32))


def test_캐시_이름은_모델_revision_접두어_자르기_청크가_하나라도_다르면_달라진다() -> None:
    info = ServerInfo(model_id="org/model", snapshot_revision="abc")
    base = cache_key(info, "", False, "f" * 64)
    assert cache_key(info, "", False, "f" * 64) == base
    assert (
        cache_key(ServerInfo(model_id="org/model", snapshot_revision="def"), "", False, "f" * 64)
        != base
    )
    assert (
        cache_key(ServerInfo(model_id="org/other", snapshot_revision="abc"), "", False, "f" * 64)
        != base
    )
    assert cache_key(info, "passage: ", False, "f" * 64) != base
    assert cache_key(info, "", True, "f" * 64) != base
    assert cache_key(info, "", False, "e" * 64) != base


def _counting_transport(calls: list[str]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        inputs: list[str] = json.loads(request.content)["inputs"]
        return httpx.Response(200, json=[[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in inputs])

    return httpx.MockTransport(handle)


def test_벡터를_한_번만_만들고_두_번째는_캐시에서_읽는다(tmp_path: Path) -> None:
    calls: list[str] = []
    client = TEIEmbeddingClient(make_clients_config(), transport=_counting_transport(calls))
    info = ServerInfo(model_id="org/model", snapshot_revision="abc")
    chunks = _chunks(3)

    first = build_or_load_vectors(
        chunks,
        client,
        info,
        passage_prefix="",
        truncate=False,
        chunks_sha="0" * 64,
        cache_dir=tmp_path,
    )
    second = build_or_load_vectors(
        chunks,
        client,
        info,
        passage_prefix="",
        truncate=False,
        chunks_sha="0" * 64,
        cache_dir=tmp_path,
    )

    assert calls == ["/embed"]
    assert first.cache_hit is False and second.cache_hit is True
    assert np.array_equal(first.vectors, second.vectors)
    assert second.index_seconds == pytest.approx(first.index_seconds)


def test_캐시의_청크_순서가_다르면_거부한다(tmp_path: Path) -> None:
    client = TEIEmbeddingClient(make_clients_config(), transport=_counting_transport([]))
    info = ServerInfo(model_id="org/model", snapshot_revision="rev1")
    chunks = _chunks(3)
    build_or_load_vectors(
        chunks,
        client,
        info,
        passage_prefix="",
        truncate=False,
        chunks_sha="0" * 64,
        cache_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="청크 순서"):
        build_or_load_vectors(
            list(reversed(chunks)),
            client,
            info,
            passage_prefix="",
            truncate=False,
            chunks_sha="0" * 64,
            cache_dir=tmp_path,
        )


def _hit(chunk_id: str, text: str) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, doc_id="d", score=0.0, text=text)


def test_리랭크는_상위_n개만_리랭커_점수로_다시_정렬한다() -> None:
    hits = [_hit("c1", "무관한 문장"), _hit("c2", "사업 기간 은 3개월"), _hit("c3", "사업 기간")]

    reranked = rerank_hits("사업 기간", hits, FakeRerankerClient(), n=2)

    assert [h.chunk_id for h in reranked] == ["c2", "c1"]  # c3은 n 밖이라 버린다
    assert reranked[0].score > reranked[1].score


def test_리랭크_후보가_없으면_빈_목록() -> None:
    assert rerank_hits("질의", [], FakeRerankerClient(), n=20) == []


def test_빈_청크가_있으면_임베딩_요청_전에_ValueError(tmp_path: Path) -> None:
    """#62: 청킹이 빈 청크를 버리므로, 들어오면 TEI 400 대신 chunk_id를 담은 오류로 멈춘다."""
    chunks = [
        Chunk(chunk_id=f"d:m:b{i}", doc_id="d", method="block", source_ids=[f"b{i}"], text=text)
        for i, text in enumerate(["첫 문장", "", "셋째 문장", "  \n"])
    ]
    sent: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(500)

    client = TEIEmbeddingClient(make_clients_config(), transport=httpx.MockTransport(handle))

    with pytest.raises(ValueError, match=r"d:m:b1.*d:m:b3"):
        build_or_load_vectors(
            chunks,
            client,
            ServerInfo(model_id="org/model", snapshot_revision="rev1"),
            passage_prefix="",
            truncate=False,
            chunks_sha="0" * 64,
            cache_dir=tmp_path,
        )
    assert sent == []  # /embed 요청을 하나도 보내지 않았다
    assert list(tmp_path.iterdir()) == []  # 캐시도 만들지 않았다


@pytest.mark.parametrize("text", [" | ", " |  | \n |  | ", "+\n", " |  |  | √", "○○"])
def test_의미_글자가_없는_청크도_임베딩_요청_전에_ValueError(tmp_path: Path, text: str) -> None:
    """#62: 공백은 아니어도 L·N 글자가 없으면 청킹이 버리는 청크다. TEI는 받아 주지만
    멈춘다 — 청킹과 같은 기준(chunking.has_meaningful_char)을 쓴다."""
    chunks = [
        Chunk(chunk_id="d:m:b0", doc_id="d", method="block", source_ids=["b0"], text="첫 문장"),
        Chunk(chunk_id="d:m:b1", doc_id="d", method="block", source_ids=["b1"], text=text),
    ]
    sent: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(200, json=[[1.0] + [0.0] * (EMBEDDING_DIM - 1)] * 2)

    client = TEIEmbeddingClient(make_clients_config(), transport=httpx.MockTransport(handle))

    with pytest.raises(ValueError, match="d:m:b1"):
        build_or_load_vectors(
            chunks,
            client,
            ServerInfo(model_id="org/model", snapshot_revision="rev1"),
            passage_prefix="",
            truncate=False,
            chunks_sha="0" * 64,
            cache_dir=tmp_path,
        )
    assert sent == []


def test_영벡터의_검색_점수는_0이다() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    hits = DenseIndex(_chunks(2), vectors).search_vector([1.0, 0.0], top_k=2)
    assert [h.chunk_id for h in hits] == ["d:m:b0", "d:m:b1"]
    assert hits[1].score == pytest.approx(0.0)


def test_스냅샷_revision이_없으면_캐시_이름을_만들지_않는다() -> None:
    # /info의 model_sha만 있고 스냅샷 revision이 없으면 null로 캐시를 만들지 않는다
    with pytest.raises(ValueError, match="revision"):
        cache_key(ServerInfo(model_id="org/model", model_sha="abc"), "", False, "f" * 64)
