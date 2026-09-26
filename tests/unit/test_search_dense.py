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
    info = ServerInfo(model_id="org/model", model_sha="abc")
    base = cache_key(info, "", False, "f" * 64)
    assert cache_key(info, "", False, "f" * 64) == base
    assert cache_key(ServerInfo(model_id="org/model", model_sha="def"), "", False, "f" * 64) != base
    assert cache_key(ServerInfo(model_id="org/other", model_sha="abc"), "", False, "f" * 64) != base
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
    info = ServerInfo(model_id="org/model", model_sha="abc")
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
    info = ServerInfo(model_id="org/model")
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
