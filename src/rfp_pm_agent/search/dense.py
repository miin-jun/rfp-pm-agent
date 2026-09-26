"""메모리 내 벡터 검색과 리랭크 (이슈 #16 모델 선정 실험용).

#17에서 OpenSearch k-NN으로 대체되는 임시 구현이다. 서비스 검색이 아니라, 임베딩 모델
3종을 같은 청크·같은 질문으로 줄 세우기 위한 실험 도구다. 하이브리드(BM25+벡터 RRF)와
서비스용 리랭크 파이프라인은 #18에서 따로 만든다.

- 코사인 유사도: 청크 벡터와 질의 벡터를 모두 길이 1로 정규화한 뒤 내적한다. TEI에
  `normalize: true`를 보내지만, 서버 버전별 동작을 확인하지 않았으므로 여기서 한 번 더
  정규화한다(이미 길이 1이면 값이 바뀌지 않는다).
- 벡터 캐시: 청크 3천여 개를 모델마다 다시 임베딩하지 않도록 `.npy`로 저장한다. 캐시
  이름에 모델 ID·revision·passage 접두어·자르기 여부·청크 파일 해시를 넣어, 다른
  모델이나 다른 청크의 벡터가 섞이지 않게 한다(CLAUDE.md: 인덱스는 모델별).
- 동점 순서: 점수가 같으면 청크 목록 순서(앞쪽 우선)로 고정한다.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rfp_pm_agent.clients.embedding import TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import RerankerClient
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import SearchHit, ServerInfo

DEFAULT_CACHE_DIR = Path("data/cache/embeddings")

FloatMatrix = npt.NDArray[np.float32]


def _normalize_rows(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class DenseIndex:
    """청크 목록과 같은 순서의 벡터 행렬로 만든 코사인 유사도 색인."""

    def __init__(self, chunks: Sequence[Chunk], vectors: FloatMatrix) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"청크 수({len(chunks)})와 벡터 수({vectors.shape[0]})가 다르다")
        self._chunks = list(chunks)
        self._matrix = _normalize_rows(vectors.astype(np.float32))

    def search_vector(self, query_vector: Sequence[float], top_k: int) -> list[SearchHit]:
        """질의 벡터와 코사인 유사도가 높은 순서로 top_k개를 돌려준다."""
        query = _normalize_rows(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ query
        # 점수 내림차순, 동점이면 청크 순서 오름차순 (stable 정렬)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchHit(
                chunk_id=self._chunks[i].chunk_id,
                doc_id=self._chunks[i].doc_id,
                score=float(scores[i]),
                text=self._chunks[i].text,
            )
            for i in order
        ]


@dataclass
class IndexedVectors:
    """청크 벡터와 그 벡터를 만든 조건. 캐시에서 읽었으면 index_seconds는 처음 만들 때 값이다."""

    vectors: FloatMatrix
    truncated: list[int]
    index_seconds: float
    cache_path: Path
    cache_hit: bool


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_key(info: ServerInfo, passage_prefix: str, truncate: bool, chunks_sha: str) -> str:
    """모델·revision·접두어·자르기·청크 파일이 모두 같을 때만 같은 이름이 된다."""
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", info.model_id)
    condition = json.dumps(
        {"sha": info.model_sha, "prefix": passage_prefix, "truncate": truncate},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(condition.encode("utf-8")).hexdigest()[:12]
    return f"{model}__{digest}__{chunks_sha[:12]}"


def build_or_load_vectors(
    chunks: Sequence[Chunk],
    client: TEIEmbeddingClient,
    info: ServerInfo,
    *,
    passage_prefix: str,
    truncate: bool,
    chunks_sha: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> IndexedVectors:
    """캐시가 있으면 읽고, 없으면 청크 전체를 passage로 임베딩해 저장한다.

    캐시 메타(.json)에 청크 ID 순서를 적어 두고, 읽을 때 지금 청크 순서와 다르면
    ValueError를 낸다 — 벡터와 청크의 대응이 어긋난 채 검색되는 것을 막는다.
    """
    key = cache_key(info, passage_prefix, truncate, chunks_sha)
    npy_path = cache_dir / f"{key}.npy"
    meta_path = cache_dir / f"{key}.json"
    chunk_ids = [c.chunk_id for c in chunks]
    if npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["chunk_ids"] != chunk_ids:
            raise ValueError(f"벡터 캐시 {npy_path}의 청크 순서가 지금 청크 파일과 다르다")
        return IndexedVectors(
            vectors=np.load(npy_path).astype(np.float32),
            truncated=list(meta["truncated"]),
            index_seconds=float(meta["index_seconds"]),
            cache_path=npy_path,
            cache_hit=True,
        )

    start = time.perf_counter()
    report = client.embed_with_report([c.text for c in chunks], input_type="passage")
    index_seconds = time.perf_counter() - start
    vectors = np.asarray(report.vectors, dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, vectors)
    meta_path.write_text(
        json.dumps(
            {
                "model": info.model_dump(),
                "passage_prefix": passage_prefix,
                "truncate": truncate,
                "chunks_sha256": chunks_sha,
                "chunk_ids": chunk_ids,
                "truncated": report.truncated,
                "index_seconds": index_seconds,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return IndexedVectors(
        vectors=vectors,
        truncated=report.truncated,
        index_seconds=index_seconds,
        cache_path=npy_path,
        cache_hit=False,
    )


def rerank_hits(
    query: str, hits: Sequence[SearchHit], reranker: RerankerClient, n: int
) -> list[SearchHit]:
    """상위 n개를 리랭커 점수로 다시 정렬해 돌려준다. score는 리랭커 점수로 바뀐다.

    n개 밖의 결과는 버린다(이슈 #16: 후보 N=20을 다시 정렬하고 상위 10으로 채점).
    동점이면 1차 검색 순서를 유지한다.
    """
    candidates = list(hits[:n])
    if not candidates:
        return []
    scores = reranker.rerank(query, [h.text for h in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i].model_copy(update={"score": float(scores[i])}) for i in order]
