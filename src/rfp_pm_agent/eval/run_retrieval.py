"""검색 평가 실행 — 임베딩·리랭커 선정 실험 (이슈 #16).

qa_v2(57문항, 답 있음 51)를 block_requirement 청크로 검색해 채점한다. 한 번 실행에
검색 조건 하나(BM25 / 임베딩 모델 1개 / 임베딩 + 리랭커)를 돌린다. TEI 서버는 한 번에
모델 하나만 띄우므로(learning-log 2026-09-24, RAM), 모델을 바꿀 때마다 서버를 다시 띄우고
이 명령을 다시 실행한다. 측정 절차는 docs/model-selection-measurement.md.

기록 (data/eval/results/model_selection/, os 모드는 data/eval/results/hybrid/ — `--out-dir`로 바꿀 수 있다):
- runs.jsonl: 실행 한 줄(RetrievalRunRecord) — 모델 ID·revision, TEI 버전·이미지, GPU,
  접두어, 자르기 여부·자른 청크 수, 색인 시간, 지연 p50·p95, 집계 점수
- {run_id}.questions.jsonl: 답 있는 문항별 결과 — 실행 간 McNemar 비교에 쓴다
- failures/{run_id}.jsonl: Recall@10(전부 적중)을 못 맞힌 문항의 상위 10개

결정 규칙(이슈 #16, 측정 전 고정)은 `--compare`로 적용한다: Recall@10 전부 적중 수가
가장 많은 실행과 나머지를 McNemar 정확검정(양측)으로 비교하고, p ≥ 0.05면 동률로 표시한다.
동률일 때 가벼운 모델 고르기(VRAM → 색인 시간 → 지연)는 사람이 기록을 보고 한다.

MRR·NDCG 함수는 소유자가 구현했다(retrieval.py). 그 전에 남긴 기록은 그 칸이 비어 있다.
LLM을 부르지 않으므로 비용은 $0이다.

OpenSearch 모드(#18 PR ①): `--os-bm25`·`--os-knn`은 메모리 색인 대신 별칭(`OPENSEARCH_INDEX_ALIAS`)을
검색한다(`search/opensearch.py`). 채점의 정답 묶음은 여전히 `--chunks` 파일에서 뽑으므로, 실행 전에
별칭이 가리키는 인덱스의 chunk_id 집합이 청크 파일과 같은지 확인하고 다르면 멈춘다. `--os-knn`은
인덱스 문서의 embedding_model이 지금 TEI 서버의 "{모델ID}@{revision}"과 같은지도 확인한다 —
다른 모델 벡터로 만든 질의를 섞어 검색하지 않기 위해서다.

하이브리드(#18 PR ②): `--os-hybrid`는 `hybrid_search`(OpenSearch RRF)로 검색하고, `--rerank`를 붙이면 그
상위 `--rerank-n`개(1~50)를 리랭크한다. 리랭크는 `--dense`와 `--os-hybrid`에만 붙는다. 검증용으로 답 있는
문항마다 같은 질문의 앱 RRF(`rrf_fuse`, bm25_search·knn_search 후보 50씩) 상위 10과 hybrid 상위 10의 겹침
수를 기록한다 — 측정 반복이 끝난 뒤 따로 검색하므로 지연에 들어가지 않는다.

CLI: `uv run python -m rfp_pm_agent.eval.run_retrieval --bm25`
     `uv run python -m rfp_pm_agent.eval.run_retrieval --dense [--rerank]`
     `uv run python -m rfp_pm_agent.eval.run_retrieval --os-bm25`
     `uv run python -m rfp_pm_agent.eval.run_retrieval --os-knn`
     `uv run python -m rfp_pm_agent.eval.run_retrieval --os-hybrid [--rerank --rerank-n 50]`
     `uv run python -m rfp_pm_agent.eval.run_retrieval --compare <run_id> <run_id> ...`
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from opensearchpy import OpenSearch

from rfp_pm_agent.clients.embedding import TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import TEIRerankerClient
from rfp_pm_agent.clients.tei_revision import (
    DEFAULT_COMPOSE_FILE,
    tei_image_tag,
    volume_revision_reader,
)
from rfp_pm_agent.config import ClientsConfig, OpenSearchConfig
from rfp_pm_agent.eval.qa_v2 import DEFAULT_QA_V2, load_questions_v2
from rfp_pm_agent.eval.retrieval import V2_TOP_K, mcnemar_exact_p, score_run_v2
from rfp_pm_agent.ingest.chunking import DEFAULT_PARSED_DIR, load_chunks
from rfp_pm_agent.ingest.index_chunks import (
    ChunkHashes,
    IndexedMeta,
    alias_targets,
    chunk_hashes,
    fetch_indexed,
    load_documents,
    resolve_embedding_model,
)
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Document
from rfp_pm_agent.schemas.eval import (
    EvalQuestionV2,
    LatencyStats,
    QuestionResult,
    RetrievalRunRecord,
    RetrievalRunScore,
    RetrieverName,
    SearchHit,
    ServerInfo,
    TokenizerName,
)
from rfp_pm_agent.search.bm25 import Bm25Index
from rfp_pm_agent.search.dense import (
    DEFAULT_CACHE_DIR,
    DenseIndex,
    build_or_load_vectors,
    file_sha256,
    rerank_hits,
)
from rfp_pm_agent.search.hybrid import rrf_fuse
from rfp_pm_agent.search.opensearch import (
    KNN_CANDIDATES,
    RRF_RANK_CONSTANT,
    bm25_search,
    hybrid_search,
    knn_search,
    search_alias,
)
from rfp_pm_agent.search.tokenize import get_tokenizer

DEFAULT_CHUNKS_FILE = Path("data/chunks/block_requirement.jsonl")
DEFAULT_OUT_DIR = Path("data/eval/results/model_selection")
# #18 OpenSearch 모드의 기록 위치. #16 모델 선정 기록과 섞지 않는다
DEFAULT_HYBRID_OUT_DIR = Path("data/eval/results/hybrid")

# 이슈 #16 결정 규칙에서 고정한 값
DEFAULT_RERANK_N = 20
# 리랭크 후보 N의 상한: 검색기별 후보가 50(KNN_CANDIDATES)이라 그보다 큰 N은 받을 후보가 없다 (#18 결정 5A)
MAX_RERANK_N = KNN_CANDIDATES
# 리랭크를 붙일 수 있는 방식
RERANK_RETRIEVERS: tuple[RetrieverName, ...] = ("dense", "os-hybrid")
SIGNIFICANCE = 0.05

# 지연 측정: 앞 문항 몇 개로 워밍업하고, 문항 전체를 몇 번 돌려 표본을 모은다
DEFAULT_WARMUP = 5
DEFAULT_LATENCY_REPEATS = 3

FAILURE_TEXT_CHARS = 200

logger = logging.getLogger(__name__)

SearchFn = Callable[[str], list[SearchHit]]
RerankFn = Callable[[str, list[SearchHit]], list[SearchHit]]


# --- 환경 기록 ---


def server_info(raw: dict[str, Any]) -> ServerInfo:
    return ServerInfo(
        model_id=str(raw.get("model_id", "")),
        model_sha=raw.get("model_sha"),
        tei_version=raw.get("version"),
        max_input_length=raw.get("max_input_length"),
    )


def gpu_snapshot() -> dict[str, str] | None:
    """nvidia-smi로 GPU 이름·드라이버·사용 중 메모리(MiB)를 읽는다. 없으면 None."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    name, driver, used = (part.strip() for part in out.splitlines()[0].split(","))
    return {"name": name, "driver": driver, "memory_used_mib": used}


def latency_stats(samples_ms: Sequence[float]) -> LatencyStats | None:
    if not samples_ms:
        return None
    arr = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        samples=len(samples_ms),
    )


# --- 검색 실행 ---


def run_searches(
    questions: Sequence[EvalQuestionV2],
    search: SearchFn,
    rerank: RerankFn | None,
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, list[SearchHit]], list[float], list[float]]:
    """문항마다 검색(과 리랭크)을 하고, 워밍업 뒤 repeats번의 처리 시간을 모은다.

    채점에 쓰는 결과는 첫 측정 반복의 결과다(검색은 결정적이라 반복마다 같다).
    """
    for question in questions[:warmup]:
        hits = search(question.question)
        if rerank is not None:
            rerank(question.question, hits)

    rankings: dict[str, list[SearchHit]] = {}
    search_ms: list[float] = []
    rerank_ms: list[float] = []
    for repeat in range(max(repeats, 1)):
        for question in questions:
            start = time.perf_counter()
            hits = search(question.question)
            search_ms.append((time.perf_counter() - start) * 1000)
            if rerank is not None:
                start = time.perf_counter()
                hits = rerank(question.question, hits)
                rerank_ms.append((time.perf_counter() - start) * 1000)
            if repeat == 0:
                rankings[question.question_id] = hits
    return rankings, search_ms, rerank_ms


def write_questions(path: Path, results: Sequence[QuestionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


def write_failures(
    path: Path, results: Sequence[QuestionResult], rankings: dict[str, list[SearchHit]]
) -> None:
    """Recall@10(전부 적중)을 못 맞힌 문항의 상위 10개 (chunk_id, 점수, text 앞부분)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            if result.recall_all_at_10:
                continue
            row = {
                "question_id": result.question_id,
                "top_hits": [
                    {
                        "rank": rank,
                        "chunk_id": hit.chunk_id,
                        "score": round(hit.score, 4),
                        "text_head": hit.text[:FAILURE_TEXT_CHARS],
                    }
                    for rank, hit in enumerate(rankings[result.question_id][:10], start=1)
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_run(path: Path, record: RetrievalRunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def _fmt_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_scores(title: str, scores: RetrievalRunScore) -> str:
    lines = [
        f"  {title}",
        f"    답 있음 {scores.answerable}문항 (답 없음 {scores.excluded_no_answer}문항 제외)",
        f"    {'구분':<20} {'R@5 전부':>9} {'R@10 전부':>10} {'R@10 비율':>11} {'MRR':>6} {'NDCG@10':>8}",
    ]
    for s in scores.slices:
        lines.append(
            f"    {s.label:<20} {s.recall_all_at_5:>4}/{s.total:<4} "
            f"{s.recall_all_at_10:>4}/{s.total:<5} {s.recall_frac_at_10:>6.1f}/{s.total:<4} "
            f"{_fmt_metric(s.mrr):>6} {_fmt_metric(s.ndcg_at_10):>8}"
        )
    for na in scores.no_answer:
        top1 = "-" if na.top1_score is None else f"{na.top1_score:.4f}"
        lines.append(f"    답 없음 {na.question_id}: 1위 {na.top1_chunk_id} 점수 {top1}")
    return "\n".join(lines)


def default_out_dir(retriever: RetrieverName) -> Path:
    """`--out-dir`를 주지 않았을 때의 기록 위치. os 모드는 hybrid, 나머지는 #16 model_selection."""
    return DEFAULT_HYBRID_OUT_DIR if retriever.startswith("os-") else DEFAULT_OUT_DIR


def _mismatched(
    indexed: Mapping[str, IndexedMeta], expected: Mapping[str, ChunkHashes], field: str
) -> list[str]:
    return sorted(
        cid for cid, h in expected.items() if getattr(indexed[cid], field) != getattr(h, field)
    )


def check_alias_index(
    client: OpenSearch,
    alias: str,
    chunks: Sequence[Chunk],
    documents: Mapping[str, Document],
    passage_prefix: str,
) -> tuple[str, dict[str, IndexedMeta]]:
    """별칭이 가리키는 실제 인덱스 이름과 그 인덱스 문서의 메타를 돌려준다.

    채점은 청크 파일에서 정답을 뽑으므로, 인덱스가 청크 파일과 다르면 점수가 검색이 아니라
    데이터 차이를 잰다. 아래 중 하나라도 맞으면 ValueError로 멈춘다.
    - 별칭이 인덱스 하나를 가리키지 않는다
    - 인덱스의 chunk_id 집합이 청크 파일과 다르다
    - content_hash가 다르다: chunk_id는 같은데 text(또는 EMBED_PASSAGE_PREFIX)가 색인 때와 다르다.
      파서를 고친 뒤 다시 색인하지 않은 경우다
    - metadata_hash가 다르다: doc_id·page 등 메타데이터가 파싱 결과와 다르다
    해시는 색인 모듈과 같은 함수(`index_chunks.chunk_hashes`)로 계산한다.
    """
    targets = alias_targets(client, alias)
    if len(targets) != 1:
        raise ValueError(f"별칭 {alias}가 가리키는 인덱스가 {targets}다 — 하나여야 한다")
    [index_name] = targets
    indexed = fetch_indexed(client, index_name)
    chunk_ids = {c.chunk_id for c in chunks}
    missing = sorted(chunk_ids - set(indexed))
    extra = sorted(set(indexed) - chunk_ids)
    if missing or extra:
        raise ValueError(
            f"{index_name}의 문서가 청크 파일과 다르다: 인덱스에 없음 {len(missing)}개 "
            f"{missing[:5]}, 인덱스에만 있음 {len(extra)}개 {extra[:5]}"
        )
    expected = chunk_hashes(chunks, documents, passage_prefix)
    for field in ("content_hash", "metadata_hash"):
        differ = _mismatched(indexed, expected, field)
        if differ:
            raise ValueError(
                f"{index_name}의 {field}가 청크 파일과 다른 문서 {len(differ)}개 {differ[:5]} — "
                "청크 파일·파싱 결과로 다시 색인한 뒤 측정한다"
            )
    return index_name, indexed


def _resolve_revision_reader(
    cfg: ClientsConfig,
    compose_file: Path,
    revision_reader: Callable[[str], str] | None,
) -> tuple[str | None, Callable[[str], str]]:
    """(TEI 이미지 태그, revision 읽기 함수). revision_reader를 주지 않으면 TEI 볼륨을 읽는다."""
    tei_image = tei_image_tag(compose_file)
    if revision_reader is None:
        if tei_image is None:
            raise ValueError(f"{compose_file}에서 TEI 이미지를 찾지 못해 revision을 읽을 수 없다")
        revision_reader = volume_revision_reader(cfg, compose_file)
    return tei_image, revision_reader


def _check_query_embedder(
    query_embedder: TEIEmbeddingClient,
    cfg: ClientsConfig,
    revision_reader: Callable[[str], str],
    index_name: str,
    indexed_meta: Mapping[str, IndexedMeta],
) -> ServerInfo:
    """인덱스 문서의 embedding_model이 질의 TEI 서버와 같은지 확인하고 서버 정보를 돌려준다."""
    raw_info = query_embedder.info()
    # 색인과 같은 규칙(model_sha가 있으면 그 값, 없으면 볼륨 스냅샷)으로 만든 이름
    query_model = resolve_embedding_model(raw_info, cfg, revision_reader)
    index_models = {meta.embedding_model for meta in indexed_meta.values()}
    if index_models != {query_model}:
        raise ValueError(
            f"{index_name}의 embedding_model {sorted(index_models)}이 질의 임베딩 "
            f"서버({query_model})와 다르다"
        )
    info = server_info(raw_info)
    return info.model_copy(update={"snapshot_revision": query_model.split("@", 1)[1]})


def _make_rerank(
    cfg: ClientsConfig,
    reranker: TEIRerankerClient | None,
    revision_reader: Callable[[str], str],
    rerank_n: int,
) -> tuple[RerankFn, ServerInfo]:
    rerank_client = reranker or TEIRerankerClient(cfg)
    info = server_info(rerank_client.info())
    info = info.model_copy(update={"snapshot_revision": revision_reader(info.model_id)})

    def rerank(q: str, hits: list[SearchHit]) -> list[SearchHit]:
        return rerank_hits(q, hits, rerank_client, rerank_n)

    return rerank, info


def app_rrf_overlaps(
    client: OpenSearch,
    embedder: TEIEmbeddingClient,
    questions: Mapping[str, str],
) -> dict[str, int]:
    """문항마다 OpenSearch RRF 상위 10과 앱 RRF 상위 10의 chunk_id 겹침 수 (#18 결정 3).

    questions는 question_id → 질문. 앱 RRF는 bm25_search·knn_search를 후보 KNN_CANDIDATES개씩
    받아 `rrf_fuse`(RRF_RANK_CONSTANT)로 합친다 — hybrid_search와 같은 후보·같은 k다.
    """
    overlaps: dict[str, int] = {}
    for question_id, question in questions.items():
        [vector] = embedder.embed([question], input_type="query")
        hybrid = hybrid_search(client, question, vector, V2_TOP_K)
        app = rrf_fuse(
            [
                bm25_search(client, question, KNN_CANDIDATES),
                knn_search(client, vector, KNN_CANDIDATES),
            ],
            V2_TOP_K,
            RRF_RANK_CONSTANT,
        )
        overlaps[question_id] = len({h.chunk_id for h in hybrid} & {h.chunk_id for h in app})
    return overlaps


def run(
    *,
    retriever: RetrieverName,
    use_rerank: bool,
    qa_file: Path = DEFAULT_QA_V2,
    chunks_file: Path = DEFAULT_CHUNKS_FILE,
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    out_dir: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    compose_file: Path = DEFAULT_COMPOSE_FILE,
    tokenizer: TokenizerName = "bigram",
    rerank_n: int = DEFAULT_RERANK_N,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_LATENCY_REPEATS,
    seed: int = 0,
    config: ClientsConfig | None = None,
    embedder: TEIEmbeddingClient | None = None,
    reranker: TEIRerankerClient | None = None,
    revision_reader: Callable[[str], str] | None = None,
    os_client: OpenSearch | None = None,
) -> RetrievalRunRecord:
    """검색 조건 하나를 돌려 채점하고 기록을 남긴다.

    embedder·reranker를 주지 않으면 config(없으면 환경변수)로 TEI 클라이언트를 만든다.
    테스트는 가짜 transport를 넣은 클라이언트를 준다. revision_reader(모델 ID → 스냅샷
    해시)를 주지 않으면 TEI 볼륨을 docker로 읽는다(`clients.tei_revision`). os_client를 주지
    않으면 `OPENSEARCH_URL`로 OpenSearch 클라이언트를 만든다(os 모드). os 모드는 parsed_dir의
    파싱 결과로 인덱스 메타데이터를 대조한다. out_dir가 None이면 `default_out_dir(retriever)`.
    """
    if use_rerank and retriever not in RERANK_RETRIEVERS:
        if retriever == "bm25":
            raise ValueError("BM25 + 리랭크 조합은 이번 실험 범위가 아니다 (이슈 #16)")
        raise ValueError(f"리랭크는 dense·os-hybrid에만 붙는다 ({retriever})")
    out_dir = out_dir if out_dir is not None else default_out_dir(retriever)
    questions = load_questions_v2(qa_file)
    chunks = load_chunks(chunks_file)
    ran_at = datetime.now(UTC)
    gpu_before = gpu_snapshot()
    top_k = max(V2_TOP_K, rerank_n) if use_rerank else V2_TOP_K

    record_fields: dict[str, Any] = {}
    overlap_embedder: TEIEmbeddingClient | None = None
    if retriever == "bm25":
        bm25 = Bm25Index(chunks, get_tokenizer(tokenizer))
        search: SearchFn = lambda q: bm25.search(q, top_k)
        rerank: RerankFn | None = None
        name = f"bm25-{tokenizer}"
        record_fields["tokenizer"] = tokenizer
    elif retriever == "dense":
        cfg = config or ClientsConfig.from_env()
        embedder = embedder or TEIEmbeddingClient(cfg)
        tei_image, revision_reader = _resolve_revision_reader(cfg, compose_file, revision_reader)

        info = server_info(embedder.info())
        if cfg.embed_model_id and info.model_id != cfg.embed_model_id:
            raise ValueError(
                f"TEI 서버 모델({info.model_id})이 EMBED_MODEL_ID({cfg.embed_model_id})와 다르다"
            )
        info = info.model_copy(update={"snapshot_revision": revision_reader(info.model_id)})
        indexed = build_or_load_vectors(
            chunks,
            embedder,
            info,
            passage_prefix=cfg.embed_passage_prefix,
            truncate=cfg.embed_truncate,
            chunks_sha=file_sha256(chunks_file),
            cache_dir=cache_dir,
        )
        index = DenseIndex(chunks, indexed.vectors)

        def search(q: str) -> list[SearchHit]:
            [vector] = embedder.embed([q], input_type="query")
            return index.search_vector(vector, top_k)

        rerank = None
        name = f"dense-{info.model_id.split('/')[-1]}"
        record_fields |= {
            "embed": info,
            "query_prefix": cfg.embed_query_prefix,
            "passage_prefix": cfg.embed_passage_prefix,
            "truncate": cfg.embed_truncate,
            "truncated_chunks": len(indexed.truncated),
            # build_or_load_vectors가 의미 글자 없는 청크를 거절하므로 성공한 실행은 늘 0이다 (#62)
            "empty_chunks": 0,
            "empty_chunk_ids": [],
            "index_seconds": indexed.index_seconds,
            "vector_cache": str(indexed.cache_path),
            "tei_image": tei_image,
        }
        if use_rerank:
            rerank, rerank_info = _make_rerank(cfg, reranker, revision_reader, rerank_n)
            name += f"+rerank-{rerank_info.model_id.split('/')[-1]}"
            record_fields |= {"rerank": rerank_info, "rerank_n": rerank_n}
    elif retriever in ("os-bm25", "os-knn", "os-hybrid"):
        cfg = config or ClientsConfig.from_env()
        client = os_client or OpenSearch(hosts=[OpenSearchConfig.from_env().url])
        alias = search_alias()
        documents = load_documents((c.doc_id for c in chunks), parsed_dir)
        index_name, indexed_meta = check_alias_index(
            client, alias, chunks, documents, cfg.embed_passage_prefix
        )
        record_fields |= {
            "index_alias": alias,
            "index_name": index_name,
            "index_doc_count": len(indexed_meta),
        }
        rerank = None
        if retriever == "os-bm25":
            search = lambda q: bm25_search(client, q, top_k)
            name = "os-bm25"
        else:
            query_embedder = embedder or TEIEmbeddingClient(cfg)
            tei_image, revision_reader = _resolve_revision_reader(
                cfg, compose_file, revision_reader
            )
            info = _check_query_embedder(
                query_embedder, cfg, revision_reader, index_name, indexed_meta
            )
            model_name = info.model_id.split("/")[-1]
            record_fields |= {
                "embed": info,
                "query_prefix": cfg.embed_query_prefix,
                "tei_image": tei_image,
            }
            if retriever == "os-knn":

                def search(q: str) -> list[SearchHit]:
                    [vector] = query_embedder.embed([q], input_type="query")
                    return knn_search(client, vector, top_k)

                name = f"os-knn-{model_name}"
            else:

                def search(q: str) -> list[SearchHit]:
                    [vector] = query_embedder.embed([q], input_type="query")
                    return hybrid_search(client, q, vector, top_k)

                name = f"os-hybrid-{model_name}"
                overlap_embedder = query_embedder
                record_fields |= {
                    "rrf_rank_constant": RRF_RANK_CONSTANT,
                    "candidates": KNN_CANDIDATES,
                }
                if use_rerank:
                    rerank, rerank_info = _make_rerank(cfg, reranker, revision_reader, rerank_n)
                    name += f"+rerank-{rerank_info.model_id.split('/')[-1]}-n{rerank_n}"
                    record_fields |= {"rerank": rerank_info, "rerank_n": rerank_n}
    else:
        raise ValueError(f"알 수 없는 retriever: {retriever}")

    rankings, search_ms, rerank_ms = run_searches(
        questions, search, rerank, warmup=warmup, repeats=repeats
    )
    scores, results = score_run_v2(questions, chunks, rankings)
    gpu_after = gpu_snapshot()
    if overlap_embedder is not None:
        # 측정 반복이 끝난 뒤 따로 검색한다 — 지연 표본에 넣지 않는다
        question_text = {q.question_id: q.question for q in questions}
        overlaps = app_rrf_overlaps(
            client, overlap_embedder, {r.question_id: question_text[r.question_id] for r in results}
        )
        results = [
            r.model_copy(update={"app_rrf_overlap_at_10": overlaps[r.question_id]}) for r in results
        ]
        if overlaps:
            record_fields |= {
                "app_rrf_overlap_at_10_mean": sum(overlaps.values()) / len(overlaps),
                "app_rrf_overlap_at_10_min": min(overlaps.values()),
            }

    run_id = f"{ran_at.strftime('%Y%m%dT%H%M%SZ')}-{name}"
    questions_file = out_dir / f"{run_id}.questions.jsonl"
    failures_file = out_dir / "failures" / f"{run_id}.jsonl"
    write_questions(questions_file, results)
    write_failures(failures_file, results, rankings)
    record = RetrievalRunRecord(
        run_id=run_id,
        ran_at=ran_at.isoformat(),
        qa_file=str(qa_file),
        chunks_file=str(chunks_file),
        chunk_count=len(chunks),
        retriever=retriever,
        top_k=top_k,
        gpu=gpu_before["name"] if gpu_before else None,
        gpu_driver=gpu_before["driver"] if gpu_before else None,
        gpu_memory_used_mib_before=int(gpu_before["memory_used_mib"]) if gpu_before else None,
        gpu_memory_used_mib_after=int(gpu_after["memory_used_mib"]) if gpu_after else None,
        seed=seed,
        search_latency=latency_stats(search_ms),
        rerank_latency=latency_stats(rerank_ms),
        scores=scores,
        questions_file=str(questions_file),
        failures_file=str(failures_file),
        **record_fields,
    )
    append_run(out_dir / "runs.jsonl", record)
    logger.info("%s", format_scores(f"{run_id} ({len(chunks)}청크)", scores))
    return record


# --- 결정 규칙 적용 ---


def load_question_results(path: Path) -> dict[str, QuestionResult]:
    with path.open(encoding="utf-8") as f:
        rows = [QuestionResult.model_validate_json(line) for line in f if line.strip()]
    return {r.question_id: r for r in rows}


def compare(run_ids: Sequence[str], out_dir: Path = DEFAULT_OUT_DIR) -> list[dict[str, Any]]:
    """Recall@10(전부 적중)이 가장 많은 실행을 1위로 두고 나머지를 McNemar로 비교한다.

    1위가 여럿이면 run_ids에서 앞에 적은 것을 1위로 둔다. 두 실행의 문항 집합이 다르면
    ValueError를 낸다 — 다른 평가 세트끼리는 문항별 대조가 성립하지 않는다.
    """
    records: dict[str, RetrievalRunRecord] = {}
    with (out_dir / "runs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = RetrievalRunRecord.model_validate_json(line)
                records[rec.run_id] = rec
    missing = [r for r in run_ids if r not in records]
    if missing:
        raise ValueError(f"runs.jsonl에 없는 run_id: {missing}")
    per_run = {r: load_question_results(Path(records[r].questions_file)) for r in run_ids}
    hits = {r: sum(q.recall_all_at_10 for q in per_run[r].values()) for r in run_ids}
    best = max(run_ids, key=lambda r: hits[r])  # max는 동점이면 앞의 것을 고른다
    rows: list[dict[str, Any]] = []
    for r in run_ids:
        if set(per_run[r]) != set(per_run[best]):
            raise ValueError(f"{r}와 {best}의 문항 집합이 다르다")
        b = sum(
            per_run[best][q].recall_all_at_10 and not per_run[r][q].recall_all_at_10
            for q in per_run[best]
        )
        c = sum(
            per_run[r][q].recall_all_at_10 and not per_run[best][q].recall_all_at_10
            for q in per_run[best]
        )
        p = mcnemar_exact_p(b, c)
        rows.append(
            {
                "run_id": r,
                "recall_all_at_10": hits[r],
                "total": len(per_run[r]),
                "recall_frac_at_10": sum(q.recall_frac_at_10 for q in per_run[r].values()),
                "best_only": b,
                "this_only": c,
                "p_value": p,
                "tie_with_best": r == best or p >= SIGNIFICANCE,
            }
        )
    return rows


def format_compare(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "  1위 대비 McNemar 정확검정(양측), p ≥ 0.05면 동률 (이슈 #16 결정 규칙)",
        f"    {'run_id':<60} {'R@10 전부':>10} {'R@10 비율':>10} {'1위만':>5} {'이쪽만':>5} {'p':>7}  판정",
    ]
    for row in rows:
        verdict = "동률" if row["tie_with_best"] else "1위보다 낮음"
        lines.append(
            f"    {row['run_id']:<60} {row['recall_all_at_10']:>4}/{row['total']:<5} "
            f"{row['recall_frac_at_10']:>6.1f}    {row['best_only']:>5} {row['this_only']:>5} "
            f"{row['p_value']:>7.4f}  {verdict}"
        )
    lines.append(
        "    동률이면 VRAM → 색인 시간 → 지연 순으로 가벼운 모델을 고른다 (runs.jsonl 참고)"
    )
    return "\n".join(lines)


def _rerank_n(value: str) -> int:
    """`--rerank-n` 값 검사: 1 이상 MAX_RERANK_N 이하 정수만 받는다."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"정수가 아니다: {value!r}") from None
    if not 1 <= n <= MAX_RERANK_N:
        raise argparse.ArgumentTypeError(f"1 이상 {MAX_RERANK_N} 이하여야 한다: {n}")
    return n


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="임베딩·리랭커 선정 검색 평가 (이슈 #16).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bm25", action="store_true", help="BM25 기준선 (TEI 불필요)")
    mode.add_argument("--dense", action="store_true", help="TEI 임베딩 서버로 벡터 검색")
    mode.add_argument("--os-bm25", action="store_true", help="OpenSearch 별칭 BM25(nori) (#18)")
    mode.add_argument("--os-knn", action="store_true", help="OpenSearch 별칭 k-NN (#18, TEI 필요)")
    mode.add_argument(
        "--os-hybrid",
        action="store_true",
        help="OpenSearch hybrid(BM25 + k-NN, RRF) (#18 PR ②, TEI 필요)",
    )
    mode.add_argument("--compare", nargs="+", metavar="RUN_ID", help="결정 규칙 적용")
    parser.add_argument(
        "--rerank", action="store_true", help="--dense·--os-hybrid 결과 상위 N개를 리랭크"
    )
    parser.add_argument(
        "--rerank-n",
        type=_rerank_n,
        default=DEFAULT_RERANK_N,
        help=f"리랭크할 후보 수 (1~{MAX_RERANK_N}, 기본 {DEFAULT_RERANK_N})",
    )
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_V2)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="os 모드에서 인덱스 메타데이터를 대조할 파싱 결과",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"기록 위치 (기본: os 모드 {DEFAULT_HYBRID_OUT_DIR}, 나머지 {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tokenizer", choices=["bigram", "whitespace"], default="bigram")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_LATENCY_REPEATS)
    parser.add_argument(
        "--seed", type=int, default=0, help="기록용. 지금 검색·채점에는 무작위 요소가 없다"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """`uv run python -m rfp_pm_agent.eval.run_retrieval` 진입점."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.rerank and not (args.dense or args.os_hybrid):
        parser.error("--rerank는 --dense와 --os-hybrid에만 붙는다")
    if args.compare:
        out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
        logger.info("%s", format_compare(compare(args.compare, out_dir)))
        return 0
    retriever: RetrieverName = (
        "bm25"
        if args.bm25
        else "os-bm25"
        if args.os_bm25
        else "os-knn"
        if args.os_knn
        else "os-hybrid"
        if args.os_hybrid
        else "dense"
    )
    run(
        retriever=retriever,
        use_rerank=args.rerank,
        qa_file=args.qa,
        chunks_file=args.chunks,
        parsed_dir=args.parsed_dir,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        tokenizer=args.tokenizer,
        rerank_n=args.rerank_n,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
