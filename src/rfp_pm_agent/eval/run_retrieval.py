"""검색 평가 실행 — 임베딩·리랭커 선정 실험 (이슈 #16).

qa_v2(57문항, 답 있음 51)를 block_requirement 청크로 검색해 채점한다. 한 번 실행에
검색 조건 하나(BM25 / 임베딩 모델 1개 / 임베딩 + 리랭커)를 돌린다. TEI 서버는 한 번에
모델 하나만 띄우므로(learning-log 2026-09-24, RAM), 모델을 바꿀 때마다 서버를 다시 띄우고
이 명령을 다시 실행한다. 측정 절차는 docs/model-selection-measurement.md.

기록 (data/eval/results/model_selection/):
- runs.jsonl: 실행 한 줄(RetrievalRunRecord) — 모델 ID·revision, TEI 버전·이미지, GPU,
  접두어, 자르기 여부·자른 청크 수, 색인 시간, 지연 p50·p95, 집계 점수
- {run_id}.questions.jsonl: 답 있는 문항별 결과 — 실행 간 McNemar 비교에 쓴다
- failures/{run_id}.jsonl: Recall@10(전부 적중)을 못 맞힌 문항의 상위 10개

결정 규칙(이슈 #16, 측정 전 고정)은 `--compare`로 적용한다: Recall@10 전부 적중 수가
가장 많은 실행과 나머지를 McNemar 정확검정(양측)으로 비교하고, p ≥ 0.05면 동률로 표시한다.
동률일 때 가벼운 모델 고르기(VRAM → 색인 시간 → 지연)는 사람이 기록을 보고 한다.

MRR·NDCG 함수가 아직 없으면(소유자 구현 대기) 그 칸은 비우고 나머지를 계산한다.
LLM을 부르지 않으므로 비용은 $0이다.

CLI: `uv run python -m rfp_pm_agent.eval.run_retrieval --bm25`
     `uv run python -m rfp_pm_agent.eval.run_retrieval --dense [--rerank]`
     `uv run python -m rfp_pm_agent.eval.run_retrieval --compare <run_id> <run_id> ...`
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rfp_pm_agent.clients.embedding import TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import TEIRerankerClient
from rfp_pm_agent.config import ClientsConfig
from rfp_pm_agent.eval.qa_v2 import DEFAULT_QA_V2, load_questions_v2
from rfp_pm_agent.eval.retrieval import V2_TOP_K, mcnemar_exact_p, score_run_v2
from rfp_pm_agent.eval.run_chunk_eval import load_chunks
from rfp_pm_agent.schemas.eval import (
    EvalQuestionV2,
    LatencyStats,
    QuestionResult,
    RetrievalRunRecord,
    RetrievalRunScore,
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
from rfp_pm_agent.search.tokenize import get_tokenizer

DEFAULT_CHUNKS_FILE = Path("data/chunks/block_requirement.jsonl")
DEFAULT_OUT_DIR = Path("data/eval/results/model_selection")
DEFAULT_COMPOSE_FILE = Path("docker-compose.yml")

# 이슈 #16 결정 규칙에서 고정한 값
DEFAULT_RERANK_N = 20
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


def tei_image_tag(compose_file: Path) -> str | None:
    """docker-compose.yml에 적힌 TEI 이미지 태그. 실제로 떠 있는 컨테이너 이미지는 아니다."""
    if not compose_file.exists():
        return None
    match = re.search(r"image:\s*(\S*text-embeddings-inference\S*)", compose_file.read_text())
    return match.group(1) if match else None


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


def run(
    *,
    retriever: str,
    use_rerank: bool,
    qa_file: Path = DEFAULT_QA_V2,
    chunks_file: Path = DEFAULT_CHUNKS_FILE,
    out_dir: Path = DEFAULT_OUT_DIR,
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
) -> RetrievalRunRecord:
    """검색 조건 하나를 돌려 채점하고 기록을 남긴다.

    embedder·reranker를 주지 않으면 config(없으면 환경변수)로 TEI 클라이언트를 만든다.
    테스트는 가짜 transport를 넣은 클라이언트를 준다.
    """
    questions = load_questions_v2(qa_file)
    chunks = load_chunks(chunks_file)
    ran_at = datetime.now(UTC)
    gpu_before = gpu_snapshot()
    top_k = max(V2_TOP_K, rerank_n) if use_rerank else V2_TOP_K

    record_fields: dict[str, Any] = {}
    if retriever == "bm25":
        if use_rerank:
            raise ValueError("BM25 + 리랭크 조합은 이번 실험 범위가 아니다 (이슈 #16)")
        bm25 = Bm25Index(chunks, get_tokenizer(tokenizer))
        search: SearchFn = lambda q: bm25.search(q, top_k)
        rerank: RerankFn | None = None
        name = f"bm25-{tokenizer}"
        record_fields["tokenizer"] = tokenizer
    elif retriever == "dense":
        cfg = config or ClientsConfig.from_env()
        embedder = embedder or TEIEmbeddingClient(cfg)
        info = server_info(embedder.info())
        if cfg.embed_model_id and info.model_id != cfg.embed_model_id:
            raise ValueError(
                f"TEI 서버 모델({info.model_id})이 EMBED_MODEL_ID({cfg.embed_model_id})와 다르다"
            )
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
            "index_seconds": indexed.index_seconds,
            "vector_cache": str(indexed.cache_path),
            "tei_image": tei_image_tag(compose_file),
        }
        if use_rerank:
            rerank_client = reranker or TEIRerankerClient(cfg)
            rerank_info = server_info(rerank_client.info())

            def rerank(q: str, hits: list[SearchHit]) -> list[SearchHit]:
                return rerank_hits(q, hits, rerank_client, rerank_n)

            name += f"+rerank-{rerank_info.model_id.split('/')[-1]}"
            record_fields |= {"rerank": rerank_info, "rerank_n": rerank_n}
    else:
        raise ValueError(f"알 수 없는 retriever: {retriever}")

    rankings, search_ms, rerank_ms = run_searches(
        questions, search, rerank, warmup=warmup, repeats=repeats
    )
    scores, results = score_run_v2(questions, chunks, rankings)
    gpu_after = gpu_snapshot()

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
        retriever="bm25" if retriever == "bm25" else "dense",
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="임베딩·리랭커 선정 검색 평가 (이슈 #16).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bm25", action="store_true", help="BM25 기준선 (TEI 불필요)")
    mode.add_argument("--dense", action="store_true", help="TEI 임베딩 서버로 벡터 검색")
    mode.add_argument("--compare", nargs="+", metavar="RUN_ID", help="결정 규칙 적용")
    parser.add_argument("--rerank", action="store_true", help="--dense 결과 상위 N개를 리랭크")
    parser.add_argument("--rerank-n", type=int, default=DEFAULT_RERANK_N)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_V2)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
    args = build_arg_parser().parse_args(argv)
    if args.compare:
        logger.info("%s", format_compare(compare(args.compare, args.out_dir)))
        return 0
    run(
        retriever="bm25" if args.bm25 else "dense",
        use_rerank=args.rerank,
        qa_file=args.qa,
        chunks_file=args.chunks,
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
