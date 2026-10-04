"""청킹 방식 비교 평가 실행 (이슈 #13).

방식 3개 × 토큰화 2개를 BM25로 검색해 hit@1·3·5와 최고 점수를 낸다. 실행할
때마다 조합 하나당 data/eval/runs.jsonl에 한 줄씩 덧붙이고, hit@5를 못 맞힌
질문의 상위 5개 청크를 data/eval/failures/{run_id}.jsonl에 남긴다. 두 파일은
run_id로 이어진다.

검색 범위는 방식별로 10개 문서의 청크 전체다. 문서별로 나누지 않는다.

LLM을 부르지 않으므로 이 실행의 비용은 $0이다.

CLI: `uv run python -m rfp_pm_agent.eval.run_chunk_eval`
     `uv run python -m rfp_pm_agent.eval.run_chunk_eval --ceiling-only`
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rfp_pm_agent.eval.retrieval import TOP_K, ceiling_only, score_run
from rfp_pm_agent.ingest.chunking import load_chunks
from rfp_pm_agent.schemas.eval import (
    EvalQuestion,
    RunRecord,
    RunScore,
    SearchHit,
    TokenizerName,
    TypeScore,
)
from rfp_pm_agent.search.bm25 import DEFAULT_B, DEFAULT_K1, Bm25Index
from rfp_pm_agent.search.tokenize import get_tokenizer

DEFAULT_QA_FILE = Path("data/eval/qa_v1.jsonl")
DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_RUNS_FILE = Path("data/eval/runs.jsonl")
DEFAULT_FAILURES_DIR = Path("data/eval/failures")

METHODS = ("block", "requirement", "block_requirement")
TOKENIZER_NAMES: tuple[TokenizerName, ...] = ("bigram", "whitespace")

# failures 파일에 남길 청크 text 앞부분 길이. 전체를 남기면 파일이 커지고,
# 너무 짧으면 "목차 줄이 상위를 차지했다" 같은 판단을 할 수 없다
FAILURE_TEXT_CHARS = 200

logger = logging.getLogger(__name__)


def load_questions(path: Path) -> list[EvalQuestion]:
    """평가 세트 jsonl을 읽어 검증한다. 형식이 어긋나면 그 줄에서 멈춘다."""
    with path.open(encoding="utf-8") as f:
        return [EvalQuestion.model_validate_json(line) for line in f if line.strip()]


def append_run(path: Path, record: RunRecord) -> None:
    """runs.jsonl에 한 줄 덧붙인다. 기존 줄은 다시 쓰지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def write_failures(path: Path, missed: dict[str, list[SearchHit]]) -> None:
    """틀린 질문마다 상위 청크를 한 줄씩 쓴다 (chunk_id, 점수, text 앞부분)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for question_id in sorted(missed):
            row = {
                "question_id": question_id,
                "top_hits": [
                    {
                        "rank": rank,
                        "chunk_id": hit.chunk_id,
                        "score": round(hit.score, 4),
                        "text_head": hit.text[:FAILURE_TEXT_CHARS],
                    }
                    for rank, hit in enumerate(missed[question_id], start=1)
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_type_score(label: str, score: TypeScore) -> str:
    return (
        f"    {label:<8} n={score.total:<3} "
        f"hit@1={score.hit_at_1:<3} hit@3={score.hit_at_3:<3} hit@5={score.hit_at_5:<3} "
        f"최고={score.ceiling}"
    )


def format_scores(title: str, scores: RunScore) -> str:
    lines = [
        f"  {title}",
        _format_type_score("전체", scores.overall),
        _format_type_score("요구사항", scores.requirement),
        _format_type_score("일반", scores.general),
    ]
    if scores.missed_question_ids:
        lines.append(
            f"    틀림({len(scores.missed_question_ids)}): " + ", ".join(scores.missed_question_ids)
        )
    return "\n".join(lines)


def run(
    *,
    qa_file: Path = DEFAULT_QA_FILE,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    runs_file: Path = DEFAULT_RUNS_FILE,
    failures_dir: Path = DEFAULT_FAILURES_DIR,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    top_k: int = TOP_K,
    ceiling_only_mode: bool = False,
) -> list[RunRecord]:
    """방식 × 토큰화 조합을 모두 돌리고 기록한 RunRecord 목록을 돌려준다."""
    questions = load_questions(qa_file)
    ran_at = datetime.now(UTC)
    stamp = ran_at.strftime("%Y%m%dT%H%M%SZ")
    records: list[RunRecord] = []

    for method in METHODS:
        chunks_file = chunks_dir / f"{method}.jsonl"
        chunks = load_chunks(chunks_file)

        if ceiling_only_mode:
            # 최고 점수는 검색과 무관하므로 토큰화와 상관없이 한 번만 낸다
            scores = ceiling_only(questions, chunks)
            logger.info("%s", format_scores(f"{method} (최고 점수만)", scores))
            continue

        for tokenizer_name in TOKENIZER_NAMES:
            run_id = f"{stamp}-{method}-{tokenizer_name}"
            index = Bm25Index(chunks, get_tokenizer(tokenizer_name), k1=k1, b=b)
            scores, missed = score_run(questions, chunks, index.search, top_k=top_k)

            failures_file = failures_dir / f"{run_id}.jsonl"
            write_failures(failures_file, missed)

            record = RunRecord(
                run_id=run_id,
                ran_at=ran_at.isoformat(),
                qa_file=str(qa_file),
                chunks_file=str(chunks_file),
                method=method,
                tokenizer=tokenizer_name,
                k1=k1,
                b=b,
                top_k=top_k,
                chunk_count=len(chunks),
                scores=scores,
                failures_file=str(failures_file),
            )
            append_run(runs_file, record)
            records.append(record)
            logger.info(
                "%s", format_scores(f"{method} / {tokenizer_name} ({len(chunks)}청크)", scores)
            )

    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="청킹 방식 3개를 BM25 검색으로 비교한다 (이슈 #13)."
    )
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_FILE, help="평가 세트 jsonl")
    parser.add_argument(
        "--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR, help="청크 jsonl이 있는 디렉터리"
    )
    parser.add_argument(
        "--runs", type=Path, default=DEFAULT_RUNS_FILE, help="실행 기록을 덧붙일 jsonl"
    )
    parser.add_argument(
        "--failures-dir",
        type=Path,
        default=DEFAULT_FAILURES_DIR,
        help="틀린 질문 상위 청크를 남길 디렉터리",
    )
    parser.add_argument("--k1", type=float, default=DEFAULT_K1, help=f"BM25 k1 (기본 {DEFAULT_K1})")
    parser.add_argument(
        "--b", type=float, default=DEFAULT_B, help=f"BM25 b, 문서 길이 보정 (기본 {DEFAULT_B})"
    )
    parser.add_argument(
        "--ceiling-only",
        action="store_true",
        help="검색 없이 최고 점수만 낸다. runs.jsonl과 failures 파일을 쓰지 않는다",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """`uv run python -m rfp_pm_agent.eval.run_chunk_eval` 진입점."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    run(
        qa_file=args.qa,
        chunks_dir=args.chunks_dir,
        runs_file=args.runs,
        failures_dir=args.failures_dir,
        k1=args.k1,
        b=args.b,
        ceiling_only_mode=args.ceiling_only,
    )


if __name__ == "__main__":
    main()
