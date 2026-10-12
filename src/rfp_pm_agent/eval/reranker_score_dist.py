"""리랭커 1위 점수 분포 측정 — 답 있음/답 없음을 점수로 가를 수 있는지 (이슈 #19).

qa_v2 문항 전부를 서비스 경로 `search_documents(top_k=10, 기본 설정)`로 검색하고, 문항마다
1위 점수·상위 10 점수·정답 청크의 순위와 점수를 JSONL로 남긴다. 문항은 세 그룹으로 나눈다.

- A: 답 있음 + 모든 evidence의 정답 청크가 상위 10 안 (검색 평가의 Recall@10 "전부 적중"과 같은 기준)
- B: 답 있음 + 정답 청크가 상위 10에 다 들어오지 못함
- C: 답 없음 (`answer`가 null)

정답 청크는 `gold_chunk_ids`의 `block_requirement`(색인된 청킹 방식) 기준이다.
ADR-0002 "후속"과 data-design.md 5절 `score` 설명이 #19로 넘긴 확인이고, 결과 해석은
docs/adr/0006-grounded-answer-refusal-and-citation.md에 있다.

OpenSearch·tei-embed·tei-rerank가 떠 있어야 한다. LLM을 부르지 않으므로 이 실행의 비용은 $0이다.

CLI: `uv run python -m rfp_pm_agent.eval.reranker_score_dist --out data/eval/results/answer/<파일>.jsonl`
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from rfp_pm_agent.eval.qa_v2 import load_questions_v2
from rfp_pm_agent.schemas.eval import EvalQuestionV2
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.tools.search_documents import SearchDeps, search_documents

DEFAULT_QA_FILE = Path("data/eval/qa_v2.jsonl")
# 색인된 청킹 방식. gold_chunk_ids에서 이 키의 정답 청크를 쓴다
GOLD_METHOD = "block_requirement"
TOP_K = 10
# 1위 청크 text에서 남길 앞부분 길이. 답 없음 문항의 1위가 어떤 청크인지 보는 용도다
TOP1_TEXT_CHARS = 50
# "1위 점수 < t 이면 거절"로 볼 임계값 후보
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)

Group = Literal["A", "B", "C"]


class EvidenceHit(BaseModel):
    """evidence 하나에 대해, 그것을 담은 청크 중 상위 10에서 가장 높은 순위. 없으면 모두 None."""

    rank: int | None
    chunk_id: str | None
    score: float | None


class ScoreDistRow(BaseModel):
    """문항 하나의 측정 결과(JSONL 한 줄)."""

    qid: str
    group: Group
    answerable: bool
    tags: list[str]
    question: str
    top1_score: float | None
    top10_scores: list[float]
    top10_chunk_ids: list[str]
    gold_all_in_top10: bool
    gold_any_in_top10: bool
    gold_best_rank: int | None
    gold_best_score: float | None
    gold_per_evidence: list[EvidenceHit]
    top1_chunk_id: str | None
    top1_text_head: str | None
    latency_ms: float


class ThresholdRow(BaseModel):
    """임계값 t에서 "1위 점수 < t 이면 거절"했을 때의 그룹별 문항 수."""

    threshold: float
    a_rejected: int  # 정답을 찾았는데 거절(잘못 거절)
    b_rejected: int
    c_passed: int  # 답이 없는데 거절하지 못함


def build_row(
    question: EvalQuestionV2, hits: Sequence[DocumentHit], latency_ms: float
) -> ScoreDistRow:
    """문항 하나와 그 검색 결과(순위 순)로 측정 결과 한 줄을 만든다.

    evidence마다 그것을 담은 청크 중 가장 높은 순위를 찾는다. 모든 evidence가 상위 10에 있으면 A,
    하나라도 없으면 B, 답 없음 문항은 C다.
    """
    answerable = question.answer is not None
    rank_of = {h.chunk_id: h.rank for h in hits}
    score_of = {h.chunk_id: h.score for h in hits}

    per_evidence: list[EvidenceHit] = []
    for chunk_ids in question.gold_chunk_ids.get(GOLD_METHOD, []):
        found = sorted((rank_of[c], c) for c in chunk_ids if c in rank_of)
        if found:
            rank, chunk_id = found[0]
            per_evidence.append(EvidenceHit(rank=rank, chunk_id=chunk_id, score=score_of[chunk_id]))
        else:
            per_evidence.append(EvidenceHit(rank=None, chunk_id=None, score=None))

    found_hits = [e for e in per_evidence if e.rank is not None]
    gold_all = answerable and len(per_evidence) > 0 and len(found_hits) == len(per_evidence)
    best = min(found_hits, key=lambda e: e.rank or 0, default=None)

    group: Group = "C" if not answerable else ("A" if gold_all else "B")
    return ScoreDistRow(
        qid=question.question_id,
        group=group,
        answerable=answerable,
        tags=list(question.tags),
        question=question.question,
        top1_score=hits[0].score if hits else None,
        top10_scores=[h.score for h in hits],
        top10_chunk_ids=[h.chunk_id for h in hits],
        gold_all_in_top10=gold_all,
        gold_any_in_top10=answerable and len(found_hits) > 0,
        gold_best_rank=best.rank if best else None,
        gold_best_score=best.score if best else None,
        gold_per_evidence=per_evidence,
        top1_chunk_id=hits[0].chunk_id if hits else None,
        top1_text_head=hits[0].text[:TOP1_TEXT_CHARS].replace("\n", " ") if hits else None,
        latency_ms=round(latency_ms, 1),
    )


def top1_scores(rows: Sequence[ScoreDistRow], group: Group) -> list[float]:
    """그 그룹 문항의 1위 점수. 검색 결과가 0건인 문항은 뺀다."""
    return [r.top1_score for r in rows if r.group == group and r.top1_score is not None]


def threshold_table(
    rows: Sequence[ScoreDistRow], thresholds: Sequence[float] = THRESHOLDS
) -> list[ThresholdRow]:
    """임계값마다 A·B에서 거절되는 문항 수와 C에서 통과하는 문항 수를 센다."""
    a, b, c = (top1_scores(rows, g) for g in ("A", "B", "C"))
    return [
        ThresholdRow(
            threshold=t,
            a_rejected=sum(x < t for x in a),
            b_rejected=sum(x < t for x in b),
            c_passed=sum(x >= t for x in c),
        )
        for t in thresholds
    ]


def format_summary(rows: Sequence[ScoreDistRow]) -> str:
    """그룹별 1위 점수 분포, 임계값 표, 답 없음 문항의 1위 청크, 지연을 사람이 읽는 글로 만든다."""
    lines = ["== 1위 점수 분포 (최소 / 25% / 중앙값 / 75% / 최대)"]
    scores: dict[Group, list[float]] = {}
    groups: tuple[Group, ...] = ("A", "B", "C")
    for group in groups:
        scores[group] = top1_scores(rows, group)
        if scores[group]:
            q = np.percentile(scores[group], [0, 25, 50, 75, 100])
            lines.append(f"{group} n={len(scores[group])} " + " / ".join(f"{v:.3f}" for v in q))
        else:
            lines.append(f"{group} n=0")

    if scores["A"] and scores["C"]:
        min_a, max_c = min(scores["A"]), max(scores["C"])
        lines.append(f"\n== A와 C가 완전히 갈리는가: min(A)={min_a:.3f} > max(C)={max_c:.3f} → ")
        lines[-1] += "갈린다" if min_a > max_c else "겹친다"
        lines.append(f"A 중 max(C) 이하: {sum(x <= max_c for x in scores['A'])}/{len(scores['A'])}")

    lines.append("\n== 임계값 t (1위 점수 < t 이면 거절): A 잘못 거절 | B 거절 | C 통과")
    lines.extend(
        f"{t.threshold} | {t.a_rejected}/{len(scores['A'])} | {t.b_rejected}/{len(scores['B'])}"
        f" | {t.c_passed}/{len(scores['C'])}"
        for t in threshold_table(rows)
    )

    lines.append("\n== C 문항의 1위")
    lines.extend(
        f"{r.qid} {r.top1_score:.3f} | {r.top1_chunk_id} | {r.top1_text_head}"
        for r in rows
        if r.group == "C" and r.top1_score is not None
    )

    latencies = [r.latency_ms for r in rows]
    if latencies:
        p50, p95 = np.percentile(latencies, [50, 95])
        lines.append(f"\n== 검색 지연(ms) n={len(latencies)} p50={p50:.0f} p95={p95:.0f}")
    return "\n".join(lines)


def run(questions: Sequence[EvalQuestionV2], deps: SearchDeps) -> list[ScoreDistRow]:
    """문항마다 `search_documents`를 1회 부르고 결과를 모은다. 지연은 그 호출 1회의 시간이다."""
    # 첫 호출의 연결 수립 시간을 지연 통계에서 빼려고 평가 문항이 아닌 질의로 1회 부른다
    search_documents("제안요청서", top_k=1, deps=deps)
    rows: list[ScoreDistRow] = []
    for question in questions:
        start = time.monotonic()
        hits = search_documents(question.question, top_k=TOP_K, deps=deps)
        rows.append(build_row(question, hits, (time.monotonic() - start) * 1000))
    return rows


def write_rows(path: Path, rows: Sequence[ScoreDistRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="리랭커 1위 점수 분포 측정 (이슈 #19).")
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--out", type=Path, required=True, help="문항별 결과를 쓸 jsonl 경로")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = run(load_questions_v2(args.qa_file), SearchDeps.from_env())
    write_rows(args.out, rows)
    print(f"{len(rows)}문항 → {args.out}")
    print(format_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
