"""이슈 #39 RTM 1단계: 요구사항×업무 관련도 점수표를 만든다.

사업마다 요구사항 전체 × 그 사업의 업무(level 1 단계 업무 제외) 모든 쌍을 TEI 리랭커로 점수 매긴다.
query = 요구사항 원문(data/parsed `Requirement.text`, 검수표 `rtm_injected_review.csv`와 같은 원문),
document = 업무 이름. 짝짓기·기준 점수는 여기서 정하지 않는다.

입력: data/eval/pms_truth_v2.json, data/synth/pms_v2.json, data/parsed/<doc_id>.json,
      data/chunks/block_requirement.jsonl(근거 청크 ID 확인), data/eval/rtm_injected_review.csv(원문 대조)
출력: data/eval/rtm_scores_v1.csv — project_id, requirement_id, chunk_id, wbs_id, score, is_injected

실행(`docker compose up -d tei-rerank` 필요): uv run python scripts/build_rtm_scores.py
"""

from __future__ import annotations

import csv
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any

from rfp_pm_agent.clients.reranker import TEIRerankerClient
from rfp_pm_agent.config import ClientsConfig
from rfp_pm_agent.schemas.document import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
TRUTH_PATH = REPO_ROOT / "data" / "eval" / "pms_truth_v2.json"
DATASET_PATH = REPO_ROOT / "data" / "synth" / "pms_v2.json"
PARSED_DIR = REPO_ROOT / "data" / "parsed"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks" / "block_requirement.jsonl"
REVIEW_PATH = REPO_ROOT / "data" / "eval" / "rtm_injected_review.csv"
OUT_PATH = REPO_ROOT / "data" / "eval" / "rtm_scores_v1.csv"

COLUMNS = ["project_id", "requirement_id", "chunk_id", "wbs_id", "score", "is_injected"]


def wbs_sort_key(wbs_id: str) -> tuple[int, ...]:
    return tuple(int(part) for part in wbs_id.split("."))


def requirement_text(text: str) -> str:
    """검수표와 같은 원문 — scripts/build_rtm_injected_review.py와 같은 규칙(줄 끝 공백만 제거)."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def parsed_texts(doc_id: str) -> dict[str, str]:
    """요구사항 ID → 원문. 같은 ID가 두 번 나오면 처음 것(pms.synth.build_requirements와 같은 규칙)."""
    doc = Document.model_validate_json((PARSED_DIR / f"{doc_id}.json").read_text("utf-8"))
    texts: dict[str, str] = {}
    for req in sorted(doc.requirements, key=lambda r: r.source_order):
        texts.setdefault(req.requirement_id, requirement_text(req.text))
    return texts


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    truth = json.loads(TRUTH_PATH.read_text("utf-8"))
    dataset = json.loads(DATASET_PATH.read_text("utf-8"))
    if dataset["as_of_date"] != truth["as_of_date"] or dataset["seed"] != truth["seed"]:
        raise SystemExit("pms_v2.json과 pms_truth_v2.json의 as_of_date·seed가 다르다")
    return truth, dataset


def build_scores(reranker: TEIRerankerClient) -> list[dict[str, str]]:
    truth, dataset = load_inputs()
    chunk_ids: set[str] = set()
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk_ids.add(json.loads(line)["chunk_id"])
    with REVIEW_PATH.open(encoding="utf-8-sig", newline="") as f:
        review = {(r["project_id"], r["requirement_id"]): r for r in csv.DictReader(f)}

    rows: list[dict[str, str]] = []
    for project in truth["projects"]:
        pid = project["project_id"]
        injected = {
            m["requirement_id"]
            for m in project["missing_requirements"]
            if m["reason"] == "injected" and m["use_for_scoring"]
        }
        texts = parsed_texts(project["doc_id"])
        requirements = [r for r in dataset["requirements"] if r["project_id"] == pid]
        if {r["requirement_id"] for r in requirements} != set(texts):
            raise SystemExit(f"{pid}: pms_v2.json 요구사항과 원문 요구사항 ID가 다르다")
        items = sorted(
            (w for w in dataset["wbs_items"] if w["project_id"] == pid and w["parent_wbs_id"]),
            key=lambda w: wbs_sort_key(w["wbs_id"]),
        )
        names = [w["name"] for w in items]

        for req in requirements:
            rid = req["requirement_id"]
            if req["source_chunk_id"] not in chunk_ids:
                raise SystemExit(
                    f"{pid}/{rid}: 근거 청크가 청크 파일에 없다 {req['source_chunk_id']}"
                )
            query = texts[rid]
            if rid in injected and review[(pid, rid)]["requirement_text"] != query:
                raise SystemExit(f"{pid}/{rid}: 검수표 원문과 다르다")
            scores = reranker.rerank(query, names)
            if len(scores) != len(items):
                raise SystemExit(f"{pid}/{rid}: 점수 {len(scores)}개, 업무 {len(items)}개")
            for item, score in zip(items, scores, strict=True):
                rows.append(
                    {
                        "project_id": pid,
                        "requirement_id": rid,
                        "chunk_id": req["source_chunk_id"],
                        "wbs_id": item["wbs_id"],
                        "score": repr(score),
                        "is_injected": str(rid in injected).lower(),
                    }
                )
        if not injected <= {r["requirement_id"] for r in requirements}:
            raise SystemExit(f"{pid}: injected 요구사항이 요구사항 목록에 없다")
    return rows


def render_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def quartiles(values: list[float]) -> str:
    q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return f"n={len(values)} 25%={q1:.4f} 중앙값={q2:.4f} 75%={q3:.4f}"


def report(rows: list[dict[str, str]]) -> None:
    """진짜 짝(wbs_requirements) vs 짝 아닌 쌍의 분포. injected 요구사항은 시험 문제라 뺀다."""
    _, dataset = load_inputs()
    linked = {
        (x["project_id"], x["requirement_id"], x["wbs_id"]) for x in dataset["wbs_requirements"]
    }
    pair: list[float] = []
    other: list[float] = []
    for row in rows:
        if row["is_injected"] == "true":
            continue
        key = (row["project_id"], row["requirement_id"], row["wbs_id"])
        (pair if key in linked else other).append(float(row["score"]))
    print(f"진짜 짝:    {quartiles(pair)}")
    print(f"짝 아닌 쌍: {quartiles(other)}")


def main() -> None:
    reranker = TEIRerankerClient(ClientsConfig.from_env())
    if not reranker.health():
        raise SystemExit(f"리랭커가 응답하지 않는다: {reranker.base_url}")
    info = reranker.info()
    print(f"리랭커 {info.get('model_id')} ({reranker.base_url})")
    started = time.perf_counter()
    rows = build_scores(reranker)
    elapsed = time.perf_counter() - started
    OUT_PATH.write_text(render_csv(rows), encoding="utf-8")
    requirements = {(r["project_id"], r["requirement_id"]) for r in rows}
    injected = {(r["project_id"], r["requirement_id"]) for r in rows if r["is_injected"] == "true"}
    print(
        f"{len(rows)}쌍 (요구사항 {len(requirements)}, injected {len(injected)}) "
        f"{elapsed:.1f}초 → {OUT_PATH.relative_to(REPO_ROOT)}"
    )
    report(rows)


if __name__ == "__main__":
    main()
