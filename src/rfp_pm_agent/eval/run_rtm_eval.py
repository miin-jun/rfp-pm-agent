"""RTM 누락 검출 채점 (이슈 #39, docs/adr/0012-rtm-reranker-mapping.md, docs/data-design.md 7절).

규칙(2026-10-18 측정 전 고정). 연결은 `rtm.mapping.build_rtm`(기준값 `rtm.report.THRESHOLD` = 0.5)이 정한다.
- 시험 50건 = `pms_truth_v2.json`의 injected 누락 정답, 판정은 사용자 검수 `data/eval/rtm_injected_judgment.csv`
  - 없음: RTM이 누락 후보로 표시해야 정답
  - 있음·애매: RTM이 업무를 하나라도 연결하면 정답
  - 해당없음: 채점 제외
- 연습 = injected가 아닌 요구사항 366건. 기준값은 이 세트의 분포로 정했다. "누락으로 잘못 부른 수"를 센다.
  `llm_omitted`(정답 아님, 원래 업무 연결 없음)는 잘못 부른 수에서 빼고 따로 표기한다

시험은 한 번만 돌린다 — 결과 파일이 이미 있으면 덮어쓰지 않고 멈춘다. 리랭커·LLM을 부르지 않는다(점수표만 읽는다).

CLI: `uv run python -m rfp_pm_agent.eval.run_rtm_eval [--out data/eval/results/rtm/rtm_eval_v1.json]`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from rfp_pm_agent.config import RtmConfig
from rfp_pm_agent.rtm.mapping import RtmResult, build_rtm
from rfp_pm_agent.rtm.report import THRESHOLD, ScoreTable, load_scores

DEFAULT_JUDGMENTS = Path("data/eval/rtm_injected_judgment.csv")
DEFAULT_TRUTH = Path("data/eval/pms_truth_v2.json")
DEFAULT_OUT = Path("data/eval/results/rtm/rtm_eval_v1.json")

Judgment = Literal["있음", "애매", "없음", "해당없음"]
Expected = Literal["missing", "linked", "excluded"]
EXPECTED: dict[Judgment, Expected] = {
    "없음": "missing",
    "있음": "linked",
    "애매": "linked",
    "해당없음": "excluded",
}


class JudgmentRow(BaseModel):
    no: int
    project_id: str
    requirement_id: str
    judgment: Judgment
    matched_wbs_ids: list[str]
    note: str


class GradedItem(BaseModel):
    no: int
    project_id: str
    requirement_id: str
    judgment: Judgment
    expected: Expected
    predicted_missing: bool
    links: list[tuple[str, float]]  # RTM이 이은 업무(점수 높은 순)
    correct: bool | None  # 해당없음은 None


def load_judgments(path: Path) -> list[JudgmentRow]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            JudgmentRow.model_validate(
                r | {"matched_wbs_ids": [x for x in r["matched_wbs_ids"].split(";") if x]}
            )
            for r in csv.DictReader(f)
        ]


def run_rtm(tables: dict[str, ScoreTable]) -> dict[str, RtmResult]:
    return {
        pid: build_rtm(t.requirement_ids, t.wbs_ids, t.scores, threshold=THRESHOLD)
        for pid, t in tables.items()
    }


def grade_test(judgments: list[JudgmentRow], results: dict[str, RtmResult]) -> list[GradedItem]:
    items: list[GradedItem] = []
    for j in judgments:
        result = results[j.project_id]
        predicted_missing = j.requirement_id in result.missing
        expected = EXPECTED[j.judgment]
        correct: bool | None
        if expected == "excluded":
            correct = None
        elif expected == "missing":
            correct = predicted_missing
        else:
            correct = not predicted_missing
        items.append(
            GradedItem(
                no=j.no,
                project_id=j.project_id,
                requirement_id=j.requirement_id,
                judgment=j.judgment,
                expected=expected,
                predicted_missing=predicted_missing,
                links=result.links[j.requirement_id],
                correct=correct,
            )
        )
    return items


def grade_practice(
    tables: dict[str, ScoreTable],
    results: dict[str, RtmResult],
    llm_omitted: set[tuple[str, str]],
) -> dict[str, Any]:
    """연습 세트(injected가 아닌 요구사항)에서 누락 후보로 나온 것. llm_omitted는 따로 센다."""
    scored: list[tuple[str, str]] = []
    false_missing: list[tuple[str, str]] = []
    omitted: list[dict[str, Any]] = []
    for pid, table in tables.items():
        missing = set(results[pid].missing)
        for rid in table.requirement_ids:
            if rid in table.injected:
                continue
            key = (pid, rid)
            if key in llm_omitted:
                omitted.append(
                    {"project_id": pid, "requirement_id": rid, "predicted_missing": rid in missing}
                )
                continue
            scored.append(key)
            if rid in missing:
                false_missing.append(key)
    return {
        "total": len(scored),
        "false_missing": len(false_missing),
        "false_missing_ids": [f"{p}/{r}" for p, r in false_missing],
        "llm_omitted": omitted,
    }


def summarize(items: list[GradedItem]) -> dict[str, Any]:
    def count(expected: Expected) -> dict[str, int]:
        group = [i for i in items if i.expected == expected]
        return {"total": len(group), "correct": sum(bool(i.correct) for i in group)}

    scored = [i for i in items if i.correct is not None]
    return {
        "judgments": dict(Counter(i.judgment for i in items)),
        "missing": count("missing"),
        "linked": count("linked"),
        "excluded": len(items) - len(scored),
        "overall": {"total": len(scored), "correct": sum(bool(i.correct) for i in scored)},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sets(
    judgments: list[JudgmentRow], tables: dict[str, ScoreTable], truth: dict[str, Any]
) -> set[tuple[str, str]]:
    """판정 파일·점수표·정답 파일의 시험 문제가 같은지 확인하고 llm_omitted 목록을 돌려준다."""
    injected_truth: set[tuple[str, str]] = set()
    llm_omitted: set[tuple[str, str]] = set()
    for p in truth["projects"]:
        for m in p["missing_requirements"]:
            key = (p["project_id"], m["requirement_id"])
            if m["reason"] == "injected" and m["use_for_scoring"]:
                injected_truth.add(key)
            elif m["reason"] == "llm_omitted":
                llm_omitted.add(key)
    judged = {(j.project_id, j.requirement_id) for j in judgments}
    in_scores = {(pid, rid) for pid, t in tables.items() for rid in t.injected}
    if not (judged == injected_truth == in_scores) or len(judged) != len(judgments):
        raise ValueError("판정 파일·정답 파일·점수표의 injected 요구사항이 서로 다르다")
    return llm_omitted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out}가 이미 있다 — 시험은 한 번만 돌린다(덮어쓰지 않음)")

    scores_path = RtmConfig.from_env().scores_path
    tables = load_scores(scores_path)
    judgments = load_judgments(args.judgments)
    truth = json.loads(args.truth.read_text("utf-8"))
    llm_omitted = check_sets(judgments, tables, truth)

    results = run_rtm(tables)
    items = grade_test(judgments, results)
    record = {
        "ran_at": datetime.now(UTC).isoformat(),
        "threshold": THRESHOLD,
        "inputs": {str(p): _sha256(p) for p in (scores_path, args.judgments, args.truth)},
        "test": summarize(items),
        "practice": grade_practice(tables, results, llm_omitted),
        "items": [i.model_dump() for i in items],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: record[k] for k in ("threshold", "test", "practice")}, ensure_ascii=False, indent=1
        )
    )
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
