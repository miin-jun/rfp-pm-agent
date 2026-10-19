"""평가 기준선 검사 (이슈 #25, ADR-0014) — 파일만 읽는다. LLM·OpenSearch·TEI를 부르지 않는다.

CI는 실제 평가를 돌리지 않는다(OpenSearch·TEI·GPU·API 키가 필요하고 비용이 든다). 대신 로컬에서 돌린 최신 평가
요약(`data/eval/latest_summary.json`)을 커밋하고, CI가 두 가지를 확인한다.

- 검사 1 (기준선): latest_summary의 수치가 `data/eval/baseline.json` 기준을 못 넘거나, 채점 조건(채점 모델·채점
  프롬프트·문항 수)이 기준선과 다르면 실패한다. 어느 항목이 얼마인지 메시지로 낸다
- 검사 2 (재측정 누락): `--base`와 비교해(CI는 PR 병합 커밋의 첫 부모 `HEAD^1`) 평가 결과에 영향을 주는 파일(`EVAL_AFFECTING_PATHS`)이 바뀌었는데
  latest_summary.json이 바뀌지 않았으면 실패한다. `--base`가 없으면(main push) 건너뛴다

latest_summary는 `answer_judge`가 근거 답변 결과를 채점할 때 자동으로 다시 쓴다(`--no-latest`로 끌 수 있다).
이미 있는 채점 결과로 다시 만들 때는 `update-latest`를 쓴다.

CLI: `uv run python -m rfp_pm_agent.eval.check_baseline check [--base origin/main]`
     `uv run python -m rfp_pm_agent.eval.check_baseline update-latest --judge-run <채점>_run.json`
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DEFAULT_BASELINE = Path("data/eval/baseline.json")
DEFAULT_LATEST = Path("data/eval/latest_summary.json")

# 바뀌면 근거 답변 평가를 다시 돌려 latest_summary.json을 갱신해야 하는 경로. "/"로 끝나면 그 아래 전부
EVAL_AFFECTING_PATHS: tuple[str, ...] = (
    "src/rfp_pm_agent/search/",  # 검색·근거 답변 코드와 답변 프롬프트(search/prompts/)
    "src/rfp_pm_agent/tools/search_documents.py",  # 서비스 검색 경로
    "src/rfp_pm_agent/schemas/answer.py",  # 근거 답변 결과 형식
    "src/rfp_pm_agent/eval/run_grounded_answer.py",  # 문항 구성·top_k
    "src/rfp_pm_agent/eval/answer_judge.py",  # 채점 규칙(규칙 판정·인용 없음 처리·입력 구성)
    # 에이전트 프롬프트 — 소유자 결정(#25)으로 넣었다. latest_summary는 근거 답변만 재므로 에이전트 품질을
    # 검증하지는 못한다. 에이전트 프롬프트를 바꾼 PR에 "평가를 다시 돌렸는가"를 묻는 표시로 쓴다(ADR-0014 한계)
    "src/rfp_pm_agent/agent/prompts/",
    "src/rfp_pm_agent/eval/prompts/",  # 채점 프롬프트(judge_*)
    "src/rfp_pm_agent/ingest/index_mapping.json",  # 인덱스 매핑
    "data/eval/qa_v2.jsonl",  # 평가 세트
    "data/eval/irrelevant_v1.jsonl",
)


class Thresholds(BaseModel):
    """기준선 통과 조건. 모두 '이상/이하'로 경계값을 포함한다."""

    answer_correct_min: int
    citation_accuracy_min: float
    refused_answerable_max: int
    no_answer_refused_min: int


class Baseline(BaseModel):
    """`data/eval/baseline.json`. 기준값은 사람이 정한다 — 코드가 갱신하지 않는다."""

    decided_at: str
    decided_by: str
    thresholds: Thresholds
    answerable_total: int
    no_answer_total: int
    judge_model: str
    judge_prompt_version: str
    scope: str  # 근거 답변 검색 범위(run_grounded_answer --scope). 범위가 다르면 같은 문항이라도 비교할 수 없다
    conditions: dict[str, Any]  # 측정 조건(기록용)
    measurements: list[dict[str, Any]]  # 근거가 된 측정(기록용)
    notes: list[str] = []


class LatestSummary(BaseModel):
    """`data/eval/latest_summary.json`. 근거 답변 평가 1회(생성 + 채점)의 요약."""

    generated_at: str
    answers_file: str
    judge_file: str
    answer_model: str
    answer_prompt_version: str
    reasoning_effort: str
    index_alias: str
    index_names: list[str]
    scope: str
    judge_model: str
    judge_prompt_version: str
    answerable: int
    answer_correct: int
    citation_judged: int
    citation_supported: int
    citation_accuracy: float | None
    refused_answerable: int
    no_answer: int
    no_answer_refused: int
    judge_errors: int


def build_latest(
    judge_record: Mapping[str, Any],
    answer_record: Mapping[str, Any],
    *,
    judge_file: str,
    generated_at: str | None = None,
) -> LatestSummary:
    """채점 `_run.json`과 근거 답변 `_run.json` 내용으로 latest_summary를 만든다."""
    judge_run, summary = judge_record["run"], judge_record["summary"]
    answer_run = answer_record["run"]
    return LatestSummary(
        generated_at=generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        answers_file=judge_run["answers_file"],
        judge_file=judge_file,
        answer_model=answer_run["model"],
        answer_prompt_version=answer_run["prompt_version"],
        reasoning_effort=answer_run["reasoning_effort"],
        index_alias=answer_run["index_alias"],
        index_names=answer_run["index_names"],
        scope=answer_run.get("scope", "all"),
        judge_model=judge_run["model"],
        judge_prompt_version=judge_run["prompt_version"],
        answerable=summary["answerable"],
        answer_correct=summary["answer_correct"],
        citation_judged=summary["citation_judged"],
        citation_supported=summary["citation_supported"],
        citation_accuracy=summary["citation_accuracy"],
        refused_answerable=len(summary["refused_answerable"]),
        no_answer=summary["no_answer"],
        no_answer_refused=summary["no_answer_refused"],
        judge_errors=len(summary["judge_errors"]),
    )


def answer_run_path(answers_file: Path) -> Path:
    """근거 답변 결과 `<stem>.jsonl`의 실행 기록 `<stem>_run.json`."""
    return answers_file.with_name(f"{answers_file.stem}_run.json")


def write_latest_from_judge_run(judge_run_path: Path, out: Path = DEFAULT_LATEST) -> LatestSummary:
    """채점 `_run.json`(과 그 안의 answers_file 실행 기록)을 읽어 latest_summary를 쓴다."""
    judge_record = json.loads(judge_run_path.read_text(encoding="utf-8"))
    answers_file = Path(judge_record["run"]["answers_file"])
    answer_record = json.loads(answer_run_path(answers_file).read_text(encoding="utf-8"))
    judge_file = judge_run_path.with_name(judge_run_path.name.removesuffix("_run.json") + ".jsonl")
    latest = build_latest(judge_record, answer_record, judge_file=str(judge_file))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return latest


def check_against_baseline(latest: LatestSummary, baseline: Baseline) -> list[str]:
    """검사 1. 실패 사유 목록(비면 통과)."""
    t = baseline.thresholds
    failures: list[str] = []
    if latest.judge_model != baseline.judge_model:
        failures.append(
            f"채점 모델이 기준선과 다르다: {latest.judge_model} (기준선 {baseline.judge_model})"
        )
    if latest.judge_prompt_version != baseline.judge_prompt_version:
        failures.append(
            f"채점 프롬프트가 기준선과 다르다: {latest.judge_prompt_version} "
            f"(기준선 {baseline.judge_prompt_version})"
        )
    if latest.scope != baseline.scope:
        failures.append(f"검색 범위가 기준선과 다르다: {latest.scope} (기준선 {baseline.scope})")
    if latest.answerable != baseline.answerable_total:
        failures.append(
            f"정답 있는 문항 수가 다르다: {latest.answerable} (기준선 {baseline.answerable_total})"
        )
    if latest.no_answer != baseline.no_answer_total:
        failures.append(
            f"답 없음·무관 문항 수가 다르다: {latest.no_answer} (기준선 {baseline.no_answer_total})"
        )
    if latest.judge_errors:
        failures.append(f"채점 오류 {latest.judge_errors}건 — 오류 없이 다시 채점해야 한다")
    if latest.answer_correct < t.answer_correct_min:
        failures.append(
            f"근거 답변 정답 {latest.answer_correct}/{latest.answerable} < 기준 {t.answer_correct_min}"
        )
    if latest.citation_accuracy is None or latest.citation_accuracy < t.citation_accuracy_min:
        value = "없음" if latest.citation_accuracy is None else f"{latest.citation_accuracy:.3f}"
        failures.append(
            f"출처 정확도 {value} ({latest.citation_supported}/{latest.citation_judged}) "
            f"< 기준 {t.citation_accuracy_min}"
        )
    if latest.refused_answerable > t.refused_answerable_max:
        failures.append(
            f"정답 있는 문항 잘못 거절 {latest.refused_answerable} > 기준 {t.refused_answerable_max}"
        )
    if latest.no_answer_refused < t.no_answer_refused_min:
        failures.append(
            f"답 없음·무관 거절 {latest.no_answer_refused}/{latest.no_answer} "
            f"< 기준 {t.no_answer_refused_min}"
        )
    return failures


def is_eval_affecting(path: str) -> bool:
    return any(path.startswith(p) if p.endswith("/") else path == p for p in EVAL_AFFECTING_PATHS)


def check_remeasured(changed: Sequence[str], latest_path: str = str(DEFAULT_LATEST)) -> list[str]:
    """검사 2. 평가에 영향을 주는 파일이 바뀌었는데 latest_summary가 안 바뀌었으면 실패 사유를 낸다."""
    affecting = sorted(p for p in changed if is_eval_affecting(p))
    latest_path = Path(os.path.normpath(latest_path)).as_posix()
    if not affecting or latest_path in changed:
        return []
    listed = "\n".join(f"  - {p}" for p in affecting)
    message = (
        f"평가에 영향을 주는 파일이 바뀌었는데 {latest_path}가 바뀌지 않았다:\n{listed}\n"
        "  근거 답변을 다시 생성·채점해(run_grounded_answer → answer_judge) latest_summary.json을 "
        "갱신해 커밋한다 — 실행 전 예상 호출 수·비용을 알린다(CLAUDE.md 비용)"
    )
    return [message]


def changed_files(base: str) -> list[str]:
    """base와 HEAD 사이에 바뀐 파일. 공통 조상을 못 찾으면(얕은 clone) 두 트리를 바로 비교한다."""
    for spec in (f"{base}...HEAD", f"{base}..HEAD"):
        result = subprocess.run(
            ["git", "diff", "--name-only", spec], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return [line for line in result.stdout.splitlines() if line]
    raise RuntimeError(f"git diff 실패: {base}\n{result.stderr}")


def run_check(baseline_path: Path, latest_path: Path, base: str | None) -> list[str]:
    baseline = Baseline.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    latest = LatestSummary.model_validate_json(latest_path.read_text(encoding="utf-8"))
    failures = check_against_baseline(latest, baseline)
    if base is not None:
        failures += check_remeasured(changed_files(base), latest_path.as_posix())
    return failures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="기준선·재측정 누락 검사")
    check.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    check.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    check.add_argument("--base", help="비교할 git 기준(예: origin/main). 없으면 검사 2를 건너뛴다")
    update = sub.add_parser(
        "update-latest", help="채점 _run.json으로 latest_summary.json을 다시 쓴다"
    )
    update.add_argument("--judge-run", type=Path, required=True)
    update.add_argument("--out", type=Path, default=DEFAULT_LATEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "update-latest":
        latest = write_latest_from_judge_run(args.judge_run, args.out)
        print(f"{args.out}: 정답 {latest.answer_correct}/{latest.answerable}")
        return 0
    failures = run_check(args.baseline, args.latest, args.base)
    if failures:
        print("평가 기준선 검사 실패:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    skipped = " (재측정 누락 검사는 --base가 없어 건너뜀)" if args.base is None else ""
    print(f"평가 기준선 검사 통과{skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
