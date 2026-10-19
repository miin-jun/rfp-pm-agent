"""평가 기준선 검사 (이슈 #25). 파일만 읽는다 — LLM·OpenSearch를 쓰지 않는다."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rfp_pm_agent.eval.check_baseline import (
    DEFAULT_BASELINE,
    DEFAULT_LATEST,
    EVAL_AFFECTING_PATHS,
    Baseline,
    LatestSummary,
    Thresholds,
    build_latest,
    changed_files,
    check_against_baseline,
    check_remeasured,
    is_eval_affecting,
    main,
    write_latest_from_judge_run,
)


def _baseline() -> Baseline:
    return Baseline(
        decided_at="2026-10-19",
        decided_by="소유자",
        thresholds=Thresholds(
            answer_correct_min=33,
            citation_accuracy_min=0.95,
            refused_answerable_max=9,
            no_answer_refused_min=11,
        ),
        answerable_total=51,
        no_answer_total=11,
        judge_model="gpt-6-sol",
        judge_prompt_version="judge_v1",
        scope="doc",
        conditions={},
        measurements=[],
    )


def _latest(**overrides: Any) -> LatestSummary:
    fields: dict[str, Any] = {
        "generated_at": "2026-10-19T00:00:00+00:00",
        "answers_file": "a.jsonl",
        "judge_file": "j.jsonl",
        "answer_model": "gpt-6-luna",
        "answer_prompt_version": "grounded_answer_v2",
        "reasoning_effort": "low",
        "index_alias": "rfp_chunks",
        "index_names": ["rfp_chunks_v2_kure"],
        "scope": "doc",
        "judge_model": "gpt-6-sol",
        "judge_prompt_version": "judge_v1",
        "answerable": 51,
        "answer_correct": 36,
        "citation_judged": 45,
        "citation_supported": 44,
        "citation_accuracy": 44 / 45,
        "refused_answerable": 6,
        "no_answer": 11,
        "no_answer_refused": 11,
        "judge_errors": 0,
    }
    fields.update(overrides)
    return LatestSummary(**fields)


# --- 검사 1: 기준선 ---


def test_기준을_모두_넘으면_통과하고_경계값은_포함한다():
    assert check_against_baseline(_latest(), _baseline()) == []
    at_edge = _latest(
        answer_correct=33, citation_accuracy=0.95, refused_answerable=9, no_answer_refused=11
    )
    assert check_against_baseline(at_edge, _baseline()) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"answer_correct": 32}, "근거 답변 정답 32/51 < 기준 33"),
        (
            {"citation_supported": 40, "citation_accuracy": 40 / 45},
            "출처 정확도 0.889 (40/45) < 기준 0.95",
        ),
        ({"citation_accuracy": None, "citation_judged": 0}, "출처 정확도 없음"),
        ({"refused_answerable": 10}, "정답 있는 문항 잘못 거절 10 > 기준 9"),
        ({"no_answer_refused": 10}, "답 없음·무관 거절 10/11 < 기준 11"),
        (
            {"judge_model": "gpt-6-luna"},
            "채점 모델이 기준선과 다르다: gpt-6-luna (기준선 gpt-6-sol)",
        ),
        ({"judge_prompt_version": "judge_v2"}, "채점 프롬프트가 기준선과 다르다"),
        ({"answerable": 52}, "정답 있는 문항 수가 다르다: 52 (기준선 51)"),
        ({"no_answer": 12, "no_answer_refused": 12}, "답 없음·무관 문항 수가 다르다"),
        ({"judge_errors": 2}, "채점 오류 2건"),
        ({"scope": "all"}, "검색 범위가 기준선과 다르다: all (기준선 doc)"),
    ],
)
def test_기준을_못_넘은_항목과_값을_메시지로_낸다(overrides: dict[str, Any], expected: str) -> None:
    failures = check_against_baseline(_latest(**overrides), _baseline())
    assert len(failures) == 1
    assert expected in failures[0]


# --- 검사 2: 재측정 누락 ---


@pytest.mark.parametrize(
    "path",
    [
        "src/rfp_pm_agent/search/prompts/grounded_answer_v3.md",
        "src/rfp_pm_agent/search/grounded_answer.py",
        "src/rfp_pm_agent/search/hybrid.py",
        "src/rfp_pm_agent/tools/search_documents.py",
        "src/rfp_pm_agent/agent/prompts/agent_v1.md",
        "src/rfp_pm_agent/eval/prompts/judge_v1.md",
        "src/rfp_pm_agent/ingest/index_mapping.json",
        "data/eval/qa_v2.jsonl",
        "data/eval/irrelevant_v1.jsonl",
        "src/rfp_pm_agent/eval/answer_judge.py",
        "src/rfp_pm_agent/eval/run_grounded_answer.py",
        "src/rfp_pm_agent/schemas/answer.py",
    ],
)
def test_평가에_영향을_주는_파일(path: str) -> None:
    assert is_eval_affecting(path)


@pytest.mark.parametrize(
    "path",
    ["README.md", "src/rfp_pm_agent/pms/rules.py", "data/eval/qa_v2.jsonl.bak", "tests/x.py"],
)
def test_평가와_무관한_파일(path: str) -> None:
    assert not is_eval_affecting(path)


def test_영향_파일이_바뀌었는데_latest가_그대로면_실패하고_재측정을_안내한다():
    changed = ["src/rfp_pm_agent/search/prompts/grounded_answer_v3.md", "README.md"]

    [failure] = check_remeasured(changed)

    assert "grounded_answer_v3.md" in failure
    assert "README.md" not in failure
    assert "latest_summary.json" in failure and "다시 생성·채점" in failure


def test_latest도_함께_바뀌었거나_영향_파일이_없으면_통과한다():
    affecting = "data/eval/qa_v2.jsonl"
    assert check_remeasured([affecting, str(DEFAULT_LATEST)]) == []
    assert check_remeasured(["README.md", "docs/adr/0014.md"]) == []
    # 정규화하지 않은 --latest 경로도 같은 파일로 본다
    assert (
        check_remeasured([affecting, str(DEFAULT_LATEST)], "./data/eval/latest_summary.json") == []
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_changed_files는_공통_조상이_없으면_두_트리를_바로_비교한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("a", "utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-q", "--orphan", "other")  # 공통 조상이 없는 기준
    (repo / "b.txt").write_text("b", "utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "other")
    other = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    monkeypatch.chdir(repo)

    # 기준(other)에는 a·b, HEAD(main)에는 a만 있다 — 3점 비교는 실패하고 2점 비교로 b만 나온다
    assert changed_files(other) == ["b.txt"]
    assert changed_files("HEAD") == []


def test_영향_경로_목록은_실제로_있는_경로다():
    root = Path(__file__).resolve().parents[2]
    for path in EVAL_AFFECTING_PATHS:
        assert (root / path).exists(), path


# --- latest_summary 만들기 ---


def _judge_record(answers_file: str) -> dict[str, Any]:
    return {
        "run": {"answers_file": answers_file, "model": "gpt-6-sol", "prompt_version": "judge_v1"},
        "summary": {
            "answerable": 51,
            "answer_correct": 36,
            "citation_judged": 45,
            "citation_supported": 44,
            "citation_accuracy": 44 / 45,
            "refused_answerable": ["q027", "q031"],
            "no_answer": 11,
            "no_answer_refused": 11,
            "judge_errors": [],
        },
    }


def _answer_record() -> dict[str, Any]:
    return {
        "run": {
            "model": "gpt-6-luna",
            "prompt_version": "grounded_answer_v2",
            "reasoning_effort": "low",
            "index_alias": "rfp_chunks",
            "index_names": ["rfp_chunks_v2_kure"],
            "scope": "doc",
        }
    }


def test_build_latest는_채점과_답변_실행_기록을_합친다():
    latest = build_latest(
        _judge_record("a.jsonl"), _answer_record(), judge_file="j.jsonl", generated_at="t"
    )
    assert (latest.answer_model, latest.judge_model) == ("gpt-6-luna", "gpt-6-sol")
    assert (latest.answer_correct, latest.refused_answerable, latest.judge_errors) == (36, 2, 0)
    assert latest.index_names == ["rfp_chunks_v2_kure"]


def test_채점_run_json으로_latest를_쓰고_check가_읽는다(tmp_path: Path) -> None:
    answers = tmp_path / "grounded_answer_x.jsonl"
    (tmp_path / "grounded_answer_x_run.json").write_text(json.dumps(_answer_record()), "utf-8")
    judge_run = tmp_path / "judge_x_run.json"
    judge_run.write_text(json.dumps(_judge_record(str(answers))), "utf-8")
    latest_path = tmp_path / "latest.json"
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(_baseline().model_dump_json(), "utf-8")

    latest = write_latest_from_judge_run(judge_run, latest_path)

    assert latest.judge_file == str(tmp_path / "judge_x.jsonl")
    assert main(["check", "--baseline", str(baseline_path), "--latest", str(latest_path)]) == 0


def test_check는_기준_미달이면_종료_코드_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    latest_path = tmp_path / "latest.json"
    latest_path.write_text(_latest(answer_correct=30).model_dump_json(), "utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(_baseline().model_dump_json(), "utf-8")

    assert main(["check", "--baseline", str(baseline_path), "--latest", str(latest_path)]) == 1
    assert "근거 답변 정답 30/51 < 기준 33" in capsys.readouterr().err


def test_레포의_baseline과_latest_summary가_스키마에_맞고_기준을_넘는다():
    root = Path(__file__).resolve().parents[2]
    baseline = Baseline.model_validate_json((root / DEFAULT_BASELINE).read_text("utf-8"))
    latest = LatestSummary.model_validate_json((root / DEFAULT_LATEST).read_text("utf-8"))
    assert check_against_baseline(latest, baseline) == []
