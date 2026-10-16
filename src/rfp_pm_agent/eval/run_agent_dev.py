"""에이전트 개발 평가 실행 — `data/eval/agent_dev_v1.jsonl` 30문항 + 기준선 (이슈 #22, ADR-0010 "평가").

에이전트 평가 자동 채점·회귀 게이트(#25)가 아니다. #22에서 정한 통과 기준(측정 전 고정, 아래 `CRITERIA`)을 확인하는 실행이다.

판정
- 규칙으로 판정(이 모듈): 툴 선택(`ok` 상태로 끝난 툴 호출의 이름 집합 = expected_tools, 프로젝트·문서 인자 일치), PMS 정답 ID,
  되묻기, 거절, 출처 유효성, LLM 호출 수, 비용, 지연
- 사람(이번에는 Claude)이 판정: 문서 답의 정답 여부. reference_answer(qa_v2 정답)와 답·인용 청크 원문을 대조해
  `<결과>_judgments.jsonl`에 적고 `--from-rows`로 요약을 다시 만든다. 판정 기록에 근거를 한 줄씩 남긴다

기준선(`--baseline`): 같은 질문의 첫 턴을 #19 `answer_question(top_k=10)`으로 두 조건에서 부른다 — 문서 지정(문항의 doc_id,
없으면 전체)과 전체 검색. 에이전트와 다른 변수는 "사업을 누가 정하나(정답 doc_id를 넣어 줌 / 전체) · 툴 선택"이다.

비용: 에이전트 30문항(31턴) LLM 약 75회 + 기준선 최대 60회(검색 0건이면 LLM을 부르지 않는다). 실행 전에 알린다(CLAUDE.md 비용).
모든 LLM 호출은 `COST_LOG_PATH`에 남는다(에이전트는 `CostLogCallback`, 기준선은 `LoggingLLMClient`).

CLI
- `uv run python -m rfp_pm_agent.eval.run_agent_dev --out-dir data/eval/results/agent [--only d1] [--baseline]`
- 요약만 다시 만들기(비용 없음): `... --from-rows <결과>.jsonl [--judgments <판정>.jsonl]`
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from rfp_pm_agent.agent.graph import PROMPT_VERSION, Agent, agent_deps_from_env
from rfp_pm_agent.clients.cost import CostLogger, compute_llm_cost
from rfp_pm_agent.clients.llm import OpenAICompatLLMClient
from rfp_pm_agent.clients.logged import LoggingLLMClient
from rfp_pm_agent.config import AgentConfig, ClientsConfig, OpenSearchConfig, SynthConfig
from rfp_pm_agent.eval.agent_dev_set import DEFAULT_OUT, load_agent_questions
from rfp_pm_agent.schemas.agent import AgentAnswer
from rfp_pm_agent.schemas.answer import GroundedAnswer
from rfp_pm_agent.schemas.eval import AgentDevQuestion
from rfp_pm_agent.search import grounded_answer
from rfp_pm_agent.search.grounded_answer import answer_question
from rfp_pm_agent.tools.search_documents import SearchDeps

PMS_TOOLS = {"get_delayed_tasks", "get_open_risks", "get_deliverables"}
BaselineScope = Literal["doc", "all"]

# 통과 기준(이슈 #22, 2026-10-16 측정 전 고정 — 1단계 보고의 제안 표를 소유자가 그대로 확정)
CRITERIA = {
    "tool_selection_min_rate": 0.90,  # 30문항 중 27 이상
    "pms_facts_min": 7,  # PMS만 8문항 중 정답 ID 재현율 100%(+정밀도) 문항 수
    "doc_vs_baseline_margin": 1,  # 문서만 8문항 정답 수 ≥ 기준선(문서 지정) − 1
    "clarify_min": 4,  # 사업 미지정 4문항 모두 되묻기(임의 사업 선택 0)
    "refuse_min": 3,  # 답 없음 4문항 중 거절
    "avg_llm_calls_max": 3.0,  # 문항당 평균 LLM 호출 수
    "max_steps_hits_max": 1,  # 상한 도달 문항 수
    "cost_max_usd": 0.30,  # 에이전트 30문항 1회
}


class TurnResult(BaseModel):
    question: str
    answer: AgentAnswer | None
    error: str | None
    latency_ms: float
    cost_usd: float


class AgentRow(BaseModel):
    qid: str
    type: str
    turns: list[TurnResult]
    checks: dict[str, bool | None]  # 규칙 판정. 해당 없는 항목은 None


class BaselineRow(BaseModel):
    qid: str
    scope: BaselineScope
    question: str
    doc_ids: list[str] | None
    answer: GroundedAnswer | None
    error: str | None
    latency_ms: float
    cost_usd: float


class Judgment(BaseModel):
    """문서 답 판정 한 건. system은 agent·baseline_doc·baseline_all."""

    qid: str
    system: Literal["agent", "baseline_doc", "baseline_all"]
    correct: bool
    # 묻는 사실은 맞지만 정답의 일부가 빠진 답(소유자 판정, 2026-10-16). correct=True로 세고, 오답으로 볼 때의 수를 따로 적는다
    partial: bool = False
    reason: str


# --- 규칙 판정 ---


def _wbs_in(text: str, wbs_id: str) -> bool:
    """WBS ID가 숫자·점에 붙지 않은 채로 나오는지("2"가 "2026"·"2.3"·"-24"에 걸리지 않게)."""
    return re.search(rf"(?<![\d.\-]){re.escape(wbs_id)}(?![\d.]\d|\d)", text) is not None


def _pms_ids_in(text: str, project_id: str, kind: Literal["R", "D"]) -> set[str]:
    """답에 나온 이 프로젝트의 Risk·산출물 ID. 줄인 꼴("R05")도 같은 프로젝트 ID로 센다."""
    found = set(re.findall(rf"{re.escape(project_id)}-{kind}\d+", text))
    for short in re.findall(rf"(?<![\w-]){kind}\d{{2,3}}(?!\d)", text):
        found.add(f"{project_id}-{short}")
    return found


def facts_ok(question: AgentDevQuestion, text: str) -> bool | None:
    """정답 ID가 답에 다 있는지(Risk·산출물은 다른 ID가 없는지까지). 볼 ID가 없으면 None."""
    facts = question.expected_facts
    checks: list[bool] = []
    if facts.wbs_ids:
        checks.append(all(_wbs_in(text, w) for w in facts.wbs_ids))
    if facts.stage_wbs_ids:
        checks.append("단계 지연" in text)
    if facts.risk_ids and question.project_id:
        checks.append(_pms_ids_in(text, question.project_id, "R") == set(facts.risk_ids))
    if facts.deliverable_ids and question.project_id:
        checks.append(_pms_ids_in(text, question.project_id, "D") == set(facts.deliverable_ids))
    if facts.requirement_ids:
        checks.append(all(r in text for r in facts.requirement_ids))
    if facts.count is not None:
        checks.append(re.search(rf"(?<!\d){facts.count}\s*(건|개)", text) is not None)
    return all(checks) if checks else None


def tool_selection_ok(question: AgentDevQuestion, answer: AgentAnswer) -> bool:
    """ok로 끝난 툴 이름 집합이 expected_tools와 같고, 인자가 대상 사업을 가리키는지."""
    ok_calls = [c for c in answer.tool_calls if c.status == "ok"]
    if {c.name for c in ok_calls} != set(question.expected_tools):
        return False
    for call in ok_calls:
        if call.name in PMS_TOOLS and call.args.get("project_id") != question.project_id:
            return False
        if call.name == "search_documents" and call.args.get("doc_ids") != [question.doc_id]:
            return False
    return True


def clarified(answer: AgentAnswer) -> bool:
    """되물었는지: ok로 끝난 툴 호출이 없고(임의 사업으로 조회하지 않음), 거절이 아니고, 묻는 문장이다."""
    asks = "?" in answer.answer or "어느" in answer.answer or "어떤 사업" in answer.answer
    return not any(c.status == "ok" for c in answer.tool_calls) and not answer.refused and asks


def check_row(question: AgentDevQuestion, turns: list[TurnResult]) -> dict[str, bool | None]:
    answers = [t.answer for t in turns]
    if any(a is None for a in answers):
        return {"error": True}
    first, last = answers[0], answers[-1]
    assert first is not None and last is not None
    checks: dict[str, bool | None] = {"error": False}
    if question.should_clarify:
        checks["clarify"] = clarified(first)
        tools = clarified(first)
        if len(turns) > 1:
            tools = tools and tool_selection_ok(question, last)
        checks["tool_selection"] = tools
    else:
        checks["tool_selection"] = tool_selection_ok(question, last)
    checks["refusal"] = last.refused == question.should_refuse
    checks["facts"] = None if last.refused else facts_ok(question, last.answer)
    if question.must_cite and not last.refused:
        checks["citation"] = bool(last.citation_map) and not last.invalid_citations
    else:
        checks["citation"] = None
    checks["no_invalid_citation"] = all(not a.invalid_citations for a in answers if a)
    checks["reached_max_steps"] = any(a.reached_max_steps for a in answers if a)
    return checks


# --- 요약 ---


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(
    questions: list[AgentDevQuestion],
    rows: list[AgentRow],
    baselines: list[BaselineRow],
    judgments: list[Judgment],
) -> dict[str, Any]:
    by_qid = {q.qid: q for q in questions}
    of_type = {
        t: [r for r in rows if by_qid[r.qid].type == t]
        for t in ("doc", "pms", "both", "unspecified", "no_answer")
    }
    judged: dict[tuple[str, str], bool] = {(j.qid, j.system): j.correct for j in judgments}

    def count(rs: list[AgentRow], key: str) -> int:
        return sum(1 for r in rs if r.checks.get(key) is True)

    llm_calls = [sum(t.answer.llm_calls for t in r.turns if t.answer) for r in rows]
    latencies = [sum(t.latency_ms for t in r.turns) for r in rows]
    agent_cost = sum(t.cost_usd for r in rows for t in r.turns)
    doc_qids = [r.qid for r in of_type["doc"]]
    partial = {(j.qid, j.system) for j in judgments if j.correct and j.partial}

    def doc_count(system: str) -> tuple[int, int]:
        """(정답 수, 그중 부분 정답 수)."""
        correct = sum(judged.get((q, system), False) for q in doc_qids)
        return correct, sum(1 for q in doc_qids if (q, system) in partial)

    def doc_text(system: str) -> str:
        correct, n_partial = doc_count(system)
        return (
            f"{correct}(부분 정답 {n_partial} 포함, 오답으로 보면 {correct - n_partial})"
            if n_partial
            else str(correct)
        )

    (doc_agent, agent_partial), (doc_base, base_partial) = (
        doc_count("agent"),
        doc_count("baseline_doc"),
    )
    doc_judged = all((q, "agent") in judged and (q, "baseline_doc") in judged for q in doc_qids)
    cite_rows = [r for r in rows if r.checks.get("citation") is not None]
    tool_ok = count(rows, "tool_selection")
    results: dict[str, dict[str, Any]] = {
        "tool_selection": {
            "value": f"{tool_ok}/{len(rows)}",
            "pass": tool_ok >= CRITERIA["tool_selection_min_rate"] * len(rows),
        },
        "pms_facts": {
            "value": f"{count(of_type['pms'], 'facts')}/{len(of_type['pms'])}",
            "pass": count(of_type["pms"], "facts") >= CRITERIA["pms_facts_min"],
        },
        "doc_answers": {
            "value": f"에이전트 {doc_text('agent')}/{len(doc_qids)} · 기준선 문서 지정 {doc_text('baseline_doc')}"
            f" · 전체 검색 {doc_text('baseline_all')}",
            "pass": (doc_agent >= doc_base - CRITERIA["doc_vs_baseline_margin"])
            if doc_judged
            else None,
            # 판정 규칙은 그대로이고, 부분 정답을 양쪽 모두 오답으로 셀 때의 결과를 함께 적는다
            "pass_if_partial_wrong": (
                doc_agent - agent_partial
                >= doc_base - base_partial - CRITERIA["doc_vs_baseline_margin"]
            )
            if doc_judged
            else None,
        },
        "clarify": {
            "value": f"{count(of_type['unspecified'], 'clarify')}/{len(of_type['unspecified'])}",
            "pass": count(of_type["unspecified"], "clarify") >= CRITERIA["clarify_min"],
        },
        "refuse": {
            "value": f"{count(of_type['no_answer'], 'refusal')}/{len(of_type['no_answer'])}",
            "pass": count(of_type["no_answer"], "refusal") >= CRITERIA["refuse_min"],
        },
        "citation": {
            "value": f"유효 출처 {count(cite_rows, 'citation')}/{len(cite_rows)}, 범위 밖 번호 있는 문항 {sum(1 for r in rows if r.checks.get('no_invalid_citation') is False)}",
            "pass": count(cite_rows, "citation") == len(cite_rows)
            and all(r.checks.get("no_invalid_citation") is not False for r in rows),
        },
        "llm_calls": {
            "value": f"평균 {statistics.mean(llm_calls):.2f}회, 상한 도달 {count(rows, 'reached_max_steps')}문항"
            if rows
            else "-",
            "pass": bool(rows)
            and statistics.mean(llm_calls) <= CRITERIA["avg_llm_calls_max"]
            and count(rows, "reached_max_steps") <= CRITERIA["max_steps_hits_max"],
        },
        "cost": {"value": f"${agent_cost:.4f}", "pass": agent_cost <= CRITERIA["cost_max_usd"]},
        "latency": {
            "value": f"p50 {_percentile(latencies, 0.5) / 1000:.1f}초 · p95 {_percentile(latencies, 0.95) / 1000:.1f}초",
            "pass": None,
        },
    }
    by_type = {
        t: {
            "n": len(rs),
            "tool_selection": count(rs, "tool_selection"),
            "facts": f"{count(rs, 'facts')}/{sum(1 for r in rs if r.checks.get('facts') is not None)}",
            "refusal_as_expected": count(rs, "refusal"),
        }
        for t, rs in of_type.items()
    }
    baseline_summary: dict[str, Any] = {}
    for scope in ("doc", "all"):
        bs = [b for b in baselines if b.scope == scope]
        if bs:
            baseline_summary[scope] = {
                "n": len(bs),
                "refused": sum(1 for b in bs if b.answer and b.answer.refused),
                "errors": sum(1 for b in bs if b.error),
                "cost_usd": round(sum(b.cost_usd for b in bs), 6),
                "refused_by_type": {
                    t: sum(
                        1 for b in bs if by_qid[b.qid].type == t and b.answer and b.answer.refused
                    )
                    for t in of_type
                },
            }
    return {
        "criteria": CRITERIA,
        "results": results,
        "all_pass": all(v["pass"] is not False for v in results.values())
        and results["doc_answers"]["pass"] is not None,
        "by_type": by_type,
        "baseline": baseline_summary,
        "agent_cost_usd": round(agent_cost, 6),
        "agent_llm_calls": sum(llm_calls),
    }


def render_md(
    questions: list[AgentDevQuestion],
    rows: list[AgentRow],
    baselines: list[BaselineRow],
    judgments: list[Judgment],
    summary: dict[str, Any],
    run_info: dict[str, Any],
) -> str:
    by_qid = {q.qid: q for q in questions}
    base = {(b.qid, b.scope): b for b in baselines}
    judged = {(j.qid, j.system): j for j in judgments}
    names = {
        "tool_selection": "툴 선택 정확도 (≥ 90%)",
        "pms_facts": "PMS 답 정답 ID (≥ 7/8)",
        "doc_answers": "문서 답 (≥ 기준선 문서 지정 − 1)",
        "clarify": "사업 미지정 되묻기 (4/4)",
        "refuse": "답 없음 거절 (≥ 3/4)",
        "citation": "출처 (유효 100%, 범위 밖 0)",
        "llm_calls": "반복 (평균 ≤ 3, 상한 도달 ≤ 1)",
        "cost": "비용 (≤ $0.30)",
        "latency": "지연 (기록만)",
    }
    mark = {True: "통과", False: "**미통과**", None: "—"}
    lines = [f"# 에이전트 개발 평가 {run_info.get('name', '')}", ""]
    lines += [f"- {k}: {v}" for k, v in run_info.items() if k != "name"]
    lines += ["", "| 기준 | 결과 | 판정 |", "|---|---|---|"]
    lines += [
        f"| {names[k]} | {v['value']} | {mark[v['pass']]} |" for k, v in summary["results"].items()
    ]
    doc_result = summary["results"]["doc_answers"]
    if doc_result.get("pass_if_partial_wrong") is not None:
        lines += [
            "",
            f"- 문서 답: 부분 정답을 오답으로 세어도 판정은 {mark[doc_result['pass_if_partial_wrong']]}",
        ]
    lines += [
        "",
        "## 문항별",
        "",
        "| qid | 유형 | 툴 | 사실 | 거절 기대대로 | 출처 | LLM | 문서 판정(에이전트/기준선 doc/all) | 툴 호출 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        q = by_qid[row.qid]
        calls = "; ".join(
            f"{c.name}({c.status})" for t in row.turns if t.answer for c in t.answer.tool_calls
        )
        llm = sum(t.answer.llm_calls for t in row.turns if t.answer)
        doc = (
            "/".join(
                ("부분" if judged[(row.qid, s)].partial else mark[judged[(row.qid, s)].correct])
                if (row.qid, s) in judged
                else "?"
                for s in ("agent", "baseline_doc", "baseline_all")
            )
            if q.type == "doc"
            else ""
        )
        c = row.checks
        lines.append(
            f"| {row.qid} | {q.type} | {mark[c.get('tool_selection')]} | {mark[c.get('facts')]} | {mark[c.get('refusal')]} | {mark[c.get('citation')]} | {llm} | {doc} | {calls} |"
        )
    lines += ["", "## 답 원문", ""]
    for row in rows:
        q = by_qid[row.qid]
        lines += [f"### {row.qid} ({q.type})", ""]
        if q.reference_answer:
            lines += [f"- 정답(qa_v2): {q.reference_answer}"]
        for t in row.turns:
            lines += [f"- 질문: {t.question}"]
            if t.answer is None:
                lines += [f"- 오류: {t.error}"]
                continue
            lines += ["- 에이전트 답:", "", "  " + t.answer.answer.replace("\n", "\n  "), ""]
            if t.answer.citation_map:
                lines += [
                    "  출처: " + " ".join(f"[{n}] {s}" for n, s in t.answer.citation_map.items()),
                    "",
                ]
        for scope in ("doc", "all"):
            b = base.get((row.qid, scope))
            if b and b.answer:
                lines += [f"- 기준선({scope}): {b.answer.answer}".replace("\n", " ")]
        for s in ("agent", "baseline_doc", "baseline_all"):
            j = judged.get((row.qid, s))
            if j:
                grade = "부분 정답" if j.correct and j.partial else "정답" if j.correct else "오답"
                lines += [f"- 판정 {s}: {grade} — {j.reason}"]
        lines.append("")
    return "\n".join(lines) + "\n"


# --- 실행 ---


def run_agent(
    agent: Agent, questions: list[AgentDevQuestion], config: ClientsConfig, run_id: str
) -> list[AgentRow]:
    rows: list[AgentRow] = []
    for q in questions:
        turns: list[TurnResult] = []
        for i, text in enumerate(q.turns):
            start = time.monotonic()
            try:
                result = agent.ask(text, thread_id=f"{run_id}-{q.qid}")
                error = None
            except Exception as e:  # noqa: BLE001 — 문항 하나의 실패를 기록하고 다음으로 간다(재시도 없음)
                result, error = None, f"{type(e).__name__}: {e}"
            latency = (time.monotonic() - start) * 1000
            cost = (
                compute_llm_cost(
                    config,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                )
                if result
                else 0.0
            )
            turns.append(
                TurnResult(
                    question=text, answer=result, error=error, latency_ms=latency, cost_usd=cost
                )
            )
            if result is None:
                break
            print(
                f"{q.qid} turn {i + 1}: llm {result.llm_calls}, tools {[c.name for c in result.tool_calls]}, ${cost:.5f}",
                flush=True,
            )
        rows.append(
            AgentRow(
                qid=q.qid,
                type=q.type,
                turns=turns,
                checks=check_row(q, turns) if turns else {"error": True},
            )
        )
    return rows


def run_baseline(
    questions: list[AgentDevQuestion], config: ClientsConfig, cost_logger: CostLogger
) -> list[BaselineRow]:
    deps = SearchDeps.from_env()
    llm = LoggingLLMClient(OpenAICompatLLMClient(config), config, cost_logger)
    rows: list[BaselineRow] = []
    scopes: tuple[BaselineScope, ...] = ("doc", "all")
    for scope in scopes:
        for q in questions:
            doc_ids = [q.doc_id] if scope == "doc" and q.doc_id else None
            start = time.monotonic()
            try:
                result: GroundedAnswer | None = answer_question(
                    q.turns[0], deps=deps, llm=llm, doc_ids=doc_ids
                )
                error = None
            except Exception as e:  # noqa: BLE001 — 기록하고 다음 문항
                result, error = None, f"{type(e).__name__}: {e}"
            usage = result.llm_usage if result else None
            cost = (
                compute_llm_cost(
                    config, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
                )
                if usage
                else 0.0
            )
            rows.append(
                BaselineRow(
                    qid=q.qid,
                    scope=scope,
                    question=q.turns[0],
                    doc_ids=doc_ids,
                    answer=result,
                    error=error,
                    latency_ms=(time.monotonic() - start) * 1000,
                    cost_usd=cost,
                )
            )
            print(
                f"baseline {scope} {q.qid}: refused={result.refused if result else None} ${cost:.5f}",
                flush=True,
            )
    return rows


def _read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.exists():
        return []
    return [
        model.model_validate_json(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows), "utf-8")


def write_outputs(
    stem: Path,
    questions: list[AgentDevQuestion],
    rows: list[AgentRow],
    baselines: list[BaselineRow],
    judgments: list[Judgment],
    run_info: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize(questions, rows, baselines, judgments)
    Path(f"{stem}_run.json").write_text(
        json.dumps({**run_info, "summary": summary}, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    Path(f"{stem}.md").write_text(
        render_md(questions, rows, baselines, judgments, summary, run_info), "utf-8"
    )
    return summary


def main(
    argv: Sequence[str] | None = None, *, now: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval/results/agent"))
    parser.add_argument("--questions", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", help="이 qid만 실행(실제 호출 확인용)")
    parser.add_argument(
        "--baseline", action="store_true", help="기준선(answer_question 문서 지정·전체 검색)도 실행"
    )
    parser.add_argument("--name", help="결과 파일 이름(기본 agent_dev_<YYYYMMDD>)")
    parser.add_argument(
        "--from-rows", type=Path, help="이 결과(.jsonl)로 요약만 다시 만든다(비용 없음)"
    )
    parser.add_argument("--judgments", type=Path, help="문서 답 판정 파일(.jsonl)")
    args = parser.parse_args(argv)

    questions = load_agent_questions(args.questions)
    if args.only:
        questions = [q for q in questions if q.qid in set(args.only)]

    if args.from_rows:
        stem = Path(str(args.from_rows).removesuffix(".jsonl"))
        rows = _read_jsonl(args.from_rows, AgentRow)
        qids = {r.qid for r in rows}
        questions = [q for q in questions if q.qid in qids]
        baselines = _read_jsonl(Path(f"{stem}_baseline.jsonl"), BaselineRow)
        judgments = _read_jsonl(args.judgments or Path(f"{stem}_judgments.jsonl"), Judgment)
        run_info = json.loads(Path(f"{stem}_run.json").read_text("utf-8"))
        run_info.pop("summary", None)
        summary = write_outputs(stem, questions, rows, baselines, judgments, run_info)
        print(json.dumps(summary["results"], ensure_ascii=False, indent=2))
        return 0

    name = args.name or f"agent_dev_{now():%Y%m%d}"
    stem = args.out_dir / name
    if Path(f"{stem}.jsonl").exists():
        raise SystemExit(f"{stem}.jsonl이 이미 있다 — 다른 --name을 쓴다")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clients = ClientsConfig.from_env()
    cost_logger = CostLogger(clients.cost_log_path)
    agent = Agent(agent_deps_from_env(cost_logger))
    run_info = {
        "name": name,
        "started_at": now().isoformat(),
        "model": clients.llm_model_dev,
        "reasoning_effort": clients.llm_reasoning_effort,
        "api": "responses",
        "agent_prompt": PROMPT_VERSION,
        "baseline_prompt": grounded_answer.PROMPT_VERSION,
        "max_steps": AgentConfig.from_env().max_steps,
        "as_of_date": SynthConfig.from_env().as_of_date.isoformat(),
        "index_alias": OpenSearchConfig.from_env().index_alias,
        "questions_file": str(args.questions),
        "only": args.only or [],
        "catalog_size": len(agent.deps.catalog),
    }
    rows = run_agent(agent, questions, clients, name)
    _write_jsonl(Path(f"{stem}.jsonl"), rows)
    baselines = run_baseline(questions, clients, cost_logger) if args.baseline else []
    if baselines:
        _write_jsonl(Path(f"{stem}_baseline.jsonl"), baselines)
    run_info["finished_at"] = now().isoformat()
    summary = write_outputs(stem, questions, rows, baselines, [], run_info)
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
