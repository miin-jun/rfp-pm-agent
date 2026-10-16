"""에이전트 개발 평가의 규칙 판정과 평가 세트 (이슈 #22). LLM·서버 없음."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from rfp_pm_agent.eval import agent_dev_set
from rfp_pm_agent.eval.run_agent_dev import (
    AgentRow,
    Judgment,
    TurnResult,
    _pms_ids_in,
    _wbs_in,
    check_row,
    clarified,
    facts_ok,
    summarize,
    tool_selection_ok,
)
from rfp_pm_agent.schemas.agent import AgentAnswer, ToolCallRecord
from rfp_pm_agent.schemas.answer import LLMUsage
from rfp_pm_agent.schemas.eval import AgentDevQuestion, AgentExpectedFacts


def _q(**overrides: Any) -> AgentDevQuestion:
    fields: dict[str, Any] = {
        "qid": "x1",
        "type": "pms",
        "turns": ["질문"],
        "doc_id": "docA",
        "project_id": "PRJ-03",
        "expected_tools": ["get_delayed_tasks"],
        "expected_facts": AgentExpectedFacts(),
        "should_clarify": False,
        "should_refuse": False,
        "must_cite": False,
        "reference_answer": None,
        "source": "test",
        "dod": False,
        "prediction": "",
    }
    fields.update(overrides)
    return AgentDevQuestion(**fields)


def _a(
    text: str = "답", calls: list[tuple[str, dict[str, Any], str]] | None = None, **overrides: Any
) -> AgentAnswer:
    fields: dict[str, Any] = {
        "answer": text,
        "refused": text == "문서에서 찾을 수 없음",
        "citations": [],
        "citation_map": {},
        "cited_chunk_ids": [],
        "invalid_citations": [],
        "tool_calls": [
            ToolCallRecord(name=n, args=a, status=s, result_count=1)  # type: ignore[arg-type]
            for n, a, s in (calls or [])
        ],
        "llm_calls": 2,
        "reached_max_steps": False,
        "project_id": None,
        "doc_ids": [],
        "usage": LLMUsage(input_tokens=1, output_tokens=1, reasoning_tokens=0),
    }
    fields.update(overrides)
    return AgentAnswer(**fields)


def _turn(answer: AgentAnswer | None) -> TurnResult:
    return TurnResult(question="q", answer=answer, error=None, latency_ms=1000.0, cost_usd=0.001)


# --- ID 찾기 ---


@pytest.mark.parametrize(
    "text,wbs_id,found",
    [
        ("WBS 2.5 지연", "2.5", True),
        ("2.5.1 업무", "2.5", False),
        ("12.5 업무", "2.5", False),
        ("기준일 2026-09-24", "2", False),
        ("단계 지연: 2 (설계)", "2", True),
        ("업무 2.3", "2", False),
        ("3.11 지연", "3.1", False),
        ("(3.1)", "3.1", True),
    ],
)
def test_wbs_in(text: str, wbs_id: str, found: bool) -> None:
    assert _wbs_in(text, wbs_id) is found


def test_pms_ids_in_reads_full_and_short_forms() -> None:
    text = "PRJ-08-R03, R05 그리고 PRJ-08-D010. 다른 사업 PRJ-07-R01"

    assert _pms_ids_in(text, "PRJ-08", "R") == {"PRJ-08-R03", "PRJ-08-R05"}
    assert _pms_ids_in(text, "PRJ-08", "D") == {"PRJ-08-D010"}


def test_facts_ok_checks_recall_precision_stage_and_count() -> None:
    risk_q = _q(expected_facts=AgentExpectedFacts(risk_ids=["PRJ-03-R05", "PRJ-03-R08"]))
    assert facts_ok(risk_q, "PRJ-03-R05, PRJ-03-R08") is True
    assert facts_ok(risk_q, "PRJ-03-R05") is False  # 재현율
    assert facts_ok(risk_q, "PRJ-03-R05, PRJ-03-R08, PRJ-03-R01") is False  # 정밀도

    stage_q = _q(expected_facts=AgentExpectedFacts(wbs_ids=["2", "2.3"], stage_wbs_ids=["2"]))
    assert facts_ok(stage_q, "단계 지연: WBS 2\n업무 지연: WBS 2.3") is True
    assert facts_ok(stage_q, "지연: WBS 2, WBS 2.3") is False

    count_q = _q(expected_facts=AgentExpectedFacts(count=6))
    assert facts_ok(count_q, "지연 업무는 6건이다") is True
    assert facts_ok(count_q, "지연 업무는 16건이다") is False

    assert facts_ok(_q(), "아무 답") is None


# --- 툴 선택·되묻기 ---


def test_tool_selection_uses_ok_calls_and_target_args() -> None:
    q = _q(type="both", expected_tools=["get_delayed_tasks", "search_documents"])
    good = _a(
        calls=[
            ("get_delayed_tasks", {"project_id": "PRJ-03"}, "ok"),
            (
                "search_documents",
                {"query": "a"},
                "invalid_args",
            ),  # 고쳐 다시 부른 실패는 세지 않는다
            ("search_documents", {"query": "a", "doc_ids": ["docA"]}, "ok"),
        ]
    )
    wrong_doc = _a(
        calls=[
            ("get_delayed_tasks", {"project_id": "PRJ-03"}, "ok"),
            ("search_documents", {"query": "a", "doc_ids": ["docB"]}, "ok"),
        ]
    )
    extra = _a(
        calls=[
            ("get_delayed_tasks", {"project_id": "PRJ-03"}, "ok"),
            ("get_open_risks", {"project_id": "PRJ-03"}, "ok"),
            ("search_documents", {"query": "a", "doc_ids": ["docA"]}, "ok"),
        ]
    )

    assert tool_selection_ok(q, good)
    assert not tool_selection_ok(q, wrong_doc)
    assert not tool_selection_ok(q, extra)


def test_clarified_requires_no_ok_call_and_a_question() -> None:
    assert clarified(_a("어느 사업인가요?", calls=[("get_open_risks", {}, "project_required")]))
    assert not clarified(
        _a("PRJ-01 기준으로 답합니다", calls=[("get_open_risks", {"project_id": "PRJ-01"}, "ok")])
    )
    assert not clarified(_a("문서에서 찾을 수 없음"))
    assert not clarified(_a("목록을 확인했습니다."))


def test_check_row_two_turns_clarify_then_answer() -> None:
    q = _q(
        type="unspecified",
        turns=["산출물", "PRJ-03이야"],
        expected_tools=["get_deliverables"],
        should_clarify=True,
        expected_facts=AgentExpectedFacts(deliverable_ids=["PRJ-03-D001"]),
    )
    turns = [
        _turn(_a("어느 사업인가요?")),
        _turn(_a("PRJ-03-D001", calls=[("get_deliverables", {"project_id": "PRJ-03"}, "ok")])),
    ]

    checks = check_row(q, turns)

    assert checks["clarify"] is True
    assert checks["tool_selection"] is True
    assert checks["facts"] is True
    assert checks["refusal"] is True


def test_check_row_citation_and_refusal() -> None:
    doc_q = _q(type="doc", expected_tools=["search_documents"], must_cite=True)
    cited = _a(
        "답 [1]", calls=[("search_documents", {"doc_ids": ["docA"]}, "ok")], citation_map={1: "[A]"}
    )
    uncited = _a("답", calls=[("search_documents", {"doc_ids": ["docA"]}, "ok")])
    bad_number = _a("답 [1][9]", citation_map={1: "[A]"}, invalid_citations=[9])

    assert check_row(doc_q, [_turn(cited)])["citation"] is True
    assert check_row(doc_q, [_turn(uncited)])["citation"] is False
    assert check_row(doc_q, [_turn(bad_number)])["no_invalid_citation"] is False

    refuse_q = _q(type="no_answer", expected_tools=[], should_refuse=True)
    checks = check_row(refuse_q, [_turn(_a("문서에서 찾을 수 없음", llm_calls=1))])
    assert (
        checks["refusal"] is True and checks["tool_selection"] is True and checks["facts"] is None
    )


def test_check_row_marks_error() -> None:
    assert check_row(_q(), [_turn(None)]) == {"error": True}


# --- 요약 ---


def _row(q: AgentDevQuestion, answer: AgentAnswer) -> AgentRow:
    turns = [_turn(answer)]
    return AgentRow(qid=q.qid, type=q.type, turns=turns, checks=check_row(q, turns))


def test_summarize_doc_criterion_waits_for_judgments() -> None:
    q = _q(qid="d1", type="doc", expected_tools=["search_documents"], must_cite=True)
    row = _row(
        q,
        _a(
            "답 [1]",
            calls=[("search_documents", {"doc_ids": ["docA"]}, "ok")],
            citation_map={1: "[A]"},
        ),
    )

    pending = summarize([q], [row], [], [])
    judged = summarize(
        [q],
        [row],
        [],
        [
            Judgment(qid="d1", system="agent", correct=False, reason="r"),
            Judgment(qid="d1", system="baseline_doc", correct=True, reason="r"),
        ],
    )
    worse = summarize(
        [q],
        [row],
        [],
        [
            Judgment(qid="d1", system="agent", correct=False, reason="r"),
            Judgment(qid="d1", system="baseline_doc", correct=False, reason="r"),
        ],
    )

    assert pending["results"]["doc_answers"]["pass"] is None
    assert pending["all_pass"] is False
    # 차이 1문항까지는 통과(기준선 − 1)
    assert judged["results"]["doc_answers"]["pass"] is True
    assert worse["results"]["doc_answers"]["pass"] is True
    assert judged["results"]["citation"]["pass"] is True
    assert judged["results"]["latency"]["pass"] is None


def test_summarize_llm_calls_and_max_steps() -> None:
    q = _q(qid="p1")
    many = _row(q, _a("x", llm_calls=6, reached_max_steps=True))

    summary = summarize([q], [many], [], [])

    assert summary["results"]["llm_calls"]["pass"] is False
    assert summary["agent_llm_calls"] == 6


# --- 평가 세트 ---


def test_eval_set_composition_and_sources() -> None:
    questions = agent_dev_set.build_questions()

    assert len(questions) == 30
    assert Counter(q.type for q in questions) == {
        "doc": 8,
        "pms": 8,
        "both": 6,
        "unspecified": 4,
        "no_answer": 4,
    }
    assert len({q.qid for q in questions}) == 30
    assert sorted(q.qid for q in questions if q.dod) == ["b1", "d1", "p1"]
    # PMS 없는 문서도 문서 문항에 있다
    assert any(q.doc_id == "63d86447109810d1" and q.project_id is None for q in questions)
    for q in questions:
        if q.type in ("pms", "both"):
            assert q.project_id and q.expected_facts.model_dump(exclude_defaults=True)
        if q.type == "doc":
            assert q.reference_answer
        assert q.prediction


def test_committed_eval_set_matches_sources() -> None:
    assert agent_dev_set.main(["check"]) == 0


def test_partial_judgment_is_reported_both_ways_without_changing_rule() -> None:
    questions = [
        _q(qid=f"d{i}", type="doc", expected_tools=["search_documents"], must_cite=True)
        for i in (1, 2)
    ]
    rows = [_row(q, _a("답")) for q in questions]
    judgments = [
        Judgment(qid="d1", system="agent", correct=True, reason="r"),
        Judgment(qid="d2", system="agent", correct=True, reason="r"),
        Judgment(qid="d1", system="baseline_doc", correct=True, reason="r"),
        Judgment(qid="d2", system="baseline_doc", correct=True, partial=True, reason="절반"),
    ]

    result = summarize(questions, rows, [], judgments)["results"]["doc_answers"]

    # 부분 정답은 정답(correct=True)으로 세고, 오답으로 볼 때의 수와 판정 영향을 함께 적는다
    assert "기준선 문서 지정 2(부분 정답 1 포함, 오답으로 보면 1)" in result["value"]
    assert result["pass"] is True
    assert result["pass_if_partial_wrong"] is True
