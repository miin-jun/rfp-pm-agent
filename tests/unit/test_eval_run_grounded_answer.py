"""근거 답변 실행 스크립트의 그룹 판정·요약 계산 (#19). 검색은 가짜, LLM은 ScriptedLLMClient다.

검색은 `run_grounded_answer.search_documents`(recording_search가 감싸는 원래 함수)를 바꿔 끼운다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rfp_pm_agent.clients.llm import ChatMessage, LLMResponse
from rfp_pm_agent.eval import run_grounded_answer as rga
from rfp_pm_agent.eval.reranker_score_dist import GOLD_METHOD
from rfp_pm_agent.schemas.answer import GroundedAnswer
from rfp_pm_agent.schemas.eval import EvalQuestionV2, IrrelevantQuestion
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search import grounded_answer
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.grounded_answer import REFUSAL_TEXT
from rfp_pm_agent.tools.search_documents import SearchDeps
from tests.fakes.fake_embedding import FakeEmbeddingClient
from tests.fakes.fake_llm import ScriptedLLMClient
from tests.fakes.fake_reranker import FakeRerankerClient
from tests.fakes.fake_search_client import RecordingSearchClient
from tests.unit.conftest import make_clients_config

CONFIG = make_clients_config(llm_price_input_per_1m=0.10, llm_price_output_per_1m=0.50)


def _cid(source_id: str) -> str:
    return f"docA:{GOLD_METHOD}:{source_id}"


def _hit(rank: int, source_id: str, **overrides: Any) -> DocumentHit:
    fields: dict[str, Any] = {
        "rank": rank,
        "chunk_id": _cid(source_id),
        "doc_id": "docA",
        "bid_title": "A 사업",
        "format": "pdf",
        "requirement_id": "SFR-001",
        "page": 14,
        "printed_page": 23,
        "block_type": None,
        "text": f"{source_id} 본문",
        "score": 1.0 - rank * 0.1,
    }
    fields.update(overrides)
    return DocumentHit(**fields)


HITS = [_hit(1, "b0001"), _hit(2, "b0002"), _hit(3, "b0003")]


def _question(qid: str, gold: list[list[str]] | None) -> EvalQuestionV2:
    """gold가 None이면 답 없음 문항."""
    if gold is None:
        return EvalQuestionV2(
            question_id=qid,
            question=f"{qid} 질문",
            evidence=[],
            doc_id="docA",
            tags=["no_answer"],
            gold_chunk_ids={GOLD_METHOD: []},
        )
    return EvalQuestionV2(
        question_id=qid,
        question=f"{qid} 질문",
        answer=f"{qid} 정답",
        evidence=["근거"] * len(gold),
        doc_id="docA",
        gold_chunk_ids={GOLD_METHOD: [[_cid(s) for s in g] for g in gold]},
    )


def _irrelevant(qid: str) -> IrrelevantQuestion:
    return IrrelevantQuestion(question_id=qid, question=f"{qid} 무관", category="일반 상식")


def _deps() -> SearchDeps:
    return SearchDeps(
        search_client=RecordingSearchClient(),
        embedder=FakeEmbeddingClient(),
        reranker=FakeRerankerClient(),
    )


def _patch_search(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, list[DocumentHit] | Exception]
) -> None:
    """질문 문자열 → 검색 결과(또는 예외). 없는 질문은 HITS."""

    def fake_search(
        query: str, top_k: int = 10, doc_ids: list[str] | None = None, *, deps: SearchDeps
    ) -> list[DocumentHit]:
        result = results.get(query, HITS)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(rga, "search_documents", fake_search)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    qa: list[EvalQuestionV2],
    irrelevant: list[IrrelevantQuestion],
    replies: list[str],
    search_results: dict[str, list[DocumentHit] | Exception] | None = None,
) -> tuple[list[rga.AnswerRow], ScriptedLLMClient]:
    _patch_search(monkeypatch, search_results or {})
    llm = ScriptedLLMClient(replies, input_tokens=1000, output_tokens=200, reasoning_tokens=50)
    rows = rga.run(qa, irrelevant, deps=_deps(), llm=llm, config=CONFIG)
    return rows, llm


# --- 그룹 판정 ---


def test_groups_are_judged_from_the_hits_the_llm_saw(monkeypatch: pytest.MonkeyPatch) -> None:
    qa = [
        _question("q001", [["b0001"], ["b0003"]]),  # 정답 둘 다 상위 안 → A
        _question("q002", [["b0001"], ["b0099"]]),  # 하나 밖 → B
        _question("q003", None),  # 답 없음 → C
    ]
    rows, llm = _run(
        monkeypatch,
        qa,
        [_irrelevant("u001")],
        ["답 [3].", "답 [2].", REFUSAL_TEXT, REFUSAL_TEXT],
    )

    assert [(r.qid, r.group) for r in rows] == [
        ("q001", "A"),
        ("q002", "B"),
        ("q003", "C"),
        ("u001", "U"),
    ]
    assert len(llm.calls) == 4
    assert rows[0].hit_chunk_ids == [h.chunk_id for h in HITS]
    assert (rows[0].gold_any_in_top10, rows[0].cited_gold) == (True, True)  # [3] = b0003 정답
    assert (rows[1].gold_any_in_top10, rows[1].cited_gold) == (True, False)  # [2]는 정답 아님
    assert (rows[2].cited_gold, rows[3].cited_gold) == (None, None)
    assert [r.refused for r in rows] == [False, False, True, True]


def test_row_records_tokens_cost_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, _ = _run(monkeypatch, [_question("q001", [["b0001"]])], [], ["답 [1][9]."])

    row = rows[0]
    assert row.citation_map == {1: "[A 사업 p.23 · SFR-001]"}
    assert row.invalid_citations == [9]
    assert (row.input_tokens, row.output_tokens, row.reasoning_tokens) == (1000, 200, 50)
    assert row.cost_usd == pytest.approx(1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6)
    assert row.citation_format_errors == []
    assert row.error is None


def test_search_error_is_recorded_and_run_continues_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qa = [_question("q001", [["b0001"]]), _question("q002", [["b0001"]])]
    rows, llm = _run(
        monkeypatch,
        qa,
        [],
        ["답 [1]."],
        {"q001 질문": SearchUnavailableError("tei-rerank", "connection refused")},
    )

    assert rows[0].error == (
        "SearchUnavailableError: 검색 서버(tei-rerank)에 연결할 수 없다: connection refused"
    )
    assert rows[0].group is None  # 검색 결과가 없어 A·B를 정할 수 없다
    assert (rows[0].refused, rows[0].input_tokens, rows[0].cost_usd) == (None, None, None)
    assert rows[1].group == "A" and rows[1].error is None
    assert len(llm.calls) == 1


def test_llm_error_keeps_the_group_from_search(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingLLM:
        def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
            raise RuntimeError("rate limit")

    _patch_search(monkeypatch, {})
    rows = rga.run(
        [_question("q001", [["b0001"]])], [], deps=_deps(), llm=FailingLLM(), config=CONFIG
    )

    assert rows[0].group == "A"
    assert rows[0].error == "RuntimeError: rate limit"
    assert rows[0].cited_gold is None


def test_recording_search_restores_the_original_function() -> None:
    before = getattr(grounded_answer, "search_documents")  # noqa: B009 — 명시적으로 내보내지 않은 이름
    with rga.recording_search():
        assert getattr(grounded_answer, "search_documents") is not before  # noqa: B009
    assert getattr(grounded_answer, "search_documents") is before  # noqa: B009


# --- 출처 형식 검사 ---


@pytest.mark.parametrize(
    ("overrides", "citation"),
    [
        ({}, "[A 사업 p.23 · SFR-001]"),
        ({"printed_page": None}, "[A 사업 PDF 15쪽 · SFR-001]"),
        ({"printed_page": None, "requirement_id": None}, "[A 사업 PDF 15쪽]"),
        ({"format": "hwp", "page": None, "printed_page": None}, "[A 사업 · SFR-001]"),
        (
            {"format": "hwpx", "page": None, "printed_page": None, "requirement_id": None},
            "[A 사업]",
        ),
    ],
)
def test_expected_citation_pattern_accepts_the_five_forms(
    overrides: dict[str, Any], citation: str
) -> None:
    assert rga.expected_citation_pattern(_hit(1, "b0001", **overrides)).fullmatch(citation)


@pytest.mark.parametrize(
    "citation",
    [
        "[A 사업 PDF 15쪽 · SFR-001]",  # 인쇄 쪽번호가 있는데 PDF 쪽
        "[A 사업 p.23]",  # 요구사항 ID 빠짐
        "[B 사업 p.23 · SFR-001]",  # 다른 문서명
        "[A 사업 p.23 · SFR-002]",  # 다른 요구사항 ID
    ],
)
def test_citation_format_errors_flags_mismatched_citations(citation: str) -> None:
    answer = GroundedAnswer(
        answer="답 [1]",
        refused=False,
        citations=[citation],
        citation_map={1: citation},
        cited_chunk_ids=[HITS[0].chunk_id],
        invalid_citations=[],
        hit_count=len(HITS),
        llm_usage=None,
    )
    assert rga.citation_format_errors(answer, HITS) == [f"[1] {citation}"]


# --- 요약 ---


def _row(qid: str, group: str | None, refused: bool | None, **overrides: Any) -> rga.AnswerRow:
    fields: dict[str, Any] = {
        "qid": qid,
        "question": "질문",
        "group": group,
        "tags": [],
        "refused": refused,
        "answer": REFUSAL_TEXT if refused else "답 [1]",
        "citations": [] if refused else ["[A 사업]"],
        "citation_map": {} if refused else {1: "[A 사업]"},
        "invalid_citations": [],
        "cited_chunk_ids": [] if refused else ["c1"],
        "hit_chunk_ids": ["c1"],
        "gold_any_in_top10": None,
        "cited_gold": None,
        "citation_format_errors": [],
        "input_tokens": 1000,
        "output_tokens": 200,
        "reasoning_tokens": 50,
        "cost_usd": 0.0002,
        "latency_ms": 10.0,
        "error": None,
    }
    fields.update(overrides)
    return rga.AnswerRow(**fields)


def _passing_rows() -> list[rga.AnswerRow]:
    """A 34(잘못 거절 3) · B 1 · C 6(거절 5) · U 5(거절 5) — 모두 기준선에 딱 맞는다."""
    rows = [_row(f"a{i:02d}", "A", i < 3) for i in range(34)]
    rows.append(_row("b01", "B", True))
    rows += [_row(f"c{i}", "C", i < 5) for i in range(6)]
    rows += [_row(f"u{i}", "U", True) for i in range(5)]
    return rows


def test_summary_passes_at_the_boundary_of_each_rule() -> None:
    summary = rga.summarize(_passing_rows())

    verdicts = {v.group: (v.total, v.refused, v.passed) for v in summary.verdicts}
    assert verdicts == {
        "A": (34, 3, True),
        "B": (1, 1, True),
        "C": (6, 5, True),
        "U": (5, 5, True),
    }
    assert summary.b_qids == ["b01"] and summary.b_refused == ["b01"]


@pytest.mark.parametrize(
    ("qid", "refused", "group"),
    [("a10", True, "A"), ("c0", False, "C"), ("u0", False, "U")],
)
def test_summary_fails_one_step_past_the_rule(qid: str, refused: bool, group: str) -> None:
    rows = [
        _row(qid, r.group, refused) if r.qid == qid else r for r in _passing_rows()
    ]  # A는 잘못 거절 4, C는 거절 4, U는 거절 4

    verdicts: dict[str, bool | None] = {v.group: v.passed for v in rga.summarize(rows).verdicts}

    assert verdicts[group] is False
    assert all(ok for g, ok in verdicts.items() if g != group)


def test_summary_marks_group_with_error_as_undecided() -> None:
    rows = _passing_rows()
    rows[-1] = _row("u4", "U", None, error="RuntimeError: x", input_tokens=None, cost_usd=None)

    verdict = next(v for v in rga.summarize(rows).verdicts if v.group == "U")

    assert (verdict.refused, verdict.errors, verdict.passed) == (4, 1, None)


def test_summary_counts_citations_tokens_cost_and_group_changes() -> None:
    rows = [
        _row("q001", "A", False),
        _row("q002", "A", False, citation_map={}, citations=[], cited_chunk_ids=[]),
        _row("q003", "B", False, invalid_citations=[15], citation_format_errors=["[1] x"]),
        _row("q004", "C", True),
        _row(
            "q005",
            None,
            None,
            error="SearchUnavailableError: x",
            input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            cost_usd=None,
        ),
    ]

    summary = rga.summarize(rows, {"q001": "A", "q003": "A", "q004": "C"})

    assert (summary.answered, summary.answered_with_citation) == (3, 2)
    assert summary.answered_without_citation == ["q002"]
    assert summary.format_error_qids == ["q003"]
    assert summary.refused_with_citation == []
    assert summary.invalid_citation_qids == ["q003"]
    assert summary.error_qids == ["q005"]
    assert summary.group_changes == ["q003 A→B"]
    assert (summary.llm_calls, summary.input_tokens, summary.output_tokens) == (4, 4000, 800)
    assert summary.reasoning_tokens == 200
    assert summary.cost_usd == pytest.approx(0.0008)


def test_comparison_table_lists_only_a_and_b_and_escapes_cells() -> None:
    qa = [_question("q001", [["b0001"]]), _question("q002", None)]
    rows = [
        _row("q001", "A", False, answer="한 줄 | 둘\n셋", cited_gold=True),
        _row("q002", "C", True),
        _row("u001", "U", True),
    ]

    table = rga.format_comparison_table(rows, qa).splitlines()

    assert len(table) == 3  # 머리 2줄 + A 1줄
    assert table[2] == "| q001 | A | 질문 | q001 정답 | 한 줄 \\| 둘<br>셋 | 아니오 | 예 |  |"


def test_load_irrelevant_reads_the_confirmed_set() -> None:
    questions = rga.load_irrelevant(Path("data/eval/irrelevant_v1.jsonl"))

    assert [q.question_id for q in questions] == ["u001", "u002", "u003", "u004", "u005"]
