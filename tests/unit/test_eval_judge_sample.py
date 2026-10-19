"""사람 판정용 문항 고르기 (이슈 #25)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rfp_pm_agent.eval.answer_judge import CitedChunk, JudgeInput
from rfp_pm_agent.eval.judge_sample import (
    QUOTAS,
    HumanJudgeItem,
    output_stem,
    select,
    stratum_of,
    to_markdown,
)


def _chunk(chunk_id: str, text: str = "원문") -> CitedChunk:
    return CitedChunk(number=1, chunk_id=chunk_id, text=text)


def _input(qid: str, **overrides: Any) -> JudgeInput:
    fields: dict[str, Any] = {
        "qid": qid,
        "question": "질문",
        "gold_answer": "정답",
        "system_answer": "답[1]",
        "refused": False,
        "cited_chunks": [_chunk("gold")],
    }
    fields.update(overrides)
    return JudgeInput(**fields)


GOLD = {"gold"}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "answered_single_cite"),
        ({"cited_chunks": [_chunk("gold"), _chunk("x")]}, "answered_multi_cite"),
        ({"cited_chunks": [_chunk("x"), _chunk("y")]}, "answered_no_gold"),
        ({"cited_chunks": []}, "answered_no_gold"),
        ({"refused": True, "cited_chunks": []}, "refused_answerable"),
        ({"refused": True, "cited_chunks": [], "gold_answer": None}, "refused_no_answer"),
        ({"gold_answer": None}, None),  # 정답 없는데 답함
        ({"error": "x", "refused": None}, None),
    ],
)
def test_묶음_판정(overrides: dict[str, Any], expected: str | None) -> None:
    assert stratum_of(_input("q001", **overrides), GOLD) == expected


def _pool() -> list[JudgeInput]:
    items: list[JudgeInput] = []
    for i in range(4):
        items.append(_input(f"q1{i}"))
        items.append(_input(f"q2{i}", cited_chunks=[_chunk("gold"), _chunk("x")]))
        items.append(_input(f"q3{i}", cited_chunks=[_chunk("x")]))
        items.append(_input(f"q4{i}", refused=True, cited_chunks=[]))
        items.append(_input(f"u5{i}", refused=True, cited_chunks=[], gold_answer=None))
    return items


def test_묶음별_개수만큼_고르고_시드가_같으면_같은_문항이다():
    gold = {item.qid: GOLD for item in _pool()}

    first = select(_pool(), gold)
    again = select(list(reversed(_pool())), gold)  # 입력 순서가 달라도 같다

    assert [p.qid for p in first] == [p.qid for p in again]
    assert [p.qid for p in first] == sorted(p.qid for p in first)
    counts = {s: sum(1 for p in first if p.stratum == s) for s in QUOTAS}
    assert counts == QUOTAS
    assert len(first) == 10


def test_후보가_모자란_묶음이_있으면_멈춘다():
    items = [i for i in _pool() if not i.qid.startswith("q3")]
    with pytest.raises(ValueError, match="answered_no_gold"):
        select(items, {item.qid: GOLD for item in items})


def test_판정지는_빈칸을_두고_묶음_이름과_LLM_판정을_싣지_않는다():
    items = [
        HumanJudgeItem(
            qid="q001",
            stratum="answered_no_gold",
            question="사업 기간은?",
            gold_answer="3개월",
            system_answer="4개월[2]",
            refused=False,
            cited_chunks=[CitedChunk(number=2, chunk_id="d:c2", text="첫 줄\n둘째 줄")],
        ),
        HumanJudgeItem(
            qid="q002",
            stratum="refused_no_answer",
            question="없는 것은?",
            gold_answer=None,
            system_answer="문서에서 찾을 수 없음",
            refused=True,
            cited_chunks=[],
        ),
    ]

    md = to_markdown(items, source="answers.jsonl")

    assert md.count("- 답 맞음 O/X:") == 2
    assert md.count("- 출처 맞음 O/X:") == 2
    assert "[2] `d:c2`" in md
    assert "  첫 줄\n  둘째 줄" in md
    assert "(답 없음 — 거절이 정답)" in md
    assert "(인용 없음)" in md
    assert "answered_no_gold" not in md and "오답" not in md


def test_출력_이름은_결과_파일_이름에서_만든다():
    path = Path("grounded_answer_20261015_doc_replay_v2.jsonl")
    assert output_stem(path) == "human_judge_20261015_doc_replay_v2"
