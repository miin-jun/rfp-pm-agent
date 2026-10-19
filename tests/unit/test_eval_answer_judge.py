"""근거 답변 LLM 채점 (이슈 #25). LLM은 tests/fakes의 ScriptedLLMClient로 대체한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rfp_pm_agent.eval.answer_judge import (
    JUDGE_PROMPT_VERSION,
    CitedChunk,
    JudgeInput,
    build_judge_inputs,
    build_user_message,
    judge_all,
    judge_one,
    load_judge_prompt,
    main,
    needs_llm,
    parse_verdict,
)
from tests.fakes.fake_llm import ScriptedLLMClient


def _row(qid: str = "q001", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "qid": qid,
        "question": "사업 기간은?",
        "refused": False,
        "answer": "3개월입니다.[2]",
        "citation_map": {"2": "[사업 PDF 1쪽]"},
        "hit_chunk_ids": ["d:c1", "d:c2", "d:c3"],
        "error": None,
    }
    row.update(overrides)
    return row


def _input(**overrides: Any) -> JudgeInput:
    fields: dict[str, Any] = {
        "qid": "q001",
        "question": "사업 기간은?",
        "gold_answer": "계약일로부터 3개월이다.",
        "system_answer": "3개월입니다.[1]",
        "refused": False,
        "cited_chunks": [CitedChunk(number=1, chunk_id="d:c1", text="사업기간: 3개월")],
    }
    fields.update(overrides)
    return JudgeInput(**fields)


def _verdict(correct: bool = True, supported: bool = True, reason: str = "같은 기간") -> str:
    return json.dumps(
        {"answer_correct": correct, "citation_supported": supported, "reason": reason},
        ensure_ascii=False,
    )


# --- 입력 만들기 ---


def test_인용_번호_n은_hit_chunk_ids의_n번째_청크_원문으로_바뀐다():
    rows = [_row(citation_map={"3": "x", "1": "y"})]
    texts = {"d:c1": "첫째", "d:c2": "둘째", "d:c3": "셋째"}

    [item] = build_judge_inputs(rows, {"q001": "정답"}, texts)

    assert [(c.number, c.chunk_id, c.text) for c in item.cited_chunks] == [
        (3, "d:c3", "셋째"),
        (1, "d:c1", "첫째"),
    ]
    assert item.gold_answer == "정답"


def test_qa_v2에_없는_무관_질문은_정답_없음으로_본다():
    [item] = build_judge_inputs(
        [_row("u001", refused=True, answer="문서에서 찾을 수 없음", citation_map={})], {}, {}
    )
    assert item.gold_answer is None
    assert item.cited_chunks == []


def test_청크_원문을_찾지_못하면_멈춘다():
    with pytest.raises(KeyError):
        build_judge_inputs([_row()], {"q001": "정답"}, {"d:c1": "첫째"})


# --- 규칙 판정 (LLM 호출 없음) ---


@pytest.mark.parametrize(
    ("gold", "refused", "correct", "reason"),
    [
        (None, True, True, "정답 없는 문항 — 거절함"),
        (None, False, False, "정답 없는 문항인데 답함"),
        ("3개월", True, False, "정답이 있는데 거절함"),
    ],
)
def test_정답_없음과_거절은_LLM_없이_판정한다(
    gold: str | None, refused: bool, correct: bool, reason: str
) -> None:
    llm = ScriptedLLMClient([])
    row = judge_one(_input(gold_answer=gold, refused=refused, cited_chunks=[]), llm)

    assert llm.calls == []
    assert (row.method, row.answer_correct, row.citation_supported) == ("rule", correct, None)
    assert row.reason == reason


def test_실행_오류_문항은_채점하지_않고_오류로_남긴다():
    llm = ScriptedLLMClient([])
    row = judge_one(_input(error="TimeoutError: x", refused=None, system_answer=None), llm)

    assert llm.calls == []
    assert row.method is None
    assert row.answer_correct is None
    assert row.error is not None and "TimeoutError" in row.error


# --- LLM 판정 ---


def test_LLM_채점은_system_프롬프트_파일과_질문_정답_답_인용청크를_보낸다():
    llm = ScriptedLLMClient([_verdict()])
    judge_one(_input(), llm, model="judge-model")

    [(messages, model)] = llm.calls
    assert model == "judge-model"
    assert messages[0].role == "system"
    assert messages[0].content == load_judge_prompt()
    user = messages[1].content
    for expected in ("사업 기간은?", "계약일로부터 3개월이다.", "3개월입니다.[1]"):
        assert expected in user
    assert '<chunk id="1">\n사업기간: 3개월\n</chunk>' in user


def test_judge_v1_프롬프트는_두_판정과_JSON_출력_형식을_적는다():
    prompt = load_judge_prompt()
    assert JUDGE_PROMPT_VERSION == "judge_v1"
    for key in ("answer_correct", "citation_supported", "reason"):
        assert key in prompt


def test_LLM_판정과_사용량을_기록한다():
    llm = ScriptedLLMClient([_verdict(True, False, "청크에 없는 날짜")], model="m1")
    row = judge_one(_input(), llm)

    assert (row.method, row.answer_correct, row.citation_supported) == ("llm", True, False)
    assert row.reason == "청크에 없는 날짜"
    assert (row.model, row.prompt_version) == ("m1", JUDGE_PROMPT_VERSION)
    assert (row.input_tokens, row.output_tokens, row.reasoning_tokens) == (100, 20, 5)
    assert row.error is None


def test_인용이_없으면_LLM이_근거_있다고_해도_citation_supported는_거짓():
    row = judge_one(_input(cited_chunks=[]), ScriptedLLMClient([_verdict(True, True)]))
    assert row.citation_supported is False
    assert "(인용한 조각 없음)" in build_user_message(_input(cited_chunks=[]))


def test_코드_블록으로_감싼_JSON은_읽는다():
    verdict = parse_verdict(f"```json\n{_verdict(False, True)}\n```")
    assert (verdict.answer_correct, verdict.citation_supported) == (False, True)


@pytest.mark.parametrize(
    "content",
    [
        "맞습니다",
        '{"answer_correct": true, "reason": "x"}',  # citation_supported 빠짐
        '판정: {"answer_correct": true, "citation_supported": true, "reason": "x"}',
    ],
)
def test_형식이_어긋난_응답은_재시도하지_않고_원문과_함께_오류로_남긴다(content: str) -> None:
    llm = ScriptedLLMClient([content])
    row = judge_one(_input(), llm)

    assert len(llm.calls) == 1
    assert row.answer_correct is None
    assert row.error is not None and row.error.startswith("채점 응답 형식 오류")
    assert row.raw_output == content


def test_LLM_예외는_그_문항만_오류로_남기고_다음_문항으로_간다():
    class _Boom(ScriptedLLMClient):
        def chat(self, messages: Any, *, model: str | None = None) -> Any:
            if not self.calls:
                self.calls.append((list(messages), model))
                raise RuntimeError("rate limit")
            return super().chat(messages, model=model)

    llm = _Boom(["unused", _verdict()])
    rows = judge_all([_input(qid="q001"), _input(qid="q002")], llm)

    assert rows[0].error == "RuntimeError: rate limit"
    assert rows[1].answer_correct is True


def test_needs_llm은_정답_있고_답한_문항만_참():
    assert needs_llm(_input()) is True
    assert needs_llm(_input(gold_answer=None)) is False
    assert needs_llm(_input(refused=True)) is False
    assert needs_llm(_input(error="x")) is False


# --- CLI ---


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    answers = tmp_path / "answers.jsonl"
    rows = [
        _row("q001"),
        _row("q002", refused=True, answer="문서에서 찾을 수 없음", citation_map={}),
        _row("u001", refused=True, answer="문서에서 찾을 수 없음", citation_map={}),
    ]
    answers.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    qa = tmp_path / "qa.jsonl"
    base = {"evidence": ["x"], "doc_id": "d", "type": "일반"}
    qa.write_text(
        "".join(
            json.dumps({"question_id": qid, "question": "q", "answer": "a"} | base) + "\n"
            for qid in ("q001", "q002")
        ),
        "utf-8",
    )
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk = {"doc_id": "d", "method": "block_requirement", "source_ids": ["b1"], "text": "t"}
    (chunks_dir / "block_requirement.jsonl").write_text(
        "".join(json.dumps(chunk | {"chunk_id": f"d:c{i}"}) + "\n" for i in (1, 2, 3)), "utf-8"
    )
    return answers, qa, chunks_dir


def test_dry_run은_LLM_호출_수만_출력한다(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers, qa, chunks_dir = _write_inputs(tmp_path)
    args = ["--answers", str(answers), "--qa-file", str(qa), "--chunks-dir", str(chunks_dir)]

    assert main([*args, "--dry-run"]) == 0
    assert "LLM 채점 1개(호출 1회)" in capsys.readouterr().out


def test_실제_채점은_model과_out이_없으면_호출_전에_멈춘다(tmp_path: Path) -> None:
    answers, qa, chunks_dir = _write_inputs(tmp_path)
    args = ["--answers", str(answers), "--qa-file", str(qa), "--chunks-dir", str(chunks_dir)]

    with pytest.raises(SystemExit):
        main(args)
