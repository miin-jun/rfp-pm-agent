"""근거 답변 LLM 채점 — 답 정확성·인용 근거성 (이슈 #25).

근거 답변 실행 결과(`run_grounded_answer`의 문항별 JSONL)를 문항마다 두 가지로 채점한다.

- answer_correct: 시스템 답이 qa_v2 정답(`answer`)의 핵심 사실을 맞게 담았나
- citation_supported: 시스템 답의 내용이 인용한 청크 원문에 실제로 있나(정답과 맞는지는 따로 본다)

LLM 없이 규칙으로 정하는 문항(호출 0회):
- 정답이 없는 문항(qa_v2 `answer`가 null인 답 없음, 무관 질문): 거절했으면 맞음. 인용 판정은 하지 않는다(None)
- 정답이 있는데 거절한 문항: 틀림. 인용 판정은 하지 않는다(None)
- 실행 오류로 답이 없는 문항: 채점하지 않고 error에 남긴다

나머지는 LLM을 문항당 1회 부른다. 프롬프트는 `eval/prompts/<JUDGE_PROMPT_VERSION>.md`이고, 응답은
JSON 객체 하나를 `JudgeVerdict`로 검증한다. 형식이 어긋나면 재시도하지 않고 error에 남긴다 — 다시 부르면
비용이 늘고, 형식 오류 자체가 채점 모델을 고를 때 볼 정보다. 인용한 청크가 없으면 LLM 판정과 상관없이
citation_supported는 False다.

인용 번호 n은 실행 때 LLM에 넣은 검색 결과의 n번째(`hit_chunk_ids[n-1]`)다(`search/grounded_answer.py`와 같다).
청크 원문은 청크 파일(`data/chunks/<GOLD_METHOD>.jsonl`)에서 읽는다.

CLI: `uv run python -m rfp_pm_agent.eval.answer_judge --answers <결과>.jsonl --dry-run` (예상 호출 수만, 비용 없음)
     `uv run python -m rfp_pm_agent.eval.answer_judge --answers <결과>.jsonl --model <채점 모델> --out <출력>.jsonl`
**두 번째 명령은 비용이 든다** — 실행 전에 --dry-run의 예상 호출 수를 사용자에게 알린다(CLAUDE.md 비용).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from rfp_pm_agent.clients.cost import CostLogger
from rfp_pm_agent.clients.llm import ChatMessage, LLMClient, OpenAICompatLLMClient
from rfp_pm_agent.clients.logged import LoggingLLMClient
from rfp_pm_agent.config import ClientsConfig
from rfp_pm_agent.eval.qa_v2 import load_questions_v2
from rfp_pm_agent.eval.reranker_score_dist import GOLD_METHOD
from rfp_pm_agent.ingest.chunking import DEFAULT_CHUNKS_DIR, load_chunks

# 프롬프트 파일 이름(확장자 제외)이 곧 버전이다. 내용을 바꾸면 파일을 고치지 않고 _v2를 새로 만든다
JUDGE_PROMPT_VERSION = "judge_v1"

DEFAULT_QA_FILE = Path("data/eval/qa_v2.jsonl")

JudgeMethod = Literal["rule", "llm"]

# LLM이 JSON 앞뒤에 코드 블록 표시(```json … ```)를 붙인 경우만 벗긴다. 그 밖의 군더더기는 형식 오류다
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class CitedChunk(BaseModel):
    """시스템 답이 인용한 청크 하나. number는 답 안의 [번호]다."""

    number: int
    chunk_id: str
    text: str


class JudgeInput(BaseModel):
    """채점 입력 한 문항. gold_answer가 None이면 정답이 없는 문항(답 없음·무관 질문)이다."""

    qid: str
    question: str
    gold_answer: str | None
    system_answer: str | None
    refused: bool | None
    cited_chunks: list[CitedChunk]
    error: str | None = None  # 실행 때 난 오류. 있으면 채점하지 않는다


class JudgeVerdict(BaseModel):
    """채점 LLM이 내는 JSON 하나 (judge_v1 프롬프트의 출력 형식)."""

    answer_correct: bool
    citation_supported: bool
    reason: str


class JudgeRow(BaseModel):
    """채점 결과 한 줄. 오류가 나면 error에 남기고 판정 필드는 비운다."""

    qid: str
    method: JudgeMethod | None  # 실행 오류로 채점하지 않았으면 None
    answer_correct: bool | None
    citation_supported: bool | None  # 규칙 판정(거절)은 인용이 없어 None
    reason: str
    prompt_version: str | None  # LLM 판정만
    model: str | None  # LLM 판정만
    input_tokens: int | None
    output_tokens: int | None  # 추론 토큰 포함
    reasoning_tokens: int | None
    raw_output: str | None  # LLM 원문 응답. 형식 오류를 확인할 때 본다
    error: str | None


def load_judge_prompt() -> str:
    """system 메시지 내용을 프롬프트 파일(`eval/prompts/<JUDGE_PROMPT_VERSION>.md`)에서 읽는다."""
    path = resources.files("rfp_pm_agent.eval").joinpath("prompts", f"{JUDGE_PROMPT_VERSION}.md")
    return path.read_text("utf-8")


def cited_numbers(row: Mapping[str, object]) -> list[int]:
    """결과 한 줄에서 유효한 인용 번호를 답에 나온 순서대로 뽑는다(JSON에서 키가 문자열이 된 것을 숫자로)."""
    citation_map = row.get("citation_map") or {}
    assert isinstance(citation_map, dict)
    return [int(number) for number in citation_map]


def build_judge_inputs(
    rows: Iterable[Mapping[str, object]],
    gold_answers: Mapping[str, str | None],
    chunk_texts: Mapping[str, str],
) -> list[JudgeInput]:
    """근거 답변 결과 줄들을 채점 입력으로 바꾼다.

    gold_answers는 qid → qa_v2 정답이다. 무관 질문처럼 qa_v2에 없는 qid는 정답 없음(None)으로 본다.
    인용 번호 n은 hit_chunk_ids[n-1]이다. 청크 원문이 chunk_texts에 없으면 KeyError — 청크 파일이
    실행 때와 달라졌다는 뜻이라 그대로 채점하면 안 된다.
    """
    inputs: list[JudgeInput] = []
    for row in rows:
        qid = str(row["qid"])
        hit_ids = row.get("hit_chunk_ids") or []
        assert isinstance(hit_ids, list)
        cited: list[CitedChunk] = []
        for number in cited_numbers(row):
            chunk_id = str(hit_ids[number - 1])
            cited.append(CitedChunk(number=number, chunk_id=chunk_id, text=chunk_texts[chunk_id]))
        answer = row.get("answer")
        refused = row.get("refused")
        error = row.get("error")
        inputs.append(
            JudgeInput(
                qid=qid,
                question=str(row["question"]),
                gold_answer=gold_answers.get(qid),
                system_answer=answer if isinstance(answer, str) else None,
                refused=refused if isinstance(refused, bool) else None,
                cited_chunks=cited,
                error=str(error) if error is not None else None,
            )
        )
    return inputs


def needs_llm(item: JudgeInput) -> bool:
    """이 문항을 LLM으로 채점해야 하면 참. 규칙 판정·실행 오류 문항은 거짓이다."""
    return item.error is None and item.gold_answer is not None and item.refused is False


def build_user_message(item: JudgeInput) -> str:
    """채점 LLM에 보낼 user 메시지. 인용 청크는 답 안의 번호 그대로 싣는다."""
    chunks = (
        "\n".join(f'<chunk id="{c.number}">\n{c.text}\n</chunk>' for c in item.cited_chunks)
        or "(인용한 조각 없음)"
    )
    return (
        f"<question>\n{item.question}\n</question>\n"
        f"<gold_answer>\n{item.gold_answer}\n</gold_answer>\n"
        f"<system_answer>\n{item.system_answer}\n</system_answer>\n"
        f"{chunks}"
    )


def parse_verdict(content: str) -> JudgeVerdict:
    """LLM 응답을 JudgeVerdict로 검증한다. 형식이 어긋나면 ValueError(pydantic ValidationError 포함)."""
    text = content.strip()
    fenced = _CODE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    return JudgeVerdict.model_validate_json(text)


def _rule_row(item: JudgeInput) -> JudgeRow:
    if item.gold_answer is None:
        correct = bool(item.refused)
        reason = "정답 없는 문항 — 거절함" if correct else "정답 없는 문항인데 답함"
    else:
        correct = False
        reason = "정답이 있는데 거절함"
    return JudgeRow(
        qid=item.qid,
        method="rule",
        answer_correct=correct,
        citation_supported=None,
        reason=reason,
        prompt_version=None,
        model=None,
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        raw_output=None,
        error=None,
    )


def _error_row(item: JudgeInput, error: str, *, method: JudgeMethod | None) -> JudgeRow:
    return JudgeRow(
        qid=item.qid,
        method=method,
        answer_correct=None,
        citation_supported=None,
        reason="",
        prompt_version=JUDGE_PROMPT_VERSION if method == "llm" else None,
        model=None,
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        raw_output=None,
        error=error,
    )


def judge_one(item: JudgeInput, llm: LLMClient, *, model: str | None = None) -> JudgeRow:
    """문항 하나를 채점한다. LLM은 needs_llm일 때만 1회 부른다. LLM 예외는 잡아 error에 남긴다."""
    if item.error is not None:
        return _error_row(item, f"실행 오류로 채점하지 않음: {item.error}", method=None)
    if not needs_llm(item):
        return _rule_row(item)

    messages = [
        ChatMessage(role="system", content=load_judge_prompt()),
        ChatMessage(role="user", content=build_user_message(item)),
    ]
    try:
        response = llm.chat(messages, model=model)
    except Exception as exc:  # noqa: BLE001 — 한 문항 실패로 전체 채점을 멈추지 않는다
        return _error_row(item, f"{type(exc).__name__}: {exc}", method="llm")

    base = _error_row(item, "", method="llm").model_copy(
        update={
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "raw_output": response.content,
        }
    )
    try:
        verdict = parse_verdict(response.content)
    except (ValidationError, ValueError) as exc:
        return base.model_copy(update={"error": f"채점 응답 형식 오류: {type(exc).__name__}"})
    return base.model_copy(
        update={
            "answer_correct": verdict.answer_correct,
            # 인용이 없으면 근거가 있을 수 없다. LLM이 true라고 해도 따르지 않는다
            "citation_supported": verdict.citation_supported and bool(item.cited_chunks),
            "reason": verdict.reason,
            "error": None,
        }
    )


def judge_all(
    items: Sequence[JudgeInput], llm: LLMClient, *, model: str | None = None
) -> list[JudgeRow]:
    """문항 순서대로 채점한다. 실패한 문항은 재시도하지 않고 다음 문항으로 간다."""
    return [judge_one(item, llm, model=model) for item in items]


def load_answer_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_inputs(answers: Path, qa_file: Path, chunks_dir: Path) -> list[JudgeInput]:
    """결과 파일·qa_v2·청크 파일을 읽어 채점 입력을 만든다(LLM 호출 없음)."""
    gold = {q.question_id: q.answer for q in load_questions_v2(qa_file)}
    texts = {c.chunk_id: c.text for c in load_chunks(chunks_dir / f"{GOLD_METHOD}.jsonl")}
    return build_judge_inputs(load_answer_rows(answers), gold, texts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--answers", type=Path, required=True, help="근거 답변 실행의 문항별 .jsonl"
    )
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="예상 LLM 호출 수만 출력한다(비용 없음)"
    )
    parser.add_argument(
        "--model", help="채점 모델(실제 채점 때 필수). 답을 만든 모델과 따로 고른다"
    )
    parser.add_argument("--out", type=Path, help="채점 결과 .jsonl(실제 채점 때 필수)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    items = load_inputs(args.answers, args.qa_file, args.chunks_dir)
    llm_count = sum(1 for item in items if needs_llm(item))
    print(
        f"문항 {len(items)}개 중 LLM 채점 {llm_count}개(호출 {llm_count}회), 규칙·제외 {len(items) - llm_count}개"
    )
    if args.dry_run:
        return 0
    if args.model is None or args.out is None:
        parser.error("실제 채점에는 --model과 --out이 필요하다(호출 수만 보려면 --dry-run)")
    if args.out.exists():
        # LLM 비용을 쓰기 전에 멈춘다. 이전 채점 결과를 지우지 않게 한다
        parser.error(f"결과 파일이 이미 있다: {args.out}")
    config = ClientsConfig.from_env()
    llm = LoggingLLMClient(OpenAICompatLLMClient(config), config, CostLogger(config.cost_log_path))
    rows = judge_all(items, llm, model=args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
