"""근거 답변 실행 — qa_v2 57문항 + 무관 질문 5문항 (이슈 #19 DoD ①~④).

문항마다 서비스 경로 `answer_question(top_k=10)`을 1회 부르고(LLM 호출 최대 1회) 결과를 JSONL로 남긴다.
문항은 네 그룹으로 나눈다. A·B는 **이번 실행에서 LLM에 넣은 검색 결과**로 다시 판정한다(기준은
`reranker_score_dist.build_row`와 같다 — 모든 evidence의 정답 청크가 상위 10 안이면 A).

- A: 답 있음 + 정답 청크가 상위 10 안 / B: 답 있음 + 상위 10에 다 들어오지 못함
- C: 답 없음(qa_v2 `answer`가 null) / U: 무관 질문(`data/eval/irrelevant_v1.jsonl`)

판정 기준(이슈 #19 DoD ②, 2026-10-12 실행 전에 고정): C 5/6 이상 거절, U 5/5 거절, A 잘못 거절 3개 이하.

실패한 문항은 재시도하지 않고 오류를 기록한 뒤 다음 문항으로 간다. LLM 호출은 `LoggingLLMClient`로 하므로
`COST_LOG_PATH`(기본 data/cost_log.jsonl)에 호출마다 비용이 남는다.

출력(`--out-dir`, 파일 이름은 `grounded_answer_<YYYYMMDD>`)
- `.jsonl`: 문항별 결과 / `_run.json`: 실행 정보(모델·추론 강도·프롬프트 버전·인덱스·시각)와 요약
- `.md`: 요약과 A·B 문항의 qa_v2 정답 ↔ LLM 답 대조 표(사람 판정 칸은 비워 둔다)

OpenSearch·tei-embed·tei-rerank와 OpenAI API가 필요하다. **이 실행은 비용이 든다** — 실행 전에 예상 호출 수
(62회)를 사용자에게 알린다(CLAUDE.md 비용).

CLI: `uv run python -m rfp_pm_agent.eval.run_grounded_answer --out-dir data/eval/results/answer`
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rfp_pm_agent.clients.cost import CostLogger, compute_llm_cost
from rfp_pm_agent.clients.llm import LLMClient, OpenAICompatLLMClient
from rfp_pm_agent.clients.logged import LoggingLLMClient
from rfp_pm_agent.config import ClientsConfig, OpenSearchConfig
from rfp_pm_agent.eval.qa_v2 import load_questions_v2
from rfp_pm_agent.eval.reranker_score_dist import GOLD_METHOD, build_row
from rfp_pm_agent.schemas.answer import GroundedAnswer
from rfp_pm_agent.schemas.eval import EvalQuestionV2, IrrelevantQuestion
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search import grounded_answer
from rfp_pm_agent.search.grounded_answer import PROMPT_VERSION, answer_question
from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K
from rfp_pm_agent.tools.search_documents import SearchDeps, search_documents

DEFAULT_QA_FILE = Path("data/eval/qa_v2.jsonl")
DEFAULT_IRRELEVANT_FILE = Path("data/eval/irrelevant_v1.jsonl")
# 10/12 리랭커 점수 분포 측정. 이번 실행의 A·B 판정과 다른 문항을 보고하는 데 쓴다
DEFAULT_PREVIOUS_GROUPS = Path("data/eval/results/answer/reranker_score_dist_20261012.jsonl")
TOP_K = DEFAULT_TOP_K  # 10

Group = Literal["A", "B", "C", "U"]

# DoD ② 통과 기준(이슈 #19, 2026-10-12 고정)
C_MIN_REFUSED = 5  # 답 없음 6문항 중
U_MIN_REFUSED = 5  # 무관 5문항 중
A_MAX_REFUSED = 3  # 정답이 상위 10 안인 문항 중 잘못 거절


class AnswerRow(BaseModel):
    """문항 하나의 실행 결과(JSONL 한 줄). 오류가 나면 error에 남기고 답 관련 필드는 비운다."""

    qid: str
    question: str
    group: Group | None  # 검색에서 오류가 나 결과가 없으면 A·B를 정할 수 없어 None
    tags: list[str]
    refused: bool | None
    answer: str | None
    citations: list[str]
    citation_map: dict[int, str]
    invalid_citations: list[int]
    cited_chunk_ids: list[str]
    hit_chunk_ids: list[str]  # LLM에 넣은 검색 결과(순위 순)
    gold_any_in_top10: bool | None  # A·B만
    cited_gold: bool | None  # A·B만 — 인용한 청크 중 정답 청크가 하나라도 있는가
    citation_format_errors: list[str]  # 결과 청크의 필드로 예상한 출처 형식과 다른 출처
    input_tokens: int | None
    output_tokens: int | None  # 추론 토큰 포함
    reasoning_tokens: int | None
    cost_usd: float | None
    latency_ms: float
    error: str | None


class RunInfo(BaseModel):
    """실행 1회의 정보."""

    started_at: str
    finished_at: str
    model: str
    reasoning_effort: str
    prompt_version: str
    index_alias: str
    index_names: list[str]
    top_k: int
    qa_file: str
    irrelevant_file: str
    question_count: int


class GroupVerdict(BaseModel):
    group: Group
    total: int
    refused: int
    errors: int
    rule: str
    passed: bool | None  # 오류가 있으면 None(판정 불가)


class Summary(BaseModel):
    # DoD ①
    answered: int  # 오류 없이 거절하지 않은 답변
    answered_with_citation: int  # 그중 유효 출처 1개 이상
    answered_without_citation: list[str]
    format_error_qids: list[str]
    refused_with_citation: list[str]  # 0이어야 한다
    # DoD ②
    verdicts: list[GroupVerdict]
    # DoD ③
    b_qids: list[str]
    b_refused: list[str]
    # 기타
    invalid_citation_qids: list[str]
    error_qids: list[str]
    group_changes: list[str]  # 10/12 측정과 그룹이 다른 문항("q013 A→B")
    # DoD ④
    llm_calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float


# --- 출처 형식 검사 (DoD ①) ---


def expected_citation_pattern(hit: DocumentHit) -> re.Pattern[str]:
    """결과 청크의 필드로 ADR-0006 결정 2의 5가지 중 어느 표기여야 하는지 정하고 그 정규식을 만든다.

    `format_citation`을 다시 부르지 않고 표에서 직접 만든다(같은 코드로 같은 코드를 검사하지 않도록).
    """
    title = re.escape(hit.bid_title)
    req = f" · {re.escape(hit.requirement_id)}" if hit.requirement_id else ""
    if hit.format == "pdf":
        page = r" p\.\d+" if hit.printed_page is not None else r" PDF \d+쪽"
        return re.compile(rf"\[{title}{page}{req}\]")
    return re.compile(rf"\[{title}{req}\]")


def citation_format_errors(answer: GroundedAnswer, hits: Sequence[DocumentHit]) -> list[str]:
    """citation_map의 출처 중 그 번호 청크에 맞는 형식이 아닌 것. 번호가 결과 범위 밖이어도 오류다."""
    errors: list[str] = []
    for number, citation in answer.citation_map.items():
        if not 1 <= number <= len(hits):
            errors.append(f"[{number}] 범위 밖: {citation}")
        elif not expected_citation_pattern(hits[number - 1]).fullmatch(citation):
            errors.append(f"[{number}] {citation}")
    return errors


# --- 문항 하나 ---


def build_answer_row(
    *,
    qid: str,
    question: str,
    tags: Sequence[str],
    gold: EvalQuestionV2 | None,
    hits: Sequence[DocumentHit] | None,
    answer: GroundedAnswer | None,
    error: str | None,
    latency_ms: float,
    config: ClientsConfig,
) -> AnswerRow:
    """실행 결과 하나를 행으로 만든다. gold가 None이면 무관 질문(U)이다.

    hits는 answer_question이 LLM에 넣은 검색 결과다(검색 전에 오류가 나면 None).
    """
    group: Group | None
    gold_any: bool | None = None
    cited_gold: bool | None = None
    if gold is None:
        group = "U"
    elif gold.answer is None:
        group = "C"
    elif hits is None:
        group = None
    else:
        score_row = build_row(gold, hits, latency_ms)
        group = "A" if score_row.gold_all_in_top10 else "B"
        gold_any = score_row.gold_any_in_top10
        gold_ids = {c for ids in gold.gold_chunk_ids.get(GOLD_METHOD, []) for c in ids}
        if answer is not None:
            cited_gold = any(c in gold_ids for c in answer.cited_chunk_ids)

    usage = answer.llm_usage if answer else None
    return AnswerRow(
        qid=qid,
        question=question,
        group=group,
        tags=list(tags),
        refused=answer.refused if answer else None,
        answer=answer.answer if answer else None,
        citations=answer.citations if answer else [],
        citation_map=answer.citation_map if answer else {},
        invalid_citations=answer.invalid_citations if answer else [],
        cited_chunk_ids=answer.cited_chunk_ids if answer else [],
        hit_chunk_ids=[h.chunk_id for h in hits] if hits is not None else [],
        gold_any_in_top10=gold_any,
        cited_gold=cited_gold,
        citation_format_errors=citation_format_errors(answer, hits) if answer and hits else [],
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        reasoning_tokens=usage.reasoning_tokens if usage else None,
        cost_usd=(
            compute_llm_cost(
                config, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
            )
            if usage
            else None
        ),
        latency_ms=round(latency_ms, 1),
        error=error,
    )


@contextmanager
def recording_search() -> Iterator[list[list[DocumentHit]]]:
    """`answer_question` 안에서 부르는 `search_documents`의 결과를 가로채 기록한다.

    A·B 판정과 출처 검사를 LLM이 실제로 본 검색 결과로 하려는 것이다. 검색을 따로 한 번 더 하면 리랭커
    점수가 호출마다 조금씩 달라(ADR-0006 "재현성") 순서가 다를 수 있다. 블록을 나가면 원래 함수로 되돌린다.
    """
    # 되돌릴 값은 grounded_answer에 있던 것이다. 감싸서 부르는 함수(이 모듈의 search_documents)와 따로 둔다
    previous = getattr(grounded_answer, "search_documents")  # noqa: B009 — 명시적으로 내보내지 않은 이름
    calls: list[list[DocumentHit]] = []

    def recording(
        query: str,
        top_k: int = DEFAULT_TOP_K,
        doc_ids: list[str] | None = None,
        *,
        deps: SearchDeps,
    ) -> list[DocumentHit]:
        hits = search_documents(query, top_k, doc_ids, deps=deps)
        calls.append(hits)
        return hits

    setattr(grounded_answer, "search_documents", recording)  # noqa: B010 — 모듈 속성 교체(되돌림은 finally)
    try:
        yield calls
    finally:
        setattr(grounded_answer, "search_documents", previous)  # noqa: B010


def run(
    qa_questions: Sequence[EvalQuestionV2],
    irrelevant: Sequence[IrrelevantQuestion],
    *,
    deps: SearchDeps,
    llm: LLMClient,
    config: ClientsConfig,
    clock: Callable[[], float] = time.monotonic,
) -> list[AnswerRow]:
    """문항마다 `answer_question`을 1회 부른다. 예외는 문항 결과에 남기고 다음 문항으로 간다(재시도 없음)."""
    items: list[tuple[str, str, list[str], EvalQuestionV2 | None]] = [
        (q.question_id, q.question, [str(t) for t in q.tags], q) for q in qa_questions
    ] + [(q.question_id, q.question, [], None) for q in irrelevant]

    rows: list[AnswerRow] = []
    with recording_search() as searches:
        for qid, question, tags, gold in items:
            before = len(searches)
            start = clock()
            answer: GroundedAnswer | None = None
            error: str | None = None
            try:
                answer = answer_question(question, deps=deps, llm=llm, top_k=TOP_K)
            except Exception as e:  # noqa: BLE001 — 문항 하나의 실패로 실행 전체를 멈추지 않는다(오류는 기록)
                error = f"{type(e).__name__}: {e}"
            latency_ms = (clock() - start) * 1000
            hits = searches[before] if len(searches) > before else None
            rows.append(
                build_answer_row(
                    qid=qid,
                    question=question,
                    tags=tags,
                    gold=gold,
                    hits=hits,
                    answer=answer,
                    error=error,
                    latency_ms=latency_ms,
                    config=config,
                )
            )
    return rows


# --- 요약 ---


def _verdict(rows: Sequence[AnswerRow], group: Group) -> GroupVerdict:
    members = [r for r in rows if r.group == group]
    refused = sum(r.refused is True for r in members)
    errors = sum(r.error is not None for r in members)
    if group == "A":
        rule, ok = f"잘못 거절 ≤ {A_MAX_REFUSED}", refused <= A_MAX_REFUSED
    elif group == "C":
        rule, ok = f"거절 ≥ {C_MIN_REFUSED}", refused >= C_MIN_REFUSED
    elif group == "U":
        rule, ok = f"거절 ≥ {U_MIN_REFUSED}", refused >= U_MIN_REFUSED
    else:
        rule, ok = "기준 없음(검색에서 생긴 결과, DoD ③)", True
    return GroupVerdict(
        group=group,
        total=len(members),
        refused=refused,
        errors=errors,
        rule=rule,
        passed=None if errors else ok,
    )


def summarize(rows: Sequence[AnswerRow], previous_groups: dict[str, str] | None = None) -> Summary:
    answered = [r for r in rows if r.error is None and r.refused is False]
    changes = [
        f"{r.qid} {previous_groups[r.qid]}→{r.group}"
        for r in rows
        if previous_groups and r.qid in previous_groups and r.group != previous_groups[r.qid]
    ]
    called = [r for r in rows if r.input_tokens is not None]
    return Summary(
        answered=len(answered),
        answered_with_citation=sum(bool(r.citation_map) for r in answered),
        answered_without_citation=[r.qid for r in answered if not r.citation_map],
        format_error_qids=[r.qid for r in rows if r.citation_format_errors],
        refused_with_citation=[r.qid for r in rows if r.refused and r.citations],
        verdicts=[_verdict(rows, g) for g in ("A", "B", "C", "U")],
        b_qids=[r.qid for r in rows if r.group == "B"],
        b_refused=[r.qid for r in rows if r.group == "B" and r.refused],
        invalid_citation_qids=[r.qid for r in rows if r.invalid_citations],
        error_qids=[r.qid for r in rows if r.error is not None],
        group_changes=changes,
        llm_calls=len(called),
        input_tokens=sum(r.input_tokens or 0 for r in called),
        output_tokens=sum(r.output_tokens or 0 for r in called),
        reasoning_tokens=sum(r.reasoning_tokens or 0 for r in called),
        cost_usd=sum(r.cost_usd or 0.0 for r in called),
    )


def format_summary(summary: Summary) -> str:
    s = summary
    ratio = f"{s.answered_with_citation}/{s.answered}" if s.answered else "0/0"
    lines = [
        "== DoD ① 출처",
        f"거절하지 않은 답변 중 유효 출처 1개 이상: {ratio}",
        f"출처 없는 답변: {s.answered_without_citation or '없음'}",
        f"출처 형식 오류: {s.format_error_qids or '없음'}",
        f"출처가 달린 거절: {s.refused_with_citation or '없음'}",
        "",
        "== DoD ② 거절 (그룹 | 문항 | 거절 | 오류 | 기준 | 판정)",
    ]
    for v in s.verdicts:
        mark = {True: "통과", False: "미통과", None: "판정 불가"}[v.passed]
        lines.append(f"{v.group} | {v.total} | {v.refused} | {v.errors} | {v.rule} | {mark}")
    lines += [
        "",
        "== DoD ③ B(정답 청크가 상위 10에 다 들어오지 못함)",
        f"B 문항: {s.b_qids}",
        f"B 중 거절: {s.b_refused}",
        "",
        "== 기타",
        f"범위 밖 인용 번호가 나온 문항: {s.invalid_citation_qids or '없음'}",
        f"오류 문항: {s.error_qids or '없음'}",
        f"10/12 측정과 그룹이 다른 문항: {s.group_changes or '없음'}",
        "",
        "== DoD ④ 비용",
        (
            f"LLM 호출 {s.llm_calls}회, 입력 {s.input_tokens:,} / 출력 {s.output_tokens:,}"
            f"(추론 {s.reasoning_tokens:,}) 토큰, ${s.cost_usd:.6f}"
        ),
    ]
    return "\n".join(lines)


def _cell(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\n", "<br>")


def format_comparison_table(
    rows: Sequence[AnswerRow], qa_questions: Sequence[EvalQuestionV2]
) -> str:
    """A·B 문항의 qa_v2 정답과 LLM 답을 나란히 놓은 마크다운 표. 사람 판정 칸은 비워 둔다."""
    gold_answer = {q.question_id: q.answer for q in qa_questions}
    lines = [
        "| qid | 그룹 | 질문 | qa_v2 정답 | LLM 답 | 거절 | 정답 청크 인용 | 사람 판정 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.group not in ("A", "B"):
            continue
        llm_answer = f"(오류) {r.error}" if r.error else r.answer
        cited = {True: "예", False: "아니오", None: ""}[r.cited_gold]
        refused = {True: "예", False: "아니오", None: ""}[r.refused]
        lines.append(
            f"| {r.qid} | {r.group} | {_cell(r.question)} | {_cell(gold_answer.get(r.qid))} "
            f"| {_cell(llm_answer)} | {refused} | {cited} |  |"
        )
    return "\n".join(lines)


# --- 입출력 ---


def load_irrelevant(path: Path) -> list[IrrelevantQuestion]:
    with path.open(encoding="utf-8") as f:
        return [IrrelevantQuestion.model_validate_json(line) for line in f if line.strip()]


def load_previous_groups(path: Path) -> dict[str, str]:
    """10/12 측정 결과의 qid → 그룹. 파일이 없으면 빈 dict."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {d["qid"]: d["group"] for d in (json.loads(line) for line in f if line.strip())}


def write_outputs(
    out_dir: Path,
    stem: str,
    rows: Sequence[AnswerRow],
    info: RunInfo,
    summary: Summary,
    qa_questions: Sequence[EvalQuestionV2],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{stem}.jsonl"
    run_path = out_dir / f"{stem}_run.json"
    md_path = out_dir / f"{stem}.md"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    run_path.write_text(
        json.dumps(
            {"run": info.model_dump(), "summary": summary.model_dump()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        f"# 근거 답변 실행 {stem}\n\n"
        f"- 모델 `{info.model}`, 추론 강도 `{info.reasoning_effort}`, 프롬프트 `{info.prompt_version}`\n"
        f"- 별칭 `{info.index_alias}` → {info.index_names}, top_k={info.top_k}, 문항 {info.question_count}\n"
        f"- 시각 {info.started_at} ~ {info.finished_at}\n\n"
        f"```\n{format_summary(summary)}\n```\n\n"
        "## A·B 문항 대조 (사람 판정 칸은 비워 둔다)\n\n"
        f"{format_comparison_table(rows, qa_questions)}\n",
        encoding="utf-8",
    )
    return [rows_path, run_path, md_path]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="근거 답변 실행 (이슈 #19). OpenAI 비용이 든다.")
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--irrelevant-file", type=Path, default=DEFAULT_IRRELEVANT_FILE)
    parser.add_argument("--previous-groups", type=Path, default=DEFAULT_PREVIOUS_GROUPS)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    from opensearchpy import OpenSearch

    from rfp_pm_agent.search.opensearch import search_alias

    args = build_arg_parser().parse_args(argv)
    qa_questions = load_questions_v2(args.qa_file)
    irrelevant = load_irrelevant(args.irrelevant_file)
    config = ClientsConfig.from_env()
    os_config = OpenSearchConfig.from_env()
    alias = search_alias()
    index_names = sorted(
        OpenSearch(hosts=[os_config.url], timeout=os_config.timeout_s).indices.get_alias(name=alias)
    )
    llm = LoggingLLMClient(OpenAICompatLLMClient(config), config, CostLogger(config.cost_log_path))

    started = datetime.now(UTC)
    rows = run(qa_questions, irrelevant, deps=SearchDeps.from_env(), llm=llm, config=config)
    finished = datetime.now(UTC)

    info = RunInfo(
        started_at=started.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        model=config.llm_model_dev,
        reasoning_effort=config.llm_reasoning_effort,
        prompt_version=PROMPT_VERSION,
        index_alias=alias,
        index_names=index_names,
        top_k=TOP_K,
        qa_file=str(args.qa_file),
        irrelevant_file=str(args.irrelevant_file),
        question_count=len(rows),
    )
    summary = summarize(rows, load_previous_groups(args.previous_groups))
    stem = f"grounded_answer_{started.astimezone().strftime('%Y%m%d')}"
    for path in write_outputs(args.out_dir, stem, rows, info, summary, qa_questions):
        print(path)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
