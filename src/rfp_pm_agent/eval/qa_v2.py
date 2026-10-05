"""평가 세트 v2 만들기·검증 (이슈 #15).

qa_v2.jsonl의 원천 정답은 evidence(원문 문자열 목록)이고, gold_chunk_ids는
청크 파일에서 코드로 뽑은 파생값이다. 이 모듈은 세 가지를 한다.

- build-from-v1: qa_v1 30문항을 같은 question_id로 v2 형식에 옮긴다. qa_v1은
  읽기만 한다. 태그는 코드로 판정할 수 있는 table과 paraphrase(겹침률 기준)만
  붙이고, 사람 판정이 필요한 태그(exact·doc_unspecified)는 비워 둔다. 출력 파일이 이미 있으면
  덮어쓰지 않고 멈춘다 — 뒤에 추가한 문항이 지워지지 않게 하기 위해서다.
- fill-gold: 모든 문항의 gold_chunk_ids를 현재 청크 파일로 다시 뽑아 쓴다.
- check: 형식·원문 일치·gold_chunk_ids 일치·태그 일관성(table 코드 판정,
  paraphrase 겹침률 ≤ PARAPHRASE_MAX_OVERLAP)을 확인하고, 청킹
  방식별 최고 점수를 낸다. 하나라도 어긋나면 종료 코드 1로 끝난다.

정답은 원문 대조로만 정한다. 검색 결과는 어디에서도 쓰지 않는다 — 검색
결과로 정답을 정하면 평가가 그 검색 방식 쪽으로 기운다.

LLM을 부르지 않으므로 비용은 $0이다.

CLI: `uv run python -m rfp_pm_agent.eval.qa_v2 check`
     `uv run python -m rfp_pm_agent.eval.qa_v2 fill-gold`
     `uv run python -m rfp_pm_agent.eval.qa_v2 build-from-v1`
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rfp_pm_agent.eval.retrieval import evidence_in_text, gold_chunk_ids, reachable
from rfp_pm_agent.eval.run_chunk_eval import METHODS
from rfp_pm_agent.ingest.chunking import DEFAULT_CHUNKS_DIR, DEFAULT_PARSED_DIR, load_chunks
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Block, Document
from rfp_pm_agent.schemas.eval import EvalQuestionV2, QuestionTag

DEFAULT_QA_V1 = Path("data/eval/qa_v1.jsonl")
DEFAULT_QA_V2 = Path("data/eval/qa_v2.jsonl")

# DoD 기준 청킹 방식. 이 방식에서 evidence마다 정답 청크가 1개 이상이어야 한다
REFERENCE_METHOD = "block_requirement"

# qa_v1에는 사람 검수 기록이 없다. 기록이 없는 검수를 채우면 거짓 기록이 된다
V1_NOTE = "사람 검수 기록 없음 — #13에서 코드 대조만"

# table 태그의 최소 표 크기. 1×1 표는 글상자처럼 레이아웃용으로 쓰여 표 안의
# 정보를 묻는 문항이 아니다
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

# paraphrase 태그의 겹침률 상한. 추가 문항 초안을 쓰기 전에 qa_v1 분포(0.00~0.36,
# 중앙값 0.114)를 보고 고정했다. 초안이 넘으면 이 값이 아니라 문항을 고친다
PARAPHRASE_MAX_OVERLAP = 0.10
PARAPHRASE_NOTE = "paraphrase: 코드 판정"

_NON_WORD = re.compile(r"[^가-힣A-Za-z0-9]")

logger = logging.getLogger(__name__)

ChunksByMethod = Mapping[str, Sequence[Chunk]]


@dataclass
class CheckReport:
    """check 결과. errors가 비어 있어야 통과다."""

    total: int = 0
    no_answer: int = 0
    # 방식별 (도달 가능 문항 수, 답이 있는 문항 수). 답 없음 문항은 분모에서 뺀다
    ceilings: dict[str, tuple[int, int]] = field(default_factory=dict)
    tag_counts: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_questions_v2(path: Path) -> list[EvalQuestionV2]:
    """jsonl을 v2 스키마로 읽는다. qa_v1 파일도 읽힌다. 형식이 어긋나면 그 줄에서 멈춘다."""
    with path.open(encoding="utf-8") as f:
        return [EvalQuestionV2.model_validate_json(line) for line in f if line.strip()]


def write_questions_v2(path: Path, questions: Iterable[EvalQuestionV2]) -> int:
    """jsonl로 쓰고 쓴 줄 수를 돌려준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for question in questions:
            f.write(question.model_dump_json() + "\n")
            written += 1
    return written


def load_documents(parsed_dir: Path, doc_ids: Iterable[str]) -> dict[str, Document]:
    """문항이 가리키는 문서만 읽는다. 파일이 없으면 FileNotFoundError."""
    return {
        doc_id: Document.model_validate_json(
            (parsed_dir / f"{doc_id}.json").read_text(encoding="utf-8")
        )
        for doc_id in sorted(set(doc_ids))
    }


def load_all_chunks(chunks_dir: Path) -> dict[str, list[Chunk]]:
    return {method: load_chunks(chunks_dir / f"{method}.jsonl") for method in METHODS}


def source_units(doc: Document) -> list[tuple[str, str]]:
    """원문 대조 단위 (단위 ID, text). 블록 text와 요구사항 text·필드 값이다.

    evidence는 이 단위 하나 안에서만 가져온다. 두 단위를 이어 붙인 evidence는
    어떤 청킹 방식으로도 한 조각에 담기지 않아 정답이 성립하지 않는다.
    """
    units = [(block.block_id, block.text) for block in doc.blocks]
    for req in doc.requirements:
        units.append((req.requirement_id, req.text))
        units.extend((f"{req.requirement_id}.{k}", v) for k, v in req.raw_fields.items())
        units.extend((f"{req.requirement_id}.{k}", v) for k, v in req.fields.items())
    return units


def evidence_source_ids(doc: Document, evidence: str) -> list[str]:
    """evidence를 담은 원문 단위 ID 목록. 비어 있으면 원문에 없는 evidence다."""
    return [unit_id for unit_id, text in source_units(doc) if evidence_in_text(evidence, text)]


def _char_bigrams(text: str) -> set[str]:
    cleaned = _NON_WORD.sub("", text)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def shared_bigrams(question: str, evidence: Sequence[str]) -> list[str]:
    """질문의 글자 bigram 중 evidence 어느 하나에라도 들어 있는 것 (정렬).

    한글·영문·숫자만 남기고 bigram을 만든다. 불용어는 빼지 않는다 — 뺄 단어
    목록이 기준값을 움직이기 때문이다. evidence가 여럿이면 각각 따로 본다
    (이어 붙이면 경계에서 원문에 없는 bigram이 생긴다).
    """
    cleaned = [_NON_WORD.sub("", e) for e in evidence]
    return sorted(b for b in _char_bigrams(question) if any(b in e for e in cleaned))


def overlap_ratio(question: str, evidence: Sequence[str]) -> float:
    """질문 글자 bigram 중 evidence와 겹치는 비율. 질문에 bigram이 없으면 0."""
    total = len(_char_bigrams(question))
    return len(shared_bigrams(question, evidence)) / total if total else 0.0


def _is_data_table(block: Block | None) -> bool:
    if block is None or block.type != "table" or not block.table:
        return False
    cols = max(len(row) for row in block.table)
    return len(block.table) >= MIN_TABLE_ROWS and cols >= MIN_TABLE_COLS


def is_table_question(question: EvalQuestionV2, doc: Document, chunks: Sequence[Chunk]) -> bool:
    """정답 청크가 모두 2×2 이상 table 블록에서 나왔으면 참 (table 태그의 코드 판정).

    REFERENCE_METHOD 청크를 기준으로 본다. 같은 evidence가 표와 문단 양쪽에 있으면
    표를 못 읽어도 문단으로 맞힐 수 있으므로 table 문항이 아니다. 요구사항 정의표는
    파서가 requirements로 옮기므로 여기서 걸러진다. 단, 파서가 요구사항 표를 놓쳐
    table 블록으로 남긴 경우(#58)는 이 판정이 구별하지 못한다 — 사람 검수에서 본다.
    """
    if not question.evidence:
        return False
    blocks = {block.block_id: block for block in doc.blocks}
    chunk_by_id = {c.chunk_id: c for c in chunks}
    gold = [cid for per in gold_chunk_ids(chunks, question) for cid in per]
    if not gold:
        return False
    return all(
        _is_data_table(blocks.get(source_id))
        for cid in gold
        for source_id in chunk_by_id[cid].source_ids
    )


def with_gold(question: EvalQuestionV2, chunks_by_method: ChunksByMethod) -> EvalQuestionV2:
    """gold_chunk_ids를 현재 청크 파일로 다시 뽑은 사본을 돌려준다."""
    gold = {method: gold_chunk_ids(chunks, question) for method, chunks in chunks_by_method.items()}
    return question.model_copy(update={"gold_chunk_ids": gold})


def build_from_v1(
    v1_questions: Sequence[EvalQuestionV2],
    docs: Mapping[str, Document],
    chunks_by_method: ChunksByMethod,
) -> list[EvalQuestionV2]:
    """qa_v1 문항을 v2로 옮긴다. 문장·정답·evidence는 그대로 두고 파생값만 붙인다."""
    reference = chunks_by_method[REFERENCE_METHOD]
    migrated: list[EvalQuestionV2] = []
    for question in v1_questions:
        tags: list[QuestionTag] = []
        note = V1_NOTE
        if is_table_question(question, docs[question.doc_id], reference):
            tags.append("table")
        if overlap_ratio(question.question, question.evidence) <= PARAPHRASE_MAX_OVERLAP:
            tags.append("paraphrase")
            note = f"{V1_NOTE}; {PARAPHRASE_NOTE}"
        base = question.model_copy(
            update={"tags": tags, "verified_by": None, "verified_at": None, "note": note}
        )
        migrated.append(with_gold(base, chunks_by_method))
    return migrated


def check(
    questions: Sequence[EvalQuestionV2],
    docs: Mapping[str, Document],
    chunks_by_method: ChunksByMethod,
) -> CheckReport:
    """v2 평가 세트를 검증하고 방식별 최고 점수를 낸다."""
    report = CheckReport(total=len(questions))
    ids = Counter(q.question_id for q in questions)
    report.errors += [f"{qid}: question_id 중복 {n}번" for qid, n in ids.items() if n > 1]

    answerable = [q for q in questions if q.evidence]
    report.no_answer = len(questions) - len(answerable)
    for q in questions:
        report.tag_counts.update(q.tags)
        if not q.evidence and not q.note.strip():
            report.errors.append(f"{q.question_id}: 답 없음 문항에 부재 확인 기록(note)이 없다")

    for q in answerable:
        doc = docs[q.doc_id]
        for i, evidence in enumerate(q.evidence):
            if not evidence_source_ids(doc, evidence):
                report.errors.append(f"{q.question_id}: evidence[{i}]가 {q.doc_id} 원문에 없다")

        expected = with_gold(q, chunks_by_method).gold_chunk_ids
        if q.gold_chunk_ids != expected:
            report.errors.append(
                f"{q.question_id}: 기록된 gold_chunk_ids가 현재 청크 파일과 다르다 (fill-gold 필요)"
            )
        for i, per in enumerate(expected.get(REFERENCE_METHOD, [])):
            if not per:
                report.errors.append(
                    f"{q.question_id}: evidence[{i}]를 담은 {REFERENCE_METHOD} 청크가 없다"
                )

        table_only = is_table_question(q, doc, chunks_by_method[REFERENCE_METHOD])
        if "table" in q.tags and not table_only:
            report.errors.append(
                f"{q.question_id}: table 태그가 있지만 정답 청크가 모두 표에서 나오지 않았다"
            )
        if "table" not in q.tags and table_only:
            report.warnings.append(f"{q.question_id}: 정답 청크가 모두 표인데 table 태그가 없다")

        ratio = overlap_ratio(q.question, q.evidence)
        if "paraphrase" in q.tags and ratio > PARAPHRASE_MAX_OVERLAP:
            report.errors.append(
                f"{q.question_id}: paraphrase 태그인데 겹침률 {ratio:.3f}가 "
                f"기준 {PARAPHRASE_MAX_OVERLAP}를 넘는다"
            )

    for method, chunks in chunks_by_method.items():
        report.ceilings[method] = (sum(reachable(chunks, q) for q in answerable), len(answerable))
    return report


def format_report(report: CheckReport) -> str:
    lines = [
        f"문항 {report.total}개 (답 없음 {report.no_answer}개는 최고 점수 분모에서 제외)",
        "태그: " + (", ".join(f"{k}={v}" for k, v in sorted(report.tag_counts.items())) or "없음"),
        "방식별 최고 점수 (모든 evidence에 도달 가능한 문항 수):",
    ]
    lines += [f"  {m:<18} {hit}/{n}" for m, (hit, n) in report.ceilings.items()]
    lines += [f"경고: {w}" for w in report.warnings]
    lines += [f"오류: {e}" for e in report.errors]
    lines.append("결과: " + ("통과" if not report.errors else f"실패 ({len(report.errors)}건)"))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="평가 세트 v2 만들기·검증 (이슈 #15).")
    parser.add_argument("command", choices=["check", "fill-gold", "build-from-v1"])
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_V2, help="v2 평가 세트 jsonl")
    parser.add_argument(
        "--v1", type=Path, default=DEFAULT_QA_V1, help="build-from-v1이 읽을 qa_v1 jsonl"
    )
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """`uv run python -m rfp_pm_agent.eval.qa_v2 <command>` 진입점. 종료 코드를 돌려준다."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    chunks_by_method = load_all_chunks(args.chunks_dir)

    if args.command == "build-from-v1":
        if args.qa.exists():
            logger.error("%s가 이미 있다. 덮어쓰지 않는다.", args.qa)
            return 1
        v1 = load_questions_v2(args.v1)
        docs = load_documents(args.parsed_dir, (q.doc_id for q in v1))
        written = write_questions_v2(args.qa, build_from_v1(v1, docs, chunks_by_method))
        logger.info("%s에 %d문항을 썼다.", args.qa, written)
        return 0

    questions = load_questions_v2(args.qa)
    docs = load_documents(args.parsed_dir, (q.doc_id for q in questions))
    if args.command == "fill-gold":
        written = write_questions_v2(args.qa, [with_gold(q, chunks_by_method) for q in questions])
        logger.info("%s의 %d문항 gold_chunk_ids를 다시 뽑았다.", args.qa, written)
        return 0

    report = check(questions, docs, chunks_by_method)
    logger.info("%s", format_report(report))
    return 0 if not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
