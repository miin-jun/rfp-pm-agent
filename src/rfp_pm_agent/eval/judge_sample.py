"""사람 판정용 문항 고르기 (이슈 #25) — LLM 채점과 사람 판정의 일치율을 재기 위한 표본.

근거 답변 결과에서 문항을 아래 묶음별 개수만큼 고른다. 묶음 안에서는 qid 순으로 정렬한 뒤
`random.Random(SEED).sample`로 뽑는다. 정답 청크는 현재 qa_v2의 `gold_chunk_ids[GOLD_METHOD]`로
다시 판정한다(결과 파일의 `cited_gold`는 실행 당시 정답 기준이라 쓰지 않는다).

| 묶음 | 조건 | 개수 |
|---|---|---|
| answered_no_gold | 답함, 인용 청크에 정답 청크 없음 (오답 후보) | 2 |
| answered_multi_cite | 답함, 정답 청크 인용, 인용 2개 이상 | 3 |
| answered_single_cite | 답함, 정답 청크 인용, 인용 1개 | 2 |
| refused_answerable | 정답이 있는데 거절 | 2 |
| refused_no_answer | 정답 없는 문항(답 없음·무관)에서 거절 | 1 |

묶음 후보가 개수보다 적으면 ValueError — 규칙을 바꿔야 하는 상황이라 다른 묶음에서 채우지 않는다.
오류가 난 문항과 정답 없는 문항에 답한 문항은 어느 묶음에도 넣지 않는다.

출력에는 LLM 채점 결과를 넣지 않는다(사람이 먼저 판정). 마크다운에는 묶음 이름도 쓰지 않는다 — "오답 후보"
같은 이름이 판정을 기울게 하기 때문이다. 문항은 qid 순이다. LLM을 부르지 않으므로 비용은 $0이다.

CLI: `uv run python -m rfp_pm_agent.eval.judge_sample --answers <결과>.jsonl --out-dir data/eval/results/answer`
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rfp_pm_agent.eval.answer_judge import (
    DEFAULT_QA_FILE,
    CitedChunk,
    JudgeInput,
    load_inputs,
)
from rfp_pm_agent.eval.qa_v2 import load_questions_v2
from rfp_pm_agent.eval.reranker_score_dist import GOLD_METHOD
from rfp_pm_agent.ingest.chunking import DEFAULT_CHUNKS_DIR

SEED = 25

Stratum = Literal[
    "answered_no_gold",
    "answered_multi_cite",
    "answered_single_cite",
    "refused_answerable",
    "refused_no_answer",
]
QUOTAS: dict[Stratum, int] = {
    "answered_no_gold": 2,
    "answered_multi_cite": 3,
    "answered_single_cite": 2,
    "refused_answerable": 2,
    "refused_no_answer": 1,
}


class HumanJudgeItem(BaseModel):
    """사람 판정 한 문항. human_* 필드는 사람이 채운다(처음에는 비어 있다)."""

    qid: str
    stratum: Stratum
    question: str
    gold_answer: str | None
    system_answer: str | None
    refused: bool | None
    cited_chunks: list[CitedChunk]
    human_answer_correct: bool | None = None
    human_citation_supported: bool | None = None
    human_note: str = ""


def stratum_of(item: JudgeInput, gold_ids: set[str]) -> Stratum | None:
    """문항이 속한 묶음. 어느 묶음에도 넣지 않는 문항(오류, 정답 없는데 답함)은 None."""
    if item.error is not None or item.refused is None:
        return None
    if item.refused:
        return "refused_no_answer" if item.gold_answer is None else "refused_answerable"
    if item.gold_answer is None:
        return None
    if not any(c.chunk_id in gold_ids for c in item.cited_chunks):
        return "answered_no_gold"
    return "answered_multi_cite" if len(item.cited_chunks) >= 2 else "answered_single_cite"


def select(
    items: Sequence[JudgeInput],
    gold_ids_by_qid: Mapping[str, set[str]],
    *,
    quotas: Mapping[Stratum, int] = QUOTAS,
    seed: int = SEED,
) -> list[HumanJudgeItem]:
    """묶음별 개수만큼 고른 문항을 qid 순으로 돌려준다. 후보가 모자란 묶음이 있으면 ValueError."""
    rng = random.Random(seed)
    candidates: dict[Stratum, list[JudgeInput]] = {s: [] for s in quotas}
    for item in sorted(items, key=lambda i: i.qid):
        stratum = stratum_of(item, gold_ids_by_qid.get(item.qid, set()))
        if stratum in candidates:
            candidates[stratum].append(item)

    picked: list[HumanJudgeItem] = []
    for stratum, count in quotas.items():
        pool = candidates[stratum]
        if len(pool) < count:
            raise ValueError(f"{stratum}: 후보 {len(pool)}개가 필요한 {count}개보다 적다")
        picked += [
            HumanJudgeItem(
                qid=item.qid,
                stratum=stratum,
                question=item.question,
                gold_answer=item.gold_answer,
                system_answer=item.system_answer,
                refused=item.refused,
                cited_chunks=item.cited_chunks,
            )
            for item in rng.sample(pool, count)
        ]
    return sorted(picked, key=lambda p: p.qid)


def to_markdown(items: Sequence[HumanJudgeItem], *, source: str) -> str:
    """채팅에 그대로 붙여 넣을 수 있는 판정지. 묶음 이름과 LLM 판정은 싣지 않는다."""
    lines = [
        f"# 사람 판정 {len(items)}문항",
        "",
        f"출처: `{source}`",
        "",
        "- **답 맞음**: 시스템 답이 정답의 핵심 사실을 빠짐없이, 틀리지 않게 담았나 (정답이 없는 문항은 거절했으면 O)",
        "- **출처 맞음**: 시스템 답의 내용이 인용한 청크에 실제로 있나 (정답과 맞는지는 따로 본다)",
        "",
    ]
    for n, item in enumerate(items, 1):
        lines += [
            f"## {n}. {item.qid}",
            "",
            f"- **질문**: {item.question}",
            f"- **정답**: {item.gold_answer if item.gold_answer is not None else '(답 없음 — 거절이 정답)'}",
            f"- **시스템 답**: {item.system_answer}",
            "- **인용 청크**:" + ("" if item.cited_chunks else " (인용 없음)"),
        ]
        for chunk in item.cited_chunks:
            lines += ["", f"  [{chunk.number}] `{chunk.chunk_id}`", "", "  ```text"]
            lines += [f"  {line}".rstrip() for line in chunk.text.splitlines()]
            lines += ["  ```"]
        lines += ["", "- 답 맞음 O/X:", "- 출처 맞음 O/X:", ""]
    return "\n".join(lines)


def output_stem(answers: Path) -> str:
    """`grounded_answer_20261015_doc_replay_v2` → `human_judge_20261015_doc_replay_v2`."""
    return "human_judge_" + answers.stem.removeprefix("grounded_answer_")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--answers", type=Path, required=True, help="근거 답변 실행의 문항별 .jsonl"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    stem = output_stem(args.answers)
    jsonl_path = args.out_dir / f"{stem}.jsonl"
    md_path = args.out_dir / f"{stem}.md"
    existing = [str(p) for p in (jsonl_path, md_path) if p.exists()]
    if existing:
        # 사람이 채우던 판정을 덮어쓰지 않는다
        parser.error(f"결과 파일이 이미 있다: {existing}")

    items = load_inputs(args.answers, args.qa_file, args.chunks_dir)
    gold_ids = {
        q.question_id: {cid for group in q.gold_chunk_ids.get(GOLD_METHOD, []) for cid in group}
        for q in load_questions_v2(args.qa_file)
    }
    picked = select(items, gold_ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in picked:
            f.write(item.model_dump_json() + "\n")
    md_path.write_text(to_markdown(picked, source=str(args.answers)), encoding="utf-8")
    print(jsonl_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
