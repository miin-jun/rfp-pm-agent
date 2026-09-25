"""검수표 data/tmp/review_v2.md 생성 (qa_v2의 q031~ + drafts.json 메타)."""

import json
import subprocess
import sys
from pathlib import Path

from rfp_pm_agent.eval.qa_v2 import (
    PARAPHRASE_MAX_OVERLAP,
    evidence_source_ids,
    load_documents,
    load_questions_v2,
    overlap_ratio,
    shared_bigrams,
)
from rfp_pm_agent.eval.retrieval import normalize
from rfp_pm_agent.schemas.document import Document

S = Path(sys.argv[1])
drafts = {d["question_id"]: d for d in json.loads((S / "drafts.json").read_text())}
questions = [q for q in load_questions_v2(Path("data/eval/qa_v2.jsonl")) if q.question_id in drafts]
docs = load_documents(Path("data/parsed"), [q.doc_id for q in questions])


def one_line(text: str, width: int = 140) -> str:
    t = " ".join(text.split())
    return t if len(t) <= width else t[:width] + "…"


def context(doc: Document, unit_id: str, evidence: str) -> tuple[str, str]:
    """블록이면 앞뒤 블록, 요구사항이면 요구사항 text 안의 앞뒤 줄."""
    req = next((r for r in doc.requirements if r.requirement_id == unit_id), None)
    if req is not None:
        lines = req.text.splitlines()
        target = normalize(evidence)
        idx = next(
            (
                i
                for i, ln in enumerate(lines)
                if normalize(ln) and normalize(ln) in target or target[:15] in normalize(ln)
            ),
            None,
        )
        if idx is None:
            return "(줄 위치 못 찾음)", "(줄 위치 못 찾음)"
        # "세부/내용:"처럼 라벨이 줄바꿈으로 쪼개진 짧은 줄은 문맥으로 쓰지 않는다
        before = next(
            (ln for ln in reversed(lines[:idx]) if len(normalize(ln)) >= 4), "(요구사항 시작)"
        )
        after = next(
            (
                ln
                for ln in lines[idx + 1 :]
                if len(normalize(ln)) >= 4 and normalize(ln) not in target
            ),
            "(요구사항 끝)",
        )
        return one_line(before), one_line(after)
    items = sorted(
        [(b.source_order, b.block_id, b.text) for b in doc.blocks]
        + [(r.source_order, r.requirement_id, r.text) for r in doc.requirements]
    )
    i = next(k for k, it in enumerate(items) if it[1] == unit_id)
    before = f"[{items[i - 1][1]}] {one_line(items[i - 1][2])}" if i > 0 else "(문서 시작)"
    after = (
        f"[{items[i + 1][1]}] {one_line(items[i + 1][2])}" if i + 1 < len(items) else "(문서 끝)"
    )
    return before, after


def absence_counts(doc: Document, terms: list[str]) -> list[str]:
    units = [b.text for b in doc.blocks] + [
        r.text + "\n" + "\n".join(r.raw_fields.values()) for r in doc.requirements
    ]
    return [f"{t} {sum(normalize(t) in normalize(u) for u in units)}건" for t in terms]


seed_out = subprocess.run(
    ["uv", "run", "python", str(S / "sample.py"), "0"], capture_output=True, text=True, check=True
).stdout

out = [
    "# qa_v2 추가 26문항 검수표 (#15)",
    "",
    "- 대상: `data/eval/qa_v2.jsonl`의 q031~q056. `verified_by`·`verified_at`은 비어 있다 — 검수 후 채운다",
    f"- paraphrase 기준: 겹침률 ≤ {PARAPHRASE_MAX_OVERLAP} (한글·영문·숫자 글자 bigram, 불용어 제거 없음). 다른 유형은 참고값으로만 적었다",
    "- 정답 선택에 검색(BM25·벡터) 결과를 쓰지 않았다. 후보는 아래 시드로 뽑은 순서대로 읽고, 쓸 수 없는 후보는 이유와 함께 건너뛰었다",
    "- 답 없음 문항의 검색어 대조는 부재 확인용이다 (공백 제거 후 원문 단위 포함 여부)",
    "- `page`는 모두 null로 두었다 (v1의 page 기준이 PDF 쪽인지 인쇄 쪽인지 미확인)",
    "",
    "## 후보 추출",
    "",
    '- 시드: `SEED = 15` (`random.Random(15)`로 슬롯 배정, 슬롯별 후보 순서는 `random.Random(f"15-{유형}-{doc_id}")`)',
    "- 배정 규칙: 표는 PDF 3개 문서 전부 + HWP 7개 중 무작위 3개, 답 없음은 10개 중 무작위 6개, 나머지 14개(말 바꾸기 6·exact 2·여러 청크 6)는 문항 수가 적은 문서부터 채워 문서당 2~3개",
    "- 후보 필터: v1 정답 청크 제외. 말 바꾸기·여러 청크는 30자 이상 문단(목차 줄 제외)+요구사항, exact는 요구사항만, 표는 2×2 이상이고 요구사항 정의표 머리글(요구사항 고유번호·분류·명칭)이 없는 table 블록",
    "",
    "```",
    seed_out.strip(),
    "```",
    "",
    "## 건너뛴 후보",
    "",
    "| 문항 | 후보 | 이유 |",
    "|---|---|---|",
]
for qid, d in drafts.items():
    for unit, reason in d.get("skipped", []):
        out.append(f"| {qid} | {unit} | {reason} |")
out.append("")

for q in questions:
    d = drafts[q.question_id]
    doc = docs[q.doc_id]
    out += [
        f"## {q.question_id} — {', '.join(q.tags)}",
        "",
        f"- 문서: `{q.doc_id}` {doc.bid_title} ({doc.format})",
        f"- 질문: {q.question}",
        f"- answer: {q.answer if q.answer is not None else '(null — 답 없음)'}",
        f"- type: {q.type}",
    ]
    for i, (e, unit) in enumerate(zip(q.evidence, d.get("units", []), strict=True)):
        before, after = context(doc, unit, e)
        out += [
            f"- evidence[{i}] (`{unit}`, 원문 단위: {', '.join(s.replace(chr(10), '') for s in evidence_source_ids(doc, e)[:3])})",
            "  ```",
            *[f"  {ln}" for ln in e.splitlines()],
            "  ```",
            f"  - 앞: {before}",
            f"  - 뒤: {after}",
            f"  - 정답 청크(block_requirement): {', '.join(q.gold_chunk_ids['block_requirement'][i])}",
        ]
    if q.evidence:
        ratio = overlap_ratio(q.question, q.evidence)
        applies = "기준 적용" if "paraphrase" in q.tags else "참고값(기준 미적용)"
        out.append(
            f"- 겹침률: {ratio:.3f} ({applies}) — 겹친 bigram: {' '.join(shared_bigrams(q.question, q.evidence)) or '없음'}"
        )
    else:
        out += [
            f"- 부재 확인 검색어: {', '.join(absence_counts(doc, d['search_terms']))}",
            f"- 목차 확인: {d['toc']}",
        ]
    out += [f"- 애매한 점: {d['ambiguity']}", "- 검수: [ ] 승인  [ ] 수정  [ ] 제외 — 메모:", ""]

Path("data/tmp").mkdir(parents=True, exist_ok=True)
Path("data/tmp/review_v2.md").write_text("\n".join(out), encoding="utf-8")
print(f"data/tmp/review_v2.md {len(out)}줄")
