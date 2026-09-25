"""drafts.json 26문항을 qa_v2.jsonl에 덧붙인다. 이미 q031 이후가 있으면 멈춘다."""

import json
import sys
from pathlib import Path

from rfp_pm_agent.eval.qa_v2 import (
    DEFAULT_QA_V2,
    evidence_source_ids,
    load_all_chunks,
    load_documents,
    load_questions_v2,
    with_gold,
    write_questions_v2,
)
from rfp_pm_agent.ingest.chunking import DEFAULT_CHUNKS_DIR, DEFAULT_PARSED_DIR
from rfp_pm_agent.schemas.eval import EvalQuestionV2, QuestionType

DRAFT_NOTE = "초안(Claude 작성), 사람 검수 전"

drafts = json.loads(Path(sys.argv[1]).read_text())
existing = load_questions_v2(DEFAULT_QA_V2)
if len(existing) != 30:
    sys.exit(f"qa_v2에 {len(existing)}문항이 있다. 30문항일 때만 덧붙인다.")

docs = load_documents(DEFAULT_PARSED_DIR, [d["doc_id"] for d in drafts])
chunks = load_all_chunks(DEFAULT_CHUNKS_DIR)
new = []
for d in drafts:
    qtype: QuestionType | None
    doc = docs[d["doc_id"]]
    if d["evidence"]:
        req_ids = {r.requirement_id for r in doc.requirements}
        from_req = [
            any(s.split(".")[0] in req_ids for s in evidence_source_ids(doc, e))
            and not any(s in {b.block_id for b in doc.blocks} for s in evidence_source_ids(doc, e))
            for e in d["evidence"]
        ]
        qtype = "요구사항" if all(from_req) else "일반"
        note = DRAFT_NOTE
    else:
        qtype = None
        note = (
            f"{DRAFT_NOTE}. 부재 확인 — 검색어(공백 제거 후 원문 단위 대조): "
            f"{', '.join(d['search_terms'])}. 목차: {d['toc']}"
        )
    q = EvalQuestionV2(
        question_id=d["question_id"],
        question=d["question"],
        answer=d["answer"],
        evidence=d["evidence"],
        doc_id=d["doc_id"],
        page=None,
        type=qtype,
        tags=d["tags"],
        note=note,
    )
    new.append(with_gold(q, chunks))

written = write_questions_v2(DEFAULT_QA_V2, [*existing, *new])
print(f"{DEFAULT_QA_V2}: {written}문항")
