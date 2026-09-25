"""평가 세트 v2 스키마·evidence 목록 판정·검증 스크립트 단위 테스트 (이슈 #15).

손으로 만든 문서·청크로 돈다. qa_v1 읽기 테스트만 레포의 data/eval/qa_v1.jsonl을
읽는다 (git에 포함된 파일이고 네트워크를 쓰지 않는다).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rfp_pm_agent.eval.qa_v2 import (
    PARAPHRASE_MAX_OVERLAP,
    PARAPHRASE_NOTE,
    V1_NOTE,
    build_from_v1,
    check,
    evidence_source_ids,
    is_table_question,
    load_questions_v2,
    main,
    overlap_ratio,
    shared_bigrams,
    with_gold,
)
from rfp_pm_agent.eval.retrieval import gold_chunk_ids, reachable
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Block, Document, Requirement
from rfp_pm_agent.schemas.eval import EvalQuestionV2

QA_V1 = Path(__file__).resolve().parents[2] / "data" / "eval" / "qa_v1.jsonl"


def _q(**overrides: Any) -> EvalQuestionV2:
    fields: dict[str, Any] = {
        "question_id": "q900",
        "question": "질문",
        "answer": "답",
        "evidence": ["사업기간 3개월"],
        "doc_id": "doc_a",
        "type": "일반",
    }
    fields.update(overrides)
    return EvalQuestionV2(**fields)


def _chunk(source_id: str, text: str, doc_id: str = "doc_a") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:block_requirement:{source_id}",
        doc_id=doc_id,
        method="block_requirement",
        source_ids=[source_id],
        text=text,
    )


def _doc(blocks: list[Block], requirements: list[Requirement] | None = None) -> Document:
    reqs = requirements or []
    return Document(
        doc_id="doc_a",
        source_file="a.hwp",
        bid_title="테스트 사업",
        format="hwp",
        parse_status="parsed",
        has_requirements=True,
        requirement_count=len(reqs),
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=blocks,
        requirements=reqs,
    )


def _paragraph(block_id: str, text: str) -> Block:
    return Block(block_id=block_id, type="paragraph", text=text, source_order=0)


def _table(block_id: str, rows: list[list[str | None]]) -> Block:
    text = "\n".join(" | ".join(c or "" for c in row) for row in rows)
    return Block(block_id=block_id, type="table", text=text, table=rows, source_order=0)


# --- 스키마 ---


def test_qa_v1_30줄이_모두_v2로_읽히고_evidence는_길이_1_목록이다():
    v1_rows = [json.loads(line) for line in QA_V1.read_text(encoding="utf-8").splitlines()]
    questions = load_questions_v2(QA_V1)

    assert len(questions) == 30
    for row, q in zip(v1_rows, questions, strict=True):
        assert q.question_id == row["question_id"]
        assert q.evidence == [row["evidence"]]
        assert q.answer == row["answer"]
        assert q.tags == []
        assert q.gold_chunk_ids == {}
        assert q.verified_by is None


def test_답_없음_문항은_태그_빈_evidence_answer_None이_모두_맞아야_한다():
    q = _q(answer=None, evidence=[], tags=["no_answer"], type=None, note="목차·동의어 확인")
    assert q.evidence == []


@pytest.mark.parametrize(
    ("answer", "evidence", "tags"),
    [
        (None, [], []),  # 태그 없음
        ("답", [], ["no_answer"]),  # answer 있음
        (None, ["원문"], ["no_answer"]),  # evidence 있음
        ("답", ["원문"], ["no_answer"]),  # 태그만 있음
        (None, ["원문"], []),  # answer만 없음
        ("답", [], []),  # evidence만 없음
    ],
)
def test_답_없음_표시가_하나라도_어긋나면_거부한다(
    answer: str | None, evidence: list[str], tags: list[str]
) -> None:
    with pytest.raises(ValidationError, match="답 없음 표시"):
        _q(answer=answer, evidence=evidence, tags=tags)


def test_gold_chunk_ids_길이가_evidence_개수와_다르면_거부한다():
    with pytest.raises(ValidationError, match="gold_chunk_ids"):
        _q(evidence=["가", "나"], gold_chunk_ids={"block_requirement": [["c1"]]})


def test_false_premise_태그는_답_있는_문항에_붙일_수_있다():
    q = _q(tags=["false_premise"])
    assert q.tags == ["false_premise"]


def test_알_수_없는_태그는_거부한다():
    with pytest.raises(ValidationError):
        _q(tags=["semantic"])


# --- evidence 목록 판정 (retrieval.py) ---


def test_gold_chunk_ids는_evidence_순서대로_담은_청크를_모두_모은다():
    chunks = [
        _chunk("b1", "사업기간 3개월"),
        _chunk("b2", "예산 1억원"),
        _chunk("b3", "사업 기간  3 개월 (반복)"),  # 공백만 다른 반복
        _chunk("b1", "예산 1억원", doc_id="doc_b"),  # 다른 문서
    ]
    q = _q(evidence=["예산 1억원", "사업기간 3개월"])

    assert gold_chunk_ids(chunks, q) == [
        ["doc_a:block_requirement:b2"],
        ["doc_a:block_requirement:b1", "doc_a:block_requirement:b3"],
    ]


def test_evidence_하나라도_어느_청크에도_없으면_도달_불가():
    chunks = [_chunk("b1", "사업기간 3개월")]
    assert reachable(chunks, _q(evidence=["사업기간 3개월"])) is True
    assert reachable(chunks, _q(evidence=["사업기간 3개월", "예산 1억원"])) is False


def test_답_없음_문항은_최고_점수_판정을_거부한다():
    q = _q(answer=None, evidence=[], tags=["no_answer"], note="확인")
    with pytest.raises(ValueError, match="답 없음"):
        reachable([_chunk("b1", "아무 내용")], q)


# --- 원문 대조·table 판정 (qa_v2.py) ---


def test_원문_대조는_블록과_요구사항_필드를_모두_본다():
    req = Requirement(
        requirement_id="SFR-001",
        prefix="SFR",
        fields={"detail": "응답 3초 이내"},
        raw_fields={"세부내용": "응답 3초 이내"},
        text="SFR-001 성능",
        source_order=1,
    )
    doc = _doc([_paragraph("b1", "사업기간 3개월")], [req])

    assert evidence_source_ids(doc, "사업기간3개월") == ["b1"]
    assert "SFR-001.세부내용" in evidence_source_ids(doc, "응답 3초 이내")
    assert evidence_source_ids(doc, "없는 문장") == []


def test_두_블록을_이어_붙인_evidence는_원문에_없다():
    doc = _doc([_paragraph("b1", "사업기간 3개월"), _paragraph("b2", "예산 1억원")])
    assert evidence_source_ids(doc, "사업기간 3개월 예산 1억원") == []


def test_정답_청크가_모두_2x2_이상_표일_때만_table_문항이다():
    grid = _table("b1", [["구분", "배점"], ["기술", "90"]])
    box = _table("b2", [["기술 90"]])  # 1×1 글상자
    para = _paragraph("b3", "기술 90점")
    doc = _doc([grid, box, para])
    chunks = [_chunk(b.block_id, b.text) for b in doc.blocks]

    assert is_table_question(_q(evidence=["기술 | 90"]), doc, chunks) is True
    # 표와 1×1 글상자 양쪽에 있으면 표를 못 읽어도 맞힐 수 있다
    assert is_table_question(_q(evidence=["기술"]), doc, chunks) is False
    assert is_table_question(_q(evidence=["기술 90점"]), doc, chunks) is False


def test_build_from_v1은_파생값과_검수_기록_없음만_붙이고_원문은_그대로_둔다():
    doc = _doc([_table("b1", [["구분", "배점"], ["기술", "90"]]), _paragraph("b2", "예산 1억")])
    chunks = {"block_requirement": [_chunk(b.block_id, b.text) for b in doc.blocks]}
    v1 = [
        # 겹침: 기술 1/4 = 0.25 > 0.10
        _q(
            question_id="q001",
            question="기술 점수는?",
            evidence=["기술 | 90"],
            verified_by="누군가",
        ),
        # 겹침 0/4
        _q(question_id="q002", question="돈은 얼마?", evidence=["예산 1억"]),
    ]

    out = build_from_v1(v1, {"doc_a": doc}, chunks)

    assert [q.tags for q in out] == [["table"], ["paraphrase"]]
    assert all(q.verified_by is None and q.verified_at is None for q in out)
    assert [q.note for q in out] == [V1_NOTE, f"{V1_NOTE}; {PARAPHRASE_NOTE}"]
    assert [(q.question, q.answer, q.evidence) for q in out] == [
        (q.question, q.answer, q.evidence) for q in v1
    ]
    assert out[0].gold_chunk_ids == {"block_requirement": [["doc_a:block_requirement:b1"]]}


def test_check는_원문에_없는_evidence와_낡은_gold를_오류로_낸다():
    doc = _doc([_paragraph("b1", "사업기간 3개월")])
    chunks = {"block_requirement": [_chunk("b1", "사업기간 3개월")]}
    good = with_gold(_q(question_id="q001"), chunks)
    stale = _q(question_id="q002")  # gold_chunk_ids 비어 있음
    missing = with_gold(_q(question_id="q003", evidence=["없는 문장"]), chunks)

    report = check([good, stale, missing], {"doc_a": doc}, chunks)

    joined = "\n".join(report.errors)
    assert "q001" not in joined
    assert "q002: 기록된 gold_chunk_ids" in joined
    assert "q003: evidence[0]가 doc_a 원문에 없다" in joined
    assert "q003: evidence[0]를 담은 block_requirement 청크가 없다" in joined
    assert report.ceilings["block_requirement"] == (2, 3)


def test_check는_답_없음을_최고_점수_분모에서_빼고_note가_없으면_오류다():
    doc = _doc([_paragraph("b1", "사업기간 3개월")])
    chunks = {"block_requirement": [_chunk("b1", "사업기간 3개월")]}
    answered = with_gold(_q(question_id="q001"), chunks)
    no_note = with_gold(
        _q(question_id="q002", answer=None, evidence=[], tags=["no_answer"], type=None), chunks
    )

    report = check([answered, no_note], {"doc_a": doc}, chunks)

    assert report.no_answer == 1
    assert report.ceilings["block_requirement"] == (1, 1)
    assert report.errors == ["q002: 답 없음 문항에 부재 확인 기록(note)이 없다"]


def test_check는_table_태그와_코드_판정이_어긋나면_알린다():
    doc = _doc([_table("b1", [["구분", "배점"], ["기술", "90"]]), _paragraph("b2", "예산 1억")])
    chunks = {"block_requirement": [_chunk(b.block_id, b.text) for b in doc.blocks]}
    untagged_table = with_gold(_q(question_id="q001", evidence=["기술 | 90"]), chunks)
    tagged_para = with_gold(_q(question_id="q002", evidence=["예산 1억"], tags=["table"]), chunks)

    report = check([untagged_table, tagged_para], {"doc_a": doc}, chunks)

    assert report.warnings == ["q001: 정답 청크가 모두 표인데 table 태그가 없다"]
    assert report.errors == ["q002: table 태그가 있지만 정답 청크가 모두 표에서 나오지 않았다"]


def test_build_from_v1은_출력_파일이_이미_있으면_덮어쓰지_않는다(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    for method in ("block", "requirement", "block_requirement"):
        (chunks_dir / f"{method}.jsonl").write_text("", encoding="utf-8")
    out = tmp_path / "qa_v2.jsonl"
    out.write_text("기존 내용\n", encoding="utf-8")

    code = main(["build-from-v1", "--qa", str(out), "--chunks-dir", str(chunks_dir)])

    assert code == 1
    assert out.read_text(encoding="utf-8") == "기존 내용\n"


# --- 겹침률 (paraphrase 기준) ---


def test_겹침률은_기호와_공백을_지운_글자_bigram으로_잰다():
    # 질문 bigram: 사업, 업예, 예산, 산은 → evidence "사업예산1억"과 겹치는 것: 사업, 업예, 예산
    assert shared_bigrams("사업 예산은?", ["□ 사업예산: 1억"]) == ["사업", "업예", "예산"]
    assert overlap_ratio("사업 예산은?", ["□ 사업예산: 1억"]) == 0.75


def test_evidence가_여럿이면_각각_따로_보고_이어_붙인_경계는_세지_않는다():
    # "가나" + "다라"를 이어 붙이면 "나다"가 생기지만 원문 어디에도 없다
    assert shared_bigrams("나다", ["가나", "다라"]) == []
    assert shared_bigrams("가나 다라", ["가나", "다라"]) == ["가나", "다라"]


def test_qa_v1의_paraphrase_판정은_E_보고의_분포와_같다():
    """기준값 0.10은 이 분포를 보고 고정했다. 측정 방식이 바뀌면 여기서 드러난다."""
    below = {
        q.question_id
        for q in load_questions_v2(QA_V1)
        if overlap_ratio(q.question, q.evidence) <= PARAPHRASE_MAX_OVERLAP
    }
    assert below == {
        "q004", "q006", "q007", "q014", "q017", "q018",
        "q020", "q021", "q022", "q025", "q026", "q030",
    }  # fmt: skip


def test_check는_paraphrase_태그인데_기준을_넘으면_실패한다():
    doc = _doc([_paragraph("b1", "사업기간 3개월")])
    chunks = {"block_requirement": [_chunk("b1", "사업기간 3개월")]}
    over = with_gold(
        _q(
            question_id="q001",
            question="사업기간은?",
            evidence=["사업기간 3개월"],
            tags=["paraphrase"],
        ),
        chunks,
    )
    under = with_gold(
        _q(
            question_id="q002",
            question="언제 끝나?",
            evidence=["사업기간 3개월"],
            tags=["paraphrase"],
        ),
        chunks,
    )

    report = check([over, under], {"doc_a": doc}, chunks)

    assert len(report.errors) == 1
    assert report.errors[0].startswith("q001: paraphrase 태그인데 겹침률")
