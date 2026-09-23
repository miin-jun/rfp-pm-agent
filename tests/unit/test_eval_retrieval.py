"""정답 판정과 hit@k 집계 단위 테스트 (이슈 #13). 손으로 만든 청크로 돈다.

판정 규칙: doc_id가 같고, 공백을 모두 지운 뒤 evidence가 청크 text에 들어 있으면
정답. 이 규칙이 느슨해지면(다른 문서 청크를 정답으로 세거나) 점수가 실제보다
높게 나오므로 양쪽 경계를 모두 고정한다.
"""

from __future__ import annotations

from rfp_pm_agent.eval.retrieval import (
    ceiling,
    ceiling_only,
    hit_at_k,
    is_correct,
    normalize,
    score_run,
)
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import EvalQuestion, SearchHit

EVIDENCE = "□사업기간: 계약일로부터3개월이내(안정화1개월포함)"


def _chunk(chunk_id: str, text: str, doc_id: str = "doc_a") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        method="block",
        source_ids=[chunk_id],
        text=text,
    )


def _question(
    question_id: str = "q001",
    doc_id: str = "doc_a",
    evidence: str = EVIDENCE,
    qtype: str = "일반",
) -> EvalQuestion:
    return EvalQuestion(
        question_id=question_id,
        question="이 사업의 수행 기간은 얼마나 되나요?",
        answer="3개월",
        evidence=evidence,
        doc_id=doc_id,
        page=1,
        type=qtype,  # type: ignore[arg-type]
    )


def _hit(chunk: Chunk, score: float) -> SearchHit:
    return SearchHit(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, score=score, text=chunk.text)


# --- 정답 판정 ---


def test_다른_문서_청크는_evidence가_들어_있어도_오답():
    chunk = _chunk("c1", f"앞말 {EVIDENCE} 뒷말", doc_id="doc_b")
    assert is_correct(chunk, _question(doc_id="doc_a")) is False


def test_같은_문서에_evidence가_그대로_있으면_정답():
    chunk = _chunk("c1", f"앞말 {EVIDENCE} 뒷말")
    assert is_correct(chunk, _question()) is True


def test_청크_쪽에_공백이_더_있어도_정답():
    chunk = _chunk("c1", "□ 사업기간 :  계약일로부터 3개월 이내(안정화 1개월 포함)")
    assert is_correct(chunk, _question()) is True


def test_evidence_쪽에_공백이_더_있어도_정답():
    chunk = _chunk("c1", "□사업기간:계약일로부터3개월이내(안정화1개월포함)")
    assert is_correct(
        chunk, _question(evidence="□ 사업기간: 계약일로부터 3개월 이내(안정화 1개월 포함)")
    )


def test_줄바꿈과_탭만_다르면_정답():
    chunk = _chunk("c1", "□사업기간:\n계약일로부터3개월\t이내(안정화1개월포함)")
    assert is_correct(chunk, _question()) is True


def test_evidence가_두_청크에_걸쳐_잘리면_어느_쪽도_오답():
    first = _chunk("c1", "□사업기간: 계약일로부터3개월")
    second = _chunk("c2", "이내(안정화1개월포함)")
    question = _question()
    assert is_correct(first, question) is False
    assert is_correct(second, question) is False
    assert ceiling([first, second], question) is False


def test_normalize는_공백만_지우고_다른_문자는_남긴다():
    assert normalize(" □사업 기간:\n3개월\t") == "□사업기간:3개월"


# --- hit@k 계산 ---


def _ranked(correct_rank: int | None, length: int = 5) -> list[SearchHit]:
    """correct_rank번째(1부터)만 정답인 검색 결과를 만든다."""
    hits = []
    for i in range(1, length + 1):
        text = f"앞말 {EVIDENCE} 뒷말" if i == correct_rank else f"상관없는 본문 {i}"
        hits.append(_hit(_chunk(f"c{i}", text), score=float(length - i)))
    return hits


def test_정답이_1위면_hit1_3_5_모두_참():
    hits, question = _ranked(1), _question()
    assert [hit_at_k(hits, question, k) for k in (1, 3, 5)] == [True, True, True]


def test_정답이_3위면_hit1은_거짓_hit3_5는_참():
    hits, question = _ranked(3), _question()
    assert [hit_at_k(hits, question, k) for k in (1, 3, 5)] == [False, True, True]


def test_정답이_6위면_셋_다_거짓():
    hits, question = _ranked(6, length=6), _question()
    assert [hit_at_k(hits, question, k) for k in (1, 3, 5)] == [False, False, False]


def test_검색_결과가_k보다_적어도_예외_없이_계산된다():
    hits, question = _ranked(2, length=2), _question()
    assert hit_at_k(hits, question, 5) is True
    assert hit_at_k([], question, 5) is False


# --- 유형별 집계와 최고 점수 ---


def test_유형별_집계가_요구사항과_일반을_갈라_세고_전체와_합이_맞는다():
    chunks = [_chunk("c1", f"앞말 {EVIDENCE}")]
    questions = [
        _question("q001", qtype="일반"),
        _question("q002", qtype="요구사항"),
        _question("q003", qtype="요구사항"),
    ]

    def search(query: str, top_k: int) -> list[SearchHit]:
        return [_hit(chunks[0], 1.0)]

    scores, missed = score_run(questions, chunks, search)
    assert scores.overall.total == 3
    assert scores.requirement.total == 2
    assert scores.general.total == 1
    assert scores.requirement.hit_at_1 + scores.general.hit_at_1 == scores.overall.hit_at_1
    assert scores.overall.hit_at_1 == 3
    assert missed == {}


def test_정답_청크가_있어도_순위_밖이면_hit5는_거짓이고_최고_점수는_참():
    target = _chunk("c99", f"앞말 {EVIDENCE}")
    noise = [_chunk(f"c{i}", f"상관없는 본문 {i}") for i in range(1, 6)]
    chunks = [*noise, target]
    question = _question()

    def search(query: str, top_k: int) -> list[SearchHit]:
        return [_hit(c, 1.0) for c in noise][:top_k]

    scores, missed = score_run([question], chunks, search)
    assert scores.overall.hit_at_5 == 0
    assert scores.overall.ceiling == 1
    assert scores.missed_question_ids == ["q001"]
    # 틀린 질문의 상위 청크가 failures 파일용으로 함께 나온다
    assert [hit.chunk_id for hit in missed["q001"]] == ["c1", "c2", "c3", "c4", "c5"]


def test_ceiling_only는_검색_없이_최고_점수만_낸다():
    chunks = [_chunk("c1", f"앞말 {EVIDENCE}")]
    questions = [_question("q001", qtype="일반"), _question("q002", doc_id="doc_b")]
    scores = ceiling_only(questions, chunks)
    assert scores.overall.ceiling == 1
    assert scores.overall.hit_at_1 == 0
    assert scores.missed_question_ids == []
