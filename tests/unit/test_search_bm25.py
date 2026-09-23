"""BM25 색인·검색 단위 테스트 (이슈 #13). 손으로 만든 청크로 돈다.

rank_bm25를 감싼 층이 하는 일 — 토큰화 갈아끼우기, k1·b 전달, 점수와 Chunk
다시 잇기, 동점 순서 고정 — 을 확인한다. 점수식 자체는 라이브러리 몫이라
여기서 값을 검산하지 않고 순위가 뜻대로 나오는지만 본다.
"""

from __future__ import annotations

from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.search.bm25 import DEFAULT_B, DEFAULT_K1, Bm25Index
from rfp_pm_agent.search.tokenize import tokenize_bigram, tokenize_whitespace


def _chunk(chunk_id: str, text: str, doc_id: str = "doc_a") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        method="block",
        source_ids=[chunk_id],
        text=text,
    )


def test_질문_토큰을_더_많이_담은_청크가_위로_온다():
    chunks = [
        _chunk("c1", "사업 개요"),
        _chunk("c2", "사업 기간 착수"),
        _chunk("c3", "착수 보고"),
    ]
    index = Bm25Index(chunks, tokenize_whitespace)
    hits = index.search("사업 기간", top_k=3)
    assert hits[0].chunk_id == "c2"


def test_흔한_토큰보다_드문_토큰이_순위를_가른다():
    # "사업"은 모든 청크에 있고 "감리"는 하나에만 있다. 감리를 담은 청크가 위여야 한다
    chunks = [
        _chunk("c1", "사업 개요"),
        _chunk("c2", "사업 범위"),
        _chunk("c3", "사업 감리"),
        _chunk("c4", "사업 일정"),
    ]
    index = Bm25Index(chunks, tokenize_whitespace)
    hits = index.search("사업 감리", top_k=4)
    assert hits[0].chunk_id == "c3"


def test_질문_토큰이_어느_청크에도_없으면_결과가_비어_있다():
    chunks = [_chunk("c1", "사업 개요"), _chunk("c2", "착수 보고")]
    index = Bm25Index(chunks, tokenize_whitespace)
    assert index.search("존재하지않는단어", top_k=5) == []


def test_토큰이_없는_질문과_빈_색인에서_예외를_내지_않는다():
    index = Bm25Index([_chunk("c1", "사업 개요")], tokenize_bigram)
    assert index.search("   ", top_k=5) == []
    assert Bm25Index([], tokenize_bigram).search("사업", top_k=5) == []


def test_동점일_때_순서가_실행마다_같다():
    # 같은 text를 담은 청크 셋. chunk_id 오름차순으로 고정된다
    chunks = [_chunk("c3", "사업 기간"), _chunk("c1", "사업 기간"), _chunk("c2", "사업 기간")]
    index = Bm25Index(chunks, tokenize_whitespace)
    ranked = [hit.chunk_id for hit in index.search("사업 기간", top_k=3)]
    assert ranked == ["c1", "c2", "c3"]
    assert ranked == [hit.chunk_id for hit in index.search("사업 기간", top_k=3)]


def test_top_k보다_많은_청크가_맞아도_top_k개만_돌려준다():
    chunks = [_chunk(f"c{i}", "사업 기간") for i in range(1, 11)]
    index = Bm25Index(chunks, tokenize_whitespace)
    assert len(index.search("사업 기간", top_k=5)) == 5


def test_결과에_doc_id와_text가_그대로_실려_온다():
    # 정답 판정이 SearchHit의 doc_id·text만 보므로 이 연결이 끊기면 점수가 어긋난다
    chunks = [_chunk("c1", "사업 기간 3개월", doc_id="doc_b")]
    hit = Bm25Index(chunks, tokenize_whitespace).search("사업", top_k=1)[0]
    assert (hit.doc_id, hit.text) == ("doc_b", "사업 기간 3개월")


def test_모든_청크에_있는_토큰으로_맞은_청크도_결과에_남는다():
    # BM25Okapi는 절반 넘는 청크에 나오는 토큰의 IDF를 epsilon × 평균 IDF로
    # 바꾸는데, 모든 토큰이 그런 경우 이 값이 음수가 되어 점수도 음수가 된다.
    # 점수 부호로 걸러내면 실제로 일치한 청크가 빠져 hit@k가 낮게 나온다
    chunks = [_chunk("c1", "사업 기간"), _chunk("c2", "사업 기간")]
    hits = Bm25Index(chunks, tokenize_whitespace).search("사업", top_k=2)
    assert [hit.chunk_id for hit in hits] == ["c1", "c2"]


def test_bigram은_띄어쓰기가_깨진_청크도_찾는다():
    chunks = [
        _chunk("c1", "□사업기간:계약일로부터3개월이내"),
        _chunk("c2", "□입찰방식:제한경쟁"),
    ]
    assert Bm25Index(chunks, tokenize_whitespace).search("사업 기간", top_k=2) == []
    hits = Bm25Index(chunks, tokenize_bigram).search("사업 기간", top_k=2)
    assert hits[0].chunk_id == "c1"


def test_k1과_b를_인자로_받아_색인에_전달한다():
    index = Bm25Index([_chunk("c1", "사업 기간")], tokenize_whitespace, k1=1.2, b=0.5)
    assert (index.k1, index.b) == (1.2, 0.5)
    default = Bm25Index([_chunk("c1", "사업 기간")], tokenize_whitespace)
    assert (default.k1, default.b) == (DEFAULT_K1, DEFAULT_B)
