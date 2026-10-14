"""근거 답변 단위 테스트 (이슈 #19).

규칙은 docs/data-design.md 5절 "근거 답변 answer_question"과 2절 "출처 표기 규칙"(ADR-0006 결정 2).

`format_citation`·`build_prompt`와 `answer_question`의 LLM 호출 이후는 소유자가 구현한다(학습 모드).
그 테스트는 `OWNER_TODO`(xfail strict)로 표시했다 — 구현하면 XPASS가 실패로 잡히므로 표시를 지운다.
표시가 없는 테스트(거절 판정, 인용 번호 파싱, 프롬프트 파일 읽기, 검색 0건·예외)는 지금 통과한다.

검색은 `grounded_answer.search_documents`를 바꿔 끼워 정해 둔 결과나 예외를 낸다. LLM은 `ScriptedLLMClient`다.
"""

from __future__ import annotations

from typing import Any

import pytest

from rfp_pm_agent.schemas.answer import GroundedAnswer, LLMUsage
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search import grounded_answer
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.grounded_answer import (
    PROMPT_VERSION,
    REFUSAL_TEXT,
    answer_question,
    build_prompt,
    extract_citation_numbers,
    format_citation,
    is_refusal,
    load_system_prompt,
)
from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K
from rfp_pm_agent.tools.search_documents import SearchDeps
from tests.fakes.fake_embedding import FakeEmbeddingClient
from tests.fakes.fake_llm import ScriptedLLMClient
from tests.fakes.fake_reranker import FakeRerankerClient
from tests.fakes.fake_search_client import RecordingSearchClient

OWNER_TODO = pytest.mark.xfail(
    strict=True, raises=NotImplementedError, reason="소유자 구현 (#19 학습 모드)"
)

QUERY = "충전기는 몇 대 설치하나요?"


def _hit(rank: int = 1, **overrides: Any) -> DocumentHit:
    """기본은 PDF 요구사항 청크(인쇄 쪽번호 23, PDF 페이지 14 = 뷰어 15쪽)."""
    fields: dict[str, Any] = {
        "rank": rank,
        "chunk_id": f"chunk-{rank}",
        "doc_id": "docA",
        "bid_title": "A 사업",
        "format": "pdf",
        "requirement_id": "SFR-001",
        "page": 14,
        "printed_page": 23,
        "block_type": None,
        "text": f"청크 본문 {rank}",
        "score": 1.0 - rank * 0.1,
    }
    fields.update(overrides)
    return DocumentHit(**fields)


def _hwp(rank: int, requirement_id: str | None) -> DocumentHit:
    return _hit(
        rank,
        doc_id="docB",
        bid_title="B 사업",
        format="hwp",
        requirement_id=requirement_id,
        page=None,
        printed_page=None,
        block_type=None if requirement_id else "paragraph",
    )


def _hits() -> list[DocumentHit]:
    """출처 5가지가 섞인 검색 결과 5건. 3·4번은 출처 문자열이 같다(`[B 사업]`)."""
    return [
        _hit(1),  # [A 사업 p.23 · SFR-001]
        _hit(2, requirement_id=None, printed_page=None, block_type="table"),  # [A 사업 PDF 15쪽]
        _hwp(3, None),  # [B 사업]
        _hwp(4, None),  # [B 사업]
        _hwp(5, "DAR-002"),  # [B 사업 · DAR-002]
    ]


def _deps() -> SearchDeps:
    return SearchDeps(
        search_client=RecordingSearchClient(),
        embedder=FakeEmbeddingClient(),
        reranker=FakeRerankerClient(),
    )


def _patch_search(
    monkeypatch: pytest.MonkeyPatch, result: list[DocumentHit] | Exception
) -> list[dict[str, Any]]:
    """`search_documents`를 정해 둔 결과(또는 예외)로 바꾸고, 받은 인자 기록을 돌려준다."""
    calls: list[dict[str, Any]] = []

    def fake_search(
        query: str,
        top_k: int = DEFAULT_TOP_K,
        doc_ids: list[str] | None = None,
        *,
        deps: SearchDeps,
    ) -> list[DocumentHit]:
        calls.append({"query": query, "top_k": top_k, "doc_ids": doc_ids, "deps": deps})
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(grounded_answer, "search_documents", fake_search)
    return calls


# --- is_refusal (Claude 구현) ---


@pytest.mark.parametrize(
    "text",
    [
        "문서에서 찾을 수 없음",
        "  문서에서 찾을 수 없음\n",
        "문서에서 찾을 수 없음.",
        "문서에서 찾을 수 없음。",
        "\t문서에서 찾을 수 없음. \n",
    ],
)
def test_is_refusal_accepts_surrounding_whitespace_and_one_trailing_period(text: str) -> None:
    assert is_refusal(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "문서에서 찾을 수 없음. 다만 사업 기간은 3개월이다 [1].",
        "문서에서 찾을 수 없음 [1]",
        "문서에서 찾을 수 없음..",
        "문서에서 찾을 수 없음.。",
        "문서에서 찾을 수 없음 .",
        "문서에서 찾을 수 없음!",
        "문서에서 찾을 수 없습니다.",
        "충전기 수는 문서에서 찾을 수 없음.",
        "",
    ],
)
def test_is_refusal_rejects_extra_text_or_other_endings(text: str) -> None:
    assert is_refusal(text) is False


# --- extract_citation_numbers (Claude 구현) ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("충전기는 10대다 [1].", [1]),
        ("10대다 [1][3].", [1, 3]),
        ("10대다 [1, 3].", [1, 3]),
        ("10대다 [1,3] 그리고 [ 2 , 4 ].", [1, 3, 2, 4]),
        ("먼저 [3], 다음 [1], 다시 [3].", [3, 1]),
        ("범위 밖도 그대로 [15] [0].", [15, 0]),
        ("두 자리 [10][12].", [10, 12]),
        ("인용이 없다.", []),
        ("", []),
    ],
)
def test_extract_citation_numbers_reads_supported_forms_in_first_seen_order(
    text: str, expected: list[int]
) -> None:
    assert extract_citation_numbers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "범위 [1-3]",
        "물결 [1~3]",
        "소괄호 (1)",
        "전각 괄호 ［1］",
        "전각 숫자 [１]",
        "요구사항 ID [SFR-001]",
        "출처 문자열 [A 사업 p.23]",
        "글자 섞임 [1번]",
        "쉼표로 끝남 [1,]",
        "빈 괄호 []",
    ],
)
def test_extract_citation_numbers_ignores_unsupported_forms(text: str) -> None:
    assert extract_citation_numbers(text) == []


# --- 프롬프트 파일 (Claude 구현: 읽기만. 내용은 소유자가 쓴다) ---


def test_load_system_prompt_reads_the_versioned_prompt_file() -> None:
    assert PROMPT_VERSION == "grounded_answer_v1"
    assert load_system_prompt().strip() != ""


# --- format_citation (소유자 구현) ---


def test_format_citation_pdf_requirement_with_printed_page() -> None:
    assert format_citation(_hit()) == "[A 사업 p.23 · SFR-001]"


def test_format_citation_pdf_requirement_without_printed_page_uses_pdf_page_plus_one() -> None:
    assert format_citation(_hit(printed_page=None)) == "[A 사업 PDF 15쪽 · SFR-001]"


def test_format_citation_pdf_block() -> None:
    hit = _hit(requirement_id=None, printed_page=None, block_type="paragraph")
    assert format_citation(hit) == "[A 사업 PDF 15쪽]"


@pytest.mark.parametrize("fmt", ["hwp", "hwpx"])
def test_format_citation_hwp_requirement(fmt: str) -> None:
    hit = _hit(format=fmt, page=None, printed_page=None)
    assert format_citation(hit) == "[A 사업 · SFR-001]"


@pytest.mark.parametrize("fmt", ["hwp", "hwpx"])
def test_format_citation_hwp_block(fmt: str) -> None:
    hit = _hit(format=fmt, requirement_id=None, page=None, printed_page=None, block_type="table")
    assert format_citation(hit) == "[A 사업]"


def test_format_citation_page_zero_is_pdf_page_one() -> None:
    """page는 0부터 센다. 0을 "없음"으로 읽지 않는다."""
    assert format_citation(_hit(page=0, printed_page=None)) == "[A 사업 PDF 1쪽 · SFR-001]"
    block = _hit(page=0, requirement_id=None, printed_page=None, block_type="paragraph")
    assert format_citation(block) == "[A 사업 PDF 1쪽]"


def test_format_citation_empty_requirement_id_is_treated_as_missing() -> None:
    pdf = _hit(requirement_id="", printed_page=None, block_type="paragraph")
    assert format_citation(pdf) == "[A 사업 PDF 15쪽]"
    hwp = _hit(format="hwp", requirement_id="", page=None, printed_page=None, block_type="table")
    assert format_citation(hwp) == "[A 사업]"


# --- build_prompt (소유자 구현) ---


@OWNER_TODO
def test_build_prompt_is_system_from_prompt_file_then_one_user_message() -> None:
    messages = build_prompt(QUERY, _hits())

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == load_system_prompt()


@OWNER_TODO
def test_build_prompt_numbers_chunks_1_to_n_in_hit_order() -> None:
    hits = _hits()

    user = build_prompt(QUERY, hits)[1].content

    position = -1
    for number, hit in enumerate(hits, start=1):
        marker_at = user.index(f"[{number}]", position + 1)
        text_at = user.index(hit.text, marker_at)
        assert position < marker_at < text_at
        position = text_at
    assert f"[{len(hits) + 1}]" not in user


@OWNER_TODO
def test_build_prompt_includes_the_question() -> None:
    assert QUERY in build_prompt(QUERY, _hits())[1].content


# --- answer_question: 검색 0건·예외 (Claude 구현) ---


def test_answer_question_passes_query_top_k_and_deps_to_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_search(monkeypatch, [])
    deps = _deps()

    answer_question(QUERY, deps=deps, llm=ScriptedLLMClient([]), top_k=7)

    assert calls == [{"query": QUERY, "top_k": 7, "doc_ids": None, "deps": deps}]


def test_answer_question_default_top_k_is_10(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_search(monkeypatch, [])

    answer_question(QUERY, deps=_deps(), llm=ScriptedLLMClient([]))

    assert calls[0]["top_k"] == 10


def test_answer_question_refuses_without_calling_llm_when_no_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(monkeypatch, [])
    llm = ScriptedLLMClient([])

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert llm.calls == []
    assert result == GroundedAnswer(
        answer=REFUSAL_TEXT,
        refused=True,
        citations=[],
        citation_map={},
        cited_chunk_ids=[],
        invalid_citations=[],
        hit_count=0,
        llm_usage=None,
    )


@pytest.mark.parametrize(
    "error",
    [SearchUnavailableError("tei-rerank", "connection refused"), ValueError("query가 비어 있다")],
)
def test_answer_question_lets_search_errors_through_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    _patch_search(monkeypatch, error)
    llm = ScriptedLLMClient([])

    with pytest.raises(type(error)) as raised:
        answer_question(QUERY, deps=_deps(), llm=llm)

    assert raised.value is error
    assert llm.calls == []


# --- answer_question: LLM 호출 이후 (소유자 구현) ---


@OWNER_TODO
def test_answer_question_calls_llm_once_with_build_prompt_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = _hits()
    _patch_search(monkeypatch, hits)
    llm = ScriptedLLMClient(["충전기는 10대다 [1]."])

    answer_question(QUERY, deps=_deps(), llm=llm)

    assert len(llm.calls) == 1
    assert llm.calls[0][0] == build_prompt(QUERY, hits)


@OWNER_TODO
def test_answer_question_turns_cited_numbers_into_citations_in_first_seen_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(monkeypatch, _hits())
    llm = ScriptedLLMClient(
        ["  표에 10대로 적혀 있다 [2]. 요구사항도 같다 [1][2]. 납품 기한은 30일이다 [5, 1].\n"],
        input_tokens=1200,
        output_tokens=80,
        reasoning_tokens=30,
    )

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert result == GroundedAnswer(
        answer="표에 10대로 적혀 있다 [2]. 요구사항도 같다 [1][2]. 납품 기한은 30일이다 [5, 1].",
        refused=False,
        citations=["[A 사업 PDF 15쪽]", "[A 사업 p.23 · SFR-001]", "[B 사업 · DAR-002]"],
        citation_map={
            2: "[A 사업 PDF 15쪽]",
            1: "[A 사업 p.23 · SFR-001]",
            5: "[B 사업 · DAR-002]",
        },
        cited_chunk_ids=["chunk-2", "chunk-1", "chunk-5"],
        invalid_citations=[],
        hit_count=5,
        llm_usage=LLMUsage(input_tokens=1200, output_tokens=80, reasoning_tokens=30),
    )


@OWNER_TODO
def test_answer_question_keeps_one_citation_for_chunks_with_the_same_citation_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """같은 HWP 문서의 블록 청크 두 개는 출처가 둘 다 `[B 사업]`이다."""
    _patch_search(monkeypatch, _hits())
    llm = ScriptedLLMClient(["두 문단에 나뉘어 있다 [3][4]."])

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert result.citations == ["[B 사업]"]
    assert result.citation_map == {3: "[B 사업]", 4: "[B 사업]"}
    assert result.cited_chunk_ids == ["chunk-3", "chunk-4"]


@OWNER_TODO
@pytest.mark.parametrize(
    "reply", [REFUSAL_TEXT, f"  {REFUSAL_TEXT}\n", f"{REFUSAL_TEXT}.", f"{REFUSAL_TEXT}。"]
)
def test_answer_question_refusal_reply_has_no_citations(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    _patch_search(monkeypatch, _hits())
    llm = ScriptedLLMClient([reply], input_tokens=1200, output_tokens=40, reasoning_tokens=30)

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert result == GroundedAnswer(
        answer=REFUSAL_TEXT,
        refused=True,
        citations=[],
        citation_map={},
        cited_chunk_ids=[],
        invalid_citations=[],
        hit_count=5,
        llm_usage=LLMUsage(input_tokens=1200, output_tokens=40, reasoning_tokens=30),
    )


@OWNER_TODO
def test_answer_question_records_out_of_range_numbers_and_does_not_cite_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(monkeypatch, _hits())
    llm = ScriptedLLMClient(["10대다 [1]. 예산은 5억이다 [15]. 기간은 1년이다 [0][15][6]."])

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert result.refused is False
    assert result.citations == ["[A 사업 p.23 · SFR-001]"]
    assert result.citation_map == {1: "[A 사업 p.23 · SFR-001]"}
    assert result.cited_chunk_ids == ["chunk-1"]
    assert result.invalid_citations == [15, 0, 6]


@OWNER_TODO
def test_answer_question_answer_without_citation_numbers_is_not_turned_into_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """출처가 없는 답변은 그대로 돌려준다. DoD ①은 실행 결과에서 센다."""
    _patch_search(monkeypatch, _hits())
    llm = ScriptedLLMClient(["충전기는 10대다."])

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert result.refused is False
    assert result.answer == "충전기는 10대다."
    assert result.citations == []
    assert result.citation_map == {}
    assert result.cited_chunk_ids == []
    assert result.invalid_citations == []


@OWNER_TODO
def test_answer_question_refusal_phrase_followed_by_explanation_is_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """거절 문구 뒤에 설명이 붙으면 거절이 아니다(is_refusal). 인용 번호는 출처로 바꾼다."""
    _patch_search(monkeypatch, _hits())
    llm = ScriptedLLMClient([f"{REFUSAL_TEXT}. 다만 사업 기간은 3개월이다 [1]."])

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert result.refused is False
    assert result.answer == f"{REFUSAL_TEXT}. 다만 사업 기간은 3개월이다 [1]."
    assert result.citation_map == {1: "[A 사업 p.23 · SFR-001]"}


@OWNER_TODO
def test_answer_question_citation_map_keeps_first_seen_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(monkeypatch, _hits())
    llm = ScriptedLLMClient(["나중 번호가 먼저 나온다 [5]. 그다음 [2][5]."])

    result = answer_question(QUERY, deps=_deps(), llm=llm)

    assert list(result.citation_map) == [5, 2]
