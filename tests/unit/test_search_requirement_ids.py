"""질문 속 요구사항 ID 정확 일치 가산 단위 테스트 (이슈 #75, 방법 A).

`extract_requirement_ids`와 `bm25_search`의 ID 가산 쿼리는 소유자가 구현한다. 이 파일은
구현 전에 #75 "결정 (2026-10-06)"의 측정 전 고정 항목을 먼저 테스트로 묶어 둔 것이다.

고정 항목
- ID 패턴: 영문 2~5자 + 하이픈 + 숫자 2~4자리, 대문자로 정규화
- 질문에 ID가 있으면 ID마다 `requirement_id` 정확 일치 조건을 should로 추가한다.
  가산은 `constant_score`(boost 100)로 한다 — `term`에 boost를 주면 keyword 필드도 BM25로
  점수가 매겨져 100이 아니라 100 × idf가 더해진다(q041 `SFR-013`은 약 +454, 조사 2026-10-06)
- 질문에 ID가 없으면 쿼리는 지금(`{"match": {"text": 질문}}`)과 완전히 같다 — body 전체를
  #75 이전 dict와 == 비교한다(`test_bm25_without_valid_id_body_is_unchanged`)

패턴 경계는 "영문 2~5자", "숫자 2~4자리"를 덩어리 전체에 적용한다고 해석했다 — `ABCDEF-013`의
뒷부분 `BCDEF-013`이나 `SFR-01234`의 앞부분 `SFR-0123`을 ID로 뽑지 않는다. 한국어 조사가
바로 붙은 `SFR-013의`는 ID로 뽑는다(파이썬 `\\b`는 한글도 단어 글자로 봐서 여기서 경계를 못 찾는다).
"""

from __future__ import annotations

from typing import Any

import pytest

from rfp_pm_agent.search.opensearch import bm25_search, extract_requirement_ids
from tests.fakes.fake_search_client import RecordingSearchClient

TEST_ALIAS = "test_alias_for_search"
BOOST = 100
EMPTY_RESPONSE: dict[str, Any] = {"hits": {"hits": []}}

# Red 단계 표시. Green(소유자 구현)이 들어가면 strict라서 XPASS가 실패로 잡힌다 — 그때 지운다
RED = pytest.mark.xfail(strict=True, reason="#75 Green 전 — 소유자가 구현한다")


@pytest.fixture(autouse=True)
def _alias_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_INDEX_ALIAS", TEST_ALIAS)


# --- extract_requirement_ids ---


def test_extracts_id_from_question() -> None:
    question = "2027학년도 고등학교 입학전형 시스템 기능개선 용역의 SFR-013 요구사항은 무엇인가요?"
    assert extract_requirement_ids(question) == ["SFR-013"]


def test_normalizes_to_uppercase() -> None:
    assert extract_requirement_ids("sfr-013 요구사항") == ["SFR-013"]
    assert extract_requirement_ids("Sfr-013 요구사항") == ["SFR-013"]


def test_keeps_question_order_and_drops_duplicates() -> None:
    question = "SER-001과 SFR-013, 그리고 sfr-013의 차이는?"
    assert extract_requirement_ids(question) == ["SER-001", "SFR-013"]


def test_returns_empty_list_without_id() -> None:
    assert extract_requirement_ids("사업 기간은 언제까지인가") == []


@pytest.mark.parametrize(
    "question",
    [
        "SFR-013의 내용",
        "SFR-013요구사항",
        "요구사항SFR-013",
        "(SFR-013)",
        "SFR-013.",
        "'SFR-013'",
    ],
)
def test_extracts_id_next_to_korean_and_punctuation(question: str) -> None:
    assert extract_requirement_ids(question) == ["SFR-013"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("AB-01", ["AB-01"]),  # 영문 2자, 숫자 2자리 (하한)
        ("ABCDE-0001", ["ABCDE-0001"]),  # 영문 5자, 숫자 4자리 (상한)
    ],
)
def test_pattern_bounds_inclusive(question: str, expected: list[str]) -> None:
    assert extract_requirement_ids(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "X-013",  # 영문 1자
        "ABCDEF-013",  # 영문 6자 — 뒤 5자만 떼어 ID로 보지 않는다
        "SFR-0",  # 숫자 1자리
        "SFR-01234",  # 숫자 5자리 — 앞 4자리만 떼어 ID로 보지 않는다
        "SFR013",  # 하이픈 없음
        "2027-2028학년도",  # 숫자-숫자
        "SFR-",
    ],
)
def test_rejects_out_of_pattern(question: str) -> None:
    assert extract_requirement_ids(question) == []


# --- bm25_search: ID가 있을 때의 쿼리 모양 ---


def _bm25_query(question: str) -> dict[str, Any]:
    client = RecordingSearchClient(EMPTY_RESPONSE)
    bm25_search(client, question, top_k=10)
    assert len(client.calls) == 1
    query: dict[str, Any] = client.calls[0]["body"]["query"]
    return query


def _is_text_match(clause: dict[str, Any], question: str) -> bool:
    """축약형 `{"match": {"text": q}}`와 전체형 `{"match": {"text": {"query": q}}}`를 모두 받는다."""
    if set(clause) != {"match"} or set(clause["match"]) != {"text"}:
        return False
    text_query = clause["match"]["text"]
    sent = text_query["query"] if isinstance(text_query, dict) else text_query
    return bool(sent == question)


def _boosted_id(clause: dict[str, Any]) -> str | None:
    """`constant_score`(filter = requirement_id term, boost 100) 조건이면 그 ID를, 아니면 None."""
    if set(clause) != {"constant_score"}:
        return None
    cs = clause["constant_score"]
    if set(cs) != {"filter", "boost"} or cs["boost"] != BOOST:
        return None
    term = cs["filter"]
    if set(term) != {"term"} or set(term["term"]) != {"requirement_id"}:
        return None
    value = term["term"]["requirement_id"]
    if isinstance(value, dict):
        if set(value) != {"value"}:  # 안쪽 boost 등 다른 인자가 붙으면 가산값이 100이 아니게 된다
            return None
        value = value["value"]
    return str(value)


def _split_bool(query: dict[str, Any], question: str) -> tuple[list[str], int]:
    """bool 쿼리에서 (should에 든 가산 ID 목록, text match 조건 수)를 돌려준다.

    text match는 must에 있어도 should에 있어도 된다. filter·must_not은 결과 집합을 바꾸므로
    허용하지 않는다(결정: "should로 추가").
    """
    assert set(query) == {"bool"}, f"ID가 있으면 bool 쿼리여야 한다: {query!r}"
    bool_query = query["bool"]
    assert not set(bool_query) - {"should", "must"}, f"허용하지 않는 bool 키: {bool_query!r}"
    should = bool_query.get("should", [])
    must = bool_query.get("must", [])
    should = should if isinstance(should, list) else [should]
    must = must if isinstance(must, list) else [must]

    boosted = [cid for c in should if (cid := _boosted_id(c)) is not None]
    others = [c for c in should if _boosted_id(c) is None] + must
    matches = [c for c in others if _is_text_match(c, question)]
    assert len(matches) == len(others), f"가산 조건과 text match 말고 다른 조건이 있다: {others!r}"
    return boosted, len(matches)


@RED
def test_bm25_with_id_adds_constant_score_should() -> None:
    question = "입학전형 시스템의 SFR-013 요구사항은 어떤 업무를 지원하나요?"
    boosted, matches = _split_bool(_bm25_query(question), question)
    assert boosted == ["SFR-013"]
    assert matches == 1


@RED
def test_bm25_with_id_keeps_original_question_in_match() -> None:
    """text match에는 질문을 그대로 보낸다 — ID를 빼거나 대문자로 바꾸지 않는다."""
    question = "sfr-013 요구사항은?"
    boosted, matches = _split_bool(_bm25_query(question), question)
    assert matches == 1
    assert boosted == ["SFR-013"]


@RED
def test_bm25_with_several_ids_adds_one_clause_each_in_order() -> None:
    question = "SER-001과 SFR-013, sfr-013의 차이는?"
    boosted, matches = _split_bool(_bm25_query(question), question)
    assert boosted == ["SER-001", "SFR-013"]
    assert matches == 1


def test_bm25_with_id_keeps_size_and_source() -> None:
    client = RecordingSearchClient(EMPTY_RESPONSE)
    bm25_search(client, "SFR-013 요구사항", top_k=7)
    body = client.calls[0]["body"]
    assert client.calls[0]["index"] == TEST_ALIAS
    assert body["size"] == 7
    assert "embedding" in body["_source"]["excludes"]


@pytest.mark.parametrize("question", ["ABCDEF-013 항목", "2027-2028학년도 일정", "SFR013 요구사항"])
def test_bm25_without_valid_id_is_plain_match(question: str) -> None:
    """ID처럼 보여도 패턴에 맞지 않으면 쿼리는 지금과 같다."""
    query = _bm25_query(question)
    assert _is_text_match(query, question), f"plain match여야 한다: {query!r}"


@pytest.mark.parametrize(
    "question",
    ["사업 기간은 언제까지인가", "ABCDEF-013 항목", "2027-2028학년도 일정", "SFR013 요구사항"],
)
def test_bm25_without_valid_id_body_is_unchanged(question: str) -> None:
    """ID가 없으면 body 전체가 #75 이전과 같다 — 키 추가·쿼리 형태 변경도 잡는다."""
    client = RecordingSearchClient(EMPTY_RESPONSE)
    bm25_search(client, question, top_k=10)
    assert client.calls[0]["body"] == {
        "size": 10,
        "_source": {"excludes": ["embedding"]},
        "query": {"match": {"text": question}},
    }
