"""근거 답변 (이슈 #19).

질문 → `search_documents` → LLM 답변 → 출처 붙이기. 규칙은 docs/data-design.md 5절 "근거 답변
answer_question", 거절 판단과 출처 표기는 docs/adr/0006-grounded-answer-refusal-and-citation.md가 기준이다.

- LLM은 프롬프트의 청크 번호(`[1]`~`[n]`)만 인용한다. 번호 → `DocumentHit` → 출처 문자열은 코드가 바꾼다
- `format_citation`은 LLM 호출과 분리돼 있다 — #22 에이전트가 같은 함수를 쓴다
- 검색 예외(`SearchUnavailableError`·`ValueError`)와 LLM 예외는 잡지 않는다

`format_citation`·`build_prompt`와 `answer_question`의 LLM 호출 이후는 소유자가 구현한다(학습 모드).
테스트: tests/unit/test_search_grounded_answer.py
"""

from __future__ import annotations

import re
from importlib import resources

from rfp_pm_agent.clients.llm import ChatMessage, LLMClient
from rfp_pm_agent.schemas.answer import GroundedAnswer, LLMUsage
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K
from rfp_pm_agent.tools.search_documents import SearchDeps, search_documents

# LLM이 근거가 없을 때 쓰는 문구. 판정은 is_refusal(앞뒤 공백과 끝 마침표 하나만 허용)
REFUSAL_TEXT = "문서에서 찾을 수 없음"
# 거절 문구 끝에 하나까지 허용하는 마침표: 반각 "."와 전각 "。"
_REFUSAL_END_MARKS = (".", "。")

# 프롬프트 파일 이름(확장자 제외)이 곧 버전이다. 내용을 바꾸면 파일을 고치지 않고 _v2를 새로 만든다
PROMPT_VERSION = "grounded_answer_v2"

# 대괄호 안에 숫자만 있거나 숫자를 쉼표로 나열한 것: [1], [1, 3]. \d는 전각 숫자도 받아 [0-9]로 적는다
_CITATION_RE = re.compile(r"\[\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\]")


def load_system_prompt() -> str:
    """system 메시지 내용을 프롬프트 파일(`search/prompts/<PROMPT_VERSION>.md`)에서 읽는다."""
    path = resources.files("rfp_pm_agent.search").joinpath("prompts", f"{PROMPT_VERSION}.md")
    return path.read_text("utf-8")


def is_refusal(text: str) -> bool:
    """LLM 답이 거절 문구인지 판정한다.

    앞뒤 공백을 떼고, 끝의 마침표(`.` 또는 `。`) 하나를 뗀 것이 `REFUSAL_TEXT`와 완전히 같을 때만 True.
    마침표 두 개(`..`), 마침표 앞 공백(`없음 .`), 문구 뒤에 설명이 붙은 답(`없음. 다만 ~`)은 거절이 아니다.
    """
    stripped = text.strip()
    if stripped.endswith(_REFUSAL_END_MARKS):
        stripped = stripped[:-1]
    return stripped == REFUSAL_TEXT


def extract_citation_numbers(text: str) -> list[int]:
    """답변에서 인용 번호를 처음 나온 순서로, 같은 번호는 한 번만 뽑는다.

    읽는 형태: `[1]`, `[1][3]`, `[1, 3]`(쉼표 나열, 공백 허용). 범위(`[1-3]`)·소괄호·전각 괄호와 숫자·
    숫자가 아닌 글자가 섞인 대괄호(`[SFR-001]`)는 읽지 않는다. 검색 결과 범위 안인지는 확인하지 않는다.
    """
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers


def format_citation(hit: DocumentHit) -> str:
    """검색 결과 한 건의 출처 문자열(ADR-0006 결정 2의 5가지).

    소유자 구현(학습 모드, #19). 규칙: docs/data-design.md 2절 "출처 표기 규칙".
    """
    head = hit.bid_title

    if hit.format == "pdf":
        if hit.printed_page is not None:
            head += f" p.{hit.printed_page}"
        elif hit.page is not None:
            head += f" PDF {hit.page + 1}쪽"

    if hit.requirement_id:
        return f"[{head} · {hit.requirement_id}]"
    return f"[{head}]"


# 청크 본문 안의 여는·닫는 구분자 태그. 대소문자·태그 앞 공백(`< /chunk`)도 잡는다
_CHUNK_TAG_RE = re.compile(r"<(\s*/?\s*chunk)", re.IGNORECASE)


def escape_chunk_tags(text: str) -> str:
    """청크 본문의 `<chunk`·`</chunk`를 `&lt;chunk`·`&lt;/chunk`로 바꾼다(그 밖의 글자는 그대로).

    한계: ASCII `<`와 `\\s` 공백만 본다. 폭 없는 공백(`<\\u200b/chunk>`)·전각 `＜/chunk＞`는 바꾸지 않는다.
    """
    return _CHUNK_TAG_RE.sub(r"&lt;\1", text)


def escape_doc_attr(title: str) -> str:
    """`doc="…"` 속성값: 구분자 태그를 바꾸고 `"`·`>`를 `&quot;`·`&gt;`로 바꾼다(문서명도 외부 데이터다)."""
    return escape_chunk_tags(title).replace('"', "&quot;").replace(">", "&gt;")


def build_prompt(query: str, hits: list[DocumentHit]) -> list[ChatMessage]:
    """LLM에 보낼 메시지: system(`load_system_prompt()`) + user(청크 `[1]`~`[n]`과 질문).

    소유자 구현(학습 모드, #19). 번호 i는 `hits[i-1]`이다.
    #86(사용자 지시로 Claude 구현): 청크마다 `<chunk id="i" doc="문서명">`…`</chunk>`로 감싸 자료 구역을 표시하고,
    질문은 마지막 청크 뒤에 둔다. 본문의 `<chunk`·`</chunk`는 `&lt;`로 바꿔 구역을 끝내거나 새로 열지 못하게 하고,
    문서명은 `escape_doc_attr`로 속성값을 벗어나지 못하게 한다.
    """
    lines = []
    for number, hit in enumerate(hits, start=1):
        text = escape_chunk_tags(hit.text)
        doc = escape_doc_attr(hit.bid_title)
        lines.append(f'<chunk id="{number}" doc="{doc}">\n[{number}] {text}\n</chunk>')
    chunks = "\n\n".join(lines)

    user_content = f"{chunks}\n\n질문: {query}"

    return [
        ChatMessage(role="system", content=load_system_prompt()),
        ChatMessage(role="user", content=user_content),
    ]


def answer_question(
    query: str,
    *,
    deps: SearchDeps,
    llm: LLMClient,
    top_k: int = DEFAULT_TOP_K,
    doc_ids: list[str] | None = None,
) -> GroundedAnswer:
    """질문에 검색 결과만 근거로 답하고 출처를 붙인다. 근거가 없으면 거절한다.

    - 검색 결과가 0건이면 LLM을 부르지 않고 거절한다
    - `SearchUnavailableError`·`ValueError`(검색)와 LLM 예외는 잡지 않는다
    """
    hits = search_documents(query, top_k, doc_ids=doc_ids, deps=deps)
    if not hits:
        return GroundedAnswer(
            answer=REFUSAL_TEXT,
            refused=True,
            citations=[],
            citation_map={},
            cited_chunk_ids=[],
            invalid_citations=[],
            hit_count=0,
            llm_usage=None,
        )

    # 여기부터 소유자 구현(학습 모드, #19) — data-design.md 5절 "근거 답변" 동작 3~6:
    # build_prompt → llm.chat 1회 → is_refusal → extract_citation_numbers → format_citation
    messages = build_prompt(query, hits)
    response = llm.chat(messages)
    usage = LLMUsage(
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens,
    )

    if is_refusal(response.content):
        return GroundedAnswer(
            answer=REFUSAL_TEXT,
            refused=True,
            citations=[],
            citation_map={},
            cited_chunk_ids=[],
            invalid_citations=[],
            hit_count=len(hits),
            llm_usage=usage,
        )

    citations: list[str] = []
    citation_map: dict[int, str] = {}
    cited_chunk_ids: list[str] = []
    invalid_citations: list[int] = []

    for number in extract_citation_numbers(response.content):
        if 1 <= number <= len(hits):
            hit = hits[number - 1]
            citation = format_citation(hit)
            citation_map[number] = citation
            if citation not in citations:
                citations.append(citation)
            if hit.chunk_id not in cited_chunk_ids:
                cited_chunk_ids.append(hit.chunk_id)
        else:
            invalid_citations.append(number)

    return GroundedAnswer(
        answer=response.content.strip(),
        refused=False,
        citations=citations,
        citation_map=citation_map,
        cited_chunk_ids=cited_chunk_ids,
        invalid_citations=invalid_citations,
        hit_count=len(hits),
        llm_usage=usage,
    )
