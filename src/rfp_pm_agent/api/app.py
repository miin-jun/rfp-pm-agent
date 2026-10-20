"""시연 화면 FastAPI 앱 (이슈 #40, docs/adr/0015-demo-ui-fastapi-single-html.md).

- `GET /`: 시연 화면 HTML 한 파일(`api/static/index.html`). `{{AS_OF_DATE}}` 자리에 기준일을 넣는다
- `POST /api/ask`: 질문 → 에이전트 `Agent.ask`(프롬프트 기본값 agent_v1, 에이전트 로직은 그대로) → 답·출처 원문·거절 여부·툴 이름·걸린 시간

질문마다 새 대화(thread_id)로 묻는다 — 시연 화면은 한 번에 질문 하나다.
실행: `uv run uvicorn rfp_pm_agent.api.app:create_app --factory` (opensearch·postgres·tei-embed·tei-rerank 필요).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from importlib import resources
from typing import Any, Protocol

import opensearchpy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from rfp_pm_agent.agent.graph import Agent, agent_deps_from_env
from rfp_pm_agent.clients.cost import CostLogger
from rfp_pm_agent.config import ClientsConfig
from rfp_pm_agent.schemas.agent import AgentAnswer
from rfp_pm_agent.schemas.api import AskRequest, AskResponse, SourceChunk
from rfp_pm_agent.search.grounded_answer import extract_citation_numbers
from rfp_pm_agent.search.opensearch import EXCLUDED_SOURCE_FIELDS, SearchClient, search_alias

logger = logging.getLogger(__name__)


class Asker(Protocol):
    """`Agent.ask`의 쓰는 부분만. 단위 테스트는 정해 둔 `AgentAnswer`를 돌려주는 가짜를 넣는다."""

    def ask(self, question: str, *, thread_id: str) -> AgentAnswer: ...


@dataclass(frozen=True)
class DemoDeps:
    agent: Asker
    search_client: SearchClient  # 출처 원문 조회(`search`만 쓴다)
    as_of_date: date  # 화면에 보여 줄 기준일(AS_OF_DATE)


def cited_chunks(answer: AgentAnswer) -> list[tuple[int, str]]:
    """답에 나온 유효 출처 번호와 chunk_id의 짝. 답에 처음 나온 순서.

    `AgentAnswer`에는 번호 → chunk_id 표가 없어 두 목록을 맞춘다. finalize는 답의 유효 번호를 나온 순서로 돌며
    chunk_id를 처음 볼 때만 `cited_chunk_ids`에 넣고, 청크 번호는 대화 안에서 청크마다 하나라(다른 번호 = 다른 청크)
    두 목록은 1:1이다. 길이가 다르면 그 전제가 깨진 것이라 ValueError.
    """
    numbers = [n for n in extract_citation_numbers(answer.answer) if n in answer.citation_map]
    if len(numbers) != len(answer.cited_chunk_ids):
        raise ValueError(
            f"출처 번호 {numbers}와 cited_chunk_ids {answer.cited_chunk_ids}의 개수가 다르다"
        )
    return list(zip(numbers, answer.cited_chunk_ids, strict=True))


def fetch_chunks(client: SearchClient, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """chunk_id로 청크 `_source`를 읽는다(검색 별칭, 임베딩 제외). 인덱스에 없는 chunk_id는 결과에 없다."""
    if not chunk_ids:
        return {}
    body = {
        "size": len(chunk_ids),
        "query": {"terms": {"chunk_id": list(chunk_ids)}},
        "_source": {"excludes": list(EXCLUDED_SOURCE_FIELDS)},
    }
    response = client.search(index=search_alias(), body=body)
    return {raw["_source"]["chunk_id"]: raw["_source"] for raw in response["hits"]["hits"]}


def build_sources(answer: AgentAnswer, client: SearchClient) -> list[SourceChunk]:
    """답의 출처 번호마다 문서 필드와 원문을 붙인다. OpenSearch에 연결하지 못하면 출처 문자열만 채운다."""
    pairs = cited_chunks(answer)
    try:
        chunks = fetch_chunks(client, [chunk_id for _, chunk_id in pairs])
    except opensearchpy.OpenSearchException:
        logger.exception("출처 원문을 읽지 못했다 — 출처 문자열만 돌려준다")
        chunks = {}
    sources: list[SourceChunk] = []
    for number, chunk_id in pairs:
        source = chunks.get(chunk_id, {})
        sources.append(
            SourceChunk(
                number=number,
                chunk_id=chunk_id,
                citation=answer.citation_map[number],
                bid_title=source.get("bid_title"),
                format=source.get("format"),
                printed_page=source.get("printed_page"),
                page=source.get("page"),
                requirement_id=source.get("requirement_id"),
                text=source.get("text"),
            )
        )
    return sources


def ask(deps: DemoDeps, question: str) -> AskResponse:
    started = time.perf_counter()
    answer = deps.agent.ask(question, thread_id=f"demo-{uuid.uuid4()}")
    sources = build_sources(answer, deps.search_client)
    return AskResponse(
        answer=answer.answer,
        refused=answer.refused,
        sources=sources,
        tools_used=[call.name for call in answer.tool_calls],
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


def deps_from_env() -> DemoDeps:
    """실제 에이전트(LLM_MODEL_DEV, agent_v1). 사업 목록을 만들려고 OpenSearch·PMS DB를 한 번 읽는다."""
    agent = Agent(agent_deps_from_env(CostLogger(ClientsConfig.from_env().cost_log_path)))
    return DemoDeps(
        agent=agent,
        search_client=agent.deps.tools.search.search_client,
        as_of_date=agent.deps.tools.pms.as_of_date,
    )


def load_page(as_of_date: date) -> str:
    template = resources.files("rfp_pm_agent.api").joinpath("static", "index.html")
    return template.read_text("utf-8").replace("{{AS_OF_DATE}}", as_of_date.isoformat())


def create_app(deps: DemoDeps | None = None) -> FastAPI:
    """앱을 만든다. deps를 주지 않으면 기동할 때 환경변수로 만든다 — 서버가 꺼져 있으면 기동이 실패한다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.deps = deps or deps_from_env()
        yield

    app = FastAPI(title="rfp-pm-agent 시연", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> str:
        return load_page(request.app.state.deps.as_of_date)

    @app.post("/api/ask")
    def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
        # 동기 함수라 FastAPI가 스레드 풀에서 실행한다(에이전트는 동기 호출, p95 약 10초)
        try:
            return ask(request.app.state.deps, body.question)
        except Exception as e:
            # 예외 메시지에는 접속 정보·키 조각이 들어갈 수 있어 화면에는 예외 이름만 보낸다(메시지는 서버 로그에)
            logger.exception("에이전트 실행 실패")
            raise HTTPException(
                status_code=502,
                detail=f"답변을 만들지 못했다: {type(e).__name__} (자세한 내용은 서버 로그)",
            ) from e

    return app
