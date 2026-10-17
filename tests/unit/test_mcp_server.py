"""MCP 서버 (이슈 #24, ADR-0011): 툴 등록·스키마, 원본 툴 호출 인자, 결과 직렬화. DB·검색 서버 없이 가짜 deps로 본다.

`server.py`는 import 시점에 `PmsDeps.from_env()`·`SearchDeps.from_env()`로 deps를 만든다. 두 함수 모두 연결은 첫
요청 때 하므로 `AS_OF_DATE`만 있으면 import된다. 테스트는 import 뒤 모듈 변수 `deps`·`search_deps`를 가짜로 바꾼다
(툴 함수가 호출 때마다 모듈 변수를 읽는다).
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult

from rfp_pm_agent.schemas.pms import PmsDataset
from rfp_pm_agent.schemas.search import DocumentHit
from rfp_pm_agent.tools import pms_tools
from rfp_pm_agent.tools import search_documents as search_tool
from rfp_pm_agent.tools.pms_tools import PmsDeps
from rfp_pm_agent.tools.search_documents import SearchDeps
from tests.fakes.fake_embedding import FakeEmbeddingClient
from tests.fakes.fake_pms_reader import FakePmsReader
from tests.fakes.fake_reranker import FakeRerankerClient
from tests.fakes.fake_search_client import RecordingSearchClient
from tests.fakes.pms_sample import AS_OF, build_sample

TOOL_NAMES = {"get_delayed_tasks", "get_open_risks", "get_deliverables", "search_documents"}


@pytest.fixture(scope="module")
def server() -> ModuleType:
    # .env가 없는 환경(CI)에서도 import되도록 기준일만 넣는다. 이미 있으면 그 값을 쓴다
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AS_OF_DATE", AS_OF.isoformat())
        return importlib.import_module("rfp_pm_agent.mcp_server.server")


@pytest.fixture(scope="module")
def dataset() -> PmsDataset:
    return build_sample()[0]


@pytest.fixture
def fake_deps(server: ModuleType, dataset: PmsDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "deps", PmsDeps(reader=FakePmsReader(dataset), as_of_date=AS_OF))
    monkeypatch.setattr(
        server,
        "search_deps",
        SearchDeps(
            search_client=RecordingSearchClient(),
            embedder=FakeEmbeddingClient(),
            reranker=FakeRerankerClient(),
        ),
    )


def _schemas(server: ModuleType) -> dict[str, dict[str, Any]]:
    tools = asyncio.run(server.mcp.list_tools())
    return {t.name: t.input_schema for t in tools}


def _spy(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, name: str, returns: Any = None
) -> list[dict[str, Any]]:
    """`module.name`을 호출 인자를 기록하는 함수로 바꾼다. returns가 없으면 원래 함수를 그대로 부른다."""
    original: Callable[..., Any] = getattr(module, name)
    calls: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, **kwargs})
        return original(*args, **kwargs) if returns is None else returns

    monkeypatch.setattr(module, name, spy)
    return calls


def _call(server: ModuleType, name: str, arguments: dict[str, Any]) -> Any:
    """MCP 클라이언트가 부르는 경로(`call_tool`)로 부르고 결과를 확인한다."""
    result = asyncio.run(server.mcp.call_tool(name, arguments))
    assert isinstance(result, CallToolResult) and not result.is_error, result
    return result.structured_content


# --- 등록·스키마 ---


def test_four_read_only_tools_are_registered(server: ModuleType) -> None:
    assert set(_schemas(server)) == TOOL_NAMES


def test_hidden_arguments_are_not_in_schema(server: ModuleType) -> None:
    for name, schema in _schemas(server).items():
        assert "deps" not in schema["properties"], name
        assert "as_of" not in schema["properties"], name


def test_visible_arguments(server: ModuleType) -> None:
    schemas = _schemas(server)
    assert list(schemas["get_delayed_tasks"]["properties"]) == ["project_id", "scope", "limit"]
    assert list(schemas["get_open_risks"]["properties"]) == ["project_id", "min_level", "limit"]
    assert set(schemas["get_deliverables"]["properties"]) == {
        "project_id",
        "status_filter",
        "requirement_id",
        "wbs_id",
        "limit",
    }
    assert set(schemas["search_documents"]["properties"]) == {"query", "doc_ids", "top_k"}


def test_pms_tools_have_no_required_arguments(server: ModuleType) -> None:
    """project_id를 빼고 부르면 오류가 아니라 사업 후보(project_required)를 돌려받아야 되묻기가 된다."""
    schemas = _schemas(server)
    for name in ("get_delayed_tasks", "get_open_risks", "get_deliverables"):
        assert schemas[name].get("required", []) == [], name


def test_search_documents_requires_doc_ids(server: ModuleType) -> None:
    schema = _schemas(server)["search_documents"]
    assert set(schema["required"]) == {"query", "doc_ids"}
    assert schema["properties"]["doc_ids"]["type"] == "array"  # null 불가 — 전체 검색을 열지 않는다


def test_search_documents_without_doc_ids_is_rejected_before_search(
    server: ModuleType, fake_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch, search_tool, "search_documents", returns=[])

    # 인자 검증 실패는 ToolError(메시지 유지)로 올라오고, 서버가 is_error 결과로 바꿔 클라이언트에 돌려준다
    with pytest.raises(ToolError, match="doc_ids"):
        asyncio.run(server.mcp.call_tool("search_documents", {"query": "SFR-001"}))
    assert calls == []


# --- 원본 툴 호출 인자·결과 ---


def test_get_delayed_tasks_passes_arguments_and_returns_dict(
    server: ModuleType, fake_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch, pms_tools, "get_delayed_tasks")

    out = _call(
        server, "get_delayed_tasks", {"project_id": "PRJ-01", "scope": "this_week", "limit": 5}
    )

    assert calls == [
        {"args": (), "project_id": "PRJ-01", "scope": "this_week", "limit": 5, "deps": server.deps}
    ]
    assert isinstance(out, dict)
    assert out["status"] == "ok" and out["project_id"] == "PRJ-01"
    assert out["as_of_date"] == AS_OF.isoformat()  # mode="json" — 날짜는 문자열


def test_get_delayed_tasks_defaults_and_project_required(
    server: ModuleType, fake_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch, pms_tools, "get_delayed_tasks")

    out = _call(server, "get_delayed_tasks", {})

    assert calls == [
        {"args": (), "project_id": None, "scope": "all", "limit": 20, "deps": server.deps}
    ]
    assert out["status"] == "project_required"
    assert [p["project_id"] for p in out["available_projects"]] == ["PRJ-01", "PRJ-02"]


def test_get_open_risks_passes_arguments_and_returns_dict(
    server: ModuleType, fake_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch, pms_tools, "get_open_risks")

    out = _call(server, "get_open_risks", {"project_id": "PRJ-02", "min_level": "high", "limit": 3})

    assert calls == [
        {"args": (), "project_id": "PRJ-02", "min_level": "high", "limit": 3, "deps": server.deps}
    ]
    assert isinstance(out, dict)
    assert out["status"] == "ok" and out["min_level"] == "high"
    assert all(r["level"] == "high" for r in out["items"])


def test_get_deliverables_passes_arguments_and_returns_dict(
    server: ModuleType, dataset: PmsDataset, fake_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = next(link for link in dataset.wbs_requirements if link.project_id == "PRJ-01")
    calls = _spy(monkeypatch, pms_tools, "get_deliverables")

    out = _call(
        server,
        "get_deliverables",
        {
            "project_id": "PRJ-01",
            "status_filter": "not_submitted",
            "requirement_id": link.requirement_id,
            "wbs_id": link.wbs_id,
            "limit": 7,
        },
    )

    assert calls == [
        {
            "args": (),
            "project_id": "PRJ-01",
            "wbs_id": link.wbs_id,
            "status_filter": "not_submitted",
            "requirement_id": link.requirement_id,
            "limit": 7,
            "deps": server.deps,
        }
    ]
    assert isinstance(out, dict)
    assert out["status"] == "ok"
    assert (out["wbs_id"], out["requirement_id"]) == (link.wbs_id, link.requirement_id)
    assert all(d["status"] == "not_submitted" for d in out["items"])


def _hit(rank: int, doc_id: str) -> DocumentHit:
    return DocumentHit(
        rank=rank,
        chunk_id=f"{doc_id}-c{rank}",
        doc_id=doc_id,
        bid_title="A 사업",
        format="pdf",
        requirement_id="SFR-001",
        page=3,
        printed_page=4,
        block_type=None,
        text="본문",
        score=0.9,
    )


def test_search_documents_passes_arguments_and_returns_list_of_dicts(
    server: ModuleType, fake_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = [_hit(1, "docA"), _hit(2, "docA")]
    calls = _spy(monkeypatch, search_tool, "search_documents", returns=hits)

    direct = server.search_documents(query="SFR-001 요구사항", doc_ids=["docA"], top_k=2)

    assert calls == [
        {
            "args": (),
            "query": "SFR-001 요구사항",
            "top_k": 2,
            "doc_ids": ["docA"],
            "deps": server.search_deps,
        }
    ]
    assert direct == [h.model_dump(mode="json") for h in hits]
    assert all(isinstance(h, dict) for h in direct)


def test_search_documents_default_top_k_through_mcp(
    server: ModuleType, fake_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch, search_tool, "search_documents", returns=[_hit(1, "docA")])

    out = _call(server, "search_documents", {"query": "SFR-001", "doc_ids": ["docA"]})

    assert calls[0]["top_k"] == 10
    # list 반환은 MCP 구조화 결과에서 {"result": [...]}로 감싸진다
    assert out == {"result": [_hit(1, "docA").model_dump(mode="json")]}


def test_tool_exception_message_does_not_reach_client(server: ModuleType, fake_deps: None) -> None:
    """ADR-0011 "한계": 원본이 던진 ValueError는 SDK가 예상하지 못한 예외로 보고 원래 메시지를 빼고 돌려준다."""
    with pytest.raises(UnexpectedToolError) as exc_info:
        asyncio.run(
            server.mcp.call_tool("get_delayed_tasks", {"project_id": "PRJ-01", "limit": 99})
        )

    assert str(exc_info.value) == "Error executing tool get_delayed_tasks"
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "limit" in str(exc_info.value.__cause__)  # 원래 메시지는 서버 쪽 예외에만 있다
