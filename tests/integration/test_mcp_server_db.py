"""MCP 서버의 PMS 툴 3개를 실제 DB와 정답 파일로 확인한다 (이슈 #24, `-m integration`).

실행: `docker compose up -d postgres`, 합성 PMS v2 적재 후 `uv run pytest tests/integration -m integration -q`.

- MCP 클라이언트가 부르는 경로(`mcp.call_tool`)로 부르고 구조화 결과(dict)를 정답과 비교한다
- deps는 서버 모듈이 import 때 만든 실제 값(`PmsDeps.from_env()`, agent_ro·`AS_OF_DATE`)을 그대로 쓴다
- 정답은 `data/eval/pms_truth_v2.json`의 PRJ-05: 지연 6건, 1위 high Risk R05, 기한 지난 미제출 7건.
  ADR-0011 수동 검증(Claude Code에서 같은 질문)과 같은 사업이다. DB가 꺼져 있으면 skip하지 않고 실패한다
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult

from rfp_pm_agent.mcp_server import server

pytestmark = pytest.mark.integration

TRUTH_PATH = Path(__file__).resolve().parents[2] / "data/eval/pms_truth_v2.json"
TRUTH: dict[str, Any] = json.loads(TRUTH_PATH.read_text("utf-8"))
PID = "PRJ-05"
PRJ: dict[str, Any] = next(p for p in TRUTH["projects"] if p["project_id"] == PID)


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(server.mcp.call_tool(name, arguments))
    assert isinstance(result, CallToolResult) and not result.is_error, result
    out: dict[str, Any] = result.structured_content
    return out


def test_server_uses_truth_as_of_date() -> None:
    assert server.deps.as_of_date == date.fromisoformat(TRUTH["as_of_date"]), (
        "AS_OF_DATE가 정답 파일 기준일과 다르다 — .env를 확인한다"
    )


def test_delayed_tasks_match_truth() -> None:
    out = _call("get_delayed_tasks", {"project_id": PID, "limit": 50})

    assert out["status"] == "ok" and out["doc_ids"] == [PRJ["doc_id"]]
    assert out["total"] == len(PRJ["delayed_tasks"]) == 6 and not out["truncated"]
    assert {(t["wbs_id"], t["delay_days"], t["status"]) for t in out["items"]} == {
        (t["wbs_id"], t["delay_days"], t["status"]) for t in PRJ["delayed_tasks"]
    }


def test_high_open_risks_match_truth() -> None:
    truth_high = [r["risk_id"] for r in PRJ["open_risks"] if r["score"] >= 15]

    out = _call("get_open_risks", {"project_id": PID, "min_level": "high"})

    assert [r["risk_id"] for r in out["items"]] == truth_high == ["PRJ-05-R05"]
    assert out["items"][0]["level"] == "high"


def test_overdue_deliverables_match_truth() -> None:
    out = _call("get_deliverables", {"project_id": PID, "status_filter": "overdue", "limit": 50})

    ids = sorted(d["deliverable_id"] for d in out["items"])
    assert ids == sorted(PRJ["overdue_unsubmitted_deliverables"])
    assert len(ids) == 7
