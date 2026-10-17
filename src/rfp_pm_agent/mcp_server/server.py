from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rfp_pm_agent.pms.errors import PmsUnavailableError
from rfp_pm_agent.pms.rules import RiskLevel
from rfp_pm_agent.schemas.pms_tools import DeliverableFilter
from rfp_pm_agent.search.errors import SearchUnavailableError
from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K
from rfp_pm_agent.tools import pms_tools
from rfp_pm_agent.tools import search_documents as search_tool

mcp = MCPServer("rfp-pm-agent")
deps = pms_tools.PmsDeps.from_env()
search_deps = search_tool.SearchDeps.from_env()


@mcp.tool()
def get_delayed_tasks(
    project_id: str | None = None,
    scope: Literal["all", "this_week"] = "all",
    limit: int = pms_tools.DEFAULT_LIMIT,
) -> dict[str, Any]:
    """공공 SI 사업의 지연 업무(WBS)를 지연 일수가 긴 순서로 돌려준다.

    - project_id: 사업 ID(예: "PRJ-05"). 없으면 status="project_required"와 사업 후보 목록을 돌려준다.
    없는 사업 ID면 status="project_not_found"와 사업 후보 목록을 돌려준다.
    두 경우 모두 사용자에게 어느 사업인지 물어본 뒤 다시 부른다.
    - scope: "all"(누적 지연 전부) 또는 "this_week"(이번 주에 새로 지연된 것만)
    - limit: 최대 건수 1~50. total에 전체 건수가 있다
    기준일은 서버 설정(AS_OF_DATE)을 쓴다.
    """
    try:
        result = pms_tools.get_delayed_tasks(
            project_id=project_id, scope=scope, limit=limit, deps=deps
        )
    except (ValueError, PmsUnavailableError) as e:
        raise ToolError(str(e)) from e
    return result.model_dump(mode="json")


@mcp.tool()
def get_open_risks(
    project_id: str | None = None,
    min_level: RiskLevel = "low",
    limit: int = pms_tools.DEFAULT_LIMIT,
) -> dict[str, Any]:
    """공공 SI 사업의 미해결 Risk를 위험도가 높은 순서로 돌려준다.

    - project_id: 사업 ID(예: "PRJ-05"). 없으면 status="project_required"와 사업 후보 목록을 돌려준다.
    없는 사업 ID면 status="project_not_found"와 사업 후보 목록을 돌려준다. 두 경우 모두 사용자에게 어느 사업인지 물어본 뒤 다시 부른다.
    - min_level: 이 등급 이상만 본다. "low"(전부), "medium"(medium·high), "high"(high만).
      등급은 점수(확률×영향) 15 이상 high, 8~14 medium, 7 이하 low
    - limit: 최대 건수 1~50. total에 전체 건수가 있다
    기준일은 서버 설정(AS_OF_DATE)을 쓴다.
    """
    try:
        result = pms_tools.get_open_risks(
            project_id=project_id, min_level=min_level, limit=limit, deps=deps
        )
    except (ValueError, PmsUnavailableError) as e:
        raise ToolError(str(e)) from e
    return result.model_dump(mode="json")


@mcp.tool()
def get_deliverables(
    project_id: str | None = None,
    status_filter: DeliverableFilter = "all",
    requirement_id: str | None = None,
    wbs_id: str | None = None,
    limit: int = pms_tools.DEFAULT_LIMIT,
) -> dict[str, Any]:
    """공공 SI 사업의 산출물 목록을 돌려준다.

    - project_id: 사업 ID(예: "PRJ-05"). 없으면 status="project_required"와 사업 후보 목록을 돌려준다.
    없는 사업 ID면 status="project_not_found"와 사업 후보 목록을 돌려준다. 두 경우 모두 사용자에게 어느 사업인지 물어본 뒤 다시 부른다.
    - status_filter: "all"(전부), "not_submitted"(미제출), "overdue"(기한 지난 미제출),
      "submitted"(제출, 승인 전), "approved"(승인)
    - wbs_id: WBS 업무(예: "2.3")와 같은 요구사항에 연결된 산출물을 볼 때.
      그 업무 자체의 산출물뿐 아니라 같은 요구사항을 맡은 다른 업무의 산출물도 포함된다.
      없는 업무 ID면 오류
    - requirement_id: 특정 요구사항(예: "SFR-005")에 연결된 산출물만 볼 때
    - limit: 최대 건수 1~50. total에 전체 건수가 있다
    기준일은 서버 설정(AS_OF_DATE)을 쓴다.
    """
    try:
        result = pms_tools.get_deliverables(
            project_id=project_id,
            wbs_id=wbs_id,
            status_filter=status_filter,
            requirement_id=requirement_id,
            limit=limit,
            deps=deps,
        )
    except (ValueError, PmsUnavailableError) as e:
        raise ToolError(str(e)) from e
    return result.model_dump(mode="json")


@mcp.tool()
def search_documents(
    query: str,
    doc_ids: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """RFP(제안요청서) 원문에서 질문과 관련된 부분을 찾아 관련도 순으로 돌려준다.

    - query: 찾을 내용(한국어). 요구사항 ID(예: "SFR-005")를 넣으면 그 요구사항이 위로 온다
    - doc_ids: 검색할 사업의 RFP 문서 ID 목록(필수). 직접 지어내지 말고, PMS 툴 결과의 doc_ids나
      사업을 지정하지 않았을 때 받는 사업 목록(available_projects)의 doc_ids를 그대로 넣는다
    - top_k: 결과 수 1~20(기본 10)
    결과마다 출처(문서명·쪽·요구사항 ID)와 본문(text)이 있다. 답할 때 출처를 함께 적는다.
    결과가 0건이면 오류가 아니라 RFP에 관련 내용이 없다는 뜻이다. 지어내지 말고 없다고 답한다.
    score는 이 호출 안에서 순서를 비교할 때만 쓴다. 다른 호출의 score와 비교하지 않는다.
    """
    if not doc_ids:
        raise ToolError("doc_ids에 문서 ID를 1개 이상 넣어야 합니다.")
    try:
        hits = search_tool.search_documents(
            query=query, top_k=top_k, doc_ids=doc_ids, deps=search_deps
        )
    except (ValueError, SearchUnavailableError) as e:
        raise ToolError(str(e)) from e
    return [h.model_dump(mode="json") for h in hits]


if __name__ == "__main__":
    mcp.run(transport="stdio")
