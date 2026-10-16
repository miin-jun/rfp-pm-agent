"""에이전트 system 프롬프트에 넣는 사업 목록 (이슈 #22, ADR-0010 결정 3).

색인된 RFP 문서(doc_id·사업명)와 PMS 프로젝트(project_id)를 묶는다. PMS 프로젝트가 없는 문서도 목록에 넣는다
(`project_id`가 None) — 그 사업은 문서 질문만 답할 수 있고 PMS 질문에는 "PMS 데이터가 없다"고 답한다.

문서 목록은 OpenSearch 별칭에서 집계 1회로 읽는다(색인된 문서만 검색할 수 있으므로 색인이 기준이다).
"""

from __future__ import annotations

from pydantic import BaseModel

from rfp_pm_agent.pms.reader import PmsReader
from rfp_pm_agent.search.grounded_answer import escape_chunk_tags
from rfp_pm_agent.search.opensearch import SearchClient, search_alias

# 집계 버킷 상한. 지금 색인 문서는 10건이다. 넘으면 목록이 잘리므로 build_catalog가 멈춘다
MAX_DOCUMENTS = 200


class CatalogEntry(BaseModel):
    doc_id: str
    bid_title: str
    project_id: str | None  # PMS 프로젝트가 없는 문서는 None


def build_catalog(search_client: SearchClient, pms_reader: PmsReader) -> list[CatalogEntry]:
    """색인 문서마다 사업명과 PMS project_id를 붙인 목록. 사업명 순.

    - 색인 집계: doc_id별 bid_title 1개(같은 문서의 청크는 bid_title이 같다)
    - PMS: 프로젝트별 문서(doc_id) 연결. 색인에 없는 PMS 문서는 목록에 넣지 않는다(검색할 수 없다)
    """
    body = {
        "size": 0,
        "aggs": {
            "docs": {
                "terms": {"field": "doc_id", "size": MAX_DOCUMENTS + 1},
                "aggs": {"title": {"terms": {"field": "bid_title", "size": 1}}},
            }
        },
    }
    response = search_client.search(index=search_alias(), body=body)
    buckets = response["aggregations"]["docs"]["buckets"]
    if len(buckets) > MAX_DOCUMENTS:
        raise ValueError(
            f"색인 문서가 {MAX_DOCUMENTS}건을 넘어 사업 목록을 프롬프트에 다 넣을 수 없다"
        )

    project_by_doc: dict[str, str] = {}
    for project in pms_reader.projects():
        for document in pms_reader.documents(project.project_id):
            project_by_doc[document.doc_id] = project.project_id

    entries = [
        CatalogEntry(
            doc_id=b["key"],
            bid_title=b["title"]["buckets"][0]["key"] if b["title"]["buckets"] else "",
            project_id=project_by_doc.get(b["key"]),
        )
        for b in buckets
    ]
    return sorted(entries, key=lambda e: (e.bid_title, e.doc_id))


def render_catalog(entries: list[CatalogEntry]) -> str:
    """system 프롬프트에 넣는 표. 사업명은 외부 데이터라 구분자 태그를 바꿔 넣고 `|`·줄바꿈을 지운다."""
    lines = ["| 사업명 | doc_id | project_id |", "|---|---|---|"]
    for e in entries:
        title = escape_chunk_tags(e.bid_title).replace("|", "/").replace("\n", " ")
        lines.append(f"| {title} | {e.doc_id} | {e.project_id or '없음(PMS 데이터 없음)'} |")
    return "\n".join(lines)
