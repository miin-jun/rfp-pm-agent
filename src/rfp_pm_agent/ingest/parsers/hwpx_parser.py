"""HWPX 스텁 파서. 이슈 #11.

수집된 HWPX 샘플에 요구사항 코드가 0건이었다(docs/parsing-exploration.md 실측 —
`^[A-Z]{3}-\\d{3}$` 셀 전체 일치 기준으로 재확인해도 0건). 실제 구조 파싱은
issue #11 범위 밖으로 미루고, `parse_status=unsupported_format`으로 표시만 하고
건너뛴다.
"""

from __future__ import annotations

from pathlib import Path

from rfp_pm_agent.schemas.document import Document


def parse_hwpx(path: str | Path, *, doc_id: str, bid_title: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_file=str(path),
        bid_title=bid_title,
        format="hwpx",
        parse_status="unsupported_format",
        has_requirements=False,
        requirement_count=0,
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=[],
        requirements=[],
    )
