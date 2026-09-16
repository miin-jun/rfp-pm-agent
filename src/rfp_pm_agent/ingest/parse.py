"""파서 진입점 — 확장자로 포맷을 감지해 해당 추출기를 호출한다. 이슈 #11.

doc_id·bid_title은 `manifest.jsonl` 항목(doc_id, notice_title)에서 가져온다.
HWP 변환(hwp5html) 결과 HTML은 `data/parsed/_hwp_html/{doc_id}.html`에 캐시한다
(data/는 git 제외 — docs/data-design.md 1절).
"""

from __future__ import annotations

from pathlib import Path

from rfp_pm_agent.ingest.parsers.hwp_parser import parse_hwp
from rfp_pm_agent.ingest.parsers.hwpx_parser import parse_hwpx
from rfp_pm_agent.ingest.parsers.pdf_parser import parse_pdf
from rfp_pm_agent.schemas.document import Document

DEFAULT_HWP_HTML_CACHE_DIR = Path("data/parsed/_hwp_html")


def parse_document(
    path: str | Path,
    *,
    doc_id: str,
    bid_title: str,
    hwp_html_cache_dir: str | Path = DEFAULT_HWP_HTML_CACHE_DIR,
) -> Document:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, doc_id=doc_id, bid_title=bid_title)
    if suffix == ".hwp":
        return parse_hwp(
            path, doc_id=doc_id, bid_title=bid_title, cache_dir=Path(hwp_html_cache_dir)
        )
    if suffix == ".hwpx":
        return parse_hwpx(path, doc_id=doc_id, bid_title=bid_title)
    raise ValueError(f"지원하지 않는 확장자: {suffix} ({path})")
