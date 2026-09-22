"""파서 진입점 — 확장자로 포맷을 감지해 해당 추출기를 호출한다. 이슈 #11.

doc_id·bid_title은 `manifest.jsonl` 항목(doc_id, notice_title)에서 가져온다.
HWP 변환(hwp5html) 결과 HTML은 `data/parsed/_hwp_html/{doc_id}.html`에 캐시한다
(data/는 git 제외 — docs/data-design.md 1절).

CLI(이슈 #52): `uv run python -m rfp_pm_agent.ingest.parse`
manifest의 문서를 파싱해 `data/parsed/<doc_id>.json`으로 저장한다.
- 원본 경로는 source_type으로 정한다 (api → data/raw/api/, manual → data/raw/manual/)
- `<doc_id>.json`이 이미 있으면 건너뛴다. parse_status="failed"로 기록된 문서도
  건너뛰므로, 다시 파싱하려면 그 json을 지운다
- 예외가 나면 parse_status="failed"로 기록하고 오류를 validation_warnings에 남긴다.
  실패가 파일로 남지 않으면 "요구사항 없음"과 "파싱 실패"가 구별되지 않는다
- 확장자가 pdf/hwp/hwpx가 아니면 Document.format에 넣을 값이 없어 파일로
  기록하지 못한다. 이 경우는 로그에만 남기고 ("unknown", "failed")로 센다
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Literal, cast, get_args

from pydantic import BaseModel, Field

from rfp_pm_agent.ingest.manifest import ManifestEntry, read_manifest
from rfp_pm_agent.ingest.parsers.hwp_parser import parse_hwp
from rfp_pm_agent.ingest.parsers.hwpx_parser import parse_hwpx
from rfp_pm_agent.ingest.parsers.pdf_parser import parse_pdf
from rfp_pm_agent.schemas.document import Document

DEFAULT_HWP_HTML_CACHE_DIR = Path("data/parsed/_hwp_html")
DEFAULT_MANIFEST_PATH = Path("data/raw/manifest.jsonl")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PARSED_DIR = Path("data/parsed")

DocFormat = Literal["pdf", "hwp", "hwpx"]

logger = logging.getLogger(__name__)


class ParseRunSummary(BaseModel):
    """한 번의 CLI 실행 결과. counts의 키는 (format, parse_status)."""

    skipped: int = 0
    counts: Counter[tuple[str, str]] = Field(default_factory=Counter)


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


def resolve_source_path(entry: ManifestEntry, raw_dir: Path) -> Path:
    """source_type(api|manual)을 raw_dir 아래 하위 디렉터리 이름으로 쓴다."""
    return raw_dir / entry.source_type / entry.file_name


def _failed_document(entry: ManifestEntry, path: Path, fmt: DocFormat, error: str) -> Document:
    return Document(
        doc_id=entry.doc_id,
        source_file=str(path),
        bid_title=entry.notice_title or "",
        format=fmt,
        parse_status="failed",
        has_requirements=False,
        requirement_count=0,
        declared_total=None,
        summary_ids=[],
        validation_warnings=[f"파싱 예외: {error}"],
        blocks=[],
        requirements=[],
    )


def parse_manifest(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    hwp_html_cache_dir: Path = DEFAULT_HWP_HTML_CACHE_DIR,
) -> ParseRunSummary:
    """manifest의 문서 중 `<parsed_dir>/<doc_id>.json`이 없는 것만 파싱해 저장한다.

    한 건의 예외가 나머지 처리를 막지 않는다."""
    summary = ParseRunSummary()
    parsed_dir.mkdir(parents=True, exist_ok=True)
    for entry in read_manifest(manifest_path):
        out_path = parsed_dir / f"{entry.doc_id}.json"
        if out_path.exists():
            summary.skipped += 1
            continue
        src = resolve_source_path(entry, raw_dir)
        try:
            doc = parse_document(
                src,
                doc_id=entry.doc_id,
                bid_title=entry.notice_title or "",
                hwp_html_cache_dir=hwp_html_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 — 한 건 실패가 전체를 멈추면 안 된다
            error = f"{type(exc).__name__}: {exc}"
            suffix = src.suffix.lower().lstrip(".")
            if suffix not in get_args(DocFormat):
                logger.error("파싱 실패(기록 불가 확장자) %s — %s", src.name, error)
                summary.counts[("unknown", "failed")] += 1
                continue
            logger.error("파싱 실패 %s — %s", src.name, error)
            doc = _failed_document(entry, src, cast(DocFormat, suffix), error)
        out_path.write_text(doc.model_dump_json(), encoding="utf-8")
        summary.counts[(doc.format, doc.parse_status)] += 1
    return summary


def count_has_requirements(parsed_dir: Path) -> int:
    """parsed_dir 전체 `<doc_id>.json` 중 has_requirements=true 건수.

    이번 실행분만 세면 재실행 시 전부 skipped라 판정할 수 없어 디렉터리 전체를 센다."""
    return sum(
        Document.model_validate_json(p.read_text(encoding="utf-8")).has_requirements
        for p in parsed_dir.glob("*.json")
    )


def log_summary(summary: ParseRunSummary) -> None:
    """형식별·parse_status별 건수. hwpx는 스텁이라 unsupported_format이 섞이므로
    합계만으로는 판단할 수 없다."""
    logger.info("건너뜀(이미 파싱됨): %d건", summary.skipped)
    for (fmt, status), n in sorted(summary.counts.items()):
        logger.info("%s / %s: %d건", fmt, status, n)
    logger.info("이번 실행 처리: %d건", sum(summary.counts.values()))


def main() -> None:
    """`uv run python -m rfp_pm_agent.ingest.parse` 진입점."""
    logging.basicConfig(level=logging.INFO)
    log_summary(parse_manifest())
    logger.info(
        "has_requirements=true (%s 전체): %d건",
        DEFAULT_PARSED_DIR,
        count_has_requirements(DEFAULT_PARSED_DIR),
    )


if __name__ == "__main__":
    main()
