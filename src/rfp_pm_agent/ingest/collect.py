"""나라장터 API 수집 + 수동 반입 등록 오케스트레이션 (이슈 #10).

두 경로 모두 같은 `manifest.jsonl`에 등록되고, `doc_id`(sha256 앞 16자)
기준으로 멱등적이다 — 두 번 실행해도 파일 수·manifest 줄 수가 늘지 않는다.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rfp_pm_agent.config import NaraApiConfig
from rfp_pm_agent.ingest.manifest import (
    ManifestEntry,
    append_entry,
    compute_hashes,
    existing_doc_ids,
    now_iso,
)
from rfp_pm_agent.ingest.nara_client import NaraApiClient, has_attachment, is_sw_related

logger = logging.getLogger(__name__)


def register_manual_files(manual_dir: str | Path, manifest_path: str | Path) -> list[ManifestEntry]:
    """`manual_dir`의 파일을 스캔해 `manifest_path`에 `source_type=manual`로
    등록한다. 이미 등록된 doc_id(source_type 무관)는 건너뛴다."""
    manual_dir = Path(manual_dir)
    known = existing_doc_ids(manifest_path)
    new_entries: list[ManifestEntry] = []
    if not manual_dir.exists():
        return new_entries

    for file_path in sorted(manual_dir.iterdir()):
        if not file_path.is_file():
            continue
        data = file_path.read_bytes()
        doc_id, sha256_hex = compute_hashes(data)
        if doc_id in known:
            continue
        entry = ManifestEntry(
            doc_id=doc_id,
            file_name=file_path.name,
            source_type="manual",
            notice_no=None,
            notice_title=None,
            url=None,
            file_size=len(data),
            sha256=sha256_hex,
            collected_at=now_iso(),
        )
        append_entry(manifest_path, entry)
        known.add(doc_id)
        new_entries.append(entry)
    return new_entries


def collect_from_api(
    *,
    client: NaraApiClient,
    manifest_path: str | Path,
    raw_api_dir: str | Path,
    limit: int = 5,
    lookback_days: int = 30,
) -> list[ManifestEntry]:
    """소프트웨어 용역 공고 중 제안요청서 첨부가 있는 공고를 최대 `limit`건
    찾아 다운로드하고 manifest에 등록한다. 멱등적이다 — 이미 manifest에
    있는 doc_id는 다시 받지 않는다. 공고 하나가 실패해도 나머지는 계속
    진행하고, 어느 공고에서 실패했는지 로그로 남긴다."""
    raw_api_dir = Path(raw_api_dir)
    known = existing_doc_ids(manifest_path)
    new_entries: list[ManifestEntry] = []

    end = datetime.now(UTC).date()
    begin = end - timedelta(days=lookback_days)
    items = client.search_service_bids(
        begin_date=begin.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")
    )
    candidates = [item for item in items if has_attachment(item) and is_sw_related(item)]

    for item in candidates:
        if len(new_entries) >= limit:
            break
        for url in item.spec_doc_urls:
            try:
                data = client.download_attachment(
                    url, context=f"notice={item.notice_no}({item.notice_title})"
                )
            except Exception:
                logger.exception(
                    "공고 %s(%s) 첨부파일 다운로드 실패: %s",
                    item.notice_no,
                    item.notice_title,
                    url,
                )
                continue

            doc_id, sha256_hex = compute_hashes(data)
            if doc_id in known:
                continue

            original_name = url.rsplit("/", 1)[-1] or "attachment"
            file_name = f"{doc_id}_{original_name}"
            raw_api_dir.mkdir(parents=True, exist_ok=True)
            (raw_api_dir / file_name).write_bytes(data)

            entry = ManifestEntry(
                doc_id=doc_id,
                file_name=file_name,
                source_type="api",
                notice_no=item.notice_no,
                notice_title=item.notice_title,
                url=url,
                file_size=len(data),
                sha256=sha256_hex,
                collected_at=now_iso(),
            )
            append_entry(manifest_path, entry)
            known.add(doc_id)
            new_entries.append(entry)
            break  # 공고 하나당 대표 첨부파일 1건만
    return new_entries


def main() -> None:
    """`uv run python -m rfp_pm_agent.ingest.collect`로 실제 수집을 실행하는
    진입점. 수동 반입 등록 → API 수집 순서로 진행한다."""
    logging.basicConfig(level=logging.INFO)
    manifest_path = "data/raw/manifest.jsonl"

    manual_new = register_manual_files("data/raw/manual", manifest_path)
    logger.info("수동 반입 등록: %d건 신규", len(manual_new))

    config = NaraApiConfig.from_env()
    client = NaraApiClient(config)
    api_new = collect_from_api(
        client=client, manifest_path=manifest_path, raw_api_dir="data/raw/api"
    )
    logger.info("API 수집: %d건 신규", len(api_new))


if __name__ == "__main__":
    main()
