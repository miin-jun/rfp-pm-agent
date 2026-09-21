"""나라장터 API 수집 + 수동 반입 등록 오케스트레이션 (이슈 #10).

두 경로 모두 같은 `manifest.jsonl`에 등록된다. 멱등성은 서로 다른 두 가지
성질을 함께 만족해야 한다 (2026-09-15 실측으로 드러남 —
docs/learning-log.md 여섯 번째 항목, docs/data-design.md 0절):

1. **같은 문서를 중복 저장하지 않는다** — `doc_id`(sha256 앞 16자)가 이미
   manifest에 있으면 다시 저장하지 않는다. `register_manual_files`와
   `collect_from_api` 둘 다 만족한다.
2. **실행 횟수와 무관하게 보유 건수가 목표치를 넘지 않는다** — `manual_dir`은
   내용이 고정이라 1번만으로 자동으로 만족되지만, `collect_from_api`의
   `limit`은 다르다. `limit`을 "이번 실행에서 새로 받을 건수"로 해석하면
   1번(중복 방지)은 지키면서도 실행할 때마다 **새로운(중복 아닌) 문서**를
   `limit`건씩 더 받아버려 총 보유 건수가 계속 늘어난다 — 실제로 1회차
   5건, 2회차 3건 추가로 8건이 됐다. 그래서 `limit`은 "manifest에 보유할
   API 수집 문서의 목표 총량"으로 정의하고, 이미 그만큼 있으면 API를
   아예 호출하지 않는다.
"""

from __future__ import annotations

import argparse
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
    read_manifest,
)
from rfp_pm_agent.ingest.nara_client import (
    NaraApiClient,
    NaraBidItem,
    has_attachment,
    is_sw_related,
    select_proposal_attachment,
)

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
    """`manifest_path`에 `source_type=api`로 이미 등록된 문서가 `limit`건
    이상이면 **API를 아예 호출하지 않고** 빈 리스트를 반환한다. `limit`은
    "이번 실행에서 새로 받을 건수"가 아니라 "manifest에 보유할 API 수집
    문서의 목표 총량"이다 — 그래야 실행 횟수와 무관하게 보유 건수가
    `limit`을 넘지 않는다(모듈 docstring의 멱등성 2번 성질). `limit`은
    상한일 뿐 목표치를 강제하지 않는다 — 이미 보유한 건수보다 낮은
    `limit`으로 실행해도 기존 파일이나 manifest 항목을 지우지 않는다. 부족한
    만큼만 소프트웨어 용역 공고 중 제안요청서 첨부가 있는 공고를 찾아
    다운로드한다. 이미 manifest에 있는 doc_id(source_type 무관, 멱등성
    1번 성질)는 다시 받지 않는다. 공고 하나가 실패해도 나머지는 계속
    진행하고, 어느 공고에서 실패했는지 로그로 남긴다."""
    raw_api_dir = Path(raw_api_dir)
    manifest_entries = read_manifest(manifest_path)
    known = {entry.doc_id for entry in manifest_entries}
    current_api_count = sum(1 for entry in manifest_entries if entry.source_type == "api")
    remaining = limit - current_api_count
    if remaining <= 0:
        logger.info(
            "API 수집 생략: 이미 %d건 보유(목표 %d건) — 목표치를 넘지 않도록 API를 호출하지 않음",
            current_api_count,
            limit,
        )
        return []

    new_entries: list[ManifestEntry] = []

    end = datetime.now(UTC).date()
    begin = end - timedelta(days=lookback_days)
    # nara_client.INQRY_DATETIME_FORMAT(YYYYMMDDHHMM, 12자리) — 8자리만 보내면
    # 서버가 부족한 HHMM을 0000(자정)으로 채워 종료일 당일 등록된 공고가 조회
    # 범위에서 빠진다(이슈 #47 실측 확인). 시작은 0000, 끝은 2359로 하루 전체를
    # 포함한다.
    inqry_bgn_dt = begin.strftime("%Y%m%d") + "0000"
    inqry_end_dt = end.strftime("%Y%m%d") + "2359"
    items = client.search_service_bids(begin_date=inqry_bgn_dt, end_date=inqry_end_dt)
    # is_sw_related가 어느 조건으로 통과시켰는지(sw_match_reason)를 manifest에
    # 남겨야 하므로(이슈 #47 문제 2), 필터링 시점에 (공고, 통과 조건) 쌍으로
    # 같이 들고 간다 — bool로만 거르면 이 정보가 사라진다.
    candidates: list[tuple[NaraBidItem, str]] = []
    for item in items:
        if not has_attachment(item):
            continue
        match_reason = is_sw_related(item)
        if match_reason is None:
            continue
        candidates.append((item, match_reason))

    for item, sw_match_reason in candidates:
        if len(new_entries) >= remaining:
            break

        attachment = select_proposal_attachment(item)
        if attachment is None:
            continue

        try:
            data = client.download_attachment(
                attachment.url, context=f"notice={item.notice_no}({item.notice_title})"
            )
        except Exception:
            logger.exception(
                "공고 %s(%s) 첨부파일 다운로드 실패: %s",
                item.notice_no,
                item.notice_title,
                attachment.file_name,
            )
            continue

        doc_id, sha256_hex = compute_hashes(data)
        if doc_id in known:
            continue

        # 저장 파일명은 attachment.url이 아니라 attachment.file_name에서 가져온다
        # — url은 실제로는 전부 downloadFile.do라 파일명 구분이 안 된다.
        file_name = f"{doc_id}_{attachment.file_name}"
        raw_api_dir.mkdir(parents=True, exist_ok=True)
        (raw_api_dir / file_name).write_bytes(data)

        entry = ManifestEntry(
            doc_id=doc_id,
            file_name=file_name,
            source_type="api",
            notice_no=item.notice_no,
            notice_title=item.notice_title,
            url=attachment.url,
            file_size=len(data),
            sha256=sha256_hex,
            collected_at=now_iso(),
            inqry_bgn_dt=inqry_bgn_dt,
            inqry_end_dt=inqry_end_dt,
            sw_match_reason=sw_match_reason,
        )
        append_entry(manifest_path, entry)
        known.add(doc_id)
        new_entries.append(entry)
    return new_entries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="나라장터 API 수집 + 수동 반입 등록을 실행한다.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "이번 실행에서 받을 건수가 아니라, manifest에 보유할 API 수집 문서의 "
            "목표 총량이다. 이미 이만큼 보유하고 있으면 API를 아예 호출하지 않는다. "
            "이 값을 낮춰도 이미 저장된 파일이나 manifest 항목은 지우지 않는다 "
            "(상한일 뿐 목표치를 강제로 맞추지 않음)."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="공고를 검색할 조회 기간(일). 오늘부터 이 일수만큼 과거까지의 공고를 조회한다.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """`uv run python -m rfp_pm_agent.ingest.collect`로 실제 수집을 실행하는
    진입점. 수동 반입 등록 → API 수집 순서로 진행한다.

    `--limit`을 낮춰도 이미 저장된 파일이나 manifest 항목은 지우지 않는다."""
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args(argv)
    manifest_path = "data/raw/manifest.jsonl"

    manual_new = register_manual_files("data/raw/manual", manifest_path)
    logger.info("수동 반입 등록: %d건 신규", len(manual_new))

    config = NaraApiConfig.from_env()
    client = NaraApiClient(config)
    api_new = collect_from_api(
        client=client,
        manifest_path=manifest_path,
        raw_api_dir="data/raw/api",
        limit=args.limit,
        lookback_days=args.lookback_days,
    )
    logger.info("API 수집: %d건 신규", len(api_new))


if __name__ == "__main__":
    main()
