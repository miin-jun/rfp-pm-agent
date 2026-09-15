import json
from pathlib import Path
from typing import Any

import httpx

from rfp_pm_agent.config import NaraApiConfig
from rfp_pm_agent.ingest.collect import collect_from_api, register_manual_files
from rfp_pm_agent.ingest.manifest import ManifestEntry, append_entry, now_iso, read_manifest
from rfp_pm_agent.ingest.nara_client import NaraApiClient

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "nara_search_response.json"


def _load_fixture() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return result


def _make_client(attachment_bytes: bytes = b"fake rfp bytes") -> NaraApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "getBidPblancListInfoServcPPSSrch" in str(request.url):
            return httpx.Response(200, json=_load_fixture())
        return httpx.Response(200, content=attachment_bytes)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    config = NaraApiConfig(api_key="test-key", base_url="https://nara.test", timeout_s=5.0)
    return NaraApiClient(config, http_client=http_client)


def test_register_manual_files_is_idempotent(tmp_path: Path) -> None:
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / "sample.hwpx").write_bytes(b"hwpx bytes")
    manifest_path = tmp_path / "manifest.jsonl"

    first_run = register_manual_files(manual_dir, manifest_path)
    second_run = register_manual_files(manual_dir, manifest_path)

    assert len(first_run) == 1
    assert first_run[0].source_type == "manual"
    assert second_run == []  # 두 번째 실행은 신규 등록 없음
    assert len(read_manifest(manifest_path)) == 1


def test_collect_from_api_saves_one_file_per_notice(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    raw_api_dir = tmp_path / "api"
    client = _make_client()

    entries = collect_from_api(
        client=client, manifest_path=manifest_path, raw_api_dir=raw_api_dir, limit=5
    )

    # 픽스처 3건 중 SW 관련 + 첨부 있는 공고는 1건뿐
    assert len(entries) == 1
    assert entries[0].source_type == "api"
    assert entries[0].notice_no == "20260901001"
    saved_files = list(raw_api_dir.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].name == entries[0].file_name
    assert saved_files[0].read_bytes() == b"fake rfp bytes"


def test_collect_from_api_is_idempotent_on_rerun(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    raw_api_dir = tmp_path / "api"

    first = collect_from_api(
        client=_make_client(), manifest_path=manifest_path, raw_api_dir=raw_api_dir
    )
    second = collect_from_api(
        client=_make_client(), manifest_path=manifest_path, raw_api_dir=raw_api_dir
    )

    assert len(first) == 1
    assert second == []  # 이미 doc_id가 manifest에 있어 다시 받지 않음
    assert len(list(raw_api_dir.iterdir())) == 1  # 파일 수가 늘지 않음
    assert len(read_manifest(manifest_path)) == 1


def test_manual_and_api_share_dedup_by_doc_id(tmp_path: Path) -> None:
    """수동으로 먼저 넣은 파일과 같은 바이트가 API로도 잡히면 건너뛴다."""
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / "already-have.pdf").write_bytes(b"fake rfp bytes")  # _make_client와 동일 바이트
    manifest_path = tmp_path / "manifest.jsonl"
    raw_api_dir = tmp_path / "api"

    manual_entries = register_manual_files(manual_dir, manifest_path)
    api_entries = collect_from_api(
        client=_make_client(), manifest_path=manifest_path, raw_api_dir=raw_api_dir
    )

    assert len(manual_entries) == 1
    assert api_entries == []  # 같은 doc_id라 API 쪽에서는 신규로 잡히지 않음
    assert not raw_api_dir.exists() or list(raw_api_dir.iterdir()) == []


def _seed_api_entries(manifest_path: Path, count: int) -> None:
    for i in range(count):
        entry = ManifestEntry(
            doc_id=f"seed{i:012x}",
            file_name=f"seed{i}.pdf",
            source_type="api",
            notice_no=f"9999000{i}",
            notice_title="이미 보유한 공고",
            url=None,
            file_size=10,
            sha256=f"seedsha{i}",
            collected_at=now_iso(),
        )
        append_entry(manifest_path, entry)


def test_collect_from_api_skips_http_call_when_target_already_met(tmp_path: Path) -> None:
    """limit은 '이번 실행의 신규 건수'가 아니라 'manifest에 보유할 목표 총량'이다.
    이미 목표치를 채웠으면 API를 아예 호출하지 않아야 한다(이슈 #10 3차 수정)."""
    manifest_path = tmp_path / "manifest.jsonl"
    raw_api_dir = tmp_path / "api"
    _seed_api_entries(manifest_path, count=5)

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json=_load_fixture())

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://nara.test", transport=transport)
    config = NaraApiConfig(api_key="test-key", base_url="https://nara.test", timeout_s=5.0)
    client = NaraApiClient(config, http_client=http_client)

    entries = collect_from_api(
        client=client, manifest_path=manifest_path, raw_api_dir=raw_api_dir, limit=5
    )

    assert entries == []
    assert call_count["n"] == 0  # 검색 API조차 호출하지 않음
    assert len(read_manifest(manifest_path)) == 5  # 보유 건수가 늘지 않음
