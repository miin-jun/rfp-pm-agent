import json
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest

from rfp_pm_agent.config import NaraApiConfig
from rfp_pm_agent.ingest import collect as collect_module
from rfp_pm_agent.ingest.collect import (
    collect_from_api,
    main,
    register_manual_files,
    split_query_range,
    to_inqry_begin,
    to_inqry_end,
    today_kst,
)
from rfp_pm_agent.ingest.collect_runs import read_runs
from rfp_pm_agent.ingest.manifest import ManifestEntry, append_entry, now_iso, read_manifest
from rfp_pm_agent.ingest.nara_client import NaraApiClient, _validate_search_range

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


def test_collect_from_api_records_query_range_in_manifest(tmp_path: Path) -> None:
    """이슈 #47 문제 3 — API로 수집한 항목은 실제로 보낸 조회 범위
    (inqryBgnDt~inqryEndDt)를 manifest에 남겨야 한다. 시작은 0000, 끝은
    2359로 하루 전체를 포함해야 한다(8자리만 보내면 서버가 HHMM을 0000으로
    채워 종료일 당일 공고가 빠진다는 게 실측으로 확인됨)."""
    manifest_path = tmp_path / "manifest.jsonl"
    raw_api_dir = tmp_path / "api"
    client = _make_client()

    entries = collect_from_api(
        client=client,
        manifest_path=manifest_path,
        raw_api_dir=raw_api_dir,
        limit=5,
        lookback_days=30,
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.inqry_bgn_dt is not None
    assert entry.inqry_end_dt is not None
    assert entry.inqry_bgn_dt.endswith("0000")
    assert entry.inqry_end_dt.endswith("2359")
    assert len(entry.inqry_bgn_dt) == 12
    assert len(entry.inqry_end_dt) == 12


def test_register_manual_files_leaves_query_range_none(tmp_path: Path) -> None:
    """수동 반입 항목은 API 조회 자체가 없으므로 조회 범위 필드가 None이어야
    한다(이슈 #47 문제 3)."""
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / "sample.hwpx").write_bytes(b"hwpx bytes")
    manifest_path = tmp_path / "manifest.jsonl"

    entries = register_manual_files(manual_dir, manifest_path)

    assert len(entries) == 1
    assert entries[0].inqry_bgn_dt is None
    assert entries[0].inqry_end_dt is None


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


def test_collect_from_api_does_not_shrink_when_limit_lowered(tmp_path: Path) -> None:
    """limit은 상한이지 목표치 강제가 아니다 — 이미 30건 보유한 상태에서
    --limit 5로 실행해도 API를 호출하지 않고, 기존 파일도 지우지 않는다."""
    manifest_path = tmp_path / "manifest.jsonl"
    raw_api_dir = tmp_path / "api"
    raw_api_dir.mkdir()
    _seed_api_entries(manifest_path, count=30)
    for i in range(30):
        (raw_api_dir / f"seed{i}.pdf").write_bytes(b"seed bytes")

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
    assert call_count["n"] == 0  # API를 호출하지 않음
    assert len(read_manifest(manifest_path)) == 30  # manifest 항목을 지우지 않음
    assert len(list(raw_api_dir.iterdir())) == 30  # 기존 파일을 지우지 않음


def test_main_passes_limit_and_lookback_days_to_collect_from_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    def fake_register_manual_files(manual_dir: Any, manifest_path: Any) -> list[Any]:
        return []

    def fake_collect_from_api(**kwargs: Any) -> list[Any]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr(collect_module, "register_manual_files", fake_register_manual_files)
    monkeypatch.setattr(collect_module, "collect_from_api", fake_collect_from_api)
    monkeypatch.setattr(
        NaraApiConfig,
        "from_env",
        classmethod(
            lambda cls: NaraApiConfig(
                api_key="test-key", base_url="https://nara.test", timeout_s=5.0
            )
        ),
    )

    main(["--limit", "30"])

    assert calls["limit"] == 30
    assert calls["lookback_days"] == 30  # 기본값


def test_main_defaults_limit_5_and_lookback_days_30(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_register_manual_files(manual_dir: Any, manifest_path: Any) -> list[Any]:
        return []

    def fake_collect_from_api(**kwargs: Any) -> list[Any]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr(collect_module, "register_manual_files", fake_register_manual_files)
    monkeypatch.setattr(collect_module, "collect_from_api", fake_collect_from_api)
    monkeypatch.setattr(
        NaraApiConfig,
        "from_env",
        classmethod(
            lambda cls: NaraApiConfig(
                api_key="test-key", base_url="https://nara.test", timeout_s=5.0
            )
        ),
    )

    main([])

    assert calls["limit"] == 5
    assert calls["lookback_days"] == 30


# --- 조회 기준 날짜 KST 고정 (이슈 #50) ---


def test_today_kst_uses_korean_date_before_9am_kst() -> None:
    """UTC 2026-09-21 23:00 = KST 2026-09-22 08:00. 나라장터 조회 시각은 KST
    기준이므로 종료일은 09-22여야 한다 — UTC 날짜(09-21)를 쓰면 당일 공고가 빠진다."""
    now = datetime(2026, 9, 21, 23, 0, tzinfo=UTC)

    assert today_kst(now) == date(2026, 9, 22)


def test_today_kst_same_day_after_9am_kst() -> None:
    now = datetime(2026, 9, 22, 1, 0, tzinfo=UTC)  # KST 10:00

    assert today_kst(now) == date(2026, 9, 22)


def test_collect_from_api_default_range_ends_on_kst_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collect_module, "today_kst", lambda now=None: date(2026, 9, 22))

    entries = collect_from_api(
        client=_make_client(),
        manifest_path=tmp_path / "manifest.jsonl",
        raw_api_dir=tmp_path / "api",
        lookback_days=30,
    )

    assert entries[0].inqry_end_dt == "202609222359"
    assert entries[0].inqry_bgn_dt == "202608230000"


# --- 조회 구간 분할과 --from/--to (이슈 #50) ---


def test_split_query_range_single_day() -> None:
    assert split_query_range(date(2026, 6, 1), date(2026, 6, 1)) == [
        (date(2026, 6, 1), date(2026, 6, 1))
    ]


def test_split_query_range_within_30_days_is_one_chunk() -> None:
    assert split_query_range(date(2026, 6, 1), date(2026, 7, 1)) == [
        (date(2026, 6, 1), date(2026, 7, 1))
    ]


def test_split_query_range_overlaps_one_day_at_boundaries() -> None:
    chunks = split_query_range(date(2026, 6, 1), date(2026, 8, 31))

    assert chunks == [
        (date(2026, 6, 1), date(2026, 7, 1)),
        (date(2026, 7, 1), date(2026, 7, 31)),
        (date(2026, 7, 31), date(2026, 8, 30)),
        (date(2026, 8, 30), date(2026, 8, 31)),
    ]
    # 다음 구간의 시작일 = 이전 구간의 종료일 (1일 겹침, 틈 없음)
    for (_, prev_end), (next_begin, _) in pairwise(chunks):
        assert next_begin == prev_end


def test_split_query_range_chunks_pass_client_range_validation() -> None:
    """분할된 모든 구간이 0000~2359로 바꿨을 때 클라이언트의 조회 범위
    검증(최대 31일)을 통과해야 한다."""
    for begin, end in split_query_range(date(2026, 1, 1), date(2026, 12, 31)):
        _validate_search_range(to_inqry_begin(begin), to_inqry_end(end))


def test_split_query_range_rejects_reversed_range() -> None:
    with pytest.raises(ValueError):
        split_query_range(date(2026, 8, 31), date(2026, 6, 1))


def test_to_inqry_formats_are_12_digits() -> None:
    assert to_inqry_begin(date(2026, 6, 1)) == "202606010000"
    assert to_inqry_end(date(2026, 6, 1)) == "202606012359"


def test_collect_from_api_calls_search_once_per_chunk(tmp_path: Path) -> None:
    searched: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "getBidPblancListInfoServcPPSSrch" in str(request.url):
            params = request.url.params
            searched.append((params["inqryBgnDt"], params["inqryEndDt"]))
            return httpx.Response(200, json=_load_fixture())
        return httpx.Response(200, content=b"fake rfp bytes")

    http_client = httpx.Client(base_url="https://nara.test", transport=httpx.MockTransport(handler))
    config = NaraApiConfig(api_key="test-key", base_url="https://nara.test", timeout_s=5.0)

    entries = collect_from_api(
        client=NaraApiClient(config, http_client=http_client),
        manifest_path=tmp_path / "manifest.jsonl",
        raw_api_dir=tmp_path / "api",
        limit=5,
        begin_date=date(2026, 6, 1),
        end_date=date(2026, 8, 31),
    )

    assert searched == [
        ("202606010000", "202607012359"),
        ("202607010000", "202607312359"),
        ("202607310000", "202608302359"),
        ("202608300000", "202608312359"),
    ]
    assert len(entries) == 1  # 구간마다 같은 공고가 와도 doc_id로 한 번만 저장
    assert entries[0].inqry_bgn_dt == "202606010000"
    assert entries[0].inqry_end_dt == "202607012359"


def _patch_main_deps(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    def fake_collect_from_api(**kwargs: Any) -> list[Any]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr(collect_module, "register_manual_files", lambda *a, **k: [])
    monkeypatch.setattr(collect_module, "collect_from_api", fake_collect_from_api)
    monkeypatch.setattr(
        NaraApiConfig,
        "from_env",
        classmethod(
            lambda cls: NaraApiConfig(
                api_key="test-key", base_url="https://nara.test", timeout_s=5.0
            )
        ),
    )


def test_main_passes_from_to_as_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_main_deps(monkeypatch, calls)

    main(["--from", "20260601", "--to", "20260831"])

    assert calls["begin_date"] == date(2026, 6, 1)
    assert calls["end_date"] == date(2026, 8, 31)


def test_main_without_from_to_passes_none_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_main_deps(monkeypatch, calls)

    main([])

    assert calls["begin_date"] is None
    assert calls["end_date"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--from", "20260601", "--to", "20260831", "--lookback-days", "10"],
        ["--from", "20260601"],
        ["--to", "20260831"],
        ["--from", "2026-06-01", "--to", "20260831"],
        ["--from", "20260831", "--to", "20260601"],
    ],
)
def test_main_rejects_invalid_range_args(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_main_deps(monkeypatch, calls)

    with pytest.raises(SystemExit):
        main(argv)

    assert calls == {}  # 수집을 시작하지 않음


# --- 구간별 실행 기록 collect_runs.jsonl (이슈 #50) ---


def _counting_client(search_calls: list[str]) -> NaraApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "getBidPblancListInfoServcPPSSrch" in str(request.url):
            search_calls.append(request.url.params["inqryBgnDt"])
            return httpx.Response(200, json=_load_fixture())
        return httpx.Response(200, content=b"fake rfp bytes")

    http_client = httpx.Client(base_url="https://nara.test", transport=httpx.MockTransport(handler))
    config = NaraApiConfig(api_key="test-key", base_url="https://nara.test", timeout_s=5.0)
    return NaraApiClient(config, http_client=http_client)


def test_collect_runs_records_every_chunk_as_not_called_when_limit_already_met(
    tmp_path: Path,
) -> None:
    """--limit 도달로 API를 안 부른 구간도 기록해야 한다 — 기록이 없으면
    '공고가 없었다'와 '조회하지 않았다'가 구별되지 않는다."""
    manifest_path = tmp_path / "manifest.jsonl"
    runs_path = tmp_path / "collect_runs.jsonl"
    _seed_api_entries(manifest_path, count=5)
    search_calls: list[str] = []

    collect_from_api(
        client=_counting_client(search_calls),
        manifest_path=manifest_path,
        raw_api_dir=tmp_path / "api",
        limit=5,
        begin_date=date(2026, 6, 1),
        end_date=date(2026, 8, 31),
        runs_path=runs_path,
    )

    runs = read_runs(runs_path)
    assert search_calls == []
    assert len(runs) == 4  # 구간 4개 모두 기록
    for run in runs:
        assert run.called is False
        assert run.skip_reason == "limit_reached"
        assert run.total_count is None
        assert run.pages_fetched == 0
        assert run.response_count == 0
        assert run.filter_passed == 0
        assert run.new_saved == 0


def test_collect_runs_records_called_chunk_counts_and_later_skips(tmp_path: Path) -> None:
    runs_path = tmp_path / "collect_runs.jsonl"
    search_calls: list[str] = []

    collect_from_api(
        client=_counting_client(search_calls),
        manifest_path=tmp_path / "manifest.jsonl",
        raw_api_dir=tmp_path / "api",
        limit=1,
        begin_date=date(2026, 6, 1),
        end_date=date(2026, 8, 31),
        runs_path=runs_path,
    )

    runs = read_runs(runs_path)
    assert search_calls == ["202606010000"]  # 1번째 구간에서 목표치 도달
    assert [(r.inqry_bgn_dt, r.inqry_end_dt) for r in runs] == [
        ("202606010000", "202607012359"),
        ("202607010000", "202607312359"),
        ("202607310000", "202608302359"),
        ("202608300000", "202608312359"),
    ]
    first = runs[0]
    assert first.called is True
    assert first.skip_reason is None
    assert first.total_count == 3
    assert first.pages_fetched == 1
    assert first.response_count == 3
    assert first.filter_passed == 1  # 픽스처 3건 중 SW 관련 + 첨부 있는 공고
    assert first.new_saved == 1
    assert all(r.called is False and r.skip_reason == "limit_reached" for r in runs[1:])
    assert len({r.run_id for r in runs}) == 1  # 같은 실행은 같은 run_id


def test_collect_runs_counts_duplicates_as_not_new(tmp_path: Path) -> None:
    """겹치는 구간에서 같은 공고가 다시 오면 필터는 통과하지만 신규 저장은 0이다."""
    runs_path = tmp_path / "collect_runs.jsonl"

    collect_from_api(
        client=_counting_client([]),
        manifest_path=tmp_path / "manifest.jsonl",
        raw_api_dir=tmp_path / "api",
        limit=5,
        begin_date=date(2026, 6, 1),
        end_date=date(2026, 7, 31),
        runs_path=runs_path,
    )

    runs = read_runs(runs_path)
    assert [(r.called, r.filter_passed, r.new_saved) for r in runs] == [
        (True, 1, 1),
        (True, 1, 0),
    ]


def test_collect_from_api_without_runs_path_writes_nothing(tmp_path: Path) -> None:
    collect_from_api(
        client=_make_client(),
        manifest_path=tmp_path / "manifest.jsonl",
        raw_api_dir=tmp_path / "api",
    )

    assert sorted(p.name for p in tmp_path.iterdir()) == ["api", "manifest.jsonl"]


def test_main_passes_collect_runs_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_main_deps(monkeypatch, calls)

    main([])

    assert str(calls["runs_path"]) == "data/raw/collect_runs.jsonl"
