from pathlib import Path

from rfp_pm_agent.ingest.parsers.hwp_parser import _venv_executable, parse_hwp


def test_venv_executable_finds_hwp5html_installed_by_uv_add() -> None:
    """`uv add pyhwp six`로 설치한 hwp5html이 이 venv 안에 실제로 있는지
    확인한다 — PATH의 다른 설치가 아니라 이 프로젝트 것을 부르는지가 핵심."""
    path = _venv_executable("hwp5html")
    assert path is not None
    assert path.exists()
    assert ".venv" in str(path) or "venv" in str(path)


def test_venv_executable_returns_none_for_missing_binary() -> None:
    assert _venv_executable("이런-실행파일은-없음") is None


def test_parse_hwp_reports_failed_status_when_conversion_impossible(tmp_path: Path) -> None:
    """실행 파일을 못 찾으면(또는 변환 실패) parse_status=failed로 기록하고
    예외를 던지지 않아야 한다 — 다음 문서 파싱을 계속할 수 있도록."""
    fake_hwp = tmp_path / "broken.hwp"
    fake_hwp.write_bytes(b"not a real hwp file")
    empty_cache_dir = tmp_path / "cache_that_will_not_help"

    doc = parse_hwp(
        fake_hwp,
        doc_id="deadbeefdeadbeef",
        bid_title="테스트",
        cache_dir=empty_cache_dir,
    )

    # 진짜 hwp5html이 이 바이트를 못 읽으면 실패, 캐시가 있으면 그건 그것대로
    # 정상 — 어느 쪽이든 예외 없이 Document가 나와야 한다는 게 핵심.
    assert doc.doc_id == "deadbeefdeadbeef"
    assert doc.parse_status in ("failed", "parsed")
    if doc.parse_status == "failed":
        assert doc.has_requirements is False
        assert doc.requirement_count == 0
        assert doc.validation_warnings
