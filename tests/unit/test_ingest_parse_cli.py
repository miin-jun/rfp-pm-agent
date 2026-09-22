"""parse.py CLI 로직(이슈 #52) — 경로 결정, 건너뛰기, 예외 처리만 본다.

실제 파일 파싱은 tests/integration/test_parse_real_files.py가 맡는다. 여기서는
`parse_document`를 가짜로 바꿔 네트워크·외부 실행 파일 없이 돈다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pytest

from rfp_pm_agent.ingest import parse as parse_mod
from rfp_pm_agent.ingest.manifest import ManifestEntry, append_entry
from rfp_pm_agent.schemas.document import Document


def _entry(doc_id: str, file_name: str, source_type: Literal["api", "manual"]) -> ManifestEntry:
    return ManifestEntry(
        doc_id=doc_id,
        file_name=file_name,
        source_type=source_type,
        notice_no=None,
        notice_title=f"title-{doc_id}",
        url=None,
        file_size=1,
        sha256=doc_id * 4,
        collected_at="2026-09-22T00:00:00+00:00",
    )


def _doc(doc_id: str, path: Path, *, has_requirements: bool = False) -> Document:
    return Document(
        doc_id=doc_id,
        source_file=str(path),
        bid_title="t",
        format="pdf",
        parse_status="parsed",
        has_requirements=has_requirements,
        requirement_count=1 if has_requirements else 0,
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=[],
        requirements=[],
    )


@pytest.fixture()
def layout(tmp_path: Path) -> dict[str, Path]:
    return {
        "manifest": tmp_path / "raw" / "manifest.jsonl",
        "raw": tmp_path / "raw",
        "parsed": tmp_path / "parsed",
        "cache": tmp_path / "parsed" / "_hwp_html",
    }


def _run(layout: dict[str, Path]) -> parse_mod.ParseRunSummary:
    return parse_mod.parse_manifest(
        manifest_path=layout["manifest"],
        raw_dir=layout["raw"],
        parsed_dir=layout["parsed"],
        hwp_html_cache_dir=layout["cache"],
    )


def test_source_type_decides_raw_subdirectory(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    append_entry(layout["manifest"], _entry("aaaa", "aaaa_a.pdf", "api"))
    append_entry(layout["manifest"], _entry("bbbb", "b.pdf", "manual"))
    seen: dict[str, Path] = {}

    def fake(path: str | Path, *, doc_id: str, bid_title: str, **_: object) -> Document:
        seen[doc_id] = Path(path)
        return _doc(doc_id, Path(path))

    monkeypatch.setattr(parse_mod, "parse_document", fake)
    _run(layout)

    assert seen["aaaa"] == layout["raw"] / "api" / "aaaa_a.pdf"
    assert seen["bbbb"] == layout["raw"] / "manual" / "b.pdf"


def test_result_saved_as_doc_id_json(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    append_entry(layout["manifest"], _entry("aaaa", "aaaa_a.pdf", "api"))
    monkeypatch.setattr(
        parse_mod,
        "parse_document",
        lambda path, *, doc_id, **_: _doc(doc_id, Path(path), has_requirements=True),
    )
    _run(layout)

    saved = Document.model_validate_json((layout["parsed"] / "aaaa.json").read_text("utf-8"))
    assert saved.doc_id == "aaaa"
    assert saved.has_requirements is True


def test_existing_json_is_skipped(layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    append_entry(layout["manifest"], _entry("aaaa", "aaaa_a.pdf", "api"))
    layout["parsed"].mkdir(parents=True)
    (layout["parsed"] / "aaaa.json").write_text("sentinel", encoding="utf-8")

    def boom(*_: object, **__: object) -> Document:
        raise AssertionError("이미 파싱한 문서를 다시 파싱함")

    monkeypatch.setattr(parse_mod, "parse_document", boom)
    summary = _run(layout)

    assert (layout["parsed"] / "aaaa.json").read_text("utf-8") == "sentinel"
    assert summary.skipped == 1


def test_exception_recorded_as_failed_and_others_continue(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    append_entry(layout["manifest"], _entry("aaaa", "aaaa_bad.hwp", "api"))
    append_entry(layout["manifest"], _entry("bbbb", "bbbb_ok.pdf", "api"))

    def fake(path: str | Path, *, doc_id: str, **_: object) -> Document:
        if doc_id == "aaaa":
            raise RuntimeError("깨진 파일")
        return _doc(doc_id, Path(path))

    monkeypatch.setattr(parse_mod, "parse_document", fake)
    summary = _run(layout)

    failed = Document.model_validate_json((layout["parsed"] / "aaaa.json").read_text("utf-8"))
    assert failed.parse_status == "failed"
    assert failed.format == "hwp"
    assert failed.source_file.endswith("aaaa_bad.hwp")
    assert any("RuntimeError" in w and "깨진 파일" in w for w in failed.validation_warnings)
    assert (layout["parsed"] / "bbbb.json").exists()
    assert summary.counts[("hwp", "failed")] == 1
    assert summary.counts[("pdf", "parsed")] == 1


def test_unknown_extension_is_logged_not_written(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """format이 Literal["pdf","hwp","hwpx"]라 Document로 기록할 수 없는 확장자."""
    append_entry(layout["manifest"], _entry("aaaa", "aaaa_x.zip", "api"))
    monkeypatch.setattr(
        parse_mod,
        "parse_document",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("지원하지 않는 확장자")),
    )
    with caplog.at_level(logging.ERROR):
        summary = _run(layout)

    assert not (layout["parsed"] / "aaaa.json").exists()
    assert summary.counts[("unknown", "failed")] == 1
    assert "aaaa_x.zip" in caplog.text


def test_summary_logged_by_format_and_status(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    append_entry(layout["manifest"], _entry("aaaa", "aaaa_a.pdf", "api"))
    monkeypatch.setattr(
        parse_mod, "parse_document", lambda path, *, doc_id, **_: _doc(doc_id, Path(path))
    )
    with caplog.at_level(logging.INFO):
        parse_mod.log_summary(_run(layout))

    assert "pdf" in caplog.text and "parsed" in caplog.text


def test_has_requirements_counted_over_all_parsed_json(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """재실행해 전부 skipped여도 parsed_dir 전체 기준으로 건수를 알 수 있어야 한다."""
    append_entry(layout["manifest"], _entry("aaaa", "aaaa_a.pdf", "api"))
    append_entry(layout["manifest"], _entry("bbbb", "bbbb_b.pdf", "api"))
    monkeypatch.setattr(
        parse_mod,
        "parse_document",
        lambda path, *, doc_id, **_: _doc(doc_id, Path(path), has_requirements=doc_id == "aaaa"),
    )
    _run(layout)
    second = _run(layout)

    assert second.skipped == 2
    assert parse_mod.count_has_requirements(layout["parsed"]) == 1
