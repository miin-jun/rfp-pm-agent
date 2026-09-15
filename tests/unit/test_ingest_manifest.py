from pathlib import Path

from rfp_pm_agent.ingest.manifest import (
    ManifestEntry,
    append_entry,
    compute_hashes,
    existing_doc_ids,
    read_manifest,
)


def test_compute_hashes_is_deterministic() -> None:
    data = b"hello rfp"

    doc_id_1, sha256_1 = compute_hashes(data)
    doc_id_2, sha256_2 = compute_hashes(data)

    assert doc_id_1 == doc_id_2
    assert sha256_1 == sha256_2
    assert len(doc_id_1) == 16
    assert sha256_1.startswith(doc_id_1)


def test_compute_hashes_differs_for_different_bytes() -> None:
    doc_id_a, _ = compute_hashes(b"file a")
    doc_id_b, _ = compute_hashes(b"file b")

    assert doc_id_a != doc_id_b


def test_read_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    entries = read_manifest(tmp_path / "does_not_exist.jsonl")

    assert entries == []


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entry = ManifestEntry(
        doc_id="abcd1234abcd1234",
        file_name="rfp.hwpx",
        source_type="manual",
        notice_no=None,
        notice_title=None,
        url=None,
        file_size=123,
        sha256="abcd1234abcd1234" + "0" * 48,
        collected_at="2026-09-15T00:00:00+00:00",
    )

    append_entry(manifest_path, entry)
    entries = read_manifest(manifest_path)

    assert entries == [entry]


def test_existing_doc_ids_ignores_source_type(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manual_entry = ManifestEntry(
        doc_id="dup0000000000000"[:16],
        file_name="a.hwpx",
        source_type="manual",
        notice_no=None,
        notice_title=None,
        url=None,
        file_size=1,
        sha256="dup0000000000000"[:16] + "1" * 48,
        collected_at="2026-09-15T00:00:00+00:00",
    )
    append_entry(manifest_path, manual_entry)

    ids = existing_doc_ids(manifest_path)

    assert "dup0000000000000"[:16] in ids
