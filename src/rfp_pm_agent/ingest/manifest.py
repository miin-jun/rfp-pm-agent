"""수집 기록 매니페스트 (`data/raw/manifest.jsonl`).

한 줄 = 파일 하나. `source_type`(api|manual)과 무관하게 `doc_id`(sha256 앞
16자)가 같으면 같은 문서로 취급해 다시 기록하지 않는다 (docs/data-design.md
0·1절). 이 파일은 `.gitignore` 예외로 커밋 대상이다 — 원본은 안 올려도
"무엇을 어디서 받았는지"는 레포에 남아야 재현 가능하다.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ManifestEntry(BaseModel):
    doc_id: str
    file_name: str
    source_type: Literal["api", "manual"]
    notice_no: str | None
    notice_title: str | None
    url: str | None
    file_size: int
    sha256: str
    collected_at: str


def compute_hashes(data: bytes) -> tuple[str, str]:
    """(doc_id, sha256_hex)를 반환한다. doc_id = sha256 앞 16자 (data-design.md 0절)."""
    digest = hashlib.sha256(data).hexdigest()
    return digest[:16], digest


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    p = Path(path)
    if not p.exists():
        return []
    entries: list[ManifestEntry] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(ManifestEntry.model_validate_json(stripped))
    return entries


def existing_doc_ids(path: str | Path) -> set[str]:
    return {entry.doc_id for entry in read_manifest(path)}


def append_entry(path: str | Path, entry: ManifestEntry) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
