"""HWP → Document 파서. 이슈 #11.

구형 HWP(바이너리)는 직접 파싱하지 않고, `hwp5html`(pyhwp, AGPLv3+)로 변환한
HTML을 파싱한다. `<table>`은 `<hp:tbl>` 대신 표준 HTML 표라 구조가 그대로
보존된다(실측: docs/parsing-exploration.md). HWP는 페이지 개념이 없어
Requirement의 pdf_page_*/printed_page_*는 항상 null이다.

라이선스 메모: pyhwp/hwp5html은 AGPLv3+다. 이 코드는 `import hwp5`를 하지 않고
**별도 프로세스로 CLI만 호출**한다 — AGPL의 파생저작물·네트워크 조항이 별개
프로세스로 실행되는 외부 도구 호출에는 일반적으로 적용되지 않는다는 실무 해석에
따른 것이며, 법률 자문은 아니다.
"""

from __future__ import annotations

import html as html_module
import re
import subprocess
import sys
from pathlib import Path

from rfp_pm_agent.ingest.parsers.common import (
    RequirementBuilder,
    collect_loose_codes,
    find_declared_total,
    table_role,
    validate_requirement_ids,
)
from rfp_pm_agent.schemas.document import Block, Document, Requirement

HWP5HTML_TIMEOUT_S = 120

_TABLE_PATTERN = re.compile(r"<table.*?</table>", re.DOTALL)
_TR_PATTERN = re.compile(r"<tr>.*?</tr>", re.DOTALL)
_TD_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")


_PARAGRAPH_END_PATTERN = re.compile(r"</p>", re.IGNORECASE)


def _strip_tags(fragment: str) -> str:
    """hwp5html의 `<p>` 사이엔 구분자가 전혀 없다(`</p><p ...>`가 바로 붙음) —
    태그를 지우기 전에 문단 경계마다 개행을 넣어 둬야 서로 다른 문단이
    글자 단위로 뭉개지지 않는다(리뷰 실측: ECR-001 등 37건 중 33건에서 발견)."""
    text = _PARAGRAPH_END_PATTERN.sub("</p>\n", fragment)
    text = _TAG_PATTERN.sub("", text)
    text = html_module.unescape(text).replace("\r", "")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _venv_executable(name: str) -> Path | None:
    """venv(`sys.executable`이 있는 디렉터리)에 설치된 실행 파일 경로. PATH의
    다른 설치가 아니라 이 프로젝트가 `uv add`한 것을 확실히 부르기 위함."""
    candidate = Path(sys.executable).parent / name
    return candidate if candidate.exists() else None


def convert_hwp_to_html(hwp_path: Path, doc_id: str, cache_dir: Path) -> Path | None:
    """`data/parsed/_hwp_html/{doc_id}.html` 캐시가 있으면 재사용하고, 없으면
    변환한다. 실패(returncode!=0, 타임아웃, 실행 파일 없음)하면 None."""
    out_path = cache_dir / f"{doc_id}.html"
    if out_path.exists():
        return out_path

    hwp5html = _venv_executable("hwp5html")
    if hwp5html is None:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [str(hwp5html), "--html", "--output", str(out_path), str(hwp_path)],
            capture_output=True,
            timeout=HWP5HTML_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0 or not out_path.exists():
        return None
    return out_path


def _table_rows(table_html: str) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for tr in _TR_PATTERN.findall(table_html):
        cells: list[str | None] = [_strip_tags(td) for td in _TD_PATTERN.findall(tr)]
        rows.append(cells)
    return rows


def _parse_html(
    raw: str,
) -> tuple[list[Block], list[Requirement], set[str], list[list[list[str | None]]]]:
    blocks: list[Block] = []
    requirements: list[Requirement] = []
    summary_ids: set[str] = set()
    all_tables: list[list[list[str | None]]] = []

    source_order = 0
    last_end = 0
    for m in _TABLE_PATTERN.finditer(raw):
        gap_text = _strip_tags(raw[last_end : m.start()])
        if gap_text:
            source_order += 1
            blocks.append(
                Block(
                    block_id=f"b{source_order:04d}",
                    type="paragraph",
                    text=gap_text,
                    source_order=source_order,
                )
            )

        table_html = m.group(0)
        rows = _table_rows(table_html)
        all_tables.append(rows)
        source_order += 1
        role, code = table_role(rows)

        if role == "definition" and code is not None:
            builder = RequirementBuilder()
            for row in rows:
                builder.add_row(row)
            requirements.append(
                Requirement(
                    requirement_id=code,
                    prefix=code[:3],
                    fields=builder.fields,
                    raw_fields=builder.raw_fields,
                    text=builder.build_text(),
                    source_order=source_order,
                )
            )
        else:
            if role == "summary":
                summary_ids |= collect_loose_codes(rows)
            blocks.append(
                Block(
                    block_id=f"b{source_order:04d}",
                    type="table",
                    text="\n".join(" | ".join(c or "" for c in row) for row in rows),
                    table=rows,
                    source_order=source_order,
                )
            )

        last_end = m.end()

    trailing = _strip_tags(raw[last_end:])
    if trailing:
        source_order += 1
        blocks.append(
            Block(
                block_id=f"b{source_order:04d}",
                type="paragraph",
                text=trailing,
                source_order=source_order,
            )
        )

    return blocks, requirements, summary_ids, all_tables


def parse_hwp(path: str | Path, *, doc_id: str, bid_title: str, cache_dir: Path) -> Document:
    hwp_path = Path(path)
    html_path = convert_hwp_to_html(hwp_path, doc_id, cache_dir)

    if html_path is None:
        return Document(
            doc_id=doc_id,
            source_file=str(path),
            bid_title=bid_title,
            format="hwp",
            parse_status="failed",
            has_requirements=False,
            requirement_count=0,
            declared_total=None,
            summary_ids=[],
            validation_warnings=["hwp5html 변환 실패 — 실행 파일 없음/타임아웃/오류"],
            blocks=[],
            requirements=[],
        )

    raw = html_path.read_text(encoding="utf-8")
    blocks, requirements, summary_ids, all_tables = _parse_html(raw)

    declared_total: int | None = None
    total_warnings: list[str] = []
    if requirements:
        # 요구사항이 없으면 declared_total은 찾지 않는다 — pdf_parser.py 참고
        # (실측: 요구사항 코드가 없는 문서에서 평가 배점표의 "합계 100"이
        # declared_total로 잘못 잡힌 사례가 있었다). all_tables는 정의표로
        # 소비된 표까지 포함한 문서 전체 표(pdf_parser.py와 범위를 맞춤).
        declared_total, total_warnings = find_declared_total(all_tables)

    requirement_ids = [r.requirement_id for r in requirements]
    validation_warnings = total_warnings + validate_requirement_ids(
        requirement_ids, summary_ids, declared_total
    )

    return Document(
        doc_id=doc_id,
        source_file=str(path),
        bid_title=bid_title,
        format="hwp",
        parse_status="parsed",
        has_requirements=len(requirements) > 0,
        requirement_count=len(requirements),
        declared_total=declared_total,
        summary_ids=sorted(summary_ids),
        validation_warnings=validation_warnings,
        blocks=blocks,
        requirements=requirements,
    )
