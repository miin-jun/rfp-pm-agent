"""이슈 #11 사전 조사: 요구사항 정의표의 실제 구조를 확인한다 (파서 구현 전).

요구사항 코드가 있는 문서 2건에서 "요구사항 정의표"가 실제로 어떻게 생겼는지
(표 1개=요구사항 1개인지, 코드가 어느 셀에 있는지, 라벨이 무엇인지, 코드가
표 밖에도 나오는지, 고유 코드 개수)를 실제 출력으로 확인한다. 파서 구현이
아니라 관찰이 목적이며, 산출물은 data/tmp/ 아래에만 둔다.

대상은 하드코딩하지 않고 data/raw/manifest.jsonl의 notice_title로 찾는다.
찾지 못하면 추측하지 않고 즉시 멈춘다.

실행: uv run python scripts/explore_requirements_table.py
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "manifest.jsonl"
RAW_API_DIR = REPO_ROOT / "data" / "raw" / "api"
HWP_DIR = REPO_ROOT / "data" / "tmp" / "hwp"
OUT_DIR = REPO_ROOT / "data" / "tmp" / "requirements_exploration"

GIVEN_CODES = ["ECR", "SFR", "PER", "INR", "DAR", "TER", "SER", "QUR", "COR", "PMR", "PSR"]
LOOSE_SEP = r"[\s\-_‐-―　]{0,2}"


def load_manifest() -> list[dict[str, Any]]:
    entries = []
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def find_manifest_entry(entries: list[dict[str, Any]], keywords: list[str]) -> dict[str, Any]:
    matches = [e for e in entries if all(k in (e.get("notice_title") or "") for k in keywords)]
    if not matches:
        titles = [e.get("notice_title") for e in entries]
        raise SystemExit(
            f"멈춤: 키워드 {keywords}에 맞는 manifest 항목을 찾지 못함.\n"
            f"manifest의 notice_title 목록: {titles}\n"
            "추측하지 않고 여기서 멈춘다 — 대상 문서를 확인해달라."
        )
    if len(matches) > 1:
        raise SystemExit(f"멈춤: 키워드 {keywords}에 맞는 항목이 여러 개: {matches}")
    return matches[0]


def resolve_source_file(doc_id: str) -> Path:
    candidates = [p for p in RAW_API_DIR.iterdir() if p.name.startswith(doc_id)]
    if not candidates:
        raise SystemExit(f"멈춤: {RAW_API_DIR}에서 doc_id={doc_id} 파일을 찾지 못함.")
    return candidates[0]


def resolve_or_convert_html(doc_id: str, hwp_path: Path) -> Path:
    candidates = [p for p in HWP_DIR.iterdir() if p.name.startswith(doc_id) and p.suffix == ".html"]
    if candidates:
        return candidates[0]

    HWP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HWP_DIR / f"{doc_id}_{hwp_path.stem}.html"
    print(f"HWP→HTML 변환 시도: {hwp_path.name}")
    result = subprocess.run(
        [
            "uvx",
            "--with",
            "six",
            "--from",
            "pyhwp",
            "hwp5html",
            "--html",
            "--output",
            str(out_path),
            str(hwp_path),
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0 or not out_path.exists():
        raise SystemExit(
            f"멈춤: hwp5html 변환 실패 (returncode={result.returncode}).\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    return out_path


def strip_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", "\x1f", html)
    text = text.replace("&#13;", "").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\x1f+", "\x1f", text)
    return text.strip("\x1f")


def code_pattern(codes: list[str]) -> re.Pattern[str]:
    alt = "|".join(codes)
    return re.compile(rf"(?:{alt}){LOOSE_SEP}\d{{1,4}}")


def analyze_hwp_html(html_path: Path) -> dict[str, Any]:
    html = html_path.read_text(encoding="utf-8")
    plain_full = strip_tags(html).replace("\x1f", " ")

    tables_raw = re.findall(r"<table.*?</table>", html, re.DOTALL)

    given_pattern = code_pattern(GIVEN_CODES)
    broad_pattern = re.compile(r"[A-Z]{2,6}-\d{2,4}")

    definition_tables = []
    summary_table_info = None
    for tbl in tables_raw:
        plain = strip_tags(tbl)
        cells = [c.strip() for c in plain.split("\x1f") if c.strip()]
        joined = " ".join(cells)
        codes_in_table = sorted(set(given_pattern.findall(joined)))

        # 라벨 문구("요구사항 고유번호")로 판별하지 않는다 — 실제로 SER-004/SER-005
        # 표는 "요구사항 교유번호"(오타)라 문구 매칭으로는 놓친다. 대신 "이 표
        # 안에 코드가 정확히 1개, 행이 2개 이상"이라는 구조로 판별한다.
        if len(codes_in_table) == 1 and len(re.findall(r"<tr>", tbl)) >= 2:
            rows = []
            for tr in re.findall(r"<tr>.*?</tr>", tbl, re.DOTALL):
                tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                rowspans = re.findall(r'rowspan="(\d+)"', tr)
                colspans = re.findall(r'colspan="(\d+)"', tr)
                cell_texts = [strip_tags(td).replace("\x1f", " ").strip() for td in tds]
                rows.append({"cells": cell_texts, "rowspans": rowspans, "colspans": colspans})
            code_match = given_pattern.search(joined)
            definition_tables.append(
                {
                    "code": code_match.group(0) if code_match else None,
                    "rows": rows,
                    "row_count": len(rows),
                }
            )
        elif len(codes_in_table) >= 2 and summary_table_info is None:
            header_tr = re.search(r"<tr>.*?</tr>", tbl, re.DOTALL)
            header_cells = []
            if header_tr:
                tds = re.findall(r"<td[^>]*>(.*?)</td>", header_tr.group(0), re.DOTALL)
                header_cells = [strip_tags(td).replace("\x1f", " ").strip() for td in tds]
            summary_table_info = {
                "distinct_codes": len(codes_in_table),
                "header_cells": header_cells,
                "raw_len": len(plain),
            }

    given_matches = given_pattern.findall(plain_full)
    broad_matches = broad_pattern.findall(plain_full)
    given_counts = Counter(given_matches)
    broad_counts = Counter(broad_matches)

    return {
        "total_tables": len(tables_raw),
        "definition_tables": definition_tables,
        "summary_table_info": summary_table_info,
        "given_code_matches_total": len(given_matches),
        "given_code_unique": len(given_counts),
        "given_code_counts": given_counts,
        "broad_prefixes": sorted({m.split("-")[0] for m in broad_matches}),
        "broad_code_counts": broad_counts,
    }


def analyze_pdf(pdf_path: Path) -> dict[str, Any]:
    doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    n_pages = doc.page_count
    full_text_parts = []
    page_tables: dict[int, list[Any]] = {}
    for i in range(n_pages):
        page = doc[i]
        text = page.get_text()  # type: ignore[no-untyped-call]
        full_text_parts.append(text)
        finder = page.find_tables()  # type: ignore[no-untyped-call]
        tables = list(finder.tables)
        if tables:
            page_tables[i + 1] = [t.extract() for t in tables]
    full_text = "".join(full_text_parts)

    given_pattern = code_pattern(GIVEN_CODES)
    broad_pattern = re.compile(r"[A-Z]{2,6}-\d{2,4}")
    given_matches = given_pattern.findall(full_text)
    broad_matches = broad_pattern.findall(full_text)
    given_counts = Counter(given_matches)
    broad_counts = Counter(broad_matches)

    definition_tables = []
    summary_table_info = None
    for page_no, tables in page_tables.items():
        for matrix in tables:
            joined = " ".join(str(c) for row in matrix for c in row if c)
            codes_in_table = sorted(set(given_pattern.findall(joined)))
            has_def_labels = any(
                cell
                and (
                    "요구사항 분류" in str(cell)
                    or "고유\n번호" in str(cell)
                    or "요구사항" in str(cell)
                    and "고유번호" in str(cell)
                )
                for row in matrix
                for cell in row
            )
            if has_def_labels and len(codes_in_table) == 1:
                definition_tables.append(
                    {"page": page_no, "code": codes_in_table[0], "matrix": matrix}
                )
            elif len(codes_in_table) >= 2 and summary_table_info is None:
                summary_table_info = {
                    "page": page_no,
                    "header_row": matrix[0] if matrix else None,
                    "distinct_codes": len(codes_in_table),
                    "rows": len(matrix),
                    "cols": len(matrix[0]) if matrix else 0,
                }

    doc.close()  # type: ignore[no-untyped-call]

    return {
        "n_pages": n_pages,
        "definition_tables": definition_tables,
        "summary_table_info": summary_table_info,
        "given_code_matches_total": len(given_matches),
        "given_code_unique": len(given_counts),
        "given_code_counts": given_counts,
        "broad_prefixes": sorted({m.split("-")[0] for m in broad_matches}),
        "broad_code_counts": broad_counts,
    }


def main() -> None:
    entries = load_manifest()

    target1_entry = find_manifest_entry(entries, ["연구행정", "AI", "플랫폼"])
    target2_entry = find_manifest_entry(entries, ["천안시", "스마트"])

    target1_source = resolve_source_file(target1_entry["doc_id"])
    target1_html = resolve_or_convert_html(target1_entry["doc_id"], target1_source)
    target2_pdf = resolve_source_file(target2_entry["doc_id"])

    print(f"대상 1 (HWP→HTML): {target1_html.relative_to(REPO_ROOT)}")
    print(f"대상 2 (PDF): {target2_pdf.relative_to(REPO_ROOT)}")

    result1 = analyze_hwp_html(target1_html)
    result2 = analyze_pdf(target2_pdf)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "target1_analysis.json").write_text(
        json.dumps(
            {
                "total_tables": result1["total_tables"],
                "summary_table_info": result1["summary_table_info"],
                "given_code_unique": result1["given_code_unique"],
                "given_code_matches_total": result1["given_code_matches_total"],
                "broad_prefixes": result1["broad_prefixes"],
                "given_code_counts": dict(result1["given_code_counts"]),
                "definition_table_count": len(result1["definition_tables"]),
                "first_3_definition_tables": result1["definition_tables"][:3],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "target2_analysis.json").write_text(
        json.dumps(
            {
                "n_pages": result2["n_pages"],
                "summary_table_info": result2["summary_table_info"],
                "given_code_unique": result2["given_code_unique"],
                "given_code_matches_total": result2["given_code_matches_total"],
                "broad_prefixes": result2["broad_prefixes"],
                "given_code_counts": dict(result2["given_code_counts"]),
                "definition_table_count": len(result2["definition_tables"]),
                "first_3_definition_tables": result2["definition_tables"][:3],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("대상 1 (HWP→HTML) 요약")
    print("=" * 60)
    print(f"- 전체 <table> 개수: {result1['total_tables']}")
    print(f"- 요구사항 정의표(1건=표 1개) 개수: {len(result1['definition_tables'])}")
    print(f"- 요약표 정보: {result1['summary_table_info']}")
    print(f"- 주어진 11개 코드 기준 고유 코드 수: {result1['given_code_unique']}")
    print(f"- 주어진 11개 코드 기준 총 매치 수: {result1['given_code_matches_total']}")
    print(f"- 느슨한 스캔(A-Z 2~6자-숫자)으로 발견된 접두어: {result1['broad_prefixes']}")
    non_2x_1 = {k: v for k, v in result1["given_code_counts"].items() if v != 2}
    print(f"- 2회가 아닌 코드(표 밖 등장 후보): {non_2x_1 or '없음'}")

    print("\n" + "=" * 60)
    print("대상 2 (PDF) 요약")
    print("=" * 60)
    print(f"- 전체 페이지 수: {result2['n_pages']}")
    print(f"- 요구사항 정의표(1건=표 1개) 개수: {len(result2['definition_tables'])}")
    print(f"- 요약표 정보: {result2['summary_table_info']}")
    print(f"- 주어진 11개 코드 기준 고유 코드 수: {result2['given_code_unique']}")
    print(f"- 주어진 11개 코드 기준 총 매치 수: {result2['given_code_matches_total']}")
    print(f"- 느슨한 스캔(A-Z 2~6자-숫자)으로 발견된 접두어: {result2['broad_prefixes']}")
    non_2x_2 = {k: v for k, v in result2["given_code_counts"].items() if v != 2}
    print(f"- 2회가 아닌 코드(표 밖 등장 후보): {non_2x_2 or '없음'}")

    print(f"\n전체 결과: {OUT_DIR.relative_to(REPO_ROOT)}/target{{1,2}}_analysis.json")


if __name__ == "__main__":
    main()
