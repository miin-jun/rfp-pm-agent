"""이슈 #11 사전 조사: data/raw/api/ 샘플 5건의 실제 구조를 관찰한다.

일회성 조사 스크립트다. 파서 구현이 아니라 관찰 보고서 생성이 목적이며,
결과는 터미널(요약)과 docs/parsing-exploration.md(전체)에 함께 출력한다.

실행: uv run python scripts/explore_samples.py
"""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path
from shutil import which

import pdfplumber
import pymupdf as fitz

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "raw" / "api"
REPORT_PATH = REPO_ROOT / "docs" / "parsing-exploration.md"

REQUIREMENT_CODES = [
    "ECR",
    "SFR",
    "PER",
    "INR",
    "DAR",
    "TER",
    "SER",
    "QUR",
    "COR",
    "PMR",
    "PSR",
]


def find_sample_files() -> list[Path]:
    return sorted(p for p in SAMPLE_DIR.iterdir() if p.is_file())


def section_1_basic_info(files: list[Path]) -> str:
    lines = ["## 1. 파일별 기본 정보\n"]
    lines.append("| 파일명 | 확장자 | 바이트 크기 | 시그니처(앞 8바이트 hex) | 판정 |")
    lines.append("|---|---|---|---|---|")
    for f in files:
        with f.open("rb") as fh:
            header = fh.read(8)
        sig = header.hex()
        size = f.stat().st_size
        ext = f.suffix.lower()
        if ext == ".hwp":
            verdict = "OLE 복합문서 (D0CF11E0)" if sig.startswith("d0cf11e0") else "예상과 다름"
        elif ext == ".hwpx":
            verdict = "ZIP (504B)" if sig.startswith("504b") else "예상과 다름"
        elif ext == ".pdf":
            verdict = "PDF (%PDF)" if header.startswith(b"%PDF") else "예상과 다름"
        else:
            verdict = "-"
        lines.append(f"| {f.name} | {ext} | {size:,} | {sig} | {verdict} |")
    return "\n".join(lines) + "\n"


def section_2_pdf_comparison(pdf_path: Path) -> tuple[str, str]:
    """(report_markdown, full_texts_for_requirement_search) 반환."""
    lines = [f"## 2. PDF 표 추출 품질 비교 — {pdf_path.name}\n"]
    all_text_chunks: list[str] = []

    # --- pdfplumber ---
    lines.append("### pdfplumber\n")
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        lines.append(f"- 전체 페이지 수: {n_pages}\n")
        lines.append("**첫 5페이지 텍스트 앞 300자**\n")
        for i, page in enumerate(pdf.pages[:5]):
            text = page.extract_text() or ""
            all_text_chunks.append(text)
            lines.append(f"- p{i + 1}: `{text[:300]!r}`")
        lines.append("")

        total_tables = 0
        first_table_rows: list[list[str | None]] = []
        for page in pdf.pages:
            tables = page.find_tables()
            total_tables += len(tables)
            if not first_table_rows and tables:
                first_table_rows = tables[0].extract()[:3]
        # 5페이지 이후 텍스트도 요구사항 ID 검색을 위해 수집
        for page in pdf.pages[5:]:
            text = page.extract_text() or ""
            all_text_chunks.append(text)

        lines.append(f"- 탐지된 표 개수(전체 페이지): {total_tables}")
        lines.append("- 첫 번째 표 앞 3행:")
        for row in first_table_rows:
            lines.append(f"  - {row}")
        lines.append("")

    pdfplumber_text = "\n".join(all_text_chunks)

    # --- PyMuPDF ---
    lines.append("### PyMuPDF (fitz)\n")
    doc = fitz.open(pdf_path)  # type: ignore[no-untyped-call]
    n_pages_fitz = doc.page_count
    lines.append(f"- 전체 페이지 수: {n_pages_fitz}\n")
    lines.append("**첫 5페이지 텍스트 앞 300자**\n")
    fitz_chunks: list[str] = []
    for i in range(min(5, n_pages_fitz)):
        text = doc[i].get_text()  # type: ignore[no-untyped-call]
        fitz_chunks.append(text)
        lines.append(f"- p{i + 1}: `{text[:300]!r}`")
    lines.append("")

    total_tables_fitz = 0
    first_table_rows_fitz: list[list[str | None]] = []
    for i in range(n_pages_fitz):
        fitz_page = doc[i]
        finder = fitz_page.find_tables()  # type: ignore[no-untyped-call]
        tables = list(finder.tables)
        total_tables_fitz += len(tables)
        if not first_table_rows_fitz and tables:
            first_table_rows_fitz = tables[0].extract()[:3]
    for i in range(5, n_pages_fitz):
        fitz_chunks.append(doc[i].get_text())  # type: ignore[no-untyped-call]

    lines.append(f"- 탐지된 표 개수(전체 페이지): {total_tables_fitz}")
    lines.append("- 첫 번째 표 앞 3행:")
    for row in first_table_rows_fitz:
        lines.append(f"  - {row}")
    lines.append("")

    fitz_text = "\n".join(fitz_chunks)
    doc.close()  # type: ignore[no-untyped-call]

    return "\n".join(lines), pdfplumber_text + "\n" + fitz_text


def section_3_hwpx_structure(hwpx_path: Path) -> tuple[str, str]:
    """(report_markdown, section0_text) 반환."""
    lines = [f"## 3. HWPX 내부 구조 — {hwpx_path.name}\n"]
    with zipfile.ZipFile(hwpx_path) as zf:
        names = zf.namelist()
        lines.append("### ZIP 내부 파일 목록\n")
        for n in names:
            lines.append(f"- {n}")
        lines.append("")

        section_candidates = [n for n in names if n.endswith("Contents/section0.xml")]
        if not section_candidates:
            section_candidates = [n for n in names if "section0.xml" in n]

        if not section_candidates:
            lines.append("**Contents/section0.xml 없음**\n")
            return "\n".join(lines), ""

        section_name = section_candidates[0]
        raw = zf.read(section_name).decode("utf-8", errors="replace")

        tag_pattern = re.compile(r"<\s*/?\s*([\w:-]+)")
        tags = tag_pattern.findall(raw)
        tag_counts = Counter(tags).most_common(20)

        lines.append(f"### {section_name} — 태그 빈도 상위 20개\n")
        lines.append("| 태그 | 등장 횟수 |")
        lines.append("|---|---|")
        for tag, count in tag_counts:
            lines.append(f"| {tag} | {count} |")
        lines.append("")

        table_tag_candidates = [
            t for t, _ in tag_counts if "tbl" in t.lower() or "table" in t.lower()
        ]
        lines.append(f"### 표 관련 태그 후보: {table_tag_candidates}\n")

        first_table_xml = ""
        for tag in table_tag_candidates:
            match = re.search(rf"<[^>]*{re.escape(tag)}[^>]*>", raw)
            if match:
                start = match.start()
                first_table_xml = raw[start : start + 500]
                break

        if first_table_xml:
            lines.append(
                f"**첫 번째 표 XML 앞 500자 (태그: {table_tag_candidates[0] if table_tag_candidates else '?'})**\n"
            )
            lines.append(f"```xml\n{first_table_xml}\n```\n")
        else:
            lines.append("**표 태그를 찾지 못함**\n")

        return "\n".join(lines), raw


def section_4_hwp_readability(hwp_files: list[Path]) -> str:
    lines = ["## 4. HWP 3건 — 현재 읽을 수 있는지 확인\n"]

    tools = {name: which(name) for name in ("libreoffice", "soffice", "hwp5txt")}
    lines.append("### 설치된 도구\n")
    for name, path in tools.items():
        lines.append(f"- `which {name}`: {path or '없음'}")
    lines.append("")

    lines.append("### HWP 파일 목록\n")
    for f in hwp_files:
        lines.append(f"- {f.name} ({f.stat().st_size:,} bytes)")
    lines.append("")

    available = next((name for name, path in tools.items() if path), None)
    if not available:
        lines.append(
            "**변환 도구가 하나도 설치되어 있지 않음 — 변환 시도 자체가 불가능. 실패로 기록.**\n"
        )
        return "\n".join(lines)

    import subprocess

    target = hwp_files[0]
    lines.append(f"### 변환 시도 — {available} 사용, 대상: {target.name}\n")
    try:
        if available == "hwp5txt":
            result = subprocess.run(
                ["hwp5txt", str(target)],
                capture_output=True,
                timeout=60,
                check=False,
            )
            output = result.stdout.decode("utf-8", errors="replace")
            ok = result.returncode == 0 and bool(output.strip())
        else:
            # libreoffice/soffice: 텍스트로 변환 후 결과 파일 읽기
            out_dir = REPO_ROOT / "scripts" / "_tmp_hwp_convert"
            out_dir.mkdir(exist_ok=True)
            result = subprocess.run(
                [
                    available,
                    "--headless",
                    "--convert-to",
                    "txt",
                    "--outdir",
                    str(out_dir),
                    str(target),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            converted = out_dir / (target.stem + ".txt")
            output = (
                converted.read_text(encoding="utf-8", errors="replace")
                if converted.exists()
                else ""
            )
            ok = result.returncode == 0 and bool(output.strip())

        lines.append(f"- 결과: {'성공' if ok else '실패'} (returncode={result.returncode})")
        if output:
            lines.append(f"- 출력 앞 300자: `{output[:300]!r}`")
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")
            lines.append(f"- stderr 앞 300자: `{stderr[:300]!r}`")
    except Exception as e:  # noqa: BLE001 — 조사 스크립트, 실패도 기록 대상
        lines.append(f"- 결과: 예외 발생 — {e!r}")

    lines.append("")
    return "\n".join(lines)


def build_requirement_pattern(code: str) -> re.Pattern[str]:
    # 구분자: 하이픈, 공백, 언더스코어, 전각 하이픈/공백, 구분자 없음
    sep = r"[\s\-_‐-―　]{0,2}"
    return re.compile(rf"{code}{sep}\d{{1,4}}", re.IGNORECASE)


def section_5_requirement_ids(texts: dict[str, str]) -> str:
    lines = ["## 5. 요구사항 ID 실제 표기\n"]

    found_examples: list[str] = []
    counts: Counter[str] = Counter()

    for source, text in texts.items():
        if not text:
            continue
        for code in REQUIREMENT_CODES:
            pattern = build_requirement_pattern(code)
            for m in pattern.finditer(text):
                counts[code] += 1
                if len(found_examples) < 20:
                    found_examples.append(f"{source}: `{m.group(0)!r}`")

    lines.append("### 원문 그대로 (최대 20개)\n")
    if found_examples:
        for ex in found_examples:
            lines.append(f"- {ex}")
    else:
        lines.append("**0건 — 대상 텍스트에서 요구사항 분류 코드를 하나도 찾지 못함.**")
    lines.append("")

    lines.append("### 코드별 등장 횟수\n")
    lines.append("| 코드 | 등장 횟수 |")
    lines.append("|---|---|")
    for code in REQUIREMENT_CODES:
        lines.append(f"| {code} | {counts.get(code, 0)} |")
    lines.append("")
    if sum(counts.values()) == 0:
        lines.append("**전체 0건.**\n")

    return "\n".join(lines)


def section_6_requirement_tables(
    pdf_first_rows: list[list[str | None]] | None, hwpx_table_xml_present: bool
) -> str:
    lines = ["## 6. 요구사항 정의 표 — 머리글 행\n"]
    if pdf_first_rows:
        header = pdf_first_rows[0] if pdf_first_rows else None
        lines.append(f"- PDF 첫 번째 표 머리글 행(추정): {header}")
    else:
        lines.append("- PDF에서 표를 찾지 못함")
    lines.append(f"- HWPX 표 태그 발견 여부: {hwpx_table_xml_present}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    files = find_sample_files()
    if len(files) != 5:
        print(f"경고: 예상한 5개 파일이 아니라 {len(files)}개 발견됨")

    pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
    hwpx_files = [f for f in files if f.suffix.lower() == ".hwpx"]
    hwp_files = [f for f in files if f.suffix.lower() == ".hwp"]

    report_parts = [
        "# 파싱 사전 조사 — 이슈 #11\n",
        "샘플 5건(.hwp 3, .hwpx 1, .pdf 1)의 실제 구조를 관찰한 결과. 파서 구현 전 참고용.\n",
    ]

    report_parts.append(section_1_basic_info(files))

    texts_for_search: dict[str, str] = {}
    pdf_first_rows: list[list[str | None]] | None = None

    if pdf_files:
        pdf_report, pdf_text = section_2_pdf_comparison(pdf_files[0])
        report_parts.append(pdf_report)
        texts_for_search[pdf_files[0].name] = pdf_text
        with pdfplumber.open(pdf_files[0]) as pdf:
            for page in pdf.pages:
                tables = page.find_tables()
                if tables:
                    pdf_first_rows = tables[0].extract()
                    break

    hwpx_table_found = False
    if hwpx_files:
        hwpx_report, hwpx_text = section_3_hwpx_structure(hwpx_files[0])
        report_parts.append(hwpx_report)
        texts_for_search[hwpx_files[0].name] = hwpx_text
        hwpx_table_found = "표 관련 태그 후보: []" not in hwpx_report

    if hwp_files:
        report_parts.append(section_4_hwp_readability(hwp_files))

    report_parts.append(section_5_requirement_ids(texts_for_search))
    report_parts.append(section_6_requirement_tables(pdf_first_rows, hwpx_table_found))

    full_report = "\n".join(report_parts)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(full_report, encoding="utf-8")

    print(f"전체 보고서 작성 완료: {REPORT_PATH.relative_to(REPO_ROOT)}\n")
    print("=" * 60)
    print("요약")
    print("=" * 60)
    print(
        f"- 조사한 파일: {len(files)}건 (pdf={len(pdf_files)}, hwpx={len(hwpx_files)}, hwp={len(hwp_files)})"
    )
    total_req = sum(
        1
        for text in texts_for_search.values()
        for code in REQUIREMENT_CODES
        for _ in build_requirement_pattern(code).finditer(text or "")
    )
    print(f"- 요구사항 ID 코드 총 매치 수: {total_req}")
    print(
        f"- HWP 변환 가능 도구: {'있음' if any(which(t) for t in ('libreoffice', 'soffice', 'hwp5txt')) else '없음'}"
    )
    print("자세한 내용은 위 보고서 파일을 확인하세요.")


if __name__ == "__main__":
    main()
