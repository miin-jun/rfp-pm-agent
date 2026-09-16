"""이슈 #11 사전 조사 2단계: 요구사항 분류 코드 0건이 추출 실패 때문인지 확인한다.

scripts/explore_samples.py에서 PDF·HWPX 텍스트 전체에 요구사항 분류 코드
(ECR/SFR/...)가 0건으로 나왔다. 이 스크립트는 그 0건이 "문서에 실제로 없어서"인지
"텍스트 추출이 부실해서"인지를 가르기 위한 확인 결과를 만든다. 파서 구현이 아니라
관찰·확인이 목적이며, 원인 판단은 하지 않는다.

산출물:
- docs/extraction-verification.md — 전체 확인 결과
- data/tmp/extracted/{doc_id}_{method}.txt — 추출한 텍스트 전문 (data/는 git 추적 제외)

실행: uv run python scripts/verify_extraction.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pdfplumber
import pymupdf as fitz

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "raw" / "api"
EXTRACT_DIR = REPO_ROOT / "data" / "tmp" / "extracted"
REPORT_PATH = REPO_ROOT / "docs" / "extraction-verification.md"

LOW_CHAR_THRESHOLD = 300

KEYWORDS = [
    "요구사항",
    "요구사항정의",
    "과업",
    "과업내용",
    "과업범위",
    "제안요청",
    "고유번호",
    "요구사항명",
    "기능요구",
    "성능요구",
    "인터페이스요구",
    "보안요구",
    "산출물",
]

NUMBERING_PATTERN = re.compile(
    r"[①-⑳]|[가-힣]\.(?=\s|$)|\d+(?:\.\d+)+\.?|\d+\.(?=\s|[가-힣A-Za-z])|[A-Z]{2,6}-\d+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+"
)


def find_sample_files() -> list[Path]:
    return sorted(p for p in SAMPLE_DIR.iterdir() if p.is_file())


def doc_id_of(path: Path) -> str:
    return path.name.split("_", 1)[0]


def dump_text(doc_id: str, method: str, text: str) -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXTRACT_DIR / f"{doc_id}_{method}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def extract_pdf_pages_pdfplumber(pdf_path: Path) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def extract_pdf_pages_pymupdf(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)  # type: ignore[no-untyped-call]
    pages = [doc[i].get_text() for i in range(doc.page_count)]  # type: ignore[no-untyped-call]
    doc.close()  # type: ignore[no-untyped-call]
    return pages


def extract_hwpx_text(hwpx_path: Path) -> str:
    """<hp:t> 요소 내용만 추출 (마크업 제외). <hp:p> 단위로 줄바꿈."""
    import zipfile

    with zipfile.ZipFile(hwpx_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("Contents/section0.xml")]
        if not names:
            names = [n for n in zf.namelist() if "section0.xml" in n]
        raw = zf.read(names[0]).decode("utf-8", errors="replace")

    paragraphs = []
    for p_match in re.finditer(r"<hp:p\b[^>]*>(.*?)</hp:p>", raw, re.DOTALL):
        texts = re.findall(r"<hp:t\b[^>]*>(.*?)</hp:t>", p_match.group(1), re.DOTALL)
        paragraph = "".join(html.unescape(t) for t in texts)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n".join(paragraphs)


def section_1_extraction_volume(
    pdf_pages_plumber: list[str], pdf_pages_fitz: list[str], hwpx_text: str
) -> str:
    lines = ["## 1. 추출량 확인\n"]

    total_plumber = sum(len(p) for p in pdf_pages_plumber)
    total_fitz = sum(len(p) for p in pdf_pages_fitz)
    lines.append(f"- PDF 총 글자 수 (pdfplumber): {total_plumber:,}")
    lines.append(f"- PDF 총 글자 수 (PyMuPDF): {total_fitz:,}")
    lines.append(f"- HWPX 총 글자 수 (hp:t 추출): {len(hwpx_text):,}\n")

    lines.append("### PDF 페이지별 글자 수\n")
    lines.append("| 페이지 | pdfplumber | PyMuPDF | 비고 |")
    lines.append("|---|---|---|---|")
    low_pages: list[int] = []
    for i, (p_text, f_text) in enumerate(zip(pdf_pages_plumber, pdf_pages_fitz, strict=True)):
        p_len, f_len = len(p_text), len(f_text)
        note = ""
        if p_len == 0 or f_len == 0:
            note = "0자 있음"
        elif p_len < LOW_CHAR_THRESHOLD or f_len < LOW_CHAR_THRESHOLD:
            note = f"{LOW_CHAR_THRESHOLD}자 미만"
        if note:
            low_pages.append(i + 1)
        lines.append(f"| {i + 1} | {p_len} | {f_len} | {note} |")
    lines.append("")

    if low_pages:
        lines.append(f"**글자 수가 낮은 페이지: {low_pages}**\n")
    else:
        lines.append(f"**{LOW_CHAR_THRESHOLD}자 미만이거나 0자인 페이지 없음.**\n")

    return "\n".join(lines)


def section_2_full_dump_paths(paths: list[Path]) -> str:
    lines = ["## 2. 전문 덤프\n"]
    for p in paths:
        lines.append(f"- {p.relative_to(REPO_ROOT)}")
    lines.append("")
    return "\n".join(lines)


def find_keyword_contexts(text: str, keyword: str, max_hits: int = 3) -> list[str]:
    contexts = []
    for m in re.finditer(re.escape(keyword), text):
        start, end = m.start(), m.end()
        ctx = text[max(0, start - 100) : end + 100]
        contexts.append(ctx)
        if len(contexts) >= max_hits:
            break
    return contexts


def section_3_korean_keywords(sources: dict[str, str]) -> tuple[str, dict[str, list[str]]]:
    lines = ["## 3. 한국어 키워드 검색\n"]
    lines.append("| 키워드 | " + " | ".join(sources.keys()) + " | 합계 |")
    lines.append("|---|" + "---|" * len(sources) + "---|")

    all_contexts: dict[str, list[str]] = {}
    for kw in KEYWORDS:
        per_source_counts = []
        total = 0
        for text in sources.values():
            count = len(re.findall(re.escape(kw), text))
            per_source_counts.append(count)
            total += count
        lines.append(f"| {kw} | " + " | ".join(str(c) for c in per_source_counts) + f" | {total} |")

        contexts: list[str] = []
        for source_name, text in sources.items():
            if len(contexts) >= 3:
                break
            for ctx in find_keyword_contexts(text, kw, max_hits=3 - len(contexts)):
                contexts.append(f"[{source_name}] `{ctx!r}`")
        all_contexts[kw] = contexts
    lines.append("")

    lines.append("### 등장 위치 앞뒤 100자 (키워드별 최대 3건)\n")
    for kw in KEYWORDS:
        lines.append(f"**{kw}**")
        if all_contexts[kw]:
            for ctx in all_contexts[kw]:
                lines.append(f"- {ctx}")
        else:
            lines.append("- (없음)")
        lines.append("")

    return "\n".join(lines), all_contexts


def section_4_pdf_tables_detail(pdf_path: Path) -> str:
    lines = ["## 4. PDF 표 14개 정밀 확인\n"]
    table_infos: list[tuple[int, int, int, int, int, list[list[str | None]]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        table_num = 0
        for page_idx, page in enumerate(pdf.pages):
            for table in page.find_tables():
                table_num += 1
                matrix = table.extract()
                rows = len(matrix)
                cols = len(matrix[0]) if rows else 0
                non_empty = sum(
                    1
                    for row in matrix
                    for cell in row
                    if cell is not None and str(cell).strip() != ""
                )
                table_infos.append((table_num, page_idx + 1, rows, cols, non_empty, matrix))

    lines.append("| # | 페이지 | 행x열 | 비어있지 않은 셀 수 |")
    lines.append("|---|---|---|---|")
    for num, page_no, rows, cols, non_empty, _ in table_infos:
        lines.append(f"| {num} | {page_no} | {rows}x{cols} | {non_empty} |")
    lines.append("")

    top3 = sorted(table_infos, key=lambda t: t[4], reverse=True)[:3]
    lines.append("### 비어있지 않은 셀이 가장 많은 표 3개 — 실제 내용 (행 5개까지)\n")
    for num, page_no, rows, cols, non_empty, matrix in top3:
        lines.append(
            f"**표 #{num} (페이지 {page_no}, {rows}x{cols}, 비어있지 않은 셀 {non_empty}개)**\n"
        )
        for row in matrix[:5]:
            lines.append(f"- {row}")
        lines.append("")

    return "\n".join(lines)


def section_5_hwpx_tables_detail(hwpx_path: Path) -> str:
    lines = ["## 5. HWPX 표 정밀 확인 (행 수 3 이상)\n"]
    import zipfile

    with zipfile.ZipFile(hwpx_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("Contents/section0.xml")]
        if not names:
            names = [n for n in zf.namelist() if "section0.xml" in n]
        raw = zf.read(names[0]).decode("utf-8", errors="replace")

    open_positions = [m.start() for m in re.finditer(r"<hp:tbl\b", raw)]
    open_positions.append(len(raw))
    segments = [
        raw[open_positions[i] : open_positions[i + 1]] for i in range(len(open_positions) - 1)
    ]
    close_tag_count = len(re.findall(r"</hp:tbl>", raw))

    lines.append(f"- 여는 태그(`<hp:tbl`) 개수 = 실제 표 개수: {len(segments)}")
    lines.append(
        f"- 닫는 태그(`</hp:tbl>`) 개수: {close_tag_count} "
        "(참고: docs/parsing-exploration.md §3의 '태그 빈도 hp:tbl=58'은 "
        "여는/닫는 태그를 합쳐 센 값이었음 — 실제 표 개수는 그 절반)\n"
    )

    qualifying = 0
    headers: list[tuple[int, int, str]] = []
    for idx, seg in enumerate(segments, start=1):
        row_match = re.search(r'rowCnt="(\d+)"', seg)
        row_cnt = int(row_match.group(1)) if row_match else 0
        if row_cnt < 3:
            continue
        qualifying += 1

        tr_match = re.search(r"<hp:tr\b[^>]*>(.*?)</hp:tr>", seg, re.DOTALL)
        header_text = "(첫 행 없음)"
        if tr_match:
            texts = re.findall(r"<hp:t\b[^>]*>(.*?)</hp:t>", tr_match.group(1), re.DOTALL)
            header_text = " | ".join(html.unescape(t) for t in texts) if texts else "(빈 셀)"
        headers.append((idx, row_cnt, header_text))

    lines.append(f"- 행 수 3 이상인 표 개수: {qualifying}\n")
    lines.append("| # | 행 수 | 첫 행(머리글) |")
    lines.append("|---|---|---|")
    for idx, row_cnt, header_text in headers:
        lines.append(f"| {idx} | {row_cnt} | {header_text} |")
    lines.append("")

    return "\n".join(lines)


def section_6_numbering_patterns(contexts_by_keyword: dict[str, list[str]]) -> str:
    lines = ["## 6. 숫자/번호 체계 패턴 (요구사항·과업 주변)\n"]

    target_contexts = contexts_by_keyword.get("요구사항", []) + contexts_by_keyword.get("과업", [])
    if not target_contexts:
        lines.append("**'요구사항' 또는 '과업'이 발견되지 않아 패턴 추출 대상 없음.**\n")
        return "\n".join(lines)

    matches: list[str] = []
    for ctx in target_contexts:
        for match in NUMBERING_PATTERN.finditer(ctx):
            matches.append(match.group(0))
            if len(matches) >= 10:
                break
        if len(matches) >= 10:
            break

    if matches:
        lines.append("원문 그대로, 발견 순서대로:\n")
        for m in matches:
            lines.append(f"- `{m!r}`")
    else:
        lines.append("**'요구사항'/'과업' 주변에서 번호 체계 패턴을 찾지 못함.**")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    files = find_sample_files()
    pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
    hwpx_files = [f for f in files if f.suffix.lower() == ".hwpx"]

    if not pdf_files or not hwpx_files:
        print("경고: PDF 또는 HWPX 샘플을 찾지 못함")
        return

    pdf_path = pdf_files[0]
    hwpx_path = hwpx_files[0]

    pdf_pages_plumber = extract_pdf_pages_pdfplumber(pdf_path)
    pdf_pages_fitz = extract_pdf_pages_pymupdf(pdf_path)
    hwpx_text = extract_hwpx_text(hwpx_path)

    pdf_doc_id = doc_id_of(pdf_path)
    hwpx_doc_id = doc_id_of(hwpx_path)

    dump_paths = [
        dump_text(pdf_doc_id, "pdfplumber", "\n".join(pdf_pages_plumber)),
        dump_text(pdf_doc_id, "pymupdf", "\n".join(pdf_pages_fitz)),
        dump_text(hwpx_doc_id, "hwpx", hwpx_text),
    ]

    sources = {
        "PDF(pdfplumber)": "\n".join(pdf_pages_plumber),
        "PDF(PyMuPDF)": "\n".join(pdf_pages_fitz),
        "HWPX": hwpx_text,
    }

    report_parts = [
        "# 텍스트 추출 확인 — 이슈 #11 (요구사항 코드 0건 원인 가리기)\n",
        (
            "요구사항 분류 코드가 0건으로 나온 것이 추출 실패 때문인지, "
            "문서에 실제로 없어서인지 확인한 결과. 원인은 판단하지 않고 관찰만 기록한다.\n"
        ),
    ]

    report_parts.append(section_1_extraction_volume(pdf_pages_plumber, pdf_pages_fitz, hwpx_text))
    report_parts.append(section_2_full_dump_paths(dump_paths))

    section_3_report, contexts_by_keyword = section_3_korean_keywords(sources)
    report_parts.append(section_3_report)

    report_parts.append(section_4_pdf_tables_detail(pdf_path))
    report_parts.append(section_5_hwpx_tables_detail(hwpx_path))
    report_parts.append(section_6_numbering_patterns(contexts_by_keyword))

    full_report = "\n".join(report_parts)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(full_report, encoding="utf-8")

    total_plumber = sum(len(p) for p in pdf_pages_plumber)
    total_fitz = sum(len(p) for p in pdf_pages_fitz)
    keyword_total = sum(
        len(re.findall(re.escape(kw), text)) for kw in KEYWORDS for text in sources.values()
    )

    print(f"전체 확인 결과 작성 완료: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"전문 덤프: {[str(p.relative_to(REPO_ROOT)) for p in dump_paths]}\n")
    print("=" * 60)
    print("요약")
    print("=" * 60)
    print(f"- PDF 총 글자 수: pdfplumber={total_plumber:,}, PyMuPDF={total_fitz:,}")
    print(f"- HWPX 총 글자 수: {len(hwpx_text):,}")
    print(f"- 13개 한국어 키워드 총 매치 수(3개 소스 합): {keyword_total}")
    print("자세한 내용은 위 보고서 파일을 확인하세요.")


if __name__ == "__main__":
    main()
