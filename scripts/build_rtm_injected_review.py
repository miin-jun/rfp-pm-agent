"""이슈 #39 사전 작업: injected 누락 정답의 사람 검수표를 만든다.

pms_truth_v2.json의 누락 정답(use_for_scoring=true, reason="injected")은 wbs_requirements 연결만
지운 것이라 비슷한 업무가 남아 있을 수 있다. 검수자가 요구사항 원문과 그 사업의 업무 전체를 보고
"비슷한 업무 없음"/"있음"을 직접 판정하도록, 판단을 끌어갈 정보(유사도·점수·정렬)는 넣지 않는다.
llm_omitted는 정답이 아니므로 넣지 않는다(ADR-0008).

입력: data/eval/pms_truth_v2.json, data/synth/pms_v2.json(업무 목록), data/parsed/<doc_id>.json(원문)
출력: data/eval/rtm_injected_review.csv, data/eval/rtm_injected_review.md(읽기용 — 판정 열 없음),
      data/eval/rtm_injected_judgment.csv(판정은 여기 한 곳에만 — 같은 순서·같은 건, no는 .md 제목 번호와 같다.
      사람이 채우므로 이미 있으면 덮어쓰지 않고 건·순서만 대조한다)
같은 입력이면 출력이 바이트 단위로 같다.

실행: uv run python scripts/build_rtm_injected_review.py
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rfp_pm_agent.schemas.document import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
TRUTH_PATH = REPO_ROOT / "data" / "eval" / "pms_truth_v2.json"
DATASET_PATH = REPO_ROOT / "data" / "synth" / "pms_v2.json"
PARSED_DIR = REPO_ROOT / "data" / "parsed"
CSV_PATH = REPO_ROOT / "data" / "eval" / "rtm_injected_review.csv"
MD_PATH = REPO_ROOT / "data" / "eval" / "rtm_injected_review.md"
JUDGMENT_PATH = REPO_ROOT / "data" / "eval" / "rtm_injected_judgment.csv"

COLUMNS = ["project_id", "requirement_id", "requirement_text", "project_wbs"]
JUDGMENT_COLUMNS = ["no", "project_id", "requirement_id", "judgment", "matched_wbs_ids", "note"]


def wbs_sort_key(wbs_id: str) -> tuple[int, ...]:
    """WBS 번호 순서("1.10"이 "1.9" 뒤)."""
    return tuple(int(part) for part in wbs_id.split("."))


def injected_ids(project: dict[str, Any]) -> list[str]:
    return [
        m["requirement_id"]
        for m in project["missing_requirements"]
        if m["reason"] == "injected" and m["use_for_scoring"]
    ]


def build_rows() -> list[dict[str, str]]:
    truth = json.loads(TRUTH_PATH.read_text("utf-8"))
    dataset = json.loads(DATASET_PATH.read_text("utf-8"))
    if dataset["as_of_date"] != truth["as_of_date"] or dataset["seed"] != truth["seed"]:
        raise SystemExit("pms_v2.json과 pms_truth_v2.json의 as_of_date·seed가 다르다")

    linked = {(x["project_id"], x["requirement_id"]) for x in dataset["wbs_requirements"]}
    rows: list[dict[str, str]] = []
    for project in truth["projects"]:
        pid = project["project_id"]
        wanted = set(injected_ids(project))
        if not wanted:
            continue
        still_linked = sorted(rid for rid in wanted if (pid, rid) in linked)
        if still_linked:
            raise SystemExit(f"{pid}: injected인데 업무 연결이 남아 있다 {still_linked}")

        items = [w for w in dataset["wbs_items"] if w["project_id"] == pid]
        items.sort(key=lambda w: wbs_sort_key(w["wbs_id"]))
        project_wbs = "\n".join(f"{w['wbs_id']} {w['name']}" for w in items)

        doc_path = PARSED_DIR / f"{project['doc_id']}.json"
        doc = Document.model_validate_json(doc_path.read_text("utf-8"))
        # 같은 ID가 두 번 나오면 처음 것 — pms.synth.build_requirements와 같은 규칙
        found: set[str] = set()
        for req in sorted(doc.requirements, key=lambda r: r.source_order):
            rid = req.requirement_id
            if rid not in wanted or rid in found:
                continue
            found.add(rid)
            rows.append(
                {
                    "project_id": pid,
                    "requirement_id": rid,
                    # 줄 끝 공백("산출정보: ")만 지운다 — pre-commit trailing-whitespace가 지우므로 재실행 결과를 맞춘다
                    "requirement_text": "\n".join(
                        line.rstrip() for line in req.text.strip().splitlines()
                    ),
                    "project_wbs": project_wbs,
                }
            )
        if found != wanted:
            raise SystemExit(f"{pid}: 원문에 없는 요구사항 {sorted(wanted - found)}")
    return rows


def render_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def render_judgment_csv(rows: list[dict[str, str]]) -> str:
    """판정만 적는 표. no는 rows 순서(1부터)이고 render_md의 제목 번호와 같다."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=JUDGMENT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for no, row in enumerate(rows, start=1):
        writer.writerow(
            {
                "no": str(no),
                "project_id": row["project_id"],
                "requirement_id": row["requirement_id"],
                "judgment": "",
                "matched_wbs_ids": "",
                "note": "",
            }
        )
    return buf.getvalue()


def render_md(rows: list[dict[str, str]]) -> str:
    lines = [
        "# RTM injected 누락 정답 검수표 (#39)",
        "",
        (
            "`data/eval/pms_truth_v2.json`의 누락 정답(`use_for_scoring: true`, "
            f'`reason: "injected"`) {len(rows)}건. 업무 연결(`wbs_requirements`)만 지운 것이라 '
            "비슷한 업무가 남아 있을 수 있다."
        ),
        "",
        (
            "판정은 `rtm_injected_judgment.csv`에서 제목의 no와 같은 행의 "
            "`judgment`(비슷한 업무 없음/있음)·`matched_wbs_ids`·`note`에 적는다. "
            "이 파일과 `rtm_injected_review.csv`는 읽기용이며 "
            "`scripts/build_rtm_injected_review.py`가 만든다."
        ),
        "",
    ]
    for no, row in enumerate(rows, start=1):
        lines += [
            f"## no {no}. {row['project_id']} / {row['requirement_id']}",
            "",
            "### 요구사항 원문",
            "",
            "```text",
            row["requirement_text"],
            "```",
            "",
            f"### {row['project_id']} 업무 목록",
            "",
            "```text",
            row["project_wbs"],
            "```",
            "",
            f"판정 기록: `rtm_injected_judgment.csv` no {no}",
            "",
        ]
    return "\n".join(lines)


def check_judgment_keys(rows: list[dict[str, str]]) -> None:
    """이미 있는 판정 파일의 (no, project_id, requirement_id)가 rows 순서와 같은지 확인한다."""
    with JUDGMENT_PATH.open(encoding="utf-8-sig", newline="") as f:
        filled = list(csv.DictReader(f))
    expected = [
        (str(no), row["project_id"], row["requirement_id"]) for no, row in enumerate(rows, start=1)
    ]
    actual = [(r["no"], r["project_id"], r["requirement_id"]) for r in filled]
    if actual != expected:
        raise SystemExit(f"{JUDGMENT_PATH.name}의 건·순서가 검수표와 다르다")


def main() -> None:
    rows = build_rows()
    CSV_PATH.write_text(render_csv(rows), encoding="utf-8-sig")
    MD_PATH.write_text(render_md(rows), encoding="utf-8")
    outputs = [CSV_PATH, MD_PATH]
    # 판정 파일은 사람이 채운 데이터다 — 있으면 덮어쓰지 않고 건·순서만 대조한다
    if JUDGMENT_PATH.exists():
        check_judgment_keys(rows)
        print(f"{JUDGMENT_PATH.relative_to(REPO_ROOT)}: 이미 있어 덮어쓰지 않음(건·순서 일치)")
    else:
        JUDGMENT_PATH.write_text(render_judgment_csv(rows), encoding="utf-8-sig")
        outputs.append(JUDGMENT_PATH)
    counts = Counter(row["project_id"] for row in rows)
    print(f"{len(rows)}건 → {', '.join(str(p.relative_to(REPO_ROOT)) for p in outputs)}")
    for pid, n in sorted(counts.items()):
        print(f"  {pid}: {n}")


if __name__ == "__main__":
    main()
