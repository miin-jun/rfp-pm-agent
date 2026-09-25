"""#15 추가 문항 후보 추출 (시드 고정 무작위). 검색은 쓰지 않는다."""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SEED = 15
rng = random.Random(SEED)

PDF = ["304e5f8b0f11f1ea", "722630192c1100a1", "9f4904d1c8ddd260"]
HWP = [
    "5c393c1c11137fdc",
    "63d86447109810d1",
    "73c61cba5e928858",
    "944b2d1427f3ee1a",
    "c6d225f1669cfadb",
    "d3e23c59f151e223",
    "e13d2d0cd453eec2",
]
ALL = PDF + HWP

# --- 슬롯 배정 ---
slots: list[tuple[str, str]] = [("table", d) for d in PDF]
slots += [("table", d) for d in rng.sample(HWP, 3)]
slots += [("no_answer", d) for d in rng.sample(ALL, 6)]
count = Counter(d for _, d in slots)
rest = ["paraphrase"] * 6 + ["exact"] * 2 + ["multi_chunk"] * 6
rng.shuffle(rest)
for t in rest:
    order = ALL[:]
    rng.shuffle(order)
    # 개수가 가장 적은 문서, 같은 유형이 없는 문서 우선
    order.sort(key=lambda d: (count[d], (t, d) in slots))
    d = order[0]
    slots.append((t, d))
    count[d] += 1
assert sum(count.values()) == 26 and all(2 <= count[d] <= 3 for d in ALL), count

# --- 후보 ---
ws = re.compile(r"\s+")
v1_sources: dict[str, set[str]] = {}
for line in Path("data/eval/qa_v2.jsonl").read_text().splitlines():
    q = json.loads(line)
    # 초안을 뽑을 때 qa_v2에는 v1 30문항만 있었다. 같은 조건이어야 시드 15로 같은 후보가 나온다
    if q["question_id"] > "q030":
        continue
    for per in q["gold_chunk_ids"]["block_requirement"]:
        for cid in per:
            v1_sources.setdefault(q["doc_id"], set()).add(cid.split(":")[2])

REQ_TABLE = re.compile(r"요구사항\s*(고유\s*번호|분류|명칭)")
TOC = re.compile(r"(·{3,}|\.{5,}|…{2,})")


def units(doc: dict, kind: str) -> list[tuple[str, str]]:
    out = []
    skip = v1_sources.get(doc["doc_id"], set())
    if kind == "table":
        for b in doc["blocks"]:
            t = b.get("table") or []
            if b["type"] != "table" or b["block_id"] in skip:
                continue
            if len(t) >= 2 and max(len(r) for r in t) >= 2 and not REQ_TABLE.search(b["text"]):
                out.append((b["block_id"], b["text"]))
        return out
    if kind != "exact":
        for b in doc["blocks"]:
            if b["type"] == "paragraph" and b["block_id"] not in skip:
                n = ws.sub("", b["text"])
                if len(n) >= 30 and not TOC.search(b["text"]):
                    out.append((b["block_id"], b["text"]))
    for r in doc["requirements"]:
        if r["requirement_id"] not in skip:
            out.append((r["requirement_id"], r["text"]))
    return out


if __name__ == "__main__":
    per_slot = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"SEED={SEED}")
    for i, (t, d) in enumerate(sorted(slots, key=lambda s: (s[1], s[0]))):
        print(f"SLOT {i:02d} {t} {d}")
    if per_slot == 0:
        sys.exit()
    for t, d in sorted(slots, key=lambda s: (s[1], s[0])):
        if t == "no_answer":
            continue
        doc = json.loads(Path(f"data/parsed/{d}.json").read_text())
        cands = units(doc, "table" if t == "table" else ("exact" if t == "exact" else "text"))
        r2 = random.Random(f"{SEED}-{t}-{d}")
        r2.shuffle(cands)
        print(f"\n######## {t} {d} {doc['bid_title'][:40]} (후보 {len(cands)})")
        for uid, text in cands[:per_slot]:
            print(f"--- [{uid}] {text[:600]!r}")
