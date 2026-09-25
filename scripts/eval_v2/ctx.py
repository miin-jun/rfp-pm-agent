"""읽기 도우미: ctx <doc> <unit> [n] — 앞뒤 n개 단위 / toc <doc> — 제목 같은 블록 목록."""

import json
import re
import sys
from pathlib import Path


def ordered(doc: dict) -> list[tuple[int, str, str, str]]:
    items = [(b["source_order"], b["block_id"], b["type"], b["text"]) for b in doc["blocks"]]
    items += [
        (r["source_order"], r["requirement_id"], "req", r["text"]) for r in doc["requirements"]
    ]
    return sorted(items)


HEAD = re.compile(
    r"^\s*([0-9]+\.|[0-9]+\)|[가-하]\.|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|제\s*[0-9]+\s*[장절조]|□|■|◯|○|<|【|\[)"
)

cmd, d = sys.argv[1], sys.argv[2]
doc = json.loads(Path(f"data/parsed/{d}.json").read_text())
items = ordered(doc)
if cmd == "ctx":
    uid, n = sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1
    i = next(k for k, it in enumerate(items) if it[1] == uid)
    for it in items[max(0, i - n) : i + n + 1]:
        mark = ">>" if it[1] == uid else "  "
        print(f"{mark} [{it[1]} {it[2]}] {it[3][:1500]!r}")
elif cmd == "toc":
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 70
    for it in items:
        t = it[3].strip()
        if it[2] == "req":
            print(f"[{it[1]}] REQ {t.splitlines()[2][:width] if len(t.splitlines()) > 2 else ''}")
        elif HEAD.match(t) and len(t) < 80:
            print(f"[{it[1]} {it[2]}] {t[:width]!r}")
        elif it[2] == "table":
            print(f"[{it[1]} table] {t[:40]!r}")
