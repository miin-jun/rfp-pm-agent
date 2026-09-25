"""답 없음 부재 확인: absent.py <doc> 검색어1 검색어2 ... — 공백 제거 후 원문 단위별 포함 여부."""

import json
import re
import sys
from pathlib import Path

ws = re.compile(r"\s+")
d, terms = sys.argv[1], sys.argv[2:]
doc = json.loads(Path(f"data/parsed/{d}.json").read_text())
units = [(b["block_id"], b["text"]) for b in doc["blocks"]]
for r in doc["requirements"]:
    units.append((r["requirement_id"], r["text"] + "\n" + "\n".join(r["raw_fields"].values())))
for t in terms:
    nt = ws.sub("", t)
    hits = [(uid, text) for uid, text in units if nt in ws.sub("", text)]
    print(f"[{t}] {len(hits)}건")
    for uid, text in hits[:4]:
        n = ws.sub("", text)
        i = n.find(nt)
        print(f"    {uid}: …{n[max(0, i - 40) : i + 60]}…")
