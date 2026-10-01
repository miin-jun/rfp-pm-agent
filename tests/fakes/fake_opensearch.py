"""네트워크를 쓰지 않는 가짜 OpenSearch 클라이언트 (opensearch-py `OpenSearch`의 일부만).

색인 모듈(`ingest/index_chunks.py`)이 쓰는 호출만 흉내 낸다.
- `indices.exists / create / exists_alias / get_alias / put_alias / refresh`
- `search`: match_all + `_source` 필드 목록 + `sort` 1개 필드 + `search_after`
- `bulk`: `index`(같은 _id가 있으면 "updated", 없으면 "created")와 `delete`
  ("deleted" 또는 404 "not_found") — 응답 모양은 OpenSearch bulk 응답과 같다
- `count`

`hidden_ids`에 넣은 문서는 저장돼 있지만 `search`에는 나오지 않는다. refresh 전이라
검색에 아직 안 보이는 문서를 흉내 내, "create로 보냈는데 updated가 돌아오는" 경우를 만든다.
"""

from __future__ import annotations

import copy
from typing import Any


class FakeIndices:
    def __init__(self, store: FakeOpenSearch) -> None:
        self._store = store

    def exists(self, *, index: str) -> bool:
        return index in self._store.indexes

    def create(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        if index in self._store.indexes:
            raise ValueError(f"resource_already_exists_exception: {index}")
        self._store.indexes[index] = {
            "settings": copy.deepcopy(body.get("settings", {})),
            "mappings": copy.deepcopy(body.get("mappings", {})),
            "docs": {},
        }
        for alias in body.get("aliases", {}):
            self._store.aliases.setdefault(alias, set()).add(index)
        self._store.create_calls += 1
        return {"acknowledged": True, "index": index}

    def exists_alias(self, *, name: str) -> bool:
        return bool(self._store.aliases.get(name))

    def get_alias(self, *, name: str) -> dict[str, Any]:
        targets = self._store.aliases.get(name)
        if not targets:
            raise KeyError(name)
        return {index: {"aliases": {name: {}}} for index in sorted(targets)}

    def put_alias(self, *, index: str, name: str) -> dict[str, Any]:
        self._store.aliases.setdefault(name, set()).add(index)
        return {"acknowledged": True}

    def refresh(self, *, index: str) -> dict[str, Any]:
        self._store.refresh_calls += 1
        return {}


class FakeOpenSearch:
    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, set[str]] = {}
        self.hidden_ids: set[str] = set()
        self.indices = FakeIndices(self)
        self.bulk_calls = 0
        self.create_calls = 0
        self.refresh_calls = 0

    # --- 테스트 도우미 ---

    def resolve(self, name: str) -> str:
        if name in self.indexes:
            return name
        targets = self.aliases.get(name, set())
        if len(targets) != 1:
            raise KeyError(f"index_not_found_exception: {name}")
        return next(iter(targets))

    def docs(self, name: str) -> dict[str, dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = self.indexes[self.resolve(name)]["docs"]
        return docs

    def snapshot(self) -> dict[str, Any]:
        """인덱스 변경 여부를 비교하기 위한 깊은 복사."""
        return copy.deepcopy({"indexes": self.indexes, "aliases": self.aliases})

    # --- opensearch-py 호출 흉내 ---

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        docs = self.docs(index)
        (sort_spec,) = body["sort"]
        (sort_field,) = sort_spec
        rows = sorted(
            ((doc_id, src) for doc_id, src in docs.items() if doc_id not in self.hidden_ids),
            key=lambda row: row[1][sort_field],
        )
        after = body.get("search_after")
        if after is not None:
            rows = [row for row in rows if row[1][sort_field] > after[0]]
        rows = rows[: body["size"]]
        fields = body.get("_source")
        hits = [
            {
                "_id": doc_id,
                "_source": {k: v for k, v in src.items() if fields is None or k in fields},
                "sort": [src[sort_field]],
            }
            for doc_id, src in rows
        ]
        return {"hits": {"hits": hits}}

    def bulk(self, *, body: list[dict[str, Any]]) -> dict[str, Any]:
        self.bulk_calls += 1
        items: list[dict[str, Any]] = []
        i = 0
        while i < len(body):
            (op, meta), *_ = body[i].items()
            docs = self.docs(meta["_index"])
            doc_id = meta["_id"]
            if op == "index":
                source = body[i + 1]
                existed = doc_id in docs
                docs[doc_id] = copy.deepcopy(source)
                result = "updated" if existed else "created"
                items.append(
                    {op: {"_id": doc_id, "result": result, "status": 200 if existed else 201}}
                )
                i += 2
            elif op == "delete":
                if doc_id in docs:
                    del docs[doc_id]
                    items.append({op: {"_id": doc_id, "result": "deleted", "status": 200}})
                else:
                    items.append({op: {"_id": doc_id, "result": "not_found", "status": 404}})
                i += 1
            else:
                raise ValueError(f"가짜 bulk가 지원하지 않는 동작: {op}")
        errors = any("error" in item[next(iter(item))] for item in items)
        return {"errors": errors, "items": items}

    def count(self, *, index: str) -> dict[str, Any]:
        return {"count": len(self.docs(index))}
