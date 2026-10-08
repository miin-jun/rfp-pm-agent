"""search_documents를 실제 서버 셋으로 부른다 (#18 PR ③, `-m tei`).

실행: `docker compose up -d opensearch tei-embed tei-rerank`(별칭이 #81 이후 인덱스를 가리키는 상태) 후
`uv run pytest tests/integration -m tei -q`. OpenSearch도 필요하지만 마커는 `tei` 하나다 — `-m integration`만
돌릴 때(TEI 없이) 실패하지 않게 하려는 것이다. 서버가 꺼져 있으면 skip하지 않고 실패한다(test_tei_live.py와 같다).
"""

from __future__ import annotations

import pytest

from rfp_pm_agent.search.hybrid import DEFAULT_TOP_K
from rfp_pm_agent.tools.search_documents import SearchDeps, search_documents

pytestmark = pytest.mark.tei

QUERY = "SFR-013 요구사항은 어떤 업무를 지원하나요?"


@pytest.fixture(scope="module")
def deps() -> SearchDeps:
    return SearchDeps.from_env()


def test_search_documents_returns_ranked_hits_with_citation_fields(deps: SearchDeps) -> None:
    hits = search_documents(QUERY, deps=deps)

    assert len(hits) == DEFAULT_TOP_K
    assert [h.rank for h in hits] == list(range(1, DEFAULT_TOP_K + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert all(h.bid_title and h.format in ("pdf", "hwp", "hwpx") for h in hits)
    # 요구사항 ID 가산(ADR-0003)이 서비스 경로에도 살아 있다
    assert "SFR-013" in {h.requirement_id for h in hits}


def test_search_documents_doc_filter_and_unknown_doc(deps: SearchDeps) -> None:
    first = search_documents(QUERY, deps=deps)[0]

    filtered = search_documents(QUERY, top_k=5, doc_ids=[first.doc_id], deps=deps)
    assert filtered and {h.doc_id for h in filtered} == {first.doc_id}

    with pytest.raises(ValueError, match="no-such-doc"):
        search_documents(QUERY, doc_ids=[first.doc_id, "no-such-doc"], deps=deps)
