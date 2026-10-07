"""run_retrieval 단위 테스트 (이슈 #16). TEI는 httpx.MockTransport로 대신하고,
nvidia-smi 호출은 막는다 — 네트워크·GPU 없이 돈다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rfp_pm_agent.clients.embedding import EMBEDDING_DIM, TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import TEIRerankerClient
from rfp_pm_agent.clients.tei_revision import parse_snapshot_listing
from rfp_pm_agent.eval import run_retrieval
from rfp_pm_agent.eval.qa_v2 import with_gold, write_questions_v2
from rfp_pm_agent.ingest.chunking import load_chunks
from rfp_pm_agent.ingest.index_chunks import chunk_hashes, load_documents
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.document import Block, Document
from rfp_pm_agent.schemas.eval import (
    EvalQuestionV2,
    QuestionResult,
    RetrievalRunRecord,
    SearchHit,
)
from tests.unit.conftest import make_clients_config

CHUNK_TEXTS = {
    "b1": "사업 예산 은 1억 원 이다",
    "b2": "사업 기간 은 3개월 이다",
    "b3": "보안 서약서 를 제출 한다",
}


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_retrieval, "gpu_snapshot", lambda: None)


def _write_data(tmp_path: Path) -> tuple[Path, Path]:
    chunks = [
        Chunk(
            chunk_id=f"d:block_requirement:{sid}",
            doc_id="d",
            method="block_requirement",
            source_ids=[sid],
            text=text,
        )
        for sid, text in CHUNK_TEXTS.items()
    ]
    chunks_file = tmp_path / "block_requirement.jsonl"
    chunks_file.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    base: dict[str, Any] = {"doc_id": "d", "question": "", "answer": "답"}
    questions = [
        EvalQuestionV2(
            **base | {"question_id": "q001", "question": "사업 예산", "evidence": ["1억 원"]}
        ),
        EvalQuestionV2(
            **base
            | {
                "question_id": "q031",
                "question": "사업 기간",
                "evidence": ["3개월", "서약서"],
                "tags": ["multi_chunk"],
            }
        ),
        EvalQuestionV2(
            **base
            | {
                "question_id": "q032",
                "question": "드론 배송",
                "answer": None,
                "evidence": [],
                "tags": ["no_answer"],
                "note": "부재 확인",
            }
        ),
    ]
    qa_file = tmp_path / "qa_v2.jsonl"
    chunk_map = {"block_requirement": chunks}
    write_questions_v2(qa_file, [with_gold(q, chunk_map) for q in questions])
    return qa_file, chunks_file


def test_bm25_실행은_기록·문항별_결과·실패_파일을_남긴다(tmp_path: Path) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    out_dir = tmp_path / "out"

    record = run_retrieval.run(
        retriever="bm25",
        use_rerank=False,
        qa_file=qa_file,
        chunks_file=chunks_file,
        out_dir=out_dir,
        compose_file=tmp_path / "없음.yml",
        warmup=1,
        repeats=2,
    )

    assert record.retriever == "bm25" and record.tokenizer == "bigram"
    assert record.scores.answerable == 2 and record.scores.excluded_no_answer == 1
    assert record.search_latency is not None and record.search_latency.samples == 2 * 3
    assert record.rerank_latency is None
    assert record.embed is None and record.gpu is None
    runs = (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(runs[0])["run_id"] == record.run_id
    results = [
        QuestionResult.model_validate_json(line)
        for line in Path(record.questions_file).read_text(encoding="utf-8").splitlines()
    ]
    assert [r.question_id for r in results] == ["q001", "q031"]
    assert results[0].recall_all_at_10 is True
    # 정답 b1이 1위 → RR = 1/1, NDCG = (1/log2 2) / (1/log2 2) = 1.0 (손계산)
    assert results[0].reciprocal_rank == pytest.approx(1.0)
    assert results[0].ndcg_at_10 == pytest.approx(1.0)
    assert Path(record.failures_file).exists()


def test_bm25와_리랭크_조합은_거부한다(tmp_path: Path) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    with pytest.raises(ValueError, match="BM25"):
        run_retrieval.run(
            retriever="bm25",
            use_rerank=True,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
        )


def _vector(text: str) -> list[float]:
    """'예산'·'기간'·그 외를 서로 다른 축에 둔 가짜 임베딩."""
    axis = 0 if "예산" in text else 1 if "기간" in text else 2
    return [1.0 if i == axis else 0.0 for i in range(EMBEDDING_DIM)]


def _embed_transport(model_id: str, seen: list[dict[str, Any]]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return httpx.Response(
                200,
                json={
                    "model_id": model_id,
                    "model_sha": "sha1",
                    "version": "1.9.4",
                    "max_input_length": 8192,
                },
            )
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=[_vector(t) for t in body["inputs"]])

    return httpx.MockTransport(handle)


def _rerank_transport() -> httpx.MockTransport:
    """'서약서'가 들어간 문서에 가장 높은 점수를 준다. 응답은 점수 내림차순."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return httpx.Response(200, json={"model_id": "org/reranker", "model_sha": "r1"})
        texts: list[str] = json.loads(request.content)["texts"]
        scored = [{"index": i, "score": 1.0 if "서약서" in t else 0.1} for i, t in enumerate(texts)]
        return httpx.Response(200, json=sorted(scored, key=lambda r: -r["score"]))

    return httpx.MockTransport(handle)


def test_dense_리랭크_실행은_모델·접두어·리랭커를_기록하고_질의는_query로_보낸다(
    tmp_path: Path,
) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    config = make_clients_config(
        embed_model_id="org/embed", embed_query_prefix="query: ", embed_passage_prefix="passage: "
    )
    seen: list[dict[str, Any]] = []
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", seen))
    reranker = TEIRerankerClient(config, transport=_rerank_transport())

    record = run_retrieval.run(
        retriever="dense",
        use_rerank=True,
        qa_file=qa_file,
        chunks_file=chunks_file,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        compose_file=tmp_path / "없음.yml",
        warmup=0,
        repeats=1,
        config=config,
        embedder=embedder,
        reranker=reranker,
        revision_reader=lambda model_id: f"snap-{model_id}",
    )

    assert record.embed is not None and record.embed.model_id == "org/embed"
    assert record.embed.model_sha == "sha1" and record.embed.tei_version == "1.9.4"
    assert record.embed.snapshot_revision == "snap-org/embed"
    assert record.rerank is not None and record.rerank.snapshot_revision == "snap-org/reranker"
    assert record.empty_chunks == 0 and record.empty_chunk_ids == []
    assert record.rerank is not None and record.rerank.model_id == "org/reranker"
    assert record.rerank_n == 20 and record.top_k == 20
    assert record.query_prefix == "query: " and record.truncated_chunks == 0
    assert record.rerank_latency is not None
    assert "dense-embed+rerank-reranker" in record.run_id
    # 첫 요청은 청크 색인(passage), 그 뒤는 질의(query)
    assert seen[0]["inputs"][0].startswith("passage: ")
    assert all(body["inputs"][0].startswith("query: ") for body in seen[1:])


def test_서버_모델이_설정과_다르면_실행하지_않는다(tmp_path: Path) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    config = make_clients_config(embed_model_id="org/embed")
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/other", []))

    with pytest.raises(ValueError, match="EMBED_MODEL_ID"):
        run_retrieval.run(
            retriever="dense",
            use_rerank=False,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            config=config,
            embedder=embedder,
            revision_reader=lambda model_id: "snap",
        )


def _write_results(path: Path, hits: dict[str, bool]) -> None:
    path.write_text(
        "".join(
            QuestionResult(
                question_id=qid,
                tags=[],
                recall_all_at_5=hit,
                recall_all_at_10=hit,
                recall_frac_at_5=float(hit),
                recall_frac_at_10=float(hit),
                top10_chunk_ids=[],
            ).model_dump_json()
            + "\n"
            for qid, hit in hits.items()
        ),
        encoding="utf-8",
    )


def _record_line(run_id: str, questions_file: Path) -> str:
    return json.dumps(
        {
            "run_id": run_id,
            "ran_at": "2026-09-26T00:00:00+00:00",
            "qa_file": "qa",
            "chunks_file": "c",
            "chunk_count": 1,
            "retriever": "dense",
            "top_k": 10,
            "seed": 0,
            "scores": {"answerable": 0, "excluded_no_answer": 0, "slices": [], "no_answer": []},
            "questions_file": str(questions_file),
            "failures_file": "f",
        }
    )


def test_compare는_1위와_각_실행을_McNemar로_비교해_동률을_판정한다(tmp_path: Path) -> None:
    # A: 8문항 모두 적중. B: 앞 2개만 적중(A만 맞힌 문항 6 → p = 0.03125 < 0.05).
    # C: 7개 적중(A만 맞힌 문항 1 → p = 1.0, 동률)
    qids = [f"q{i:03d}" for i in range(1, 9)]
    files = {name: tmp_path / f"{name}.questions.jsonl" for name in "ABC"}
    _write_results(files["A"], dict.fromkeys(qids, True))
    _write_results(files["B"], {q: i < 2 for i, q in enumerate(qids)})
    _write_results(files["C"], {q: i < 7 for i, q in enumerate(qids)})
    (tmp_path / "runs.jsonl").write_text(
        "".join(_record_line(n, files[n]) + "\n" for n in "ABC"), encoding="utf-8"
    )

    rows = {row["run_id"]: row for row in run_retrieval.compare(["B", "A", "C"], tmp_path)}

    assert rows["A"]["tie_with_best"] is True and rows["A"]["p_value"] == pytest.approx(1.0)
    assert rows["B"]["best_only"] == 6 and rows["B"]["this_only"] == 0
    assert rows["B"]["p_value"] == pytest.approx(0.03125) and rows["B"]["tie_with_best"] is False
    assert rows["C"]["p_value"] == pytest.approx(1.0) and rows["C"]["tie_with_best"] is True


def test_compare는_없는_run_id를_거부한다(tmp_path: Path) -> None:
    (tmp_path / "runs.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="없는 run_id"):
        run_retrieval.compare(["X"], tmp_path)


# --- 모델 revision (TEI 볼륨의 스냅샷 해시) ---


def test_스냅샷이_하나면_그_해시를_revision으로_쓴다() -> None:
    listing = "8b418a58414668e75532ed045c22d9ca018ae2b2\n"
    assert (
        parse_snapshot_listing("nlpai-lab/KURE-v1", listing)
        == "8b418a58414668e75532ed045c22d9ca018ae2b2"
    )


@pytest.mark.parametrize("listing", ["", "aaa\nbbb\n"])
def test_스냅샷이_0개거나_2개_이상이면_추측하지_않고_거부한다(listing: str) -> None:
    with pytest.raises(ValueError, match="revision을 정할 수 없다"):
        parse_snapshot_listing("org/model", listing)


def test_revision을_못_읽으면_색인하지_않는다(tmp_path: Path) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    config = make_clients_config(embed_model_id="org/embed")
    seen: list[dict[str, Any]] = []
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", seen))

    def failing_reader(model_id: str) -> str:
        raise RuntimeError("docker 없음")

    with pytest.raises(RuntimeError, match="docker 없음"):
        run_retrieval.run(
            retriever="dense",
            use_rerank=False,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            config=config,
            embedder=embedder,
            revision_reader=failing_reader,
        )
    assert seen == []  # /embed 요청을 하나도 보내지 않았다
    assert not (tmp_path / "cache").exists()


def test_빈_청크가_있으면_임베딩_전에_실패한다(tmp_path: Path) -> None:
    """#62: 빈 청크는 청킹에서 버린다. 남아 있으면 영벡터로 채우지 않고 멈춘다."""
    qa_file, chunks_file = _write_data(tmp_path)
    empty = Chunk(
        chunk_id="d:block_requirement:b9",
        doc_id="d",
        method="block_requirement",
        source_ids=["b9"],
        text="",
    )
    with chunks_file.open("a", encoding="utf-8") as f:
        f.write(empty.model_dump_json() + "\n")
    config = make_clients_config(embed_model_id="org/embed")
    seen: list[dict[str, Any]] = []
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", seen))

    with pytest.raises(ValueError, match="d:block_requirement:b9"):
        run_retrieval.run(
            retriever="dense",
            use_rerank=False,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            compose_file=tmp_path / "없음.yml",
            warmup=0,
            repeats=1,
            config=config,
            embedder=embedder,
            revision_reader=lambda model_id: "snap",
        )

    assert seen == []  # /embed 요청을 하나도 보내지 않았다


# --- OpenSearch 모드 (#18 PR ①) ---
# run()이 bm25_search·knn_search를 어떤 인자로 부르고 무엇을 기록하는지(연결)만 보려고
# 두 함수를 가짜로 바꿔 끼운다. 쿼리 본문은 tests/unit/test_search_opensearch.py가 본다.

OS_ALIAS = "test_alias_for_run"
OS_INDEX = "test_index_v1_embed"
QUERY_MODEL = "org/embed@sha1"  # _embed_transport가 model_sha "sha1"을 준다


def _write_parsed(tmp_path: Path) -> Path:
    """_write_data의 청크(b1·b2·b3)가 가리키는 파싱 결과 d.json을 쓴다."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    doc = Document(
        doc_id="d",
        source_file="d.pdf",
        bid_title="테스트 사업",
        format="pdf",
        parse_status="parsed",
        has_requirements=False,
        requirement_count=0,
        declared_total=None,
        summary_ids=[],
        validation_warnings=[],
        blocks=[
            Block(block_id=sid, type="paragraph", text=text, source_order=n, pdf_page=n)
            for n, (sid, text) in enumerate(CHUNK_TEXTS.items())
        ],
        requirements=[],
    )
    (parsed_dir / "d.json").write_text(doc.model_dump_json(), encoding="utf-8")
    return parsed_dir


def _fake_os(
    chunks_file: Path,
    parsed_dir: Path,
    *,
    embedding_model: str = QUERY_MODEL,
    drop: int = 0,
    passage_prefix: str = "",
    overrides: dict[str, dict[str, str]] | None = None,
) -> Any:
    """청크 파일을 색인 모듈과 같은 규칙의 해시로 채운 가짜 인덱스. overrides로 문서 필드를 바꾼다."""
    from tests.fakes.fake_opensearch import FakeOpenSearch

    fake = FakeOpenSearch()
    fake.indices.create(index=OS_INDEX, body={})
    fake.indices.put_alias(index=OS_INDEX, name=OS_ALIAS)
    chunks = load_chunks(chunks_file)
    hashes = chunk_hashes(
        chunks, load_documents((c.doc_id for c in chunks), parsed_dir), passage_prefix
    )
    for chunk in chunks[drop:]:
        fake.docs(OS_INDEX)[chunk.chunk_id] = {
            "chunk_id": chunk.chunk_id,
            "content_hash": hashes[chunk.chunk_id].content_hash,
            "embedding_model": embedding_model,
            "metadata_hash": hashes[chunk.chunk_id].metadata_hash,
        } | (overrides or {}).get(chunk.chunk_id, {})
    return fake


def _keyword_hits(chunks_file: Path, query: str, top_k: int) -> list[SearchHit]:
    """질문 단어가 든 청크를 파일 순서로 돌려준다 (점수 계산은 흉내 내지 않는다)."""
    words = query.split()
    return [
        SearchHit(chunk_id=c.chunk_id, doc_id=c.doc_id, score=1.0, text=c.text)
        for c in load_chunks(chunks_file)
        if any(w in c.text for w in words)
    ][:top_k]


def _run_os(tmp_path: Path, retriever: Any, fake: Any, **kwargs: Any) -> RetrievalRunRecord:
    qa_file, chunks_file = tmp_path / "qa_v2.jsonl", tmp_path / "block_requirement.jsonl"
    options: dict[str, Any] = {
        "use_rerank": False,
        "qa_file": qa_file,
        "chunks_file": chunks_file,
        "parsed_dir": tmp_path / "parsed",
        "out_dir": tmp_path / "out",
        "warmup": 0,
        "repeats": 1,
        "config": make_clients_config(embed_model_id="org/embed"),
        "os_client": fake,
    }
    return run_retrieval.run(retriever=retriever, **(options | kwargs))


@pytest.fixture
def os_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(청크 파일, 파싱 결과 디렉터리). 별칭 환경변수도 테스트용으로 바꾼다."""
    monkeypatch.setenv("OPENSEARCH_INDEX_ALIAS", OS_ALIAS)
    _, chunks_file = _write_data(tmp_path)
    return chunks_file, _write_parsed(tmp_path)


def test_os_bm25_실행은_별칭·실제_인덱스·문서_수를_기록하고_소유자_함수를_부른다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    fake = _fake_os(chunks_file, parsed_dir)
    calls: list[tuple[Any, str, int]] = []

    def fake_bm25(client: Any, query: str, top_k: int) -> list[SearchHit]:
        calls.append((client, query, top_k))
        return _keyword_hits(chunks_file, query, top_k)

    monkeypatch.setattr(run_retrieval, "bm25_search", fake_bm25)

    record = _run_os(tmp_path, "os-bm25", fake)

    assert record.retriever == "os-bm25" and "os-bm25" in record.run_id
    assert record.index_alias == OS_ALIAS and record.index_name == OS_INDEX
    assert record.index_doc_count == 3 and record.chunk_count == 3
    assert record.chunks_file == str(chunks_file)
    assert record.embed is None and record.tokenizer is None
    assert {c[0] for c in calls} == {fake} and {c[2] for c in calls} == {10}
    assert [c[1] for c in calls] == ["사업 예산", "사업 기간", "드론 배송"]
    assert record.scores.answerable == 2


def test_os_모드는_인덱스와_청크_파일의_chunk_id가_다르면_실행하지_않는다(
    tmp_path: Path, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    with pytest.raises(ValueError, match="인덱스에 없음 1개"):
        _run_os(tmp_path, "os-bm25", _fake_os(chunks_file, parsed_dir, drop=1))


def test_os_모드는_별칭이_인덱스_둘을_가리키면_실행하지_않는다(
    tmp_path: Path, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    fake = _fake_os(chunks_file, parsed_dir)
    fake.indices.create(index="other_index", body={})
    fake.indices.put_alias(index="other_index", name=OS_ALIAS)
    with pytest.raises(ValueError, match="하나여야 한다"):
        _run_os(tmp_path, "os-bm25", fake)


def test_os_모드는_인덱스_text가_청크_파일과_다르면_실행하지_않는다(
    tmp_path: Path, os_data: tuple[Path, Path]
) -> None:
    """chunk_id는 같아도 파서가 바뀐 뒤 다시 색인하지 않았으면 content_hash가 다르다 (#18 리뷰 권장 3)."""
    chunks_file, parsed_dir = os_data
    stale = {"d:block_requirement:b2": {"content_hash": "옛-text-해시"}}
    with pytest.raises(ValueError, match=r"content_hash.*1개.*b2"):
        _run_os(tmp_path, "os-bm25", _fake_os(chunks_file, parsed_dir, overrides=stale))


def test_os_모드는_접두어가_색인_때와_다르면_실행하지_않는다(
    tmp_path: Path, os_data: tuple[Path, Path]
) -> None:
    """content_hash는 접두어를 포함한 임베딩 입력의 해시라 EMBED_PASSAGE_PREFIX가 달라도 걸린다."""
    chunks_file, parsed_dir = os_data
    fake = _fake_os(chunks_file, parsed_dir, passage_prefix="passage: ")
    with pytest.raises(ValueError, match=r"content_hash.*3개"):
        _run_os(tmp_path, "os-bm25", fake)


def test_os_모드는_인덱스_메타데이터가_파싱_결과와_다르면_실행하지_않는다(
    tmp_path: Path, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    stale = {"d:block_requirement:b3": {"metadata_hash": "옛-메타-해시"}}
    with pytest.raises(ValueError, match=r"metadata_hash.*1개.*b3"):
        _run_os(tmp_path, "os-bm25", _fake_os(chunks_file, parsed_dir, overrides=stale))


def test_os_knn_실행은_질의를_query로_임베딩해_소유자_함수에_넘기고_모델을_기록한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    config = make_clients_config(embed_model_id="org/embed", embed_query_prefix="query: ")
    seen: list[dict[str, Any]] = []
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", seen))
    vectors: list[list[float]] = []

    def fake_knn(client: Any, query_vector: list[float], top_k: int) -> list[SearchHit]:
        vectors.append(query_vector)
        return _keyword_hits(chunks_file, "사업", top_k)

    monkeypatch.setattr(run_retrieval, "knn_search", fake_knn)

    record = _run_os(
        tmp_path,
        "os-knn",
        _fake_os(chunks_file, parsed_dir),
        compose_file=tmp_path / "없음.yml",
        config=config,
        embedder=embedder,
        revision_reader=lambda model_id: pytest.fail("model_sha가 있으면 볼륨을 읽지 않는다"),
    )

    assert record.retriever == "os-knn" and "os-knn-embed" in record.run_id
    assert record.embed is not None and record.embed.snapshot_revision == "sha1"
    assert record.query_prefix == "query: " and record.index_name == OS_INDEX
    # 질의 3개 모두 query 접두어로 임베딩됐고, 그 벡터가 그대로 knn_search에 갔다
    assert [body["inputs"][0] for body in seen] == [
        "query: 사업 예산",
        "query: 사업 기간",
        "query: 드론 배송",
    ]
    assert vectors == [_vector(q) for q in ["사업 예산", "사업 기간", "드론 배송"]]


def test_os_knn은_인덱스_embedding_model이_질의_서버와_다르면_실행하지_않는다(
    tmp_path: Path, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    config = make_clients_config(embed_model_id="org/embed")
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", []))
    with pytest.raises(ValueError, match="embedding_model"):
        _run_os(
            tmp_path,
            "os-knn",
            _fake_os(chunks_file, parsed_dir, embedding_model="org/other@sha9"),
            config=config,
            embedder=embedder,
            revision_reader=lambda model_id: "snap",
        )


@pytest.mark.parametrize("retriever", ["os-bm25", "os-knn"])
def test_os_bm25·os_knn과_리랭크_조합은_거부한다(
    tmp_path: Path, os_data: tuple[Path, Path], retriever: str
) -> None:
    """리랭크는 --dense와 --os-hybrid에만 붙는다 (#18 PR ②)."""
    chunks_file, parsed_dir = os_data
    with pytest.raises(ValueError, match="dense.*os-hybrid"):
        _run_os(tmp_path, retriever, _fake_os(chunks_file, parsed_dir), use_rerank=True)


# --- os-hybrid (#18 PR ②) ---
# hybrid_search·bm25_search·knn_search를 가짜로 바꿔 끼우고 run()의 연결과 기록만 본다.
# 앱 RRF(rrf_fuse)는 진짜 함수를 쓴다 — 겹침 수가 실제 계산 규칙으로 나오는지 보려는 것이다.

B1, B2, B3 = (f"d:block_requirement:{sid}" for sid in ("b1", "b2", "b3"))


def _hits(chunks_file: Path, chunk_ids: list[str]) -> list[SearchHit]:
    by_id = {c.chunk_id: c for c in load_chunks(chunks_file)}
    return [
        SearchHit(chunk_id=cid, doc_id="d", score=1.0, text=by_id[cid].text) for cid in chunk_ids
    ]


class _HybridFakes:
    """os-hybrid가 부르는 세 검색 함수의 가짜. 부른 인자를 순서대로 남긴다.

    - hybrid: "사업 기간"이면 [b1, b3], 그 외는 질문 단어가 든 청크(_keyword_hits)
    - bm25: 질문 단어가 든 청크, knn: 빈 목록 → 앱 RRF 상위 = 질문 단어가 든 청크
    그래서 앱 RRF 상위 10 겹침은 q001 2개({b1,b2}∩{b1,b2}), q031 1개({b1,b3}∩{b1,b2})다.
    """

    def __init__(self, chunks_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.chunks_file = chunks_file
        self.hybrid_calls: list[tuple[Any, str, list[float], int]] = []
        self.bm25_calls: list[tuple[str, int]] = []
        self.knn_calls: list[tuple[list[float], int]] = []
        monkeypatch.setattr(run_retrieval, "hybrid_search", self.hybrid)
        monkeypatch.setattr(run_retrieval, "bm25_search", self.bm25)
        monkeypatch.setattr(run_retrieval, "knn_search", self.knn)

    def hybrid(
        self, client: Any, query: str, query_vector: list[float], top_k: int
    ) -> list[SearchHit]:
        self.hybrid_calls.append((client, query, query_vector, top_k))
        if query == "사업 기간":
            return _hits(self.chunks_file, [B1, B3])[:top_k]
        return _keyword_hits(self.chunks_file, query, top_k)

    def bm25(self, client: Any, query: str, top_k: int) -> list[SearchHit]:
        self.bm25_calls.append((query, top_k))
        return _keyword_hits(self.chunks_file, query, top_k)

    def knn(self, client: Any, query_vector: list[float], top_k: int) -> list[SearchHit]:
        self.knn_calls.append((query_vector, top_k))
        return []


def _run_hybrid(tmp_path: Path, os_data: tuple[Path, Path], **kwargs: Any) -> RetrievalRunRecord:
    chunks_file, parsed_dir = os_data
    config = make_clients_config(embed_model_id="org/embed")
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", []))
    options: dict[str, Any] = {
        "compose_file": tmp_path / "없음.yml",
        "config": config,
        "embedder": embedder,
        "revision_reader": lambda model_id: f"snap-{model_id}",
    }
    return _run_os(tmp_path, "os-hybrid", _fake_os(chunks_file, parsed_dir), **(options | kwargs))


def test_os_hybrid_실행은_질의_벡터를_hybrid_search에_넘기고_RRF_조건을_기록한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path]
) -> None:
    fakes = _HybridFakes(os_data[0], monkeypatch)

    record = _run_hybrid(tmp_path, os_data)

    assert record.retriever == "os-hybrid" and "os-hybrid-embed" in record.run_id
    assert record.rrf_rank_constant == 60 and record.candidates == 50
    assert record.top_k == 10 and record.rerank is None and record.rerank_n is None
    assert record.embed is not None and record.embed.snapshot_revision == "sha1"
    assert record.index_alias == OS_ALIAS and record.index_name == OS_INDEX
    # 측정 반복의 검색: 57문항 전체(여기선 3개)를 top_k=10으로, 질의 벡터를 그대로 넘긴다
    measured = fakes.hybrid_calls[:3]
    assert [(c[1], c[3]) for c in measured] == [
        ("사업 예산", 10),
        ("사업 기간", 10),
        ("드론 배송", 10),
    ]
    assert [c[2] for c in measured] == [_vector(q) for q in ["사업 예산", "사업 기간", "드론 배송"]]
    assert record.search_latency is not None and record.search_latency.samples == 3
    results = run_retrieval.load_question_results(Path(record.questions_file))
    assert results["q031"].top10_chunk_ids == [B1, B3]  # 리랭크 없으면 hybrid 순서 그대로


def test_os_hybrid는_답_있는_문항마다_앱_RRF_상위10_겹침_수를_기록한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path]
) -> None:
    fakes = _HybridFakes(os_data[0], monkeypatch)

    record = _run_hybrid(tmp_path, os_data)

    results = {
        r.question_id: r
        for r in run_retrieval.load_question_results(Path(record.questions_file)).values()
    }
    assert results["q001"].app_rrf_overlap_at_10 == 2
    assert results["q031"].app_rrf_overlap_at_10 == 1
    # 집계는 questions.jsonl과 같은 문항(답 있는 51)으로 — 답 없는 q032(겹침 0)는 빠진다
    assert record.app_rrf_overlap_at_10_mean == pytest.approx(1.5)
    assert record.app_rrf_overlap_at_10_min == 1
    # 앱 RRF 입력은 검색기별 후보 50, k-NN에는 같은 질의 벡터
    assert fakes.bm25_calls == [("사업 예산", 50), ("사업 기간", 50)]
    assert fakes.knn_calls == [(_vector("사업 예산"), 50), (_vector("사업 기간"), 50)]
    # 겹침 계산의 hybrid 호출은 측정 반복 뒤에 따로 하고 지연 표본에 넣지 않는다
    assert [(c[1], c[3]) for c in fakes.hybrid_calls[3:]] == [("사업 예산", 10), ("사업 기간", 10)]
    assert record.search_latency is not None and record.search_latency.samples == 3


@pytest.mark.parametrize("rerank_n", [10, 20, 50])
def test_os_hybrid_리랭크는_hybrid_상위_N개를_리랭크하고_N을_기록한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path], rerank_n: int
) -> None:
    fakes = _HybridFakes(os_data[0], monkeypatch)
    config = make_clients_config(embed_model_id="org/embed")

    record = _run_hybrid(
        tmp_path,
        os_data,
        use_rerank=True,
        rerank_n=rerank_n,
        reranker=TEIRerankerClient(config, transport=_rerank_transport()),
    )

    assert f"os-hybrid-embed+rerank-reranker-n{rerank_n}" in record.run_id
    assert record.rerank is not None and record.rerank.model_id == "org/reranker"
    assert record.rerank.snapshot_revision == "snap-org/reranker"
    assert record.rerank_n == rerank_n and record.top_k == rerank_n
    assert record.rrf_rank_constant == 60 and record.candidates == 50
    assert record.rerank_latency is not None and record.rerank_latency.samples == 3
    assert [c[3] for c in fakes.hybrid_calls[:3]] == [rerank_n] * 3
    # q031 hybrid 순서는 [b1, b3]이고 리랭커가 '서약서'(b3)를 1위로 올렸다 — 채점은 리랭크 순서로 한다
    results = run_retrieval.load_question_results(Path(record.questions_file))
    assert results["q031"].top10_chunk_ids == [B3, B1]
    # 겹침은 리랭크 전 hybrid 상위 10으로 잰다
    assert results["q031"].app_rrf_overlap_at_10 == 1
    assert record.app_rrf_overlap_at_10_min == 1


@pytest.mark.parametrize("rerank_n", [10, 20, 50])
def test_os_hybrid_리랭크_채점_결과는_서비스_함수_hybrid_rerank_search와_같다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path], rerank_n: int
) -> None:
    """평가는 지연을 나눠 재려고 hybrid_search + rerank_hits를 따로 부른다. ADR-0004의 서비스 기본값은
    hybrid_rerank_search(rerank_n, top_k=10)이므로, 같은 후보·같은 리랭커에서 상위 10이 같아야 한다."""
    from rfp_pm_agent.search import hybrid as hybrid_module

    fakes = _HybridFakes(os_data[0], monkeypatch)
    monkeypatch.setattr(hybrid_module, "hybrid_search", fakes.hybrid)
    config = make_clients_config(embed_model_id="org/embed")
    reranker = TEIRerankerClient(config, transport=_rerank_transport())

    record = _run_hybrid(tmp_path, os_data, use_rerank=True, rerank_n=rerank_n, reranker=reranker)

    results = run_retrieval.load_question_results(Path(record.questions_file))
    for question_id, question in [("q001", "사업 예산"), ("q031", "사업 기간")]:
        service = hybrid_module.hybrid_rerank_search(
            _fake_os(*os_data), reranker, question, _vector(question), rerank_n, 10
        )
        assert results[question_id].top10_chunk_ids == [h.chunk_id for h in service]


@pytest.mark.parametrize("rerank_n", [0, 9, 51])
def test_os_hybrid_리랭크_N이_10_50_밖이면_실행하지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path], rerank_n: int
) -> None:
    """N<10이면 R@10을 N개로 채점하게 되고, N>50이면 받을 후보가 없다 (#18 리뷰 권장 4)."""
    fakes = _HybridFakes(os_data[0], monkeypatch)
    config = make_clients_config(embed_model_id="org/embed")
    with pytest.raises(ValueError, match="rerank_n"):
        _run_hybrid(
            tmp_path,
            os_data,
            use_rerank=True,
            rerank_n=rerank_n,
            reranker=TEIRerankerClient(config, transport=_rerank_transport()),
        )
    assert fakes.hybrid_calls == []


def test_os_hybrid는_인덱스_embedding_model이_질의_서버와_다르면_실행하지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path]
) -> None:
    fakes = _HybridFakes(os_data[0], monkeypatch)
    chunks_file, parsed_dir = os_data
    with pytest.raises(ValueError, match="embedding_model"):
        _run_hybrid(
            tmp_path,
            os_data,
            os_client=_fake_os(chunks_file, parsed_dir, embedding_model="org/other@sha9"),
        )
    assert fakes.hybrid_calls == []


def test_os_hybrid_아닌_실행은_RRF_기록_칸이_비어_있다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    monkeypatch.setattr(
        run_retrieval, "bm25_search", lambda c, q, k: _keyword_hits(chunks_file, q, k)
    )
    record = _run_os(tmp_path, "os-bm25", _fake_os(chunks_file, parsed_dir))
    assert record.rrf_rank_constant is None and record.candidates is None
    assert record.app_rrf_overlap_at_10_mean is None and record.app_rrf_overlap_at_10_min is None
    results = run_retrieval.load_question_results(Path(record.questions_file))
    assert all(r.app_rrf_overlap_at_10 is None for r in results.values())


# --- 기본 출력 위치 (#18 리뷰 권장 2) ---


@pytest.mark.parametrize(
    ("retriever", "expected"),
    [
        ("bm25", "data/eval/results/model_selection"),
        ("dense", "data/eval/results/model_selection"),
        ("os-bm25", "data/eval/results/hybrid"),
        ("os-knn", "data/eval/results/hybrid"),
        ("os-hybrid", "data/eval/results/hybrid"),
    ],
)
def test_기본_출력_위치는_os_모드면_hybrid_아니면_model_selection(
    retriever: Any, expected: str
) -> None:
    assert run_retrieval.default_out_dir(retriever) == Path(expected)


def test_out_dir를_주지_않은_os_실행은_hybrid_기본_위치에_기록한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_data: tuple[Path, Path]
) -> None:
    chunks_file, parsed_dir = os_data
    hybrid = tmp_path / "hybrid_default"
    monkeypatch.setattr(run_retrieval, "DEFAULT_HYBRID_OUT_DIR", hybrid)
    monkeypatch.setattr(
        run_retrieval, "bm25_search", lambda c, q, k: _keyword_hits(chunks_file, q, k)
    )

    record = _run_os(tmp_path, "os-bm25", _fake_os(chunks_file, parsed_dir), out_dir=None)

    assert Path(record.questions_file).parent == hybrid
    assert (hybrid / "runs.jsonl").exists()


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--bm25", "bm25"),
        ("--dense", "dense"),
        ("--os-bm25", "os-bm25"),
        ("--os-knn", "os-knn"),
        ("--os-hybrid", "os-hybrid"),
    ],
)
def test_cli_플래그가_retriever로_이어지고_out_dir는_run이_정한다(
    flag: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_retrieval, "run", lambda **kwargs: captured.update(kwargs))
    assert run_retrieval.main([flag]) == 0
    assert captured["retriever"] == expected
    assert captured["out_dir"] is None


def test_cli_parsed_dir와_out_dir가_run으로_이어진다(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_retrieval, "run", lambda **kwargs: captured.update(kwargs))
    run_retrieval.main(["--os-bm25", "--parsed-dir", "/p", "--out-dir", "/o"])
    assert captured["parsed_dir"] == Path("/p") and captured["out_dir"] == Path("/o")


@pytest.mark.parametrize("flag", ["--dense", "--os-hybrid"])
def test_cli_리랭크는_dense와_os_hybrid에_붙는다(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_retrieval, "run", lambda **kwargs: captured.update(kwargs))
    assert run_retrieval.main([flag, "--rerank", "--rerank-n", "50"]) == 0
    assert captured["use_rerank"] is True and captured["rerank_n"] == 50


@pytest.mark.parametrize("flag", ["--bm25", "--os-bm25", "--os-knn"])
def test_cli_리랭크를_다른_방식에_붙이면_인자_단계에서_거절한다(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_retrieval, "run", lambda **kwargs: pytest.fail("run을 부르면 안 된다"))
    with pytest.raises(SystemExit) as exc:
        run_retrieval.main([flag, "--rerank"])
    assert exc.value.code == 2
    assert "--rerank" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["10", "50"])
def test_cli_rerank_n은_10부터_50까지_받는다(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_retrieval, "run", lambda **kwargs: captured.update(kwargs))
    run_retrieval.main(["--os-hybrid", "--rerank", "--rerank-n", value])
    assert captured["rerank_n"] == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "1", "9", "51", "abc"])
def test_cli_rerank_n이_10_50_밖이면_인자_단계에서_거절한다(
    value: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """N<10이면 R@10을 N개 결과로 채점하게 되고(#18 리뷰 권장 4), 후보는 검색기별 50(KNN_CANDIDATES)까지라
    N>50은 리랭크할 후보가 없다 (#18 결정 5A)."""
    monkeypatch.setattr(run_retrieval, "run", lambda **kwargs: pytest.fail("run을 부르면 안 된다"))
    with pytest.raises(SystemExit) as exc:
        run_retrieval.main(["--dense", "--rerank", "--rerank-n", value])
    assert exc.value.code == 2
    assert "--rerank-n" in capsys.readouterr().err
