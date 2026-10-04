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
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import EvalQuestionV2, QuestionResult, SearchHit
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
# bm25_search·knn_search는 소유자 구현 전이라 NotImplementedError를 낸다. 여기서는 run()이
# 그 함수를 어떤 인자로 부르고 무엇을 기록하는지(연결)만 보려고 가짜 함수로 바꿔 끼운다.

OS_ALIAS = "test_alias_for_run"
OS_INDEX = "test_index_v1_embed"
QUERY_MODEL = "org/embed@sha1"  # _embed_transport가 model_sha "sha1"을 준다


def _fake_os(chunks_file: Path, *, embedding_model: str = QUERY_MODEL, drop: int = 0) -> Any:
    from tests.fakes.fake_opensearch import FakeOpenSearch

    fake = FakeOpenSearch()
    fake.indices.create(index=OS_INDEX, body={})
    fake.indices.put_alias(index=OS_INDEX, name=OS_ALIAS)
    chunks = [
        Chunk.model_validate_json(line)
        for line in chunks_file.read_text(encoding="utf-8").splitlines()
    ]
    for chunk in chunks[drop:]:
        fake.docs(OS_INDEX)[chunk.chunk_id] = {
            "chunk_id": chunk.chunk_id,
            "content_hash": "h",
            "embedding_model": embedding_model,
            "metadata_hash": "m",
        }
    return fake


def _keyword_hits(chunks_file: Path, query: str, top_k: int) -> list[SearchHit]:
    """질문 단어가 든 청크를 파일 순서로 돌려준다 (점수 계산은 흉내 내지 않는다)."""
    chunks = [
        Chunk.model_validate_json(line)
        for line in chunks_file.read_text(encoding="utf-8").splitlines()
    ]
    words = query.split()
    return [
        SearchHit(chunk_id=c.chunk_id, doc_id=c.doc_id, score=1.0, text=c.text)
        for c in chunks
        if any(w in c.text for w in words)
    ][:top_k]


@pytest.fixture
def _os_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_INDEX_ALIAS", OS_ALIAS)


@pytest.mark.usefixtures("_os_alias")
def test_os_bm25_실행은_별칭·실제_인덱스·문서_수를_기록하고_소유자_함수를_부른다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    fake = _fake_os(chunks_file)
    calls: list[tuple[Any, str, int]] = []

    def fake_bm25(client: Any, query: str, top_k: int) -> list[SearchHit]:
        calls.append((client, query, top_k))
        return _keyword_hits(chunks_file, query, top_k)

    monkeypatch.setattr(run_retrieval, "bm25_search", fake_bm25)

    record = run_retrieval.run(
        retriever="os-bm25",
        use_rerank=False,
        qa_file=qa_file,
        chunks_file=chunks_file,
        out_dir=tmp_path / "out",
        warmup=0,
        repeats=1,
        os_client=fake,
    )

    assert record.retriever == "os-bm25" and "os-bm25" in record.run_id
    assert record.index_alias == OS_ALIAS and record.index_name == OS_INDEX
    assert record.index_doc_count == 3 and record.chunk_count == 3
    assert record.chunks_file == str(chunks_file)
    assert record.embed is None and record.tokenizer is None
    assert {c[0] for c in calls} == {fake} and {c[2] for c in calls} == {10}
    assert [c[1] for c in calls] == ["사업 예산", "사업 기간", "드론 배송"]
    assert record.scores.answerable == 2


@pytest.mark.usefixtures("_os_alias")
def test_os_모드는_인덱스와_청크_파일의_chunk_id가_다르면_실행하지_않는다(tmp_path: Path) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    with pytest.raises(ValueError, match="인덱스에 없음 1개"):
        run_retrieval.run(
            retriever="os-bm25",
            use_rerank=False,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
            os_client=_fake_os(chunks_file, drop=1),
        )


@pytest.mark.usefixtures("_os_alias")
def test_os_모드는_별칭이_인덱스_둘을_가리키면_실행하지_않는다(tmp_path: Path) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    fake = _fake_os(chunks_file)
    fake.indices.create(index="other_index", body={})
    fake.indices.put_alias(index="other_index", name=OS_ALIAS)
    with pytest.raises(ValueError, match="하나여야 한다"):
        run_retrieval.run(
            retriever="os-bm25",
            use_rerank=False,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
            os_client=fake,
        )


@pytest.mark.usefixtures("_os_alias")
def test_os_knn_실행은_질의를_query로_임베딩해_소유자_함수에_넘기고_모델을_기록한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    config = make_clients_config(embed_model_id="org/embed", embed_query_prefix="query: ")
    seen: list[dict[str, Any]] = []
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", seen))
    vectors: list[list[float]] = []

    def fake_knn(client: Any, query_vector: list[float], top_k: int) -> list[SearchHit]:
        vectors.append(query_vector)
        return _keyword_hits(chunks_file, "사업", top_k)

    monkeypatch.setattr(run_retrieval, "knn_search", fake_knn)

    record = run_retrieval.run(
        retriever="os-knn",
        use_rerank=False,
        qa_file=qa_file,
        chunks_file=chunks_file,
        out_dir=tmp_path / "out",
        compose_file=tmp_path / "없음.yml",
        warmup=0,
        repeats=1,
        config=config,
        embedder=embedder,
        revision_reader=lambda model_id: pytest.fail("model_sha가 있으면 볼륨을 읽지 않는다"),
        os_client=_fake_os(chunks_file),
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


@pytest.mark.usefixtures("_os_alias")
def test_os_knn은_인덱스_embedding_model이_질의_서버와_다르면_실행하지_않는다(
    tmp_path: Path,
) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    config = make_clients_config(embed_model_id="org/embed")
    embedder = TEIEmbeddingClient(config, transport=_embed_transport("org/embed", []))
    with pytest.raises(ValueError, match="embedding_model"):
        run_retrieval.run(
            retriever="os-knn",
            use_rerank=False,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
            config=config,
            embedder=embedder,
            revision_reader=lambda model_id: "snap",
            os_client=_fake_os(chunks_file, embedding_model="org/other@sha9"),
        )


def test_os_모드와_리랭크_조합은_아직_거부한다(tmp_path: Path) -> None:
    qa_file, chunks_file = _write_data(tmp_path)
    with pytest.raises(ValueError, match="PR ②"):
        run_retrieval.run(
            retriever="os-bm25",
            use_rerank=True,
            qa_file=qa_file,
            chunks_file=chunks_file,
            out_dir=tmp_path / "out",
            os_client=_fake_os(chunks_file),
        )


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--bm25", "bm25"), ("--dense", "dense"), ("--os-bm25", "os-bm25"), ("--os-knn", "os-knn")],
)
def test_cli_플래그가_retriever로_이어진다(
    flag: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_retrieval, "run", lambda **kwargs: captured.update(kwargs))
    assert run_retrieval.main([flag]) == 0
    assert captured["retriever"] == expected
