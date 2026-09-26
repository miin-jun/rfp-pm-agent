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
from rfp_pm_agent.eval import run_retrieval
from rfp_pm_agent.eval.qa_v2 import with_gold, write_questions_v2
from rfp_pm_agent.schemas.chunk import Chunk
from rfp_pm_agent.schemas.eval import EvalQuestionV2, QuestionResult
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
    assert results[0].reciprocal_rank is None  # 소유자 구현 대기
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
    )

    assert record.embed is not None and record.embed.model_id == "org/embed"
    assert record.embed.model_sha == "sha1" and record.embed.tei_version == "1.9.4"
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
