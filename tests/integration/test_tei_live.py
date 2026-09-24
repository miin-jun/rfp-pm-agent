"""로컬 TEI 서버(docker compose의 tei-embed·tei-rerank)를 실제로 부른다.

실행: `docker compose up -d tei-embed tei-rerank` 후
`uv run pytest tests/integration -m tei -q`.

TEI가 꺼져 있으면 skip하지 않고 실패한다 — `-m tei`로 골라 실행했는데 통과로
보이면 아무것도 검증하지 않은 셈이기 때문이다(이슈 #14 결정). 설정은
`ClientsConfig.from_env()`로 읽어, `.env`의 base_url·모델 ID가 실제로 코드까지
도달하는지도 함께 확인한다.
"""

from __future__ import annotations

import math

import httpx
import pytest

from rfp_pm_agent.clients.embedding import EMBEDDING_DIM, TEIEmbeddingClient
from rfp_pm_agent.clients.reranker import TEIRerankerClient
from rfp_pm_agent.config import ClientsConfig

pytestmark = pytest.mark.tei


@pytest.fixture(scope="module")
def config() -> ClientsConfig:
    return ClientsConfig.from_env()


@pytest.fixture(scope="module")
def embedder(config: ClientsConfig) -> TEIEmbeddingClient:
    client = TEIEmbeddingClient(config)
    assert client.health(), f"TEI 임베딩 서버 응답 없음: {config.embed_base_url}"
    return client


@pytest.fixture(scope="module")
def reranker(config: ClientsConfig) -> TEIRerankerClient:
    client = TEIRerankerClient(config)
    assert client.health(), f"TEI 리랭커 서버 응답 없음: {config.rerank_base_url}"
    return client


def test_served_models_match_env_model_ids(
    config: ClientsConfig, embedder: TEIEmbeddingClient, reranker: TEIRerankerClient
) -> None:
    embed_info = httpx.get(f"{config.embed_base_url}/info").json()
    rerank_info = httpx.get(f"{config.rerank_base_url}/info").json()

    assert embed_info["model_id"] == config.embed_model_id
    assert rerank_info["model_id"] == config.rerank_model_id
    # 모델 최대 길이를 넘는 입력을 잘라서 임베딩하지 않도록 compose에서 끔
    assert embed_info["auto_truncate"] is False
    assert rerank_info["auto_truncate"] is False


def test_embedding_dim_and_determinism(embedder: TEIEmbeddingClient) -> None:
    texts = ["데이터 이관 요구사항", "보안 점검 일정"]

    first = embedder.embed(texts)
    second = embedder.embed(texts)

    assert [len(v) for v in first] == [EMBEDDING_DIM, EMBEDDING_DIM]
    assert first == second
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-3) for v in first)


def test_embedding_more_than_one_batch(config: ClientsConfig, embedder: TEIEmbeddingClient) -> None:
    n = config.tei_max_client_batch_size + 8
    texts = [f"요구사항 {i}번 문장" for i in range(n)]

    vectors = embedder.embed(texts)

    assert len(vectors) == n
    # 나눠 보낸 두 번째 묶음의 벡터가 한 건씩 보낸 결과와 같은 문장에 대응해야 한다
    single = embedder.embed([texts[-1]])[0]
    cosine = sum(a * b for a, b in zip(vectors[-1], single, strict=True))
    assert cosine > 0.999


def test_rerank_relevant_document_scores_higher(reranker: TEIRerankerClient) -> None:
    scores = reranker.rerank(
        "데이터 이관 일정", ["회의실 예약 방법", "데이터 이관은 3월에 완료한다"]
    )

    assert scores[1] > scores[0]


def test_rerank_more_than_one_batch_keeps_positions(
    config: ClientsConfig, reranker: TEIRerankerClient
) -> None:
    n = config.tei_max_client_batch_size + 8
    documents = ["회의실 예약 방법"] * n
    documents[n - 3] = "데이터 이관은 3월에 완료한다"  # 두 번째 묶음 안에 둔다

    scores = reranker.rerank("데이터 이관 일정", documents)

    assert len(scores) == n
    assert max(range(n), key=lambda i: scores[i]) == n - 3
