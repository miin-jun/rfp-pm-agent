"""네트워크를 쓰지 않는 가짜 TEI 서버 (`httpx.MockTransport` 처리 함수).

실제 `TEIEmbeddingClient`에 transport로 끼워 쓴다. 클라이언트가 붙이는 접두어·나누어 보내기를
그대로 거치므로, 서버가 실제로 받은 문자열(`embed_inputs`)과 요청 수(`embed_requests`)를 셀 수 있다.
벡터는 문자열 해시로 만든 결정적 값이다(같은 문자열이면 같은 벡터).
"""

from __future__ import annotations

import json

import httpx

from rfp_pm_agent.clients.embedding import TEIEmbeddingClient
from rfp_pm_agent.config import ClientsConfig
from tests.fakes.fake_embedding import FakeEmbeddingClient


class FakeTEIServer:
    """`/info`·`/embed`만 흉내 낸다. model_sha는 TEI 1.9.4처럼 None으로 둘 수 있다."""

    def __init__(self, model_id: str = "test-embed-model", model_sha: str | None = "rev-1") -> None:
        self.model_id = model_id
        self.model_sha = model_sha
        self.embed_inputs: list[str] = []
        self.embed_requests = 0
        self.info_requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            self.info_requests += 1
            return httpx.Response(
                200,
                json={
                    "model_id": self.model_id,
                    "model_sha": self.model_sha,
                    "max_input_length": 8192,
                    "version": "fake",
                },
            )
        if request.url.path == "/embed":
            inputs: list[str] = json.loads(request.content)["inputs"]
            self.embed_requests += 1
            self.embed_inputs.extend(inputs)
            return httpx.Response(200, json=[FakeEmbeddingClient._embed_one(t) for t in inputs])
        return httpx.Response(404)

    def client(self, config: ClientsConfig) -> TEIEmbeddingClient:
        return TEIEmbeddingClient(config, transport=httpx.MockTransport(self.handler))

    def reset_counts(self) -> None:
        self.embed_inputs = []
        self.embed_requests = 0
        self.info_requests = 0
