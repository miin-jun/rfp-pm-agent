"""네트워크를 쓰지 않는 결정적 가짜 LLM 클라이언트.

같은 입력이면 항상 같은 출력을 낸다 (마지막 user 메시지를 그대로 돌려주는 echo
방식). 응답 모양은 `rfp_pm_agent.clients.llm.LLMResponse`와 같다.
"""

from __future__ import annotations

from rfp_pm_agent.clients.llm import ChatMessage, LLMResponse


class FakeLLMClient:
    def __init__(self, *, model: str = "fake-llm") -> None:
        self._model = model

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        content = f"echo: {last_user}"
        input_tokens = sum(len(m.content.split()) for m in messages)
        output_tokens = len(content.split())
        return LLMResponse(
            content=content,
            model=model or self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
