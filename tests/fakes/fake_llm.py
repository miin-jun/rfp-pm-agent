"""네트워크를 쓰지 않는 결정적 가짜 LLM 클라이언트.

- `FakeLLMClient`: 같은 입력이면 항상 같은 출력을 낸다 (마지막 user 메시지를 그대로 돌려주는 echo 방식)
- `ScriptedLLMClient`: 정해 둔 응답을 순서대로 돌려주고 받은 메시지를 기록한다 (#19)

응답 모양은 `rfp_pm_agent.clients.llm.LLMResponse`와 같다.
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


class ScriptedLLMClient:
    """정해 둔 응답을 부른 순서대로 돌려준다. 받은 메시지와 model 인자를 `calls`에 남긴다.

    응답이 남지 않았는데 부르면 AssertionError — 예상보다 많이 부른 것을 테스트가 놓치지 않게 한다.
    토큰 수는 응답마다 같은 값(`input_tokens`·`output_tokens`·`reasoning_tokens`)이다.
    """

    def __init__(
        self,
        responses: list[str],
        *,
        model: str = "fake-llm",
        input_tokens: int = 100,
        output_tokens: int = 20,
        reasoning_tokens: int = 5,
    ) -> None:
        self._responses = list(responses)
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._reasoning_tokens = reasoning_tokens
        self.calls: list[tuple[list[ChatMessage], str | None]] = []

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        self.calls.append((list(messages), model))
        if len(self.calls) > len(self._responses):
            raise AssertionError(
                f"정해 둔 응답 {len(self._responses)}개를 넘겨 {len(self.calls)}번째로 불렀다"
            )
        return LLMResponse(
            content=self._responses[len(self.calls) - 1],
            model=model or self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            reasoning_tokens=self._reasoning_tokens,
        )
