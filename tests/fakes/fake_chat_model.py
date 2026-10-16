"""네트워크를 쓰지 않는 LangChain 채팅 모델 (이슈 #22).

`ScriptedChatModel`은 정해 둔 `AIMessage`를 부른 순서대로 돌려주고, 호출마다 받은 메시지와 바인딩 인자
(`tools`·`tool_choice`)를 `calls`에 남긴다. 응답이 남지 않았는데 부르면 AssertionError —
`ScriptedLLMClient`(#19)와 같은 규칙이다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field


def tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    """툴 호출 1건을 담은 AI 메시지."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def answer(text: str, *, input_tokens: int = 100, output_tokens: int = 20) -> AIMessage:
    """최종 답 AI 메시지(사용량 포함)."""
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "output_token_details": {"reasoning": 3},
        },
        response_metadata={"model_name": "fake-chat"},
    )


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append({"messages": list(messages), "kwargs": kwargs})
        if len(self.calls) > len(self.responses):
            raise AssertionError(
                f"정해 둔 응답 {len(self.responses)}개를 넘겨 {len(self.calls)}번째로 불렀다"
            )
        message = self.responses[len(self.calls) - 1]
        return ChatResult(generations=[ChatGeneration(message=message)])
