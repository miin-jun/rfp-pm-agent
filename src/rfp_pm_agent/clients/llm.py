"""Chat-completion LLM client.

인터페이스(Protocol)를 먼저 정의하고, OpenAI 호환 API(OpenAI 본체·vLLM·Ollama)를
호출하는 구현체를 그 아래 둔다. base_url만 바꾸면 로컬/원격 전환이 된다
(docs/tech-stack.md 5절).
"""

from __future__ import annotations

from typing import Literal, Protocol, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from rfp_pm_agent.config import ClientsConfig

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse: ...


class OpenAICompatLLMClient:
    """OpenAI 호환 `/chat/completions` 엔드포인트를 호출하는 구현체.

    `config.llm_base_url`만 바꾸면 OpenAI ↔ vLLM/Ollama(둘 다 OpenAI 호환
    API를 제공)로 전환된다. 실제 네트워크 호출은 `chat()`에서만 일어나므로,
    단위 테스트에서는 이 클래스를 만들기만 하고 `chat()`은 부르지 않는 방식으로
    (예: `client.base_url` 확인) 네트워크 없이도 base_url 배선을 검증할 수 있다.
    """

    def __init__(self, config: ClientsConfig, *, default_model: str | None = None) -> None:
        self._config = config
        self._default_model = default_model or config.llm_model_dev
        self._openai = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key or "not-needed",
            timeout=config.llm_timeout_s,
        )

    @property
    def base_url(self) -> str:
        return str(self._openai.base_url)

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        resolved_model = model or self._default_model
        response = self._openai.chat.completions.create(
            model=resolved_model,
            messages=cast(
                list[ChatCompletionMessageParam],
                [{"role": m.role, "content": m.content} for m in messages],
            ),
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
