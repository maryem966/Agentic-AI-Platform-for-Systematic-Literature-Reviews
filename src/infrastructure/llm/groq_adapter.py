"""Groq implementation of LLMPort."""

from __future__ import annotations

from typing import Any

from groq import AsyncGroq

from ...config.settings import settings
from ...domain.ports.llm_port import LLMPort, LLMResponse
from ._structured import invoke_structured


class GroqAdapter(LLMPort):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model or settings.groq_default_model
        resolved_key = api_key if api_key is not None else settings.groq_api_key
        if client is None and not resolved_key:
            raise ValueError("GROQ_API_KEY is required to configure the Groq adapter")
        self.client = client or AsyncGroq(
            api_key=resolved_key,
            timeout=timeout_seconds or settings.llm_timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        result = await self.client.chat.completions.create(**kwargs)
        choice = result.choices[0]
        usage = result.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=result.model or self._model,
            provider=self.provider_name,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
        )

    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.0,
        max_repair_attempts: int = 2,
    ) -> Any:
        return await invoke_structured(
            self, messages, response_model, temperature, max_repair_attempts
        )

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
