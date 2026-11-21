"""Ollama implementation of LLMPort."""

from __future__ import annotations

from typing import Any

from ollama import AsyncClient

from ...config.settings import settings
from ...domain.ports.llm_port import LLMPort, LLMResponse
from ._structured import invoke_structured


class OllamaAdapter(LLMPort):
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout_seconds: int | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model or settings.ollama_default_model
        self.client = client or AsyncClient(
            host=host or settings.ollama_base_url,
            timeout=timeout_seconds or settings.llm_timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        result = await self.client.chat(
            model=self._model, messages=messages, options=options, stream=False
        )
        message = result.message
        return LLMResponse(
            content=message.content,
            model=result.model or self._model,
            provider=self.provider_name,
            tokens_input=result.prompt_eval_count or 0,
            tokens_output=result.eval_count or 0,
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
            await self.client.list()
            return True
        except Exception:
            return False
