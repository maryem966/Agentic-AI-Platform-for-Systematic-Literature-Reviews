"""Primary-to-fallback routing for LLM provider failures."""

from __future__ import annotations

import logging
from typing import Any

from ...domain.ports.llm_port import LLMPort, LLMResponse

logger = logging.getLogger(__name__)


class FallbackLLMAdapter(LLMPort):
    """Use the local provider first, then the remote provider after a failure."""

    def __init__(self, primary: LLMPort, fallback: LLMPort) -> None:
        if primary is fallback:
            raise ValueError("primary and fallback providers must be different instances")
        self.primary = primary
        self.fallback = fallback

    @property
    def provider_name(self) -> str:
        return f"{self.primary.provider_name}->{self.fallback.provider_name}"

    @property
    def model_name(self) -> str:
        return f"{self.primary.model_name}->{self.fallback.model_name}"

    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            return await self.primary.invoke(messages, temperature, max_tokens)
        except Exception as primary_error:
            logger.warning("Primary LLM provider failed; trying fallback: %s", primary_error)
            return await self.fallback.invoke(messages, temperature, max_tokens)

    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.0,
        max_repair_attempts: int = 2,
    ) -> Any:
        try:
            return await self.primary.invoke_structured(
                messages, response_model, temperature, max_repair_attempts
            )
        except Exception as primary_error:
            logger.warning("Primary structured LLM call failed; trying fallback: %s", primary_error)
            return await self.fallback.invoke_structured(
                messages, response_model, temperature, max_repair_attempts
            )

    async def health_check(self) -> bool:
        return await self.primary.health_check() or await self.fallback.health_check()
