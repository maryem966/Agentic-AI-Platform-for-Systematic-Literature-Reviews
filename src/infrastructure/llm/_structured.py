"""Shared JSON repair loop for structured LLM responses."""

from __future__ import annotations

from typing import Any

from ...domain.ports.llm_port import LLMPort


async def invoke_structured(
    adapter: LLMPort,
    messages: list[dict[str, str]],
    response_model: type,
    temperature: float,
    max_repair_attempts: int,
) -> Any:
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts cannot be negative")
    schema = (
        response_model.model_json_schema() if hasattr(response_model, "model_json_schema") else {}
    )
    prompt_messages = [
        *messages,
        {
            "role": "system",
            "content": "Return only valid JSON matching this schema: " + str(schema),
        },
    ]
    last_error: Exception | None = None
    for attempt in range(max_repair_attempts + 1):
        response = await adapter.invoke(prompt_messages, temperature=temperature)
        try:
            if hasattr(response_model, "model_validate_json"):
                return response_model.model_validate_json(response.content)
            raise TypeError("response_model must expose model_validate_json()")
        except Exception as error:
            last_error = error
            if attempt == max_repair_attempts:
                break
            prompt_messages = [
                *prompt_messages,
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": f"The JSON was invalid: {error}. Return corrected JSON only.",
                },
            ]
    raise ValueError(f"LLM did not produce valid structured output: {last_error}") from last_error
