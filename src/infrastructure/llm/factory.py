"""Construct the configured default local-first LLM router."""

from __future__ import annotations

from .fallback import FallbackLLMAdapter
from .groq_adapter import GroqAdapter
from .ollama_adapter import OllamaAdapter


def create_default_llm_adapter() -> FallbackLLMAdapter:
    """Configure Ollama as primary and Groq as fallback."""
    return FallbackLLMAdapter(primary=OllamaAdapter(), fallback=GroqAdapter())
