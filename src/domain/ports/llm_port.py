"""
Port : LLMPort
Interface abstraite pour les appels LLM (implémentée par Ollama / Groq adapters).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Réponse brute d'un appel LLM."""

    content: str
    model: str
    provider: str
    tokens_input: int = 0
    tokens_output: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


class LLMPort(ABC):
    """
    Port vers le fournisseur LLM.
    La discipline de contrôle (Section 7.4) s'applique APRÈS invoke() :
    la réponse brute est validée par Pydantic avant toute utilisation.
    """

    @abstractmethod
    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Appel LLM standard. Retourne une réponse brute."""
        ...

    @abstractmethod
    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.0,
        max_repair_attempts: int = 2,
    ) -> Any:
        """
        Appel LLM avec sortie structurée Pydantic.
        Inclut la logique de repair prompt (Section 12.1).
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nom du fournisseur : 'ollama' ou 'groq'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nom du modèle utilisé."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Vérifie que le fournisseur est disponible."""
        ...
