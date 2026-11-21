"""
Entité : AgentExecution
Trace d'une exécution d'agent avec métriques de consommation LLM.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AgentExecution(BaseModel):
    """
    Historique d'exécution d'un agent.
    Enregistre les métriques de coût (tokens, appels LLM, durée) et le statut.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_id: str
    agent_name: str
    input_summary: str  # Résumé de l'entrée — PAS les données complètes
    output_summary: str | None = None
    status: Literal[
        "running",
        "completed",
        "failed",
        "escalated",
    ] = "running"
    llm_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    llm_provider: str  # Ex: "ollama", "groq"
    duration_ms: int = 0
    error_message: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "escalated")

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    def mark_completed(
        self,
        output_summary: str,
        duration_ms: int,
        llm_calls: int = 0,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> AgentExecution:
        return self.model_copy(
            update={
                "status": "completed",
                "output_summary": output_summary,
                "duration_ms": duration_ms,
                "llm_calls": self.llm_calls + llm_calls,
                "tokens_input": self.tokens_input + tokens_input,
                "tokens_output": self.tokens_output + tokens_output,
                "completed_at": datetime.utcnow(),
            }
        )

    def mark_failed(self, error_message: str, duration_ms: int) -> AgentExecution:
        return self.model_copy(
            update={
                "status": "failed",
                "error_message": error_message,
                "duration_ms": duration_ms,
                "completed_at": datetime.utcnow(),
            }
        )

    def mark_escalated(self, reason: str) -> AgentExecution:
        return self.model_copy(
            update={
                "status": "escalated",
                "error_message": reason,
                "completed_at": datetime.utcnow(),
            }
        )
