"""
Entité : ResearchProject
Projet de revue systématique de littérature et son cycle de vie.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Statuts valides et transitions autorisées
PROJECT_TRANSITIONS: dict[str, list[str]] = {
    "draft": ["protocol_defined", "archived"],
    "protocol_defined": ["searching", "archived"],
    "searching": ["screening", "archived"],
    "screening": ["extracting", "archived"],
    "extracting": ["writing", "archived"],
    "writing": ["reviewing", "archived"],
    "reviewing": ["completed", "writing", "archived"],
    "completed": ["archived"],
    "archived": [],
}


class ResearchProject(BaseModel):
    """Projet SLR avec son cycle de vie complet."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=3, max_length=255)
    topic: str = Field(..., min_length=10)
    status: Literal[
        "draft",
        "protocol_defined",
        "searching",
        "screening",
        "extracting",
        "writing",
        "reviewing",
        "completed",
        "archived",
    ] = "draft"
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    protocol_id: str | None = None  # Version active du protocole

    model_config = {"frozen": False}

    def transition_to(self, new_status: str) -> ResearchProject:
        """
        Retourne un nouveau projet avec le statut mis à jour.
        Lève ValueError si la transition n'est pas autorisée.
        """
        allowed = PROJECT_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Transition invalide : '{self.status}' → '{new_status}'. "
                f"Transitions autorisées depuis '{self.status}' : {allowed}"
            )
        return self.model_copy(
            update={
                "status": new_status,
                "updated_at": datetime.utcnow(),
            }
        )

    def can_transition_to(self, new_status: str) -> bool:
        """Vérifie si la transition est autorisée sans la déclencher."""
        return new_status in PROJECT_TRANSITIONS.get(self.status, [])

    def attach_protocol(self, protocol_id: str) -> ResearchProject:
        """Attache un protocole et passe au statut 'protocol_defined'."""
        project = self.transition_to("protocol_defined")
        return project.model_copy(update={"protocol_id": protocol_id})
