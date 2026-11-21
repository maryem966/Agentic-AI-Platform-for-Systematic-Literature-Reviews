"""
Entités : ResearchProtocol, ResearchQuestion, DateRange, ProtocolModification
Protocole de recherche versionné — toute modification est justifiée et tracée.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class DateRange(BaseModel):
    """Plage d'années pour filtrer les publications."""

    start: int | None = None  # Année de début (incluse)
    end: int | None = None    # Année de fin (incluse)

    @model_validator(mode="after")
    def validate_range(self) -> "DateRange":
        if self.start and self.end and self.start > self.end:
            raise ValueError(
                f"L'année de début ({self.start}) doit être ≤ à l'année de fin ({self.end})"
            )
        return self


class ResearchQuestion(BaseModel):
    """Question de recherche d'une SLR (ex: RQ1, RQ2...)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=10)
    category: str | None = None  # Ex: "RQ1", "population", "intervention", "outcome"


class ProtocolModification(BaseModel):
    """
    Entrée dans l'historique des modifications du protocole.
    Garantit la reproductibilité scientifique : chaque changement est justifié.
    """

    modified_at: datetime = Field(default_factory=datetime.utcnow)
    modified_by: str
    field_changed: str                # Champ modifié
    old_value: str | None = None      # Valeur avant modification
    new_value: str                    # Valeur après modification
    justification: str = Field(       # Obligatoire — traçabilité scientifique
        ..., min_length=10
    )


class ResearchProtocol(BaseModel):
    """
    Protocole de recherche versionné pour une SLR.
    Toute modification incrémente la version et nécessite une revalidation humaine.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_id: str
    version: int = 1
    research_questions: list[ResearchQuestion] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    target_databases: list[str] = Field(
        default_factory=lambda: ["openalex", "crossref", "semantic_scholar"]
    )
    date_range: DateRange | None = None
    language_filter: list[str] = Field(default_factory=list)
    validated_by: str | None = None
    validated_at: datetime | None = None
    modification_history: list[ProtocolModification] = Field(default_factory=list)

    @property
    def is_validated(self) -> bool:
        return self.validated_by is not None and self.validated_at is not None

    def validate_postconditions(self) -> tuple[bool, list[str]]:
        """
        Vérifie les postconditions du Planning Agent (Section 7.3).
        Returns (valid, list_of_errors).
        """
        errors: list[str] = []
        if len(self.research_questions) < 1:
            errors.append("Au moins 1 question de recherche est requise.")
        if len(self.inclusion_criteria) < 2:
            errors.append("Au moins 2 critères d'inclusion sont requis.")
        if len(self.exclusion_criteria) < 2:
            errors.append("Au moins 2 critères d'exclusion sont requis.")
        return len(errors) == 0, errors

    def mark_validated(self, validated_by: str) -> "ResearchProtocol":
        """Retourne le protocole marqué comme validé par un humain."""
        return self.model_copy(
            update={
                "validated_by": validated_by,
                "validated_at": datetime.utcnow(),
            }
        )

    def apply_modification(
        self,
        modified_by: str,
        field_changed: str,
        old_value: str | None,
        new_value: str,
        justification: str,
        **updated_fields: object,
    ) -> "ResearchProtocol":
        """
        Applique une modification, incrémente la version et invalide la validation.
        La revalidation humaine est requise après toute modification.
        """
        modification = ProtocolModification(
            modified_by=modified_by,
            field_changed=field_changed,
            old_value=old_value,
            new_value=new_value,
            justification=justification,
        )
        return self.model_copy(
            update={
                **updated_fields,
                "version": self.version + 1,
                "modification_history": [*self.modification_history, modification],
                "validated_by": None,   # Revalidation requise
                "validated_at": None,
            }
        )
