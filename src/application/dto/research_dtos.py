"""
DTOs : Modèles de transport entre couches applicatives.
Ces modèles ne sont PAS des entités du domaine.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# --- Projet ---

class CreateResearchProjectDTO(BaseModel):
    title: str = Field(..., min_length=3, max_length=255)
    topic: str = Field(..., min_length=10)
    created_by: str


class ResearchProjectResponseDTO(BaseModel):
    id: str
    title: str
    topic: str
    status: str
    created_by: str
    protocol_id: str | None = None


# --- Protocole ---

class ResearchQuestionDTO(BaseModel):
    text: str = Field(..., min_length=10)
    category: str | None = None


class DefineProtocolDTO(BaseModel):
    research_id: str
    research_questions: list[ResearchQuestionDTO] = Field(..., min_length=1)
    inclusion_criteria: list[str] = Field(..., min_length=2)
    exclusion_criteria: list[str] = Field(..., min_length=2)
    target_databases: list[str] = Field(
        default=["openalex", "crossref", "semantic_scholar"]
    )
    start_year: int | None = None
    end_year: int | None = None
    language_filter: list[str] = Field(default_factory=list)
    defined_by: str


class ModifyProtocolDTO(BaseModel):
    research_id: str
    protocol_id: str
    field_changed: str
    new_value: str
    justification: str = Field(..., min_length=10)
    modified_by: str


# --- Validation humaine ---

class HumanValidationDTO(BaseModel):
    research_id: str
    review_type: str
    reviewer_id: str
    decision: str
    justification: str = Field(..., min_length=5)


# --- Screening ---

class ScreeningArbitrationDTO(BaseModel):
    publication_id: str
    decision: str  # "include" ou "exclude"
    exclusion_reason: str | None = None
    justification: str


# --- Requête RAG ---

class RAGQueryDTO(BaseModel):
    research_id: str
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=10, ge=1, le=50)
    strategy: str = Field(default="hybrid")  # "bm25", "dense", "hybrid", "graph"
