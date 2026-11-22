"""
Modèles SQLAlchemy ORM pour PostgreSQL (Source de vérité métier).
Correspondance 1-to-1 avec les entités du Domaine.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles ORM."""
    pass


class ProjectModel(Base):
    """Table : research_projects"""
    __tablename__ = "research_projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="draft", nullable=False)
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    protocol_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    protocols: Mapped[list[ProtocolModel]] = relationship("ProtocolModel", back_populates="project", cascade="all, delete-orphan")


class ProtocolModel(Base):
    """Table : research_protocols"""
    __tablename__ = "research_protocols"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    research_id: Mapped[str] = mapped_column(String(36), ForeignKey("research_projects.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    research_questions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    inclusion_criteria: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    exclusion_criteria: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    target_databases: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    date_range: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    language_filter: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    validated_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    modification_history: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)

    project: Mapped[ProjectModel] = relationship("ProjectModel", back_populates="protocols")


class SearchRunModel(Base):
    """Table : search_runs"""
    __tablename__ = "search_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    research_id: Mapped[str] = mapped_column(String(36), ForeignKey("research_projects.id"), nullable=False)
    protocol_id: Mapped[str] = mapped_column(String(36), nullable=False)
    protocol_version: Mapped[int] = mapped_column(Integer, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    queries: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    results_by_source: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict, nullable=False)
    duplicates_removed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    agent_execution_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


class PublicationModel(Base):
    """Table : publications"""
    __tablename__ = "publications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    research_id: Mapped[str] = mapped_column(String(36), ForeignKey("research_projects.id"), index=True, nullable=False)
    external_ids: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    journal: Mapped[str | None] = mapped_column(String(255), nullable=True)
    document_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ScreeningDecisionModel(Base):
    """Table : screening_decisions"""
    __tablename__ = "screening_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    publication_id: Mapped[str] = mapped_column(String(36), ForeignKey("publications.id"), index=True, nullable=False)
    research_id: Mapped[str] = mapped_column(String(36), ForeignKey("research_projects.id"), index=True, nullable=False)
    protocol_id: Mapped[str] = mapped_column(String(36), nullable=False)
    screening_stage: Mapped[str] = mapped_column(String(50), nullable=False)  # "title_abstract" ou "full_text"
    decision: Mapped[str] = mapped_column(String(20), nullable=False)  # "include", "exclude", "uncertain"
    exclusion_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    criteria_applied: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    agent: Mapped[str] = mapped_column(String(50), nullable=False)  # "deterministic" ou "llm"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    human_review_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class EvidenceModel(Base):
    """Table : evidence_registry"""
    __tablename__ = "evidence_registry"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    publication_id: Mapped[str] = mapped_column(String(36), ForeignKey("publications.id"), index=True, nullable=False)
    research_id: Mapped[str] = mapped_column(String(36), ForeignKey("research_projects.id"), index=True, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(100), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)  # Passage source exact
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(100), nullable=True)
    evidence_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "result", "method", etc.
    extracted_by_agent: Mapped[str] = mapped_column(String(50), nullable=False)
    extraction_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    verification_status: Mapped[str] = mapped_column(String(50), default="extracted", nullable=False)
    verification_method: Mapped[str | None] = mapped_column(String(100), nullable=True)
    human_review_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ClaimModel(Base):
    """Table : claims"""
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    rq_id: Mapped[str] = mapped_column(String(36), nullable=False)
    report_version_id: Mapped[str] = mapped_column(String(36), nullable=False)
    section: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    verification_method: Mapped[str] = mapped_column(String(50), default="semantic", nullable=False)
    contradicting_evidence_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ClaimEvidenceLinkModel(Base):
    """Table : claim_evidence_links (Table de liaison N-N)"""
    __tablename__ = "claim_evidence_links"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(36), ForeignKey("claims.id"), index=True, nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(36), ForeignKey("evidence_registry.id"), index=True, nullable=False)
    link_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "supports", "contradicts", etc.
    link_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_by: Mapped[str] = mapped_column(String(20), default="auto", nullable=False)
