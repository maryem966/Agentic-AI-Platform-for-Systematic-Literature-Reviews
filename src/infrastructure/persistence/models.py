"""SQLAlchemy ORM mappings for the PostgreSQL schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

JSON_VALUE = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


class ResearchProjectORM(Base):
    __tablename__ = "research_projects"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    protocol_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_protocols.id", ondelete="SET NULL")
    )


class ResearchProtocolORM(Base):
    __tablename__ = "research_protocols"
    __table_args__ = (UniqueConstraint("research_id", "version"),)
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    research_questions: Mapped[list[dict[str, Any]]] = mapped_column(JSON_VALUE, default=list)
    inclusion_criteria: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    exclusion_criteria: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    target_databases: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    date_range: Mapped[dict[str, Any] | None] = mapped_column(JSON_VALUE)
    language_filter: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    validated_by: Mapped[str | None] = mapped_column(Text)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    modification_history: Mapped[list[dict[str, Any]]] = mapped_column(JSON_VALUE, default=list)


class PublicationORM(Base):
    __tablename__ = "publications"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    external_ids: Mapped[dict[str, str]] = mapped_column(JSON_VALUE, default=dict)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[dict[str, Any]]] = mapped_column(JSON_VALUE, default=list)
    year: Mapped[int | None] = mapped_column(Integer)
    abstract: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    journal: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("full_text_documents.id", ondelete="SET NULL")
    )


class FullTextDocumentORM(Base):
    __tablename__ = "full_text_documents"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    publication_id: Mapped[str] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"))
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)
    extraction_status: Mapped[str] = mapped_column(String(20), default="pending")
    extraction_error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    extractor_version: Mapped[str | None] = mapped_column(Text)


class AgentExecutionORM(Base):
    __tablename__ = "agent_executions"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    agent_name: Mapped[str] = mapped_column(Text)
    input_summary: Mapped[str] = mapped_column(Text)
    output_summary: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="running")
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens_input: Mapped[int] = mapped_column(Integer, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, default=0)
    llm_provider: Mapped[str] = mapped_column(Text)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class HumanReviewORM(Base):
    __tablename__ = "human_reviews"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    review_type: Mapped[str] = mapped_column(String(40))
    reviewer_id: Mapped[str] = mapped_column(Text)
    context_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE, default=dict)
    decision: Mapped[str] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_to_decision_seconds: Mapped[int | None] = mapped_column(Integer)


class SearchRunORM(Base):
    __tablename__ = "search_runs"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    protocol_id: Mapped[str] = mapped_column(ForeignKey("research_protocols.id"))
    protocol_version: Mapped[int] = mapped_column(Integer)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    results_by_source: Mapped[dict[str, int]] = mapped_column(JSON_VALUE, default=dict)
    duplicates_removed: Mapped[int] = mapped_column(Integer, default=0)
    agent_execution_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_executions.id", ondelete="SET NULL")
    )


class SearchQueryORM(Base):
    __tablename__ = "search_queries"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    search_run_id: Mapped[str] = mapped_column(ForeignKey("search_runs.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(30))
    query_string: Mapped[str] = mapped_column(Text)
    filters_applied: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE, default=dict)
    results_count: Mapped[int] = mapped_column(Integer, default=0)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    http_status: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)


class ScreeningDecisionORM(Base):
    __tablename__ = "screening_decisions"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    publication_id: Mapped[str] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"))
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    protocol_id: Mapped[str] = mapped_column(ForeignKey("research_protocols.id"))
    screening_stage: Mapped[str] = mapped_column(String(30))
    decision: Mapped[str] = mapped_column(String(20))
    exclusion_reason: Mapped[str | None] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text)
    criteria_applied: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    agent: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    human_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_reviews.id", ondelete="SET NULL")
    )
    evaluator_2_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("screening_decisions.id", ondelete="SET NULL")
    )
    agreement_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkORM(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("full_text_documents.id", ondelete="CASCADE")
    )
    publication_id: Mapped[str] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)
    section_type: Mapped[str] = mapped_column(String(20), default="text")
    chunk_index: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(Text)
    embedding_version: Mapped[str] = mapped_column(Text)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExtractionRecordORM(Base):
    __tablename__ = "extraction_records"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    publication_id: Mapped[str] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"))
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    agent_execution_id: Mapped[str] = mapped_column(ForeignKey("agent_executions.id"))
    extracted_fields: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE, default=dict)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    extraction_confidence: Mapped[float] = mapped_column(Float)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    human_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_reviews.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvidenceORM(Base):
    __tablename__ = "evidence"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    publication_id: Mapped[str] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)
    evidence_type: Mapped[str] = mapped_column(String(20))
    extracted_by_agent: Mapped[str] = mapped_column(Text)
    extraction_confidence: Mapped[float] = mapped_column(Float)
    verification_status: Mapped[str] = mapped_column(String(24), default="extracted")
    verification_method: Mapped[str | None] = mapped_column(Text)
    human_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_reviews.id", ondelete="SET NULL")
    )
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReportVersionORM(Base):
    __tablename__ = "report_versions"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    research_id: Mapped[str] = mapped_column(ForeignKey("research_projects.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE, default=dict)
    generated_by_agent: Mapped[str] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    overall_quality_score: Mapped[float | None] = mapped_column(Float)
    human_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    human_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_reviews.id", ondelete="SET NULL")
    )
    previous_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("report_versions.id", ondelete="SET NULL")
    )


class ClaimORM(Base):
    __tablename__ = "claims"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    rq_id: Mapped[str] = mapped_column(Text)
    report_version_id: Mapped[str] = mapped_column(
        ForeignKey("report_versions.id", ondelete="CASCADE")
    )
    section: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="pending")
    verification_method: Mapped[str] = mapped_column(String(20), default="semantic")
    contradicting_evidence_ids: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClaimEvidenceLinkORM(Base):
    __tablename__ = "claim_evidence_links"
    __table_args__ = (UniqueConstraint("claim_id", "evidence_id"),)
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"))
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id", ondelete="CASCADE"))
    link_type: Mapped[str] = mapped_column(String(24))
    link_confidence: Mapped[float] = mapped_column(Float)
    created_by: Mapped[str] = mapped_column(String(10))


class AuditLogORM(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        CheckConstraint("action <> ''", name="ck_audit_action_nonempty"),
        Index("ix_audit_resource", "resource_type", "resource_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    research_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_projects.id", ondelete="SET NULL")
    )
    actor: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    resource_type: Mapped[str] = mapped_column(Text)
    resource_id: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


@event.listens_for(AuditLogORM, "before_update")
def _reject_audit_update(mapper: Any, connection: Any, target: AuditLogORM) -> None:
    raise ValueError("audit_log entries are immutable")


@event.listens_for(AuditLogORM, "before_delete")
def _reject_audit_delete(mapper: Any, connection: Any, target: AuditLogORM) -> None:
    raise ValueError("audit_log entries are immutable")
