"""Async PostgreSQL repository implementations."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.chunk import Chunk
from ...domain.entities.claim import Claim, ClaimEvidenceLink
from ...domain.entities.evidence import Evidence
from ...domain.entities.extraction_record import ExtractionRecord
from ...domain.entities.full_text_document import FullTextDocument
from ...domain.entities.publication import Author, Publication
from ...domain.entities.research_project import ResearchProject
from ...domain.entities.research_protocol import ResearchProtocol
from ...domain.entities.screening_decision import ScreeningDecision
from ...domain.entities.search_run import SearchRun
from ...domain.ports.evidence_port import EvidencePort
from ...domain.ports.protocol_repository import ProtocolRepository
from ...domain.ports.publication_repository import PublicationRepository
from ...domain.ports.research_project_repository import ResearchProjectRepository
from ...domain.services.prisma_counter import PRISMACounter, PRISMAFlow
from .models import (
    AuditLogORM,
    ChunkORM,
    ClaimEvidenceLinkORM,
    ClaimORM,
    EvidenceORM,
    ExtractionRecordORM,
    FullTextDocumentORM,
    PublicationORM,
    ResearchProjectORM,
    ResearchProtocolORM,
    ScreeningDecisionORM,
    SearchQueryORM,
    SearchRunORM,
)


def _publication_from_row(row: PublicationORM) -> Publication:
    data = {key: getattr(row, key) for key in Publication.model_fields}
    data["authors"] = [Author.model_validate(author) for author in (row.authors or [])]
    return Publication.model_validate(data)


def _project_from_row(row: ResearchProjectORM) -> ResearchProject:
    return ResearchProject.model_validate(
        {key: getattr(row, key) for key in ResearchProject.model_fields}
    )


def _protocol_from_row(row: ResearchProtocolORM) -> ResearchProtocol:
    return ResearchProtocol.model_validate(
        {key: getattr(row, key) for key in ResearchProtocol.model_fields}
    )


def _evidence_from_row(row: EvidenceORM) -> Evidence:
    return Evidence.model_validate({key: getattr(row, key) for key in Evidence.model_fields})


def _chunk_from_row(row: ChunkORM) -> Chunk:
    return Chunk.model_validate({key: getattr(row, key) for key in Chunk.model_fields})


def _document_from_row(row: FullTextDocumentORM) -> FullTextDocument:
    return FullTextDocument.model_validate(
        {key: getattr(row, key) for key in FullTextDocument.model_fields}
    )


class FullTextDocumentRepo:
    """Persist document extraction state; hashes make repeat ingestion idempotent."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, document: FullTextDocument) -> FullTextDocument:
        row = await self.session.get(FullTextDocumentORM, document.id)
        if row is None:
            row = (
                await self.session.scalars(
                    select(FullTextDocumentORM).where(
                        FullTextDocumentORM.file_hash == document.file_hash
                    )
                )
            ).first()
        values = document.model_dump(mode="python")
        if row is None:
            row = FullTextDocumentORM(**values)
            self.session.add(row)
        else:
            for key, value in values.items():
                if key != "id":
                    setattr(row, key, value)
        await self.session.flush()
        return _document_from_row(row)

    async def find_by_id(self, document_id: str) -> FullTextDocument | None:
        row = await self.session.get(FullTextDocumentORM, document_id)
        return _document_from_row(row) if row else None


class ChunkRepo:
    """Persist chunk metadata idempotently by document and chunk index."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_many(self, chunks: list[Chunk]) -> list[Chunk]:
        saved_rows: list[ChunkORM] = []
        for chunk in chunks:
            row = await self.session.get(ChunkORM, chunk.id)
            if row is None:
                row = (
                    await self.session.scalars(
                        select(ChunkORM).where(
                            ChunkORM.document_id == chunk.document_id,
                            ChunkORM.chunk_index == chunk.chunk_index,
                        )
                    )
                ).first()
            values = chunk.model_dump(mode="python")
            if row is None:
                row = ChunkORM(**values)
                self.session.add(row)
            else:
                for key, value in values.items():
                    if key != "id":
                        setattr(row, key, value)
            saved_rows.append(row)
        await self.session.flush()
        return [_chunk_from_row(row) for row in saved_rows]

    async def list_for_publications(self, publication_ids: list[str]) -> list[Chunk]:
        if not publication_ids:
            return []
        rows = await self.session.scalars(
            select(ChunkORM)
            .where(ChunkORM.publication_id.in_(publication_ids))
            .order_by(ChunkORM.publication_id, ChunkORM.chunk_index)
        )
        return [_chunk_from_row(row) for row in rows]


def _claim_from_row(row: ClaimORM) -> Claim:
    return Claim.model_validate({key: getattr(row, key) for key in Claim.model_fields})


def _extraction_record_from_row(row: ExtractionRecordORM) -> ExtractionRecord:
    return ExtractionRecord.model_validate(
        {key: getattr(row, key) for key in ExtractionRecord.model_fields}
    )


class ProjectRepo(ResearchProjectRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, project: ResearchProject) -> ResearchProject:
        row = await self.session.get(ResearchProjectORM, project.id)
        values = project.model_dump()
        if row is None:
            row = ResearchProjectORM(**values)
            self.session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        await self.session.flush()
        return _project_from_row(row)

    async def find_by_id(self, project_id: str) -> ResearchProject | None:
        row = await self.session.get(ResearchProjectORM, project_id)
        return _project_from_row(row) if row else None

    async def list_projects(self, limit: int = 100, offset: int = 0) -> list[ResearchProject]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset cannot be negative")
        result = await self.session.scalars(
            select(ResearchProjectORM)
            .order_by(ResearchProjectORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_project_from_row(row) for row in result]


class ProtocolRepo(ProtocolRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, protocol: ResearchProtocol) -> ResearchProtocol:
        row = await self.session.get(ResearchProtocolORM, protocol.id)
        values = protocol.model_dump(mode="python")
        values["research_questions"] = [
            item.model_dump(mode="json") for item in protocol.research_questions
        ]
        values["date_range"] = (
            protocol.date_range.model_dump(mode="json") if protocol.date_range else None
        )
        values["modification_history"] = [
            item.model_dump(mode="json") for item in protocol.modification_history
        ]
        if row is None:
            row = ResearchProtocolORM(**values)
            self.session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        await self.session.flush()
        await self.session.execute(
            update(ResearchProjectORM)
            .where(ResearchProjectORM.id == protocol.research_id)
            .values(protocol_id=protocol.id)
        )
        return _protocol_from_row(row)

    async def find_by_id(self, protocol_id: str) -> ResearchProtocol | None:
        row = await self.session.get(ResearchProtocolORM, protocol_id)
        return _protocol_from_row(row) if row else None

    async def find_latest_by_research(self, research_id: str) -> ResearchProtocol | None:
        result = await self.session.scalars(
            select(ResearchProtocolORM)
            .where(ResearchProtocolORM.research_id == research_id)
            .order_by(ResearchProtocolORM.version.desc())
            .limit(1)
        )
        row = result.first()
        return _protocol_from_row(row) if row else None


class PublicationRepo(PublicationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, publication: Publication) -> Publication:
        row = await self.session.get(PublicationORM, publication.id)
        if publication.doi:
            by_doi = await self.find_by_doi(publication.doi)
            if by_doi:
                row = await self.session.get(PublicationORM, by_doi.id)
        values = publication.model_dump(mode="python")
        values["authors"] = [author.model_dump(mode="json") for author in publication.authors]
        if row is None:
            row = PublicationORM(**values)
            self.session.add(row)
        else:
            for key, value in values.items():
                if key != "id":
                    setattr(row, key, value)
        await self.session.flush()
        return _publication_from_row(row)

    async def find_by_id(self, publication_id: str) -> Publication | None:
        row = await self.session.get(PublicationORM, publication_id)
        return _publication_from_row(row) if row else None

    async def find_by_doi(self, doi: str) -> Publication | None:
        normalized = (
            doi.strip().lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        )
        result = await self.session.scalars(
            select(PublicationORM).where(func.lower(PublicationORM.doi) == normalized)
        )
        row = result.first()
        return _publication_from_row(row) if row else None

    async def find_by_external_id(self, source: str, external_id: str) -> Publication | None:
        key = {
            "openalex": "openalex_id",
            "crossref": "doi",
            "semantic_scholar": "semantic_scholar_id",
        }.get(source, source)
        result = await self.session.scalars(
            select(PublicationORM).where(
                PublicationORM.external_ids[key].as_string() == external_id
            )
        )
        row = result.first()
        return _publication_from_row(row) if row else None

    async def find_by_research_id(self, research_id: str) -> list[Publication]:
        latest = (
            select(
                ScreeningDecisionORM.publication_id,
                func.row_number()
                .over(
                    partition_by=ScreeningDecisionORM.publication_id,
                    order_by=(
                        ScreeningDecisionORM.created_at.desc(),
                        ScreeningDecisionORM.id.desc(),
                    ),
                )
                .label("rank"),
            )
            .where(ScreeningDecisionORM.research_id == research_id)
            .subquery()
        )
        result = await self.session.scalars(
            select(PublicationORM)
            .join(latest, latest.c.publication_id == PublicationORM.id)
            .where(latest.c.rank == 1)
            .order_by(PublicationORM.title)
        )
        return [_publication_from_row(row) for row in result]

    async def find_included_by_research_id(self, research_id: str) -> list[Publication]:
        latest = (
            select(
                ScreeningDecisionORM.publication_id,
                ScreeningDecisionORM.decision,
                func.row_number()
                .over(
                    partition_by=ScreeningDecisionORM.publication_id,
                    order_by=(
                        ScreeningDecisionORM.created_at.desc(),
                        ScreeningDecisionORM.id.desc(),
                    ),
                )
                .label("rank"),
            )
            .where(ScreeningDecisionORM.research_id == research_id)
            .subquery()
        )
        result = await self.session.scalars(
            select(PublicationORM)
            .join(latest, latest.c.publication_id == PublicationORM.id)
            .where(latest.c.rank == 1, latest.c.decision == "include")
            .order_by(PublicationORM.title)
        )
        return [_publication_from_row(row) for row in result]

    async def count_by_research_id(self, research_id: str) -> int:
        return len(await self.find_by_research_id(research_id))

    async def bulk_save(self, publications: list[Publication]) -> list[Publication]:
        return [await self.save(publication) for publication in publications]


class EvidenceRepo(EvidencePort):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_evidence(self, evidence: Evidence) -> Evidence:
        statement = select(EvidenceORM).where(
            EvidenceORM.chunk_id == evidence.chunk_id,
            EvidenceORM.evidence_type == evidence.evidence_type,
        )
        row = (await self.session.scalars(statement)).first()
        values = evidence.model_dump(mode="python")
        if row is None:
            row = EvidenceORM(**values)
            self.session.add(row)
        else:
            for key, value in values.items():
                if key != "id":
                    setattr(row, key, value)
        await self.session.flush()
        return _evidence_from_row(row)

    async def find_evidence_by_id(self, evidence_id: str) -> Evidence | None:
        row = await self.session.get(EvidenceORM, evidence_id)
        return _evidence_from_row(row) if row else None

    async def find_evidence_by_research(self, research_id: str) -> list[Evidence]:
        rows = await self.session.scalars(
            select(EvidenceORM)
            .where(EvidenceORM.research_id == research_id)
            .order_by(EvidenceORM.created_at)
        )
        return [_evidence_from_row(row) for row in rows]

    async def find_evidence_by_publication(
        self, publication_id: str, research_id: str
    ) -> list[Evidence]:
        rows = await self.session.scalars(
            select(EvidenceORM)
            .where(
                EvidenceORM.publication_id == publication_id, EvidenceORM.research_id == research_id
            )
            .order_by(EvidenceORM.created_at)
        )
        return [_evidence_from_row(row) for row in rows]

    async def find_trustworthy_evidence(self, research_id: str) -> list[Evidence]:
        rows = await self.session.scalars(
            select(EvidenceORM)
            .where(
                EvidenceORM.research_id == research_id,
                EvidenceORM.verification_status.in_(("auto_verified", "human_validated")),
            )
            .order_by(EvidenceORM.created_at)
        )
        return [_evidence_from_row(row) for row in rows]

    async def update_verification_status(
        self,
        evidence_id: str,
        new_status: str,
        verification_method: str | None = None,
        human_review_id: str | None = None,
    ) -> Evidence:
        row = await self.session.get(EvidenceORM, evidence_id)
        if row is None:
            raise LookupError(f"Evidence not found: {evidence_id}")
        row.verification_status = new_status
        if verification_method is not None:
            row.verification_method = verification_method
        if human_review_id is not None:
            row.human_review_id = human_review_id
        await self.session.flush()
        return _evidence_from_row(row)

    async def save_claim(self, claim: Claim) -> Claim:
        row = await self.session.get(ClaimORM, claim.id)
        values = claim.model_dump(mode="python")
        if row is None:
            row = ClaimORM(**values)
            self.session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        await self.session.flush()
        return _claim_from_row(row)

    async def find_claim_by_id(self, claim_id: str) -> Claim | None:
        row = await self.session.get(ClaimORM, claim_id)
        return _claim_from_row(row) if row else None

    async def find_claims_by_report(self, report_version_id: str) -> list[Claim]:
        rows = await self.session.scalars(
            select(ClaimORM)
            .where(ClaimORM.report_version_id == report_version_id)
            .order_by(ClaimORM.created_at)
        )
        return [_claim_from_row(row) for row in rows]

    async def link_claim_to_evidence(self, link: ClaimEvidenceLink) -> ClaimEvidenceLink:
        row = await self.session.get(ClaimEvidenceLinkORM, link.id)
        if row is None:
            row = (
                await self.session.scalars(
                    select(ClaimEvidenceLinkORM).where(
                        ClaimEvidenceLinkORM.claim_id == link.claim_id,
                        ClaimEvidenceLinkORM.evidence_id == link.evidence_id,
                    )
                )
            ).first()
        if row is None:
            row = ClaimEvidenceLinkORM(**link.model_dump())
            self.session.add(row)
        await self.session.flush()
        return ClaimEvidenceLink.model_validate(
            {key: getattr(row, key) for key in ClaimEvidenceLink.model_fields}
        )

    async def get_evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        rows = await self.session.scalars(
            select(EvidenceORM)
            .join(ClaimEvidenceLinkORM, ClaimEvidenceLinkORM.evidence_id == EvidenceORM.id)
            .where(ClaimEvidenceLinkORM.claim_id == claim_id)
            .order_by(EvidenceORM.created_at)
        )
        return [_evidence_from_row(row) for row in rows]

    async def get_claims_without_evidence(self, report_version_id: str) -> list[Claim]:
        has_links = (
            select(ClaimEvidenceLinkORM.id)
            .where(ClaimEvidenceLinkORM.claim_id == ClaimORM.id)
            .exists()
        )
        rows = await self.session.scalars(
            select(ClaimORM).where(ClaimORM.report_version_id == report_version_id, ~has_links)
        )
        return [_claim_from_row(row) for row in rows]


class ScreeningRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, decision: ScreeningDecision) -> ScreeningDecision:
        values = decision.model_dump()
        row = await self.session.get(ScreeningDecisionORM, decision.id)
        if row is None:
            row = ScreeningDecisionORM(**values)
            self.session.add(row)
        await self.session.flush()
        return ScreeningDecision.model_validate(
            {key: getattr(row, key) for key in ScreeningDecision.model_fields}
        )

    async def find_by_research(self, research_id: str) -> list[ScreeningDecision]:
        rows = await self.session.scalars(
            select(ScreeningDecisionORM)
            .where(ScreeningDecisionORM.research_id == research_id)
            .order_by(ScreeningDecisionORM.created_at)
        )
        return [
            ScreeningDecision.model_validate(
                {key: getattr(row, key) for key in ScreeningDecision.model_fields}
            )
            for row in rows
        ]

    async def calculate_prisma_flow(self, research_id: str) -> PRISMAFlow:
        """Derive counts from persisted search runs, screening decisions and documents."""
        run_rows = await self.session.scalars(
            select(SearchRunORM)
            .where(SearchRunORM.research_id == research_id)
            .order_by(SearchRunORM.executed_at.desc(), SearchRunORM.id.desc())
        )
        latest_run = run_rows.first()
        identified = dict(latest_run.results_by_source) if latest_run else {}
        duplicates = latest_run.duplicates_removed if latest_run else 0

        rows = await self.session.scalars(
            select(ScreeningDecisionORM)
            .where(ScreeningDecisionORM.research_id == research_id)
            .order_by(ScreeningDecisionORM.created_at, ScreeningDecisionORM.id)
        )
        latest: dict[tuple[str, str], Any] = {}
        for row in rows:
            latest[(row.publication_id, row.screening_stage)] = row
        title = [row for (pub, stage), row in latest.items() if stage == "title_abstract"]
        full = [row for (pub, stage), row in latest.items() if stage == "full_text"]
        title_excluded = [row for row in title if row.decision == "exclude"]
        full_excluded = [row for row in full if row.decision == "exclude"]
        included = sum(row.decision == "include" for row in full)
        included_title = sum(row.decision == "include" for row in title)
        doc_statuses: dict[str, int] = {}
        if included_title:
            docs = await self.session.execute(
                select(FullTextDocumentORM.extraction_status, func.count())
                .where(
                    FullTextDocumentORM.publication_id.in_(
                        [row.publication_id for row in title if row.decision == "include"]
                    )
                )
                .group_by(FullTextDocumentORM.extraction_status)
            )
            doc_statuses = dict(docs.all())
        inaccessible = doc_statuses.get("inaccessible", 0) + doc_statuses.get("failed", 0)

        title_reason_counts: dict[str, int] = {}
        for row in title_excluded:
            reason = row.exclusion_reason or "other"
            title_reason_counts[reason] = title_reason_counts.get(reason, 0) + 1

        full_reason_counts: dict[str, int] = {}
        for row in full_excluded:
            reason = row.exclusion_reason or "other"
            full_reason_counts[reason] = full_reason_counts.get(reason, 0) + 1

        flow = PRISMACounter().build_from_counts(
            identified_by_source=identified,
            duplicates_removed=duplicates,
            excluded_title_abstract=len(title_excluded),
            excluded_title_abstract_reasons=title_reason_counts,
            inaccessible=inaccessible,
            excluded_full_text=len(full_excluded),
            excluded_full_text_reasons=full_reason_counts,
            included=included,
        )
        flow.screened_title_abstract = len(title)
        flow.assessed_full_text = included_title
        return flow


class AuditLogRepo:
    """Append-only audit journal. No update/delete API is intentionally exposed."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def append(
        self,
        *,
        audit_id: str,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        payload: dict[str, Any],
        research_id: str | None = None,
    ) -> AuditLogORM:
        if (
            not actor.strip()
            or not action.strip()
            or not resource_type.strip()
            or not resource_id.strip()
        ):
            raise ValueError("actor, action, resource_type and resource_id are required")
        entry = AuditLogORM(
            id=audit_id,
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload,
            research_id=research_id,
        )
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def list_for_resource(self, resource_type: str, resource_id: str) -> list[AuditLogORM]:
        rows = await self.session.scalars(
            select(AuditLogORM)
            .where(
                AuditLogORM.resource_type == resource_type, AuditLogORM.resource_id == resource_id
            )
            .order_by(AuditLogORM.created_at, AuditLogORM.id)
        )
        return list(rows)


class SearchRunRepo:
    """Persist search runs and their verbatim per-source queries atomically."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, run: SearchRun) -> SearchRun:
        values = {
            "id": run.id,
            "research_id": run.research_id,
            "protocol_id": run.protocol_id,
            "protocol_version": run.protocol_version,
            "executed_at": run.executed_at,
            "results_by_source": run.results_by_source,
            "duplicates_removed": run.duplicates_removed,
            "agent_execution_id": run.agent_execution_id,
        }
        row = await self.session.get(SearchRunORM, run.id)
        if row is None:
            self.session.add(SearchRunORM(**values))
        else:
            for key, value in values.items():
                setattr(row, key, value)
        for query in run.queries:
            query_row = await self.session.get(SearchQueryORM, query.id)
            query_values = query.model_dump(mode="python")
            if query_row is None:
                self.session.add(SearchQueryORM(**query_values))
            else:
                for key, value in query_values.items():
                    setattr(query_row, key, value)
        await self.session.flush()
        return run


class ExtractionRecordRepo:
    """Repository for structured extraction records linked to publications and source chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, record: ExtractionRecord) -> ExtractionRecord:
        row = await self.session.get(ExtractionRecordORM, record.id)
        values = record.model_dump(mode="python")
        if row is None:
            row = ExtractionRecordORM(**values)
            self.session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        await self.session.flush()
        return _extraction_record_from_row(row)

    async def find_by_id(self, record_id: str) -> ExtractionRecord | None:
        row = await self.session.get(ExtractionRecordORM, record_id)
        return _extraction_record_from_row(row) if row else None

    async def find_by_research(self, research_id: str) -> list[ExtractionRecord]:
        rows = await self.session.scalars(
            select(ExtractionRecordORM)
            .where(ExtractionRecordORM.research_id == research_id)
            .order_by(ExtractionRecordORM.created_at)
        )
        return [_extraction_record_from_row(row) for row in rows]

    async def find_by_publication(self, publication_id: str) -> list[ExtractionRecord]:
        rows = await self.session.scalars(
            select(ExtractionRecordORM)
            .where(ExtractionRecordORM.publication_id == publication_id)
            .order_by(ExtractionRecordORM.created_at)
        )
        return [_extraction_record_from_row(row) for row in rows]
