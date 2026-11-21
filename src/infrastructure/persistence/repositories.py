"""Async PostgreSQL repository implementations."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.claim import Claim, ClaimEvidenceLink
from ...domain.entities.evidence import Evidence
from ...domain.entities.publication import Author, Publication
from ...domain.entities.research_project import ResearchProject
from ...domain.entities.research_protocol import ResearchProtocol
from ...domain.entities.screening_decision import ScreeningDecision
from ...domain.ports.evidence_port import EvidencePort
from ...domain.ports.protocol_repository import ProtocolRepository
from ...domain.ports.publication_repository import PublicationRepository
from ...domain.ports.research_project_repository import ResearchProjectRepository
from .models import (
    AuditLogORM,
    ClaimEvidenceLinkORM,
    ClaimORM,
    EvidenceORM,
    PublicationORM,
    ResearchProjectORM,
    ResearchProtocolORM,
    ScreeningDecisionORM,
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


def _claim_from_row(row: ClaimORM) -> Claim:
    return Claim.model_validate({key: getattr(row, key) for key in Claim.model_fields})


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
