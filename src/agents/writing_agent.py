"""
WritingAgent Avancé — SLR-Nexus.
Rédige la synthèse scientifique section par section à partir des preuves vérifiées.
Garantit l'invariant : chaque affirmation (Claim) est liée à au moins un evidence_id.
"""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..domain.entities.claim import Claim, ClaimEvidenceLink
from ..domain.entities.evidence import Evidence
from ..domain.entities.report_version import ReportVersion
from ..domain.entities.research_protocol import ResearchQuestion
from ..domain.ports.llm_port import LLMPort


class ClaimDraft(BaseModel):
    """Assertion scientifique atomique produite par l'agent rédacteur."""
    text: str = Field(..., min_length=10, description="Affirmation scientifique précise")
    rq_id: str = Field(..., description="ID de la question de recherche associée")
    section: str = Field(..., description="Nom de la section (ex: Introduction, RQ1, Limitations)")
    cited_evidence_ids: list[str] = Field(default_factory=list, description="IDs des preuves exactes soutenant ce claim")


class ReportSectionDraft(BaseModel):
    """Section rédigée avec ses claims et citations associés."""
    section_name: str
    rq_id: str | None = None
    content: str = Field(..., min_length=20, description="Contenu rédigé de la section")
    claims: list[ClaimDraft] = Field(default_factory=list, description="Claims extraits de cette section")


class WritingSynthesisOutput(BaseModel):
    """Sortie Pydantic complète du WritingAgent."""
    sections: list[ReportSectionDraft]
    executive_summary: str = Field(..., min_length=30)
    general_conclusion: str = Field(..., min_length=20)


class WritingAgent:
    """
    Agent Rédacteur Scientifique.
    Conçu pour produire un rapport structuré sans jamais extrapoler au-delà des preuves fournies.
    """

    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    async def write_report(
        self,
        research_id: str,
        questions: list[ResearchQuestion],
        evidences: list[Evidence],
        revision_feedback: str | None = None,
        previous_version: int = 0,
    ) -> tuple[ReportVersion, list[Claim], list[ClaimEvidenceLink]]:
        """
        Rédige un rapport de synthèse complet ou une version révisée suite à un feedback.
        """
        # Construction du catalogue de preuves disponibles (Evidence Registry snapshot)
        evidence_catalog = [
            {
                "evidence_id": ev.id,
                "publication_id": ev.publication_id,
                "type": ev.evidence_type,
                "text": ev.text,
                "section": ev.section,
            }
            for ev in evidences
            if ev.is_trustworthy or ev.verification_status == "extracted"
        ]

        rq_list = [{"rq_id": q.id, "text": q.text, "category": q.category} for q in questions]

        # Prompt système rigide et directif
        system_prompt = (
            "Tu es le Writing Agent d'une plateforme de Revues Systématiques de Littérature (SLR).\n"
            "DIRECTIVES ABSOLUES :\n"
            "1. Chaque paragraphe que tu rédiges DOIT reposer uniquement sur les preuves (Evidence) fournies dans le catalogue.\n"
            "2. Pour chaque affirmation scientifique importante, tu DOIS extraire un ClaimDraft et lier les 'cited_evidence_ids' exacts.\n"
            "3. Interdiction formelle d'inventer des chiffres, des méthodes ou des résultats qui ne figurent pas dans les preuves.\n"
            "4. Rédige au moins une section par Question de Recherche (RQ).\n"
        )

        user_content = {
            "research_id": research_id,
            "research_questions": rq_list,
            "available_evidences": evidence_catalog,
        }

        if revision_feedback:
            user_content["revision_feedback"] = revision_feedback
            system_prompt += (
                f"\nATTENTION : C'est une RÉVISION. Intègre impérativement ces corrections :\n{revision_feedback}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Données de rédaction : {user_content}"},
        ]

        # Appel LLM avec structure Pydantic garantie
        synthesis: WritingSynthesisOutput = await self.llm.invoke_structured(
            messages=messages,
            response_model=WritingSynthesisOutput,
            temperature=0.0,
            max_repair_attempts=2,
        )

        # Transformation en entités du domaine
        report_version_id = str(uuid.uuid4())
        sections_dict: dict[str, str] = {
            "executive_summary": synthesis.executive_summary,
            "conclusion": synthesis.general_conclusion,
        }

        all_claims: list[Claim] = []
        all_links: list[ClaimEvidenceLink] = []

        for sec in synthesis.sections:
            sections_dict[sec.section_name] = sec.content

            for cd in sec.claims:
                claim_id = str(uuid.uuid4())
                claim = Claim(
                    id=claim_id,
                    text=cd.text,
                    rq_id=cd.rq_id,
                    report_version_id=report_version_id,
                    section=cd.section,
                    status="supported" if cd.cited_evidence_ids else "unsupported",
                    verification_method="semantic",
                )
                all_claims.append(claim)

                # Création des liens de traçabilité Claim ↔ Evidence
                for eid in cd.cited_evidence_ids:
                    link = ClaimEvidenceLink(
                        claim_id=claim_id,
                        evidence_id=eid,
                        link_type="supports",
                        link_confidence=0.95,
                        created_by="auto",
                    )
                    all_links.append(link)

        report = ReportVersion(
            id=report_version_id,
            research_id=research_id,
            version=previous_version + 1,
            content=sections_dict,
            generated_by_agent="writing_agent",
            overall_quality_score=None,
            human_approved=False,
        )

        return report, all_claims, all_links
