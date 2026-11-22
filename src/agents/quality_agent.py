"""
QualityAgent Avancé — SLR-Nexus.
Industrialisation et extension des concepts du TD08 :
1. Analyse formelle de couverture des Questions de Recherche (RQ Coverage).
2. Contrôle rigoureux du Grounding et des citations orphelines.
3. Vérification des contradictions et conformité avec les preuves (Claims).
4. Évaluation sémantique LLM-as-a-Judge.
5. Verdict tri-directionnel raisonné (ACCEPT, REVISE, ESCALATE) avec désignation de la cible de réparation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..domain.entities.claim import Claim, ClaimEvidenceLink
from ..domain.entities.evidence import Evidence
from ..domain.entities.report_version import ReportVersion
from ..domain.entities.research_protocol import ResearchQuestion
from ..domain.ports.llm_port import LLMPort


class LLMJudgeReportEvaluation(BaseModel):
    """Évaluation sémantique produite par le composant LLM-as-a-Judge."""
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Cohérence logique de la synthèse")
    scientific_rigor_score: float = Field(..., ge=0.0, le=1.0, description="Rigueur scientifique du vocabulaire")
    critique_findings: list[str] = Field(default_factory=list, description="Remarques et faiblesses identifiées")
    recommended_verdict: Literal["ACCEPT", "REVISE", "ESCALATE"]


class QualityVerdict(BaseModel):
    """Verdict final consolidé du QualityAgent."""
    verdict: Literal["ACCEPT", "REVISE", "ESCALATE"]
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    rq_coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    missing_rq_ids: list[str]
    grounding_ratio: float = Field(..., ge=0.0, le=1.0)
    unsupported_claim_ids: list[str]
    contradicted_claim_ids: list[str]
    repair_target: Literal["writing", "retrieval", "human"] | None = None
    feedback_for_repair: str | None = None


class QualityAgent:
    """
    Quality Agent (Juge de Qualité et Superviseur de Validation).
    Combine des métriques déterministes strictes et un LLM-as-a-Judge.
    """

    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    def compute_deterministic_metrics(
        self,
        report: ReportVersion,
        required_questions: list[ResearchQuestion],
        claims: list[Claim],
        links: list[ClaimEvidenceLink],
        available_evidences: list[Evidence],
    ) -> dict[str, Any]:
        """
        Calcule les métriques déterministes pures (TODO 2 et TODO 3 du TD08).
        """
        # 1. Couverture des RQ (TODO 2 du TD08)
        required_rq_ids = {rq.id for rq in required_questions}
        
        # RQ couvertes soit par les sections du rapport, soit par les claims associés
        covered_rq_ids: set[str] = set()
        for claim in claims:
            if claim.rq_id in required_rq_ids:
                covered_rq_ids.add(claim.rq_id)

        # Vérification également dans les clés de sections
        for section_title in report.content.keys():
            for rq in required_questions:
                if rq.id in section_title or (rq.category and rq.category.lower() in section_title.lower()):
                    covered_rq_ids.add(rq.id)

        rq_coverage = len(covered_rq_ids) / max(1, len(required_rq_ids))
        missing_rqs = list(required_rq_ids - covered_rq_ids)

        # 2. Grounding et Ancrage des Preuves (TODO 3 du TD08)
        evidence_ids_set = {ev.id for ev in available_evidences}
        linked_claim_ids = {link.claim_id for link in links if link.evidence_id in evidence_ids_set}

        unsupported_claims: list[str] = []
        contradicted_claims: list[str] = []

        for c in claims:
            if c.status == "contradicted":
                contradicted_claims.append(c.id)
            elif c.status == "unsupported" or c.id not in linked_claim_ids:
                unsupported_claims.append(c.id)

        total_claims = len(claims)
        supported_claims_count = total_claims - len(unsupported_claims) - len(contradicted_claims)
        grounding_ratio = supported_claims_count / max(1, total_claims) if total_claims > 0 else 0.0

        return {
            "rq_coverage": rq_coverage,
            "missing_rqs": missing_rqs,
            "grounding_ratio": grounding_ratio,
            "unsupported_claims": unsupported_claims,
            "contradicted_claims": contradicted_claims,
        }

    async def evaluate_report(
        self,
        report: ReportVersion,
        required_questions: list[ResearchQuestion],
        claims: list[Claim],
        links: list[ClaimEvidenceLink],
        available_evidences: list[Evidence],
        revision_count: int = 0,
        max_revisions: int = 3,
    ) -> QualityVerdict:
        """
        Évalue le rapport complet et détermine le verdict et la boucle de révision.
        """
        # Étape 1 : Métriques déterministes (Règles formelles)
        metrics = self.compute_deterministic_metrics(
            report, required_questions, claims, links, available_evidences
        )

        # Étape 2 : LLM-as-a-Judge (TODO 4 du TD08)
        judge_prompt = (
            "Tu es le Quality Judge d'une revue systématique de littérature.\n"
            "Évalue la qualité rédactionnelle, la cohérence et l'honnêteté scientifique de cet extrait de rapport.\n"
            "Vérifie que les affirmations sont tempérées et ne surinterprètent pas les faits.\n"
        )
        judge_payload = {
            "report_sections": report.content,
            "claims_count": len(claims),
            "rq_coverage_ratio": metrics["rq_coverage"],
            "grounding_ratio": metrics["grounding_ratio"],
        }

        judge_decision: LLMJudgeReportEvaluation = await self.llm.invoke_structured(
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Données à évaluer : {judge_payload}"},
            ],
            response_model=LLMJudgeReportEvaluation,
            temperature=0.0,
        )

        # Étape 3 : Consolidation du score global
        # Note globale = 40% couverture RQ + 40% grounding preuves + 20% appréciation LLM
        overall_score = (
            0.40 * metrics["rq_coverage"]
            + 0.40 * metrics["grounding_ratio"]
            + 0.20 * ((judge_decision.coherence_score + judge_decision.scientific_rigor_score) / 2.0)
        )

        # Étape 4 : Détermination du Verdict (TODO 5 du TD08 - Réflexion Bornée)
        verdict: Literal["ACCEPT", "REVISE", "ESCALATE"]
        repair_target: Literal["writing", "retrieval", "human"] | None = None
        feedback_lines: list[str] = []

        # Critères stricts d'acceptation
        has_critical_failures = (
            metrics["rq_coverage"] < 0.80
            or metrics["grounding_ratio"] < 0.85
            or len(metrics["contradicted_claims"]) > 0
        )

        if not has_critical_failures and judge_decision.recommended_verdict == "ACCEPT":
            verdict = "ACCEPT"
        else:
            # Besoin de révision ou escalade
            if revision_count >= max_revisions:
                verdict = "ESCALATE"
                repair_target = "human"
                feedback_lines.append(
                    f"Budget de révisions épuisé ({revision_count}/{max_revisions}). "
                    f"Couverture RQ: {metrics['rq_coverage']:.2f}, Grounding: {metrics['grounding_ratio']:.2f}."
                )
            else:
                verdict = "REVISE"
                # Ciblage intelligent du réparateur
                if metrics["rq_coverage"] < 0.80 and len(metrics["missing_rqs"]) > 0:
                    repair_target = "retrieval"  # Des preuves manquent pour certaines RQ
                    feedback_lines.append(
                        f"Questions de recherche non couvertes : {metrics['missing_rqs']}. Recherche de preuves requise."
                    )
                else:
                    repair_target = "writing"  # Les preuves existent mais l'écriture est mal ancrée
                    if metrics["unsupported_claims"]:
                        feedback_lines.append(
                            f"{len(metrics['unsupported_claims'])} affirmations manquent de preuves citées."
                        )
                    if metrics["contradicted_claims"]:
                        feedback_lines.append(
                            f"{len(metrics['contradicted_claims'])} affirmations contredisent les sources."
                        )

                feedback_lines.extend(judge_decision.critique_findings)

        feedback_str = "\n".join(feedback_lines) if feedback_lines else "Rapport conforme aux exigences scientifiques."

        return QualityVerdict(
            verdict=verdict,
            overall_quality_score=round(overall_score, 3),
            rq_coverage_ratio=round(metrics["rq_coverage"], 3),
            missing_rq_ids=metrics["missing_rqs"],
            grounding_ratio=round(metrics["grounding_ratio"], 3),
            unsupported_claim_ids=metrics["unsupported_claims"],
            contradicted_claim_ids=metrics["contradicted_claims"],
            repair_target=repair_target,
            feedback_for_repair=feedback_str,
        )
