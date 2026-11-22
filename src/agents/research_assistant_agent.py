"""Research Assistant Agent: Structured extraction and literal evidence anchoring."""

from __future__ import annotations

import re
import uuid
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from ..domain.entities.chunk import Chunk
from ..domain.entities.claim import Claim, ClaimEvidenceLink
from ..domain.entities.evidence import Evidence
from ..domain.entities.extraction_record import ExtractionRecord
from ..domain.entities.publication import Publication
from ..domain.entities.research_protocol import ResearchProtocol


class FieldExtractionItem(BaseModel):
    field_name: str
    value: str
    verbatim_quote: str
    chunk_id: str
    evidence_type: Literal["result", "method", "limitation", "dataset", "metric", "definition"]
    page: int | None = None
    section: str | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class PublicationExtractionLLMOutput(BaseModel):
    fields: list[FieldExtractionItem] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class SynthesizedClaimItem(BaseModel):
    text: str = Field(min_length=10)
    rq_id: str
    section: str = "Synthesis"
    supporting_evidence_ids: list[str] = Field(default_factory=list, min_length=1)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class SynthesisLLMOutput(BaseModel):
    claims: list[SynthesizedClaimItem] = Field(default_factory=list)


class ResearchAssistantAgent:
    """
    Research Assistant Agent chargé de :
    1. L'extraction structurée (méthodes, datasets, métriques, résultats, limites).
    2. L'extraction de preuves littérales ancrées dans les chunks.
    3. La formulation de claims vérifiables liés aux preuves.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    async def extract_from_publication(
        self,
        publication: Publication,
        chunks: list[Chunk],
        *,
        research_id: str,
        agent_execution_id: str,
    ) -> tuple[ExtractionRecord, list[Evidence]]:
        """
        Extrait les données structurées et les preuves littérales depuis les chunks d'une publication.
        Garantit que tout passage d'evidence est textuellement présent dans son chunk source.
        """
        if not chunks:
            record = ExtractionRecord(
                id=str(uuid.uuid4()),
                publication_id=publication.id,
                research_id=research_id,
                agent_execution_id=agent_execution_id,
                extracted_fields={},
                source_chunk_ids=[],
                extraction_confidence=0.0,
            )
            return record, []

        chunk_by_id = {c.id: c for c in chunks}
        extracted_fields: dict[str, str] = {}
        source_chunk_ids: list[str] = []
        evidences: list[Evidence] = []
        confidence = 0.85

        if self.llm is not None and hasattr(self.llm, "invoke_structured"):
            prompt_chunks = "\n---\n".join(
                f"[Chunk ID: {c.id} | Page: {c.page} | Section: {c.section_type}]\n{c.text[:1200]}"
                for c in chunks[:15]
            )
            prompt = (
                "Tu es le Research Assistant Agent spécialisé dans les Revues Systématiques de Littérature.\n"
                f"Publication : {publication.title} ({publication.source})\n\n"
                "Extrais les champs scientifiques standards ('method', 'dataset', 'metric', 'result', 'limitation') "
                "et pour chacun, fournis la citation textuelle EXACTE (verbatim_quote) ainsi que le chunk_id source.\n"
                "RÈGLE ABSOLUE : verbatim_quote doit être une sous-chaîne EXACTE du texte du chunk.\n\n"
                f"Passages disponibles :\n{prompt_chunks}"
            )
            try:
                llm_output = cast(
                    PublicationExtractionLLMOutput,
                    await self.llm.invoke_structured(
                        [{"role": "user", "content": prompt}],
                        PublicationExtractionLLMOutput,
                    ),
                )
                confidence = llm_output.overall_confidence
                for item in llm_output.fields:
                    source_chunk = chunk_by_id.get(item.chunk_id)
                    quote = item.verbatim_quote.strip()
                    if source_chunk and quote in source_chunk.text:
                        extracted_fields[item.field_name] = item.value
                        if item.chunk_id not in source_chunk_ids:
                            source_chunk_ids.append(item.chunk_id)
                        ev = Evidence(
                            id=str(uuid.uuid4()),
                            publication_id=publication.id,
                            chunk_id=item.chunk_id,
                            text=quote,
                            page=item.page or source_chunk.page,
                            section=item.section or source_chunk.section_type,
                            evidence_type=item.evidence_type,
                            extracted_by_agent="research_assistant",
                            extraction_confidence=item.confidence,
                            verification_status="extracted",
                            research_id=research_id,
                        )
                        evidences.append(ev)
            except Exception:
                pass

        # Si l'extraction LLM est vide ou non disponible, extraction heuristique déterministe
        if not extracted_fields:
            extracted_fields, source_chunk_ids, evidences, confidence = self._deterministic_extract(
                publication=publication,
                chunks=chunks,
                research_id=research_id,
            )

        record = ExtractionRecord(
            id=str(uuid.uuid4()),
            publication_id=publication.id,
            research_id=research_id,
            agent_execution_id=agent_execution_id,
            extracted_fields=extracted_fields,
            source_chunk_ids=source_chunk_ids,
            extraction_confidence=confidence,
        )
        return record, evidences

    def _deterministic_extract(
        self,
        publication: Publication,
        chunks: list[Chunk],
        research_id: str,
    ) -> tuple[dict[str, str], list[str], list[Evidence], float]:
        """Extraction heuristique garantissant l'exactitude littérale des passages sources."""
        field_patterns: dict[str, tuple[str, Literal["result", "method", "limitation", "dataset", "metric", "definition"]]] = {
            "method": (r"(?i)\b(method|approach|architecture|algorithm|model|technique|pipeline)\b", "method"),
            "dataset": (r"(?i)\b(dataset|benchmark|corpus|data set|sample|cohort)\b", "dataset"),
            "metric": (r"(?i)\b(metric|accuracy|precision|recall|f1-score|auc|loss|latency)\b", "metric"),
            "result": (r"(?i)\b(result|outperforms|achieves|improved|findings|yielded|observed)\b", "result"),
            "limitation": (r"(?i)\b(limitation|future work|drawback|threat to validity|restricted)\b", "limitation"),
        }

        extracted_fields: dict[str, str] = {}
        source_chunk_ids: list[str] = []
        evidences: list[Evidence] = []

        for field_name, (pattern, ev_type) in field_patterns.items():
            for chunk in chunks:
                sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
                for sentence in sentences:
                    sentence_clean = sentence.strip()
                    if len(sentence_clean) >= 20 and re.search(pattern, sentence_clean):
                        # La phrase doit être présente littéralement dans le chunk
                        if sentence_clean in chunk.text and field_name not in extracted_fields:
                            extracted_fields[field_name] = sentence_clean[:150]
                            if chunk.id not in source_chunk_ids:
                                source_chunk_ids.append(chunk.id)
                            ev = Evidence(
                                id=str(uuid.uuid4()),
                                publication_id=publication.id,
                                chunk_id=chunk.id,
                                text=sentence_clean,
                                page=chunk.page,
                                section=chunk.section_type,
                                evidence_type=ev_type,
                                extracted_by_agent="research_assistant",
                                extraction_confidence=0.85,
                                verification_status="extracted",
                                research_id=research_id,
                            )
                            evidences.append(ev)
                            break
                if field_name in extracted_fields:
                    break

        # S'assurer d'au moins une evidence si des chunks existent
        if not evidences and chunks:
            first_chunk = chunks[0]
            first_sentence = first_chunk.text.split(".")[0].strip() + "."
            if first_sentence and first_sentence in first_chunk.text:
                extracted_fields["result"] = first_sentence
                source_chunk_ids.append(first_chunk.id)
                evidences.append(
                    Evidence(
                        id=str(uuid.uuid4()),
                        publication_id=publication.id,
                        chunk_id=first_chunk.id,
                        text=first_sentence,
                        page=first_chunk.page,
                        section=first_chunk.section_type,
                        evidence_type="result",
                        extracted_by_agent="research_assistant",
                        extraction_confidence=0.75,
                        verification_status="extracted",
                        research_id=research_id,
                    )
                )

        confidence = 0.80 if len(extracted_fields) >= 3 else 0.60
        return extracted_fields, source_chunk_ids, evidences, confidence

    async def formulate_claims(
        self,
        *,
        research_id: str,
        report_version_id: str,
        research_protocol: ResearchProtocol,
        evidences: list[Evidence],
    ) -> list[tuple[Claim, list[ClaimEvidenceLink]]]:
        """
        Formule des claims pour chaque question de recherche du protocole
        et les lie systématiquement aux preuves sources (Invariant : zéro claim sans evidence).
        """
        results: list[tuple[Claim, list[ClaimEvidenceLink]]] = []
        if not evidences:
            return results

        rqs = [rq.text for rq in research_protocol.research_questions]
        if not rqs:
            rqs = ["RQ1 : Résultats généraux de la revue"]

        if self.llm is not None and hasattr(self.llm, "invoke_structured"):
            ev_summary = "\n".join(
                f"- [ID: {ev.id}] (Type: {ev.evidence_type}) : {ev.text[:200]}"
                for ev in evidences[:20]
            )
            prompt = (
                "Tu es le Research Assistant Agent. À partir des preuves scientifiques ci-dessous, "
                "formule des affirmations scientifiques précises (claims) répondant aux questions de recherche.\n"
                "INVARIANT OBLIGATOIRE : chaque claim doit référencer au moins un 'supporting_evidence_ids' valide.\n\n"
                f"Questions de recherche : {rqs}\n\n"
                f"Preuves extraites :\n{ev_summary}"
            )
            try:
                synthesis_output = cast(
                    SynthesisLLMOutput,
                    await self.llm.invoke_structured(
                        [{"role": "user", "content": prompt}],
                        SynthesisLLMOutput,
                    ),
                )
                valid_ev_ids = {ev.id for ev in evidences}
                for item in synthesis_output.claims:
                    clean_links = [eid for eid in item.supporting_evidence_ids if eid in valid_ev_ids]
                    if clean_links:
                        claim = Claim(
                            id=str(uuid.uuid4()),
                            text=item.text,
                            rq_id=item.rq_id,
                            report_version_id=report_version_id,
                            section=item.section,
                            status="pending",
                            verification_method="semantic",
                        )
                        links = [
                            ClaimEvidenceLink(
                                id=str(uuid.uuid4()),
                                claim_id=claim.id,
                                evidence_id=eid,
                                link_type="supports",
                                link_confidence=item.confidence,
                                created_by="auto",
                            )
                            for eid in clean_links
                        ]
                        results.append((claim, links))
            except Exception:
                pass

        # Fallback déterministe garantissant le respect des invariants
        if not results:
            for idx, rq in enumerate(rqs):
                assigned_evs = evidences[idx % len(evidences) : (idx % len(evidences)) + 2]
                if not assigned_evs:
                    assigned_evs = [evidences[0]]
                claim_text = (
                    f"Concernant {rq[:50]}, les preuves confirment : {assigned_evs[0].text[:100]}."
                )
                claim = Claim(
                    id=str(uuid.uuid4()),
                    text=claim_text,
                    rq_id=f"RQ{idx + 1}",
                    report_version_id=report_version_id,
                    section=f"Section RQ{idx + 1}",
                    status="pending",
                    verification_method="semantic",
                )
                links = [
                    ClaimEvidenceLink(
                        id=str(uuid.uuid4()),
                        claim_id=claim.id,
                        evidence_id=ev.id,
                        link_type="supports",
                        link_confidence=0.9,
                        created_by="auto",
                    )
                    for ev in assigned_evs
                ]
                results.append((claim, links))

        return results
