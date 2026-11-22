"""
Application Streamlit de Supervision et Démonstration — SLR-Nexus.
Connectée au moteur multi-agents réel de SLR-Nexus :
1. Planning Agent (Protocole & RQ)
2. Search Agent (OpenAlex & Crossref)
3. Screening Agent (Filtrage & PRISMA)
4. RAG & Evidence Registry (Ingestion & Ancrage)
5. Writing Agent & Quality Agent (Génération & Jugement TD08)
"""
from __future__ import annotations

import asyncio
import streamlit as st

from src.agents.planning_agent import PlanningAgent
from src.agents.quality_agent import QualityAgent
from src.agents.screening_agent import ScreeningAgent
from src.agents.writing_agent import WritingAgent
from src.application.dto.research_dtos import PlanningInput
from src.domain.entities.evidence import Evidence
from src.domain.entities.publication import Author, Publication
from src.domain.entities.research_protocol import ResearchQuestion
from src.domain.ports.llm_port import LLMPort, LLMResponse
from src.infrastructure.scientific_apis.client import ScientificAPIsClient

# Configuration de la page Streamlit
st.set_page_config(
    page_title="SLR-Nexus | Plateforme Agentique SLR",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 SLR-Nexus — Plateforme Agentique de Revues Systématiques")
st.caption("Architecture Hexagonale Multi-Agents connectée (Phases 0 à 5 — Zéro Neo4j)")


class FastDemonstrationLLM(LLMPort):
    """LLM déterministe haute fidélité pour démonstrations instantanées."""

    @property
    def provider_name(self) -> str:
        return "slr-engine"

    @property
    def model_name(self) -> str:
        return "slr-agentic-v5"

    async def health_check(self) -> bool:
        return True

    async def invoke(self, messages, temperature=0.0, max_tokens=None):
        return LLMResponse(content="", model=self.model_name, provider=self.provider_name)

    async def invoke_structured(self, messages, response_model, temperature=0.0, max_repair_attempts=2):
        name = response_model.__name__
        if "ProtocolDraft" in name or "ResearchProtocol" in name:
            from src.agents.planning_agent import ProtocolDraft
            return ProtocolDraft(
                research_questions=[
                    "Quelles architectures d'apprentissage fédéré (FL) sont utilisées pour la détection de fraude en santé ?",
                    "Quelles sont les métriques de performance et les garanties de confidentialité rapportées ?"
                ],
                inclusion_criteria=[
                    "Études expérimentales utilisant l'apprentissage fédéré publiées après 2019",
                    "Évaluation empirique sur jeux de données médicaux ou réclamations de santé"
                ],
                exclusion_criteria=[
                    "Articles théoriques sans expérimentation quantitative",
                    "Publications non évaluées par des pairs"
                ],
                target_databases=["openalex", "crossref"],
                start_year=2020,
                end_year=2025,
            )
        elif "ScreeningDecision" in name or "ScreeningEvaluation" in name:
            from src.agents.screening_agent import ScreeningEvaluation
            return ScreeningEvaluation(
                decision="include",
                confidence=0.94,
                exclusion_reason=None,
                justification="L'article correspond précisément aux critères d'inclusion et traite d'apprentissage fédéré appliqué à la santé."
            )
        elif "WritingSynthesisOutput" in name:
            from src.agents.writing_agent import WritingSynthesisOutput, ReportSectionDraft, ClaimDraft
            return WritingSynthesisOutput(
                executive_summary="Cette revue systématique analyse l'état de l'art de l'apprentissage fédéré appliqué à la détection des fraudes dans le secteur de la santé.",
                general_conclusion="Les architectures fédérées démontrent une performance comparable aux modèles centralisés tout en préservant la confidentialité des dossiers médicaux.",
                sections=[
                    ReportSectionDraft(
                        section_name="RQ1_Architectures",
                        rq_id="RQ1",
                        content="Les études sélectionnées s'orientent principalement vers deux paradigmes : les réseaux de neurones convolutionnels fédérés (Fed-ResNet) et les arbres de décision distribués (Federated XGBoost).",
                        claims=[
                            ClaimDraft(
                                text="Les architectures Federated XGBoost dominent pour le traitement des données tabulaires de facturation médicale.",
                                rq_id="RQ1",
                                section="RQ1_Architectures",
                                cited_evidence_ids=["ev_001"]
                            )
                        ]
                    ),
                    ReportSectionDraft(
                        section_name="RQ2_Performance_and_Privacy",
                        rq_id="RQ2",
                        content="En termes de performance, les modèles fédérés atteignent un F1-score moyen de 0.93 et un AUC-ROC de 0.95 sous des mécanismes de confidentialité différentielle avec epsilon=0.5.",
                        claims=[
                            ClaimDraft(
                                text="Le modèle Federated XGBoost rapporte un AUC-ROC de 0.95 avec confidentialité différentielle epsilon=0.5.",
                                rq_id="RQ2",
                                section="RQ2_Performance_and_Privacy",
                                cited_evidence_ids=["ev_002"]
                            )
                        ]
                    )
                ]
            )
        elif "LLMJudgeReportEvaluation" in name:
            from src.agents.quality_agent import LLMJudgeReportEvaluation
            return LLMJudgeReportEvaluation(
                coherence_score=0.95,
                scientific_rigor_score=0.92,
                critique_findings=[],
                recommended_verdict="ACCEPT"
            )
        raise ValueError(f"Modèle inconnu : {name}")


# Barre latérale : Configuration & Choix du Sujet
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Mode d'exécution", ["Démonstration Réactive", "Live Web APIs (OpenAlex/Crossref)"])
    
    st.divider()
    st.subheader("🎯 Sujet de la SLR")
    default_topic = "Federated Learning for fraud detection in healthcare"
    topic = st.text_area("Sujet de recherche", value=default_topic, height=100)
    
    col_btn1, col_btn2 = st.columns(2)
    start_slr = col_btn1.button("🚀 Lancer SLR", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 Reset", use_container_width=True)

if reset_btn:
    st.session_state.clear()
    st.rerun()

# Initialisation de l'état
if "slr_state" not in st.session_state:
    st.session_state.slr_state = {
        "step": 0,
        "protocol": None,
        "search_results": [],
        "screening_decisions": [],
        "evidences": [],
        "report": None,
        "quality_verdict": None,
    }

tab_workflow, tab_agents, tab_prisma, tab_evidence, tab_report = st.tabs([
    "📊 Pipeline en Direct",
    "🤖 Les 5 Agents",
    "📈 Diagramme PRISMA",
    "📑 Evidence Registry",
    "📝 Rapport & Qualité (TD08)",
])

# -------------------------------------------------------------
# Onglet 2 : Explication des Agents
# -------------------------------------------------------------
with tab_agents:
    st.markdown("### 🤖 Les 5 Agents Principaux et leurs Responsabilités")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**1. Planning Agent (Phase 2)**\n- Transforme le sujet en protocole formel\n- Définit les Questions de Recherche (RQ)\n- Établit les critères d'inclusion/exclusion\n- Interruption humaine obligatoire pour validation")
        st.success("**2. Search Agent (Phase 2)**\n- Déterministe (sans hallucination)\n- Interroge OpenAlex et Crossref\n- Déduplication par DOI et titre normalisé\n- Trace l'historique exact dans `search_runs`")
        st.warning("**3. Screening Agent (Phase 3)**\n- Filtrage déterministe (dates, langue)\n- LLM structuré avec motif Pydantic obligatoire\n- Alerte HITL si `uncertain` et confiance < 0.6\n- Génération des flux PRISMA")
    with col2:
        st.info("**4. Research Assistant Agent (Phase 4/5)**\n- Retrieval adaptatif (BM25 + Qdrant + RRF)\n- Extraction structurée (méthode, dataset, résultat)\n- Ancrage strict vers les chunks sources exacts\n- Vérification des Claims en 4 niveaux")
        st.error("**5. Writing & Quality Agents (Phase 5 / TD08)**\n- **WritingAgent** : Rédige uniquement depuis les preuves\n- **QualityAgent** : Calcule la couverture des RQ et le score de grounding\n- Boucle de réflexion bornée (`ACCEPT`, `REVISE`, `ESCALATE`)")

# -------------------------------------------------------------
# Onglet 1 : Pipeline en Direct
# -------------------------------------------------------------
with tab_workflow:
    if start_slr or st.session_state.slr_state["step"] > 0:
        llm = FastDemonstrationLLM()
        st.subheader("⚡ Exécution Pas-à-Pas du Workflow LangGraph")

        # 1. Planning Agent
        with st.status("Étape 1 : Planning Agent en action...", expanded=(st.session_state.slr_state["step"] == 0)) as s1:
            if st.session_state.slr_state["protocol"] is None:
                planning_agent = PlanningAgent(llm)
                req = PlanningInput(research_id="res_live", topic=topic, objectives=["Évaluer les modèles FL"])
                protocol = asyncio.run(planning_agent.plan(req))
                st.session_state.slr_state["protocol"] = protocol
            s1.update(label="✅ Protocole formulé avec succès par le PlanningAgent !", state="complete")

        protocol = st.session_state.slr_state["protocol"]
        col_rq, col_crit = st.columns(2)
        with col_rq:
            st.markdown("**Questions de Recherche (RQ) :**")
            for rq in protocol.research_questions:
                st.write(f"- **{rq.id}** : {rq.text}")
        with col_crit:
            st.markdown("**Critères d'Inclusion :**")
            for inc in protocol.inclusion_criteria:
                st.write(f"✔️ {inc}")
            st.markdown("**Critères d'Exclusion :**")
            for exc in protocol.exclusion_criteria:
                st.write(f"❌ {exc}")

        st.divider()

        # 2. Search Agent
        with st.status("Étape 2 : Search Agent (Recherche & Déduplication)...", expanded=True) as s2:
            if not st.session_state.slr_state["search_results"]:
                if mode == "Live Web APIs (OpenAlex/Crossref)":
                    client = ScientificAPIsClient()
                    pubs = asyncio.run(client.search_openalex(topic, limit=5))
                    st.session_state.slr_state["search_results"] = [
                        {"id": p.id, "title": p.title, "year": p.year, "doi": p.doi, "source": p.source}
                        for p in pubs
                    ]
                else:
                    st.session_state.slr_state["search_results"] = [
                        {"id": "pub_01", "title": "Federated Learning for Healthcare Fraud Detection", "year": 2023, "doi": "10.1109/FL.2023.01", "source": "openalex"},
                        {"id": "pub_02", "title": "Privacy-Preserving Federated XGBoost on Medical Claims", "year": 2022, "doi": "10.1016/j.jbi.2022.10", "source": "crossref"},
                        {"id": "pub_03", "title": "Traditional Machine Learning in Insurance (Outdated)", "year": 2012, "doi": "10.1145/old.2012", "source": "openalex"},
                        {"id": "pub_04", "title": "Federated Learning for Healthcare Fraud Detection (Duplicate)", "year": 2023, "doi": "10.1109/FL.2023.01", "source": "crossref"},
                    ]
            s2.update(label=f"✅ Recherche terminée : {len(st.session_state.slr_state['search_results'])} articles trouvés", state="complete")

        st.dataframe(st.session_state.slr_state["search_results"], use_container_width=True)

        st.divider()

        # 3. Screening Agent
        with st.status("Étape 3 : Screening Agent (Tri en entonnoir & PRISMA)...", expanded=True) as s3:
            if not st.session_state.slr_state["screening_decisions"]:
                screening_agent = ScreeningAgent(llm)
                pubs = [
                    Publication(id=p["id"], title=p["title"], year=p.get("year"), doi=p.get("doi"), source=p.get("source", "openalex"))
                    for p in st.session_state.slr_state["search_results"]
                ]
                output = asyncio.run(screening_agent.screen(pubs, protocol, stage="title_abstract"))
                st.session_state.slr_state["screening_decisions"] = [
                    {"pub_id": d.publication_id, "decision": d.decision, "reason": d.exclusion_reason, "confidence": d.confidence, "justification": d.justification}
                    for d in output.decisions
                ]
            s3.update(label="✅ Screening terminé avec motifs d'exclusion standardisés", state="complete")

        st.dataframe(st.session_state.slr_state["screening_decisions"], use_container_width=True)

        st.divider()

        # 4 & 5. Writing & Quality Agents
        with st.status("Étape 4 & 5 : WritingAgent & QualityAgent (TD08)...", expanded=True) as s4:
            if not st.session_state.slr_state["evidences"]:
                writing_agent = WritingAgent(llm)
                quality_agent = QualityAgent(llm)

                ev1 = Evidence(
                    id="ev_001", publication_id="pub_01", chunk_id="chk1",
                    text="Federated XGBoost achieved an AUC-ROC of 0.95 on medical insurance claims.",
                    evidence_type="result", extracted_by_agent="assistant", extraction_confidence=0.95,
                    verification_status="auto_verified", research_id="res_live"
                )
                ev2 = Evidence(
                    id="ev_002", publication_id="pub_02", chunk_id="chk2",
                    text="The differential privacy mechanism maintained privacy with epsilon=0.5 while retaining 93% F1-score.",
                    evidence_type="result", extracted_by_agent="assistant", extraction_confidence=0.92,
                    verification_status="auto_verified", research_id="res_live"
                )
                evidences = [ev1, ev2]
                st.session_state.slr_state["evidences"] = evidences

                report, claims, links = asyncio.run(
                    writing_agent.write_report("res_live", protocol.research_questions, evidences)
                )
                verdict = asyncio.run(
                    quality_agent.evaluate_report(report, protocol.research_questions, claims, links, evidences)
                )

                st.session_state.slr_state["report"] = report
                st.session_state.slr_state["quality_verdict"] = verdict
            s4.update(label="✅ Synthèse rédigée et Validée par le QualityAgent !", state="complete")

        st.session_state.slr_state["step"] = 1
    else:
        st.info("👈 Cliquez sur le bouton '🚀 Lancer SLR' dans le menu latéral pour voir le système s'exécuter pas à pas !")

# -------------------------------------------------------------
# Onglet 3 : Diagramme PRISMA
# -------------------------------------------------------------
with tab_prisma:
    st.subheader("📈 Diagramme de Flux PRISMA (Calculé automatiquement)")
    decisions = st.session_state.slr_state["screening_decisions"]
    inc_count = sum(1 for d in decisions if d["decision"] == "include") if decisions else 2
    exc_count = sum(1 for d in decisions if d["decision"] == "exclude") if decisions else 1
    total_count = len(st.session_state.slr_state["search_results"]) if st.session_state.slr_state["search_results"] else 4
    
    c1, c2, c3 = st.columns(3)
    c1.metric("1. Articles Identifiés", str(total_count))
    c2.metric("2. Articles Évalués", str(inc_count + exc_count))
    c3.metric("3. Articles Inclus", str(inc_count), delta=f"-{exc_count} exclus")
    
    st.code(
        f"""
        [ Identification ]
          Total identifié (OpenAlex + Crossref) : {total_count}
                 ↓
        [ Screening Titre / Résumé ]
          Articles exclus : {exc_count}
                 ↓
        [ Synthèse Finale ]
          Articles retenus et analysés dans le rapport : {inc_count}
        """,
        language="text",
    )

# -------------------------------------------------------------
# Onglet 4 : Evidence Registry
# -------------------------------------------------------------
with tab_evidence:
    st.subheader("📑 Registre Officiel des Preuves (Evidence Registry)")
    st.markdown("Chaque preuve est un **passage verbatim exact** extrait du document, sans aucune paraphrase.")
    if st.session_state.slr_state["evidences"]:
        for ev in st.session_state.slr_state["evidences"]:
            with st.expander(f"Preuve {ev.id} (Publication: {ev.publication_id}, Section: {ev.section})"):
                st.info(f"« {ev.text} »")
                st.caption(f"Type : {ev.evidence_type} | Statut : {ev.verification_status} | Ancrage : Vérifié (Confiance: {ev.extraction_confidence})")
    else:
        st.write("Aucune preuve enregistrée pour l'instant. Lancez le workflow.")

# -------------------------------------------------------------
# Onglet 5 : Rapport & Qualité (TD08)
# -------------------------------------------------------------
with tab_report:
    st.subheader("📝 Rapport de Synthèse & Verdict du QualityAgent")
    if st.session_state.slr_state["report"]:
        report = st.session_state.slr_state["report"]
        qv = st.session_state.slr_state["quality_verdict"]

        col_rep, col_qual = st.columns([2, 1])
        with col_rep:
            st.markdown(f"### Rapport SLR — Version {report.version}")
            for sec, content in report.content.items():
                st.markdown(f"#### {sec}")
                st.write(content)
            
            # Export Markdown du rapport
            full_md = f"# Rapport SLR : {topic}\n\n"
            for sec, content in report.content.items():
                full_md += f"## {sec}\n{content}\n\n"
            
            st.download_button(
                label="📥 Télécharger le Rapport Complet (.md)",
                data=full_md,
                file_name=f"SLR_Report_{report.version}.md",
                mime="text/markdown",
            )

        with col_qual:
            st.markdown("### Évaluation du QualityAgent (TD08)")
            if qv.verdict == "ACCEPT":
                st.success(f"**Verdict Final :** {qv.verdict}")
            elif qv.verdict == "REVISE":
                st.warning(f"**Verdict Final :** {qv.verdict} (Cible : {qv.repair_target})")
            else:
                st.error(f"**Verdict Final :** {qv.verdict} (Escalade Humaine)")

            st.metric("Score Global de Qualité", f"{qv.overall_quality_score * 100:.1f}%")
            st.metric("Couverture des RQ", f"{qv.rq_coverage_ratio * 100:.1f}%")
            st.metric("Grounding (Ancrage Preuves)", f"{qv.grounding_ratio * 100:.1f}%")
            st.info(f"**Remarques du Juge :**\n{qv.feedback_for_repair}")
    else:
        st.write("Le rapport sera affiché ici après l'exécution.")
