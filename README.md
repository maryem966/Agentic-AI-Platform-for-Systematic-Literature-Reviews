# SLR-Nexus

> **Plateforme intelligente d'assistance aux Revues Systématiques de Littérature**

[![CI](https://github.com/maryem966/Agentic-AI-Platform-for-Systematic-Literature-Reviews/actions/workflows/ci.yml/badge.svg)](https://github.com/maryem966/Agentic-AI-Platform-for-Systematic-Literature-Reviews/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Présentation

SLR-Nexus répond à une question précise :

> **Comment automatiser les étapes d'une revue systématique tout en garantissant que chaque affirmation du rapport final est scientifiquement traçable jusqu'à son passage source dans une publication réelle ?**

La plateforme s'appuie sur :
- Une **architecture hexagonale** (monolithe modulaire) garantissant la séparation domaine/infrastructure
- Cinq **agents LLM spécialisés** orchestrés par LangGraph
- Un **Evidence Registry** liant chaque affirmation à son passage source exact
- Un **RAG hybride** (BM25 + Dense + Reranking + Graph) évalué sur un Golden Dataset

---

## Architecture

```
INTERFACES  │  FastAPI · Streamlit · CLI · Tests
            ↕
APPLICATION │  Use Cases · Coordination des agents
            ↕
DOMAINE     │  Entités · Règles métier · Ports (interfaces)  ← 0 dépendance externe
            ↕
INFRA       │  PostgreSQL · Qdrant · Neo4j · Ollama · Groq · APIs scientifiques
```

**Règle fondamentale** : `src/domain/` ne contient aucun `import` vers `langchain`, `langgraph`, `fastapi`, `sqlalchemy`, `qdrant_client` ou `neo4j`.

---

## Démarrage rapide

```bash
# 1. Cloner
git clone https://github.com/maryem966/Agentic-AI-Platform-for-Systematic-Literature-Reviews.git
cd Agentic-AI-Platform-for-Systematic-Literature-Reviews

# 2. Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos valeurs réelles

# 3. Démarrer les services (PostgreSQL + Qdrant)
docker-compose -f docker/docker-compose.yml up -d
# Au premier démarrage d'un volume PostgreSQL vide, le schéma initial est appliqué automatiquement.

# 4. Installer les dépendances Python
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt

# 5. Lancer les tests unitaires (sans LLM ni BDD)
pytest tests/unit/ -v

# 6. Vérifier l'isolation du domaine
python scripts/check_domain_imports.py

# 7. Construire et lancer l'image applicative (API squelette + healthcheck)
docker build -f docker/Dockerfile -t slr-nexus .
docker run --rm -p 8000:8000 slr-nexus
# Vérifier : http://localhost:8000/health
```

La migration initiale est versionnée dans `docker/migrations/001_initial_schema.sql`.
Les vérifications de compatibilité des versions sont décrites dans `docs/dependency-compatibility.md`.

---

## Stack technique

| Couche | Technologie | Version |
|--------|-------------|---------|
| Langage | Python | 3.11.x |
| Validation | Pydantic | v2.9.2 |
| Orchestration | LangGraph | 0.2.42 |
| Composants LLM | LangChain | 0.3.7 |
| API | FastAPI | 0.115.5 |
| Interface | Streamlit | 1.40.1 |
| BDD relationnelle | PostgreSQL | 16 |
| Recherche vectorielle | Qdrant | 1.12.1 |
| Graphe | Neo4j | 5.26.0 (Phase 6+) |
| LLM local | Ollama | 0.4.2 |
| LLM distant | Groq | 0.12.0 |

---

## Roadmap

| Phase | Description | Durée | Statut |
|-------|-------------|-------|--------|
| **0** | Fondations, domaine complet, Docker, CI/CD | 2 sem | 🟡 En cours |
| **1** | Adapters PostgreSQL, Qdrant, LLM, APIs scientifiques | 2 sem | ⬜ |
| **2** | LangGraph, Planning Agent, Search Agent, HITL | 3 sem | ⬜ |
| **3** | Screening Agent, PDF ingestion, Evidence Registry | 3 sem | ⬜ |
| **4** | RAG hybride complet + Golden Dataset + benchmark | 3 sem | ⬜ |
| **5** | Research Assistant Agent + Claim Verifier 4 niveaux | 3 sem | ⬜ |
| **6** | Neo4j + Graph Retrieval + HybridGraphRAG | 3 sem | ⬜ |
| **7** | Writing Agent + Critic + Quality + rapport validé | 2 sem | ⬜ |
| **8** | Mémoire épisodique + sémantique + isolation | 2 sem | ⬜ |
| **9** | API FastAPI complète + Streamlit + Structlog | 3 sem | ⬜ |
| **10** | Tests E2E, documentation, démo | 2 sem | ⬜ |

---

## Règles invariantes

1. `src/domain/` — **zéro import externe** (vérifiée par CI et `scripts/check_domain_imports.py`)
2. **Discipline de contrôle** : `Decide → Validate (Pydantic) → Authorize → Act → Observe`
3. **Zéro claim** dans le rapport final sans `evidence_id` validé
4. **Zéro requête Cypher générée par LLM** — uniquement les templates de `src/config/cypher_templates.py`
5. **Idempotence** de toutes les opérations d'indexation et d'insertion

---

## Organisation du code

```
src/
├── domain/           ← Entités + Ports + Services (pur Python + Pydantic)
├── application/      ← Use Cases + DTOs
├── infrastructure/   ← Adapters (PostgreSQL, Qdrant, Neo4j, LLM, APIs)
├── agents/           ← 5 agents + Supervisor
├── orchestration/    ← LangGraph (State, Graph, Routing, Checkpointer)
├── rag/              ← Ingestion, Chunking, Indexation, Retrieval, Évaluation
├── evidence/         ← Registry, Verifier, CitationChecker
├── memory/           ← Épisodique (PostgreSQL) + Sémantique (Qdrant)
├── api/              ← FastAPI routes + middleware
└── config/           ← Settings, CypherTemplates, Procedures
```

---

## Contribution

Voir [docs/contributing.md](docs/contributing.md) pour les guidelines.

---

## Licence

MIT
