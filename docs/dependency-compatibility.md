# Compatibilité des versions

## Environnement ciblé

- Python supporté : `3.11.x` (la configuration du projet refuse les autres versions).
- Les dépendances directes sont épinglées dans `requirements.txt`.
- La CI installe ensemble les versions critiques Pydantic, LangChain, LangGraph et leurs intégrations, puis exécute `pip check` et vérifie leurs imports.

## Portée de la validation

La CI vérifie la résolution de la pile critique sur Python 3.11 à chaque exécution. Elle ne constitue pas un lock transitif reproductible : les dépendances indirectes restent résolues par pip. Une installation complète des bibliothèques de modèles et d'évaluation RAG doit être validée séparément avant leur utilisation.

## Historique

| Date | Python | Résultat | Détail |
|---|---|---|---|
| 2026-11-21 | 3.11.1 | Partiel | Import du domaine et 65 tests locaux réussis. La pile LangChain/LangGraph complète n'était pas installée dans l'environnement local. |
| CI | 3.11.x | À exécuter | Le workflow installe ensemble les versions critiques épinglées et vérifie résolution/importation. |
| 2026-11-22 | 3.11.x | Conflit corrigé | `ollama==0.4.2` requiert `httpx<0.28`; le pin direct est ajusté à `httpx==0.27.2`. L'installation complète reste à confirmer par la CI. |
| 2026-11-22 | 3.11.1 | Phase 2 | `langgraph==0.2.60` fournit `interrupt()` et `Command`. `langchain-core==0.3.23` respecte ses exclusions; `langgraph-checkpoint==2.0.7` satisfait le minimum exigé par `langgraph-checkpoint-postgres==2.0.6`. Imports, compilation et `pip check` locaux réussis. |

## Accès PostgreSQL Phase 2

Le test de reprise réelle se trouve dans `tests/integration/test_phase2_postgres_recovery.py`.
Il doit être lancé avec `RUN_POSTGRES_CHECKPOINTER_TESTS=1`. Le conteneur PostgreSQL est sain,
mais le test local reçoit actuellement un rejet d'authentification pour `slr_user`; aucun fichier
`.env` n'est présent dans le workspace. Copier `.env.example` vers `.env` et renseigner un mot de
passe qui correspond au rôle déjà initialisé dans PostgreSQL avant d'exécuter ce test. Le test
supprime uniquement ses checkpoints de test après une exécution réussie.
