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
