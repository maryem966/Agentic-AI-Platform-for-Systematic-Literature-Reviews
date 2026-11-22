# Phase 4 — RAG hybride et évaluation

## État et limites actuelles

Le code de retrieval, d'évaluation et le gabarit d'annotation sont prêts. Le Golden Dataset reste **à annoter par des humains** : le dépôt ne contient pas de SLR de référence ni de jugements humains. Aucun score, modèle gagnant ou résultat supérieur à 0,70 n'est prérempli ou simulé. La sélection et le gel des modèles ne peuvent être validés qu'après cette annotation et l'exécution du benchmark.

## Golden Dataset

Le schéma impose 20 à 30 questions, une SLR source, une réponse de référence, des publications avec PDF et SHA-256, des chunks pertinents annotés, des décisions de screening avec codes d'exclusion, ainsi que des extractions validées et leurs chunks sources. Chaque annotation conserve l'annotateur, la date et l'état d'adjudication. `GoldenDataset.validate_pdf_assets()` vérifie que les PDF existent sous le dossier du dataset et correspondent à leurs hashes.

Créer 25 lignes d'annotation vides :

```powershell
.venv\Scripts\python.exe scripts\create_golden_dataset_template.py
```

Compléter `data/golden_dataset/annotations.json` à partir d'une SLR de référence, déposer les PDF dans `data/golden_dataset/pdfs/`, puis indiquer leurs chemins relatifs et hashes SHA-256. Annoter les IDs de chunks déjà indexés en PostgreSQL. Le template vide échoue volontairement au contrôle de complétude.

## Retrieval et métriques

- BM25 Okapi utilise `rank-bm25` lorsqu'il est installé, avec tokenisation déterministe et fallback interne pour les tests avant installation des extras.
- Le retrieval dense accepte Sentence Transformers et le store Qdrant; la fusion hybride utilise RRF avec `rrf_k=60` par défaut.
- Le Cross-Encoder rerank les candidats de texte narratif. Les chunks `table` et `formula` ne sont jamais envoyés au reranker.
- Le routeur applique des règles de complexité explicites et mesurables (longueur, comparaisons, relations et conjonctions). Le rapport compare le routage et le baseline hybride.
- Les métriques par stratégie sont Recall@5, Recall@10, MRR, nDCG@10 et latence. Les benchmarks d'embedding mesurent la dimension, latence d'encodage et mémoire GPU; ceux du reranker mesurent Recall@10 avant/après et coût.
- RAGAS est branché sur Faithfulness, AnswerRelevancy, ContextPrecision et ContextRecall avec l'adaptateur Ollama configurable.

Les appels Cross-Encoder traitent les paires question/passage et sont donc mesurés sur le même ensemble de candidats que le baseline hybride; la décision d'en payer le coût dépend du gain observé sur les annotations humaines. [Documentation Sentence Transformers Cross-Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html).

## Exécution du benchmark

Après annotation, indexation phase 3, installation des dépendances épinglées, démarrage PostgreSQL/Qdrant/Ollama et disponibilité des modèles :

```powershell
.venv\Scripts\python.exe scripts\benchmark_phase4.py `
  --dataset data\golden_dataset\annotations.json `
  --embedding-model <modèle-embedding-1>@<commit-immuable> `
  --embedding-model <modèle-embedding-2>@<commit-immuable> `
  --reranker-model <modèle-cross-encoder-1>@<commit-immuable> `
  --device cpu
```

Par défaut, le script exécute RAGAS pour chaque stratégie. `--skip-ragas` sert uniquement à un benchmark retrieval plus rapide. Le rapport JSON est écrit dans `reports/phase4_benchmark.json`; il inclut chaque question et les IDs récupérés pour audit.

Après examen humain des résultats, ajouter `--select-embedding <modèle@commit testé> --select-reranker <modèle@commit testé>`. Le script vérifie que les révisions exactes ont été testées et génère un fragment `.env` adjacent au rapport avec provider, noms, révisions immuables, dimension d'embedding et une collection Qdrant versionnée. Appliquer ensuite ce fragment à la configuration de déploiement et réindexer les chunks dans cette collection. Si le Cross-Encoder ne montre pas de gain utile au regard de sa latence/mémoire, ne pas le sélectionner.

Le seuil `Hybrid Recall@10 > 0.70` est rapporté comme objectif mesuré, jamais garanti par le code. Les changements de dépendances restent sous les versions déjà épinglées dans `requirements.txt`; les modèles et leurs révisions doivent être gelés à partir du rapport annoté.
