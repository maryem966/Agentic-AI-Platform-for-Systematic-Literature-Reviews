# Phase 3 — Screening, PDF et Evidence Registry

## Flux implémenté

- Le screening titre/résumé applique d'abord les signaux déterministes explicites, puis appelle le LLM avec une sortie Pydantic structurée. Les critères sont conservés dans chaque décision; les motifs sont limités aux codes du domaine. Une information manquante produit `uncertain`, jamais une exclusion automatique.
- `uncertain` avec une confiance inférieure à 0,6 suspend LangGraph. `POST /api/v1/research/{id}/validate` accepte `include` ou `exclude` à cette étape et demande un motif standardisé pour une exclusion. Les décisions et revues humaines sont persistées.
- Les PDF sont téléchargés uniquement depuis HTTPS public. Chaque redirection est contrôlée contre les adresses privées/locales; taille, MIME et signature PDF sont vérifiés. L'extraction conserve les passages source et signale les formulations d'injection potentielles sans modifier le document. Le texte transmis au LLM est encadré comme donnée non fiable.
- Le texte est découpé avec une fenêtre de 512 tokens estimés par un tokenizer déterministe, un chevauchement de 64 et une fusion des fenêtres finales de moins de 100 tokens. Les tables et formules sont classées par heuristiques. Les métadonnées sont enregistrées dans PostgreSQL; les embeddings sont indexés dans Qdrant avec les identifiants de publication, document et chunk. Le tokenizer exact sera aligné au modèle d'embedding retenu après le benchmark de Phase 4.
- `EvidenceRegistry` valide l'existence du chunk, l'identité de sa publication et l'exactitude littérale de l'extrait avant insertion. Il expose création, lecture, liste, transition des six statuts et suppression.
- `GET /api/v1/research/{id}/prisma` recalcule le flux depuis les exécutions de recherche, décisions de screening et états des documents stockés.

## Modèle d'embedding temporaire

La configuration par défaut est `all-minilm` avec 384 dimensions, cohérente avec la collection Qdrant déjà configurée. Le modèle se change via `EMBEDDING_MODEL`; l'adapter refuse une dimension différente de `EMBEDDING_VECTOR_SIZE` afin d'éviter un index incohérent. PyMuPDF reste la dépendance d'extraction définie dans `requirements.txt`.

## Vérification locale

Les tests unitaires couvrent le chevauchement et les offsets des chunks, la fusion de petits chunks, le traitement prudent d'un abstract manquant et le refus d'un fichier sans signature PDF. La compilation Python, Ruff et les tests ciblés phase 2/domaine passent.

Le poste courant n'a pas PyMuPDF installé dans son environnement virtuel. L'extraction d'un vrai PDF et un run sur cinq publications nécessitent l'installation des dépendances et des services locaux PostgreSQL, Qdrant, Ollama ainsi que des accès aux sources scientifiques. Aucun corpus ni secret/API externe n'a été inventé pour simuler ce livrable.
