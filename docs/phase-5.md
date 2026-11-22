# Phase 5 — Research Assistant Agent et Claim Verifier 4 niveaux

## Architecture implémentée

La Phase 5 implémente les composants d'assistance à la recherche scientifique et de vérification rigoureuse des affirmations (claims) en conformité avec la section 11.1 du Cahier des Charges et les invariants fondamentaux de SLR-Nexus.

```
Publications incluses (Screening)
               │
               ▼
┌──────────────────────────────┐
│   Research Assistant Agent   │
├──────────────────────────────┤
│ 1. Extraction structurée     │ ──► ExtractionRecord (method, dataset, metric, result, limitation)
│ 2. Ancrage littéral          │ ──► Evidence (passage source exact, chunk_id, non-paraphrasé)
│ 3. Formulation de synthèses  │ ──► Claim + ClaimEvidenceLink (zéro claim sans preuve)
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  FourLevelClaimVerifier      │
├──────────────────────────────┤
│ Niveau 1 : Existence         │ ── Vérifie l'existence de chaque evidence_id dans le registre
│ Niveau 2 : Provenance        │ ── Vérifie la concordance chunk/publication et l'exactitude textuelle
│ Niveau 3 : Sémantique        │ ── Mesure la similarité cosinus (embedding claim ↔ passage source)
│ Niveau 4 : Entailment (NLI)  │ ── Évalue l'inférence logique (entailment, neutral, contradiction)
└──────────────────────────────┘
               │
               ▼
Mise à jour du cycle de vie des preuves :
- auto_verified si entailment validé (score >= seuil)
- auto_rejected / contested si contradiction ou support insuffisant (requiert revue humaine)
```

## 1. Research Assistant Agent (`src/agents/research_assistant_agent.py`)

- **Extraction structurée** : extrait les champs scientifiques normalisés (`method`, `dataset`, `metric`, `result`, `limitation`) enregistrés dans `ExtractionRecord` et ancrés dans des `source_chunk_ids` traçables.
- **Ancrage littéral des preuves** : chaque `Evidence` conserve le passage textuel littéral exact extrait du chunk (INVARIANT : aucun texte n'est paraphrasé ou altéré).
- **Formulation de Claims** : formule des affirmations synthétiques répondant aux questions de recherche du protocole (`rq_id`) et génère systématiquement les liens `ClaimEvidenceLink` (INVARIANT : aucun claim dans le rapport sans preuve source).
- **Support hybride LLM / Heuristique** : utilise la sortie structurée Pydantic (`PublicationExtractionLLMOutput`, `SynthesisLLMOutput`) avec fallback déterministe garantissant le passage des tests hors-ligne sans dépendance réseau.

## 2. Claim Verifier à 4 niveaux (`src/evidence/verifier.py`)

Le service implémente l'interface abstraite du domaine `ClaimVerifier` (`src/domain/services/claim_verifier.py`) en 4 niveaux séquentiels :

1. **Niveau 1 — Existence** :
   - Échoue immédiatement avec `status="unsupported"` et `verification_method="existence"` si aucune preuve n'est liée ou si un identifiant d'evidence est introuvable dans le registre.
2. **Niveau 2 — Provenance** :
   - Vérifie la cohérence entre le chunk source et la publication citée.
   - Contrôle que le texte de la preuve est strictement contenu dans le chunk source.
3. **Niveau 3 — Pertinence sémantique** :
   - Calcule la similarité cosinus entre les embeddings vectoriels du claim et du passage source.
   - Rejette les affirmations dont la similarité est inférieure au seuil (`similarity_threshold`, configurable).
4. **Niveau 4 — Entailment (NLI)** :
   - Évalue la relation logique entre le passage source (prémisse) et le claim (hypothèse) :
     - `entailment` : claim accepté (`status="supported"`), ne nécessite pas d'arbitrage humain bloquant.
     - `contradiction` : claim contesté (`status="contradicted"`), enregistre `contradicting_evidence_ids` et signale `requires_human_review=True`.
     - `neutral` : claim partiellement supporté (`status="partially_supported"`), nécessite une revue humaine.
- **Détection des contradictions globales** (`find_contradicting_evidence`) :
   - Scanne le registre de preuves pour identifier les publications affirmant des résultats opposés ou contradictoires avant l'approbation finale.

## 3. Intégration dans le workflow LangGraph (`src/orchestration/workflow.py`)

Le nœud `evidence_extraction` remplace le stub antérieur :
- Charge le protocole et les publications retenues au screening.
- Orchestre l'extraction via `ResearchAssistantAgent`.
- Persiste les preuves dans `EvidenceRegistry` et les enregistrements d'extraction dans `ExtractionRecordRepo`.
- Exécute le `FourLevelClaimVerifier` et associe les `ClaimEvidenceLink`.
- Fait progresser les statuts des preuves (`auto_verified`, `auto_rejected`, `contested`).

## 4. Vérification et tests

La suite de tests unitaires dédiée `tests/unit/test_phase5_assistant_and_verifier.py` valide :
- L'échec au niveau 1 lors d'absence ou d'inexistence de preuve.
- L'échec au niveau 2 lors d'une incohérence chunk/publication ou d'une preuve non textuellement conforme.
- L'échec au niveau 3 lors d'une divergence sémantique (vecteurs orthogonaux).
- Le succès au niveau 4 sur entailment, et la détection d'une contradiction logique.
- La recherche des contradictions croisées à l'échelle du registre.
- L'extraction structurée et la formulation de claims par le Research Assistant Agent.
- L'exécution avec mock de sortie structurée LLM.
