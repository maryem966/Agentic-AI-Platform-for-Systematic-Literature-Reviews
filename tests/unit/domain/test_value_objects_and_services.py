"""
Tests unitaires : Value Objects et Services du domaine.
Aucune dépendance externe — pas de LLM, pas de base de données.
"""
from __future__ import annotations

import pytest

from src.domain.value_objects.evidence_type import EvidenceType
from src.domain.value_objects.screening_stage import ScreeningStage, ScreeningDecisionValue
from src.domain.value_objects.claim_status import ClaimStatus
from src.domain.value_objects.memory_policy import MemoryPolicy
from src.domain.services.entity_normalizer import EntityNormalizer
from src.domain.services.prisma_counter import PRISMACounter, PRISMAFlow


# ============================================================
# Value Objects
# ============================================================

class TestEvidenceType:

    def test_all_types_defined(self):
        types = {e.value for e in EvidenceType}
        assert "result" in types
        assert "method" in types
        assert "limitation" in types
        assert "dataset" in types
        assert "metric" in types
        assert "definition" in types

    def test_string_comparison(self):
        assert EvidenceType.RESULT == "result"
        assert EvidenceType.METHOD == "method"


class TestScreeningStage:

    def test_stages(self):
        assert ScreeningStage.TITLE_ABSTRACT == "title_abstract"
        assert ScreeningStage.FULL_TEXT == "full_text"

    def test_decision_values(self):
        assert ScreeningDecisionValue.INCLUDE == "include"
        assert ScreeningDecisionValue.EXCLUDE == "exclude"
        assert ScreeningDecisionValue.UNCERTAIN == "uncertain"


class TestClaimStatus:

    def test_all_statuses(self):
        statuses = {s.value for s in ClaimStatus}
        assert "supported" in statuses
        assert "contradicted" in statuses
        assert "pending" in statuses


class TestMemoryPolicy:

    def test_default_isolated(self):
        policy = MemoryPolicy()
        assert policy.scope == "research"
        assert policy.allow_cross_research is False
        assert policy.is_isolated is True
        assert policy.provenance_required is True

    def test_cross_research_policy(self):
        policy = MemoryPolicy(scope="user", allow_cross_research=True)
        assert policy.is_isolated is False

    def test_system_policy(self):
        policy = MemoryPolicy(scope="system", allow_cross_research=True)
        assert policy.scope == "system"


# ============================================================
# EntityNormalizer
# ============================================================

class TestEntityNormalizer:

    def setup_method(self):
        self.normalizer = EntityNormalizer()

    def test_normalize_method_rf(self):
        assert self.normalizer.normalize_method("rf") == "Random Forest"
        assert self.normalizer.normalize_method("RF") == "Random Forest"
        assert self.normalizer.normalize_method("Random Forest") == "Random Forest"
        assert self.normalizer.normalize_method("random forest") == "Random Forest"

    def test_normalize_method_svm(self):
        assert self.normalizer.normalize_method("svm") == "Support Vector Machine"

    def test_normalize_method_unknown(self):
        assert self.normalizer.normalize_method("CustomModel") == "CustomModel"

    def test_normalize_metric_f1(self):
        assert self.normalizer.normalize_metric("f1") == "F1-score"
        assert self.normalizer.normalize_metric("F1-score") == "F1-score"
        assert self.normalizer.normalize_metric("f1-score") == "F1-score"

    def test_normalize_metric_auc(self):
        assert self.normalizer.normalize_metric("auc") == "AUC-ROC"
        assert self.normalizer.normalize_metric("auc-roc") == "AUC-ROC"

    def test_normalize_dataset_strips(self):
        assert self.normalizer.normalize_dataset("  My Dataset  ") == "My Dataset"


# ============================================================
# PRISMACounter
# ============================================================

class TestPRISMACounter:

    def test_build_from_counts_consistent(self):
        counter = PRISMACounter()
        flow = counter.build_from_counts(
            identified_by_source={"openalex": 100, "crossref": 50},
            duplicates_removed=20,
            excluded_title_abstract=80,
            excluded_title_abstract_reasons={
                "wrong_population": 40,
                "wrong_intervention": 40,
            },
            inaccessible=5,
            excluded_full_text=10,
            excluded_full_text_reasons={"insufficient_data": 10},
            included=35,
        )
        assert flow.total_identified == 150
        assert flow.after_deduplication == 130
        # assessed = 130 - 80 = 50
        assert flow.assessed_full_text == 50

    def test_prisma_flow_consistency_check(self):
        """Un PRISMAFlow cohérent ne retourne pas d'erreurs."""
        counter = PRISMACounter()
        flow = counter.build_from_counts(
            identified_by_source={"openalex": 60},
            duplicates_removed=10,
            excluded_title_abstract=20,
            excluded_title_abstract_reasons={"wrong_population": 20},
            inaccessible=2,
            excluded_full_text=8,
            excluded_full_text_reasons={"insufficient_data": 8},
            included=20,
        )
        errors = flow.validate_consistency()
        assert errors == [], f"Erreurs inattendues : {errors}"

    def test_prisma_text_generation(self):
        flow = PRISMAFlow(
            identified_by_source={"openalex": 42},
            total_identified=42,
            included=15,
        )
        text = flow.to_prisma_text()
        assert "42" in text
        assert "15" in text
