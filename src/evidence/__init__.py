"""Registre de preuves et vérificateur scientifique de claims en 4 niveaux."""

from .registry import EvidenceRegistry
from .verifier import FourLevelClaimVerifier, NLIEvaluationOutput, cosine_similarity

__all__ = [
    "EvidenceRegistry",
    "FourLevelClaimVerifier",
    "NLIEvaluationOutput",
    "cosine_similarity",
]
