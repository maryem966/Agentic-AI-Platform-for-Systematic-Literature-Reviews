"""Value Object : ClaimStatus"""

from enum import Enum


class ClaimStatus(str, Enum):
    """Statut de vérification d'un Claim."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    PENDING = "pending"
