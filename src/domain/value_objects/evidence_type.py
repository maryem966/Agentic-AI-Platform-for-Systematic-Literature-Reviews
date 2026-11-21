"""Value Object : EvidenceType"""

from enum import Enum


class EvidenceType(str, Enum):
    """Types de preuves scientifiques extraites des publications."""

    RESULT = "result"
    METHOD = "method"
    LIMITATION = "limitation"
    DATASET = "dataset"
    METRIC = "metric"
    DEFINITION = "definition"
