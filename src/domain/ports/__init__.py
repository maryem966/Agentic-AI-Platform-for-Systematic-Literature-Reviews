"""Ports (interfaces abstraites) du domaine SLR-Nexus."""
from .publication_repository import PublicationRepository
from .vector_store import VectorStore, VectorSearchResult
from .llm_port import LLMPort, LLMResponse
from .evidence_port import EvidencePort
from .kg_repository import KGRepository, GraphResult

__all__ = [
    "PublicationRepository",
    "VectorStore",
    "VectorSearchResult",
    "LLMPort",
    "LLMResponse",
    "EvidencePort",
    "KGRepository",
    "GraphResult",
]
