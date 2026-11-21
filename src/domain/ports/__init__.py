"""Ports (interfaces abstraites) du domaine SLR-Nexus."""

from .evidence_port import EvidencePort
from .kg_repository import GraphResult, KGRepository
from .llm_port import LLMPort, LLMResponse
from .protocol_repository import ProtocolRepository
from .publication_repository import PublicationRepository
from .research_project_repository import ResearchProjectRepository
from .vector_store import VectorSearchResult, VectorStore

__all__ = [
    "PublicationRepository",
    "VectorStore",
    "VectorSearchResult",
    "LLMPort",
    "LLMResponse",
    "EvidencePort",
    "KGRepository",
    "GraphResult",
    "ResearchProjectRepository",
    "ProtocolRepository",
]
