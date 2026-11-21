"""
Configuration centralisée — Pydantic BaseSettings.
Toutes les valeurs proviennent des variables d'environnement ou de .env.
JAMAIS de secrets dans le code source.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration de l'application SLR-Nexus."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environnement ---
    environment: str = "development"
    log_level: str = "INFO"

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "slr_nexus"
    postgres_user: str = "slr_user"
    postgres_password: str = "changeme"
    database_url: str = (
        "postgresql+asyncpg://slr_user:changeme@localhost:5432/slr_nexus"
    )

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""

    # --- Neo4j (Phase 6+) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "changeme"

    # --- Ollama (LLM local par défaut) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2:3b"

    # --- Groq (LLM distant, fallback) ---
    groq_api_key: str = ""
    groq_default_model: str = "llama-3.1-8b-instant"

    # --- APIs scientifiques ---
    semantic_scholar_api_key: str = ""
    crossref_email: str = ""

    # --- Workflow ---
    max_tokens_per_workflow: int = 100000
    max_revisions: int = 3
    screening_confidence_threshold: float = 0.6
    human_review_timeout_screening_hours: int = 24
    human_review_timeout_neo4j_hours: int = 1

    # --- LLM Fallback (Section 12.2) ---
    llm_max_repair_attempts: int = 2
    llm_retry_delay_seconds: float = 1.0
    llm_timeout_seconds: int = 60

    # --- Retrieval ---
    retrieval_top_k: int = 20
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    min_chunk_tokens: int = 100

    # --- Évaluation ---
    semantic_similarity_threshold: float = 0.7

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


# Instance singleton — importée partout dans le projet
settings = Settings()
