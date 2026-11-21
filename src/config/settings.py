"""
Configuration centralisée — Pydantic BaseSettings.
Toutes les valeurs proviennent des variables d'environnement ou de .env.
JAMAIS de secrets dans le code source.
"""

from __future__ import annotations

from urllib.parse import quote_plus

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
    database_url: str = ""

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_chunks_collection: str = "chunks"
    qdrant_memory_collection: str = "semantic_memory"
    embedding_vector_size: int = 384

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
    openalex_base_url: str = "https://api.openalex.org"
    crossref_base_url: str = "https://api.crossref.org"
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    scientific_api_max_retries: int = 4
    scientific_api_backoff_seconds: float = 0.5

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

    @property
    def resolved_database_url(self) -> str:
        """Use the explicit URL override or safely assemble it from POSTGRES_* settings."""
        if self.database_url.strip():
            return self.database_url
        username = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        database = quote_plus(self.postgres_db)
        return (
            f"postgresql+asyncpg://{username}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{database}"
        )


# Instance singleton — importée partout dans le projet
settings = Settings()
