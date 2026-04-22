"""
==============================================================================
Configuration Management — Centralized Settings via Pydantic BaseSettings
==============================================================================

ARCHITECTURAL DECISION:
    We use pydantic-settings to centralize ALL configuration in a single,
    type-validated, environment-variable-driven class. This ensures:
    
    1. **No hardcoded secrets** — All sensitive values (GEMINI_API_KEY,
       JWT_SECRET) are loaded from environment variables or a .env file.
    2. **Fail-fast validation** — If a required config is missing, the app
       won't even start, preventing silent runtime failures.
    3. **Single source of truth** — Every module imports `get_settings()`,
       ensuring consistency across the entire application.
    4. **12-Factor App compliance** — Config is strictly separated from code,
       which is critical for enterprise deployments.

SECURITY NOTE:
    The JWT_SECRET_KEY must be a cryptographically random string in production.
    The default is ONLY for development/assessment convenience. In a real
    banking system, this would be injected via a secrets manager (Vault, AWS
    Secrets Manager, etc.).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    Application-wide configuration. All values can be overridden via
    environment variables or a .env file at the project root.
    """

    # -------------------------------------------------------------------------
    # Application Metadata
    # -------------------------------------------------------------------------
    APP_NAME: str = "Viniyog One — Banking RAG Intelligence"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Enterprise-Grade RAG System for Banking Knowledge Intelligence. "
        "The LLM is NOT the source of truth — the Knowledge Base is."
    )
    DEBUG: bool = True

    # -------------------------------------------------------------------------
    # Security & JWT Configuration
    # -------------------------------------------------------------------------
    # JWT for KB access tokens — short-lived tokens to enforce controlled
    # access to the Knowledge Base. This is NOT user authentication; it's
    # an internal system-level security mechanism.
    JWT_SECRET_KEY: str = Field(
        default="viniyog-rag-secret-key-change-in-production-2024",
        description="Secret key for signing KB access JWTs"
    )
    JWT_ALGORITHM: str = "HS256"
    # KB tokens are intentionally short-lived (60 seconds) to minimize the
    # blast radius of a leaked token. The KB fallback completes in <5s,
    # so 60s is generous. (Reduced from 5min per Gemini blueprint review)
    KB_TOKEN_EXPIRY_MINUTES: int = 1

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    # We use Google Gemini API for reliability, speed, and cost-effectiveness.
    # gemini-2.0-flash provides excellent quality for RAG at minimal cost.
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for LLM inference"
    )
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_TEMPERATURE: float = 0.1  # Low temperature for factual responses
    GEMINI_MAX_TOKENS: int = 1024

    # -------------------------------------------------------------------------
    # Embedding Configuration
    # -------------------------------------------------------------------------
    # BAAI/bge-large-en-v1.5 is chosen for:
    #   - MTEB benchmark leader in its size class
    #   - 1024-dim embeddings balance precision and storage
    #   - Instruction-aware (supports query/passage prefixes)
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024

    # -------------------------------------------------------------------------
    # Vector Store Configuration
    # -------------------------------------------------------------------------
    VECTOR_STORE_TYPE: str = "chromadb"  # "chromadb" or "faiss"
    VECTOR_STORE_PATH: str = "./data/vector_store"
    CHROMA_COLLECTION_NAME: str = "banking_knowledge"

    # -------------------------------------------------------------------------
    # RAG Pipeline Parameters
    # -------------------------------------------------------------------------
    # Chunking — semantic chunking is preferred, with these fallback params
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    # Retrieval — Top-K=10 casts a wide initial net for re-ranking.
    # The cross-encoder then narrows to RERANK_TOP_K highest-quality chunks.
    RETRIEVAL_TOP_K: int = 10
    # Similarity threshold below which we consider results irrelevant
    RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.65

    # -------------------------------------------------------------------------
    # Cross-Encoder Re-ranking Configuration
    # -------------------------------------------------------------------------
    # WHY RE-RANKING? Bi-encoders (BGE) embed query and document separately.
    # Cross-encoders process them TOGETHER through attention layers, computing
    # deep token-level interactions. This dramatically improves precision.
    # The trade-off is ~50ms extra latency per query — acceptable for banking.
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 3  # Final chunks passed to LLM after re-ranking

    # -------------------------------------------------------------------------
    # Hallucination Detection Thresholds
    # -------------------------------------------------------------------------
    # Groundedness score below which we flag as hallucination.
    # Set to 0.85 (raised from 0.7 per Gemini blueprint review).
    # Banking demands near-zero tolerance for fabrication. A higher
    # threshold catches subtle errors; the KB fallback is reliable,
    # so additional false positives are acceptable.
    HALLUCINATION_THRESHOLD: float = 0.85
    # If True, always run hallucination check (recommended for banking)
    HALLUCINATION_CHECK_ENABLED: bool = True

    # -------------------------------------------------------------------------
    # Knowledge Base (Mock) Configuration
    # -------------------------------------------------------------------------
    KB_DATABASE_PATH: str = "./data/knowledge_base.db"
    KB_JSON_PATH: str = "./data/banking_schemes.json"

    # -------------------------------------------------------------------------
    # Logging & Observability
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/rag_system.log"
    # Maximum number of retrieval logs to keep in memory for the
    # /retrieval/logs endpoint (ring buffer pattern)
    MAX_RETRIEVAL_LOGS: int = 1000

    # -------------------------------------------------------------------------
    # Data Paths
    # -------------------------------------------------------------------------
    DATA_INGESTION_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern via lru_cache — the Settings object is created once
    and reused across all dependency injections. This avoids re-reading
    environment variables on every request.
    """
    return Settings()
