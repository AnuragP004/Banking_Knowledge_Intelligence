"""
==============================================================================
Viniyog One — Enterprise RAG for Banking Knowledge Intelligence
==============================================================================
Main Application Entry Point

ARCHITECTURAL OVERVIEW:
    This application implements an Enterprise-Grade RAG (Retrieval-Augmented
    Generation) system for the banking domain. The core architectural
    principle is:
    
        ╔══════════════════════════════════════════════╗
        ║  THE LLM IS NOT THE SOURCE OF TRUTH.        ║
        ║  THE KNOWLEDGE BASE IS THE SOURCE OF TRUTH. ║
        ╚══════════════════════════════════════════════╝
    
    The system enforces this principle through:
    1. Mandatory hallucination detection on every LLM response
    2. Token-based KB fallback when hallucination is detected
    3. Source attribution on all responses
    4. Role-based access control on all endpoints

SYSTEM ARCHITECTURE:
    
    ┌──────────────────────────────────────────────────────────────┐
    │                      FastAPI Application                     │
    │                                                              │
    │  ┌─────────┐    ┌──────────────┐    ┌───────────────────┐   │
    │  │  /query  │───▶│ RAG Pipeline │───▶│ Hallucination     │   │
    │  │         │    │              │    │ Detection         │   │
    │  └─────────┘    └──────┬───────┘    └────────┬──────────┘   │
    │                        │                      │              │
    │                        ▼                      ▼              │
    │  ┌──────────────────────────┐    ┌────────────────────────┐  │
    │  │    Vector Store          │    │   KB Fallback          │  │
    │  │  (ChromaDB/FAISS)       │    │   /kb/token → /kb/fetch│  │
    │  └──────────────────────────┘    └────────────────────────┘  │
    │                                                              │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │  Observability: /health, /retrieval/logs, /evaluate     │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────┘

REQUEST LIFECYCLE:
    1. Request arrives → RBAC middleware validates role
    2. Query endpoints → RAG pipeline processes query
    3. LLM generates draft response
    4. Hallucination detector evaluates groundedness
    5. If grounded (score > 0.7) → return response with sources
    6. If hallucinated (score ≤ 0.7) → drop draft
       a. Generate KB token (POST /kb/token)
       b. Fetch verified data (POST /kb/fetch with JWT)
       c. Regenerate response using ONLY KB data
       d. Return KB-grounded response
    7. All operations logged for observability

STARTUP SEQUENCE:
    1. Load configuration from environment/.env
    2. Initialize logging (loguru → file + stderr)
    3. Mount API router with all endpoints
    4. Register startup/shutdown event handlers
    5. Begin accepting requests
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.api.routes import router as api_router


# =============================================================================
# Logging Configuration
# =============================================================================

def configure_logging():
    """
    Configure loguru for structured logging.
    
    WHY LOGURU OVER STANDARD LOGGING?
    - Zero-config structured output (JSON-ready)
    - Built-in log rotation and retention
    - Exception formatting with full traceback
    - Thread-safe without additional handlers
    - Perfect for observability requirements in banking
    """
    settings = get_settings()

    # Ensure log directory exists
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler and add our own
    logger.remove()

    # Console output (human-readable for development)
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File output (structured for log aggregation)
    logger.add(
        settings.LOG_FILE,
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    logger.info("Logging configured successfully")


# =============================================================================
# Lifespan Event Handler
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler — manages startup and shutdown.
    
    STARTUP:
    - Configure logging
    - Ensure data directories exist
    - Initialize vector store connection (future)
    - Load embedding model (future)
    - Verify LLM API connectivity (future)
    
    SHUTDOWN:
    - Flush logs
    - Close database connections
    - Persist any in-memory state
    """
    # --- STARTUP ---
    configure_logging()
    settings = get_settings()

    logger.info("=" * 70)
    logger.info(f"  {settings.APP_NAME}")
    logger.info(f"  Version: {settings.APP_VERSION}")
    logger.info(f"  Debug Mode: {settings.DEBUG}")
    logger.info("=" * 70)

    # Create required data directories
    for directory in [
        settings.DATA_INGESTION_PATH,
        settings.PROCESSED_DATA_PATH,
        settings.VECTOR_STORE_PATH,
        Path(settings.KB_DATABASE_PATH).parent,
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

    # -------------------------------------------------------------------
    # Initialize Knowledge Base (Source of Truth)
    # -------------------------------------------------------------------
    from app.services.kb_service import get_kb_service
    kb_service = get_kb_service()
    kb_service.initialize()
    kb_health = kb_service.health_check()
    logger.info(
        f"Knowledge Base initialized | status={kb_health['status']} | "
        f"schemes={kb_health.get('scheme_count', 0)}"
    )

    # -------------------------------------------------------------------
    # Initialize Vector Store & RAG Pipeline
    # -------------------------------------------------------------------
    from app.services.vector_store_service import get_vector_store_service
    vector_service = get_vector_store_service()
    vector_service.initialize()
    vs_stats = vector_service.get_collection_stats()
    logger.info(
        f"Vector store initialized | vectors={vs_stats['total_vectors']} | "
        f"model={vs_stats['embedding_model']}"
    )

    # Run ingestion + chunking + indexing only if store is empty
    if vs_stats["total_vectors"] == 0:
        logger.info("Empty vector store detected — running data pipeline...")
        from app.services.ingestion_service import IngestionService
        from app.services.chunking_service import ChunkingService

        ingestion = IngestionService()
        chunker = ChunkingService()

        # Ingest banking schemes
        documents = ingestion.ingest_banking_schemes(settings.KB_JSON_PATH)
        logger.info(f"Ingested {len(documents)} banking scheme documents")

        # Also ingest any raw data files
        raw_docs = ingestion.ingest_directory(settings.DATA_INGESTION_PATH)
        documents.extend(raw_docs)

        # Chunk all documents
        chunks = chunker.chunk_documents(documents, strategy="semantic")
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Index into vector store
        indexed = vector_service.index_chunks(chunks)
        logger.info(f"Indexed {indexed} chunks into vector store")
    else:
        logger.info(
            f"Vector store already populated with {vs_stats['total_vectors']} vectors — "
            f"skipping ingestion pipeline"
        )

    logger.info("Application startup complete — ready to serve requests")

    yield  # Application is running

    # --- SHUTDOWN ---
    logger.info("Application shutdown initiated — cleaning up resources")
    # TODO: Close connections, flush buffers
    logger.info("Shutdown complete")


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Application factory pattern.
    
    WHY FACTORY PATTERN?
    - Enables creating multiple app instances for testing
    - Separates configuration from instantiation
    - Follows FastAPI best practices for large applications
    - Makes dependency injection testing straightforward
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",          # Swagger UI
        redoc_url="/redoc",        # ReDoc alternative
        openapi_url="/openapi.json",
        # Custom metadata for API documentation
        openapi_tags=[
            {
                "name": "RAG Pipeline",
                "description": (
                    "Core query endpoints. The /query endpoint processes user questions "
                    "through retrieval, generation, and hallucination detection. "
                    "The /query/debug endpoint adds full pipeline tracing for evaluators."
                ),
            },
            {
                "name": "Knowledge Base",
                "description": (
                    "Secure Knowledge Base access. Token-based authentication ensures "
                    "that KB data is only accessed through the controlled fallback path. "
                    "Direct access is prohibited — even for admin users."
                ),
            },
            {
                "name": "Evaluation",
                "description": (
                    "Testing and benchmarking endpoints for evaluating RAG pipeline "
                    "performance, groundedness, and hallucination detection accuracy."
                ),
            },
            {
                "name": "Observability",
                "description": (
                    "Monitoring and analysis endpoints for chunking strategy inspection, "
                    "retrieval quality tracking, and operational logging."
                ),
            },
            {
                "name": "System",
                "description": (
                    "Infrastructure endpoints including health checks for load balancer "
                    "integration and system status monitoring."
                ),
            },
        ],
    )

    # -------------------------------------------------------------------------
    # CORS Middleware
    # -------------------------------------------------------------------------
    # In production, origins would be restricted to specific domains.
    # For assessment, we allow all origins for easy testing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Mount API Router
    # -------------------------------------------------------------------------
    # All routes are mounted under the root prefix.
    # In production, versioned prefixes (/api/v1/) would be used.
    app.include_router(api_router, prefix="", tags=["API"])

    return app


# =============================================================================
# Application Instance
# =============================================================================
app = create_app()


# =============================================================================
# Direct Execution
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
