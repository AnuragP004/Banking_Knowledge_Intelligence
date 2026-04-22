"""
==============================================================================
API Schemas — Pydantic Request/Response Models
==============================================================================

All API contracts are defined here to:
1. Enforce input validation at the boundary (reject bad data early)
2. Provide automatic OpenAPI documentation
3. Decouple internal representations from API surface
4. Enable contract testing
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# Query Endpoints
# =============================================================================

class QueryRequest(BaseModel):
    """Standard user query payload."""
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The user's natural language question about banking schemes/policies",
        examples=["What are the eligibility criteria for the PM Mudra Yojana?"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (default 5, max 20)",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source attribution in the response",
    )


class SourceAttribution(BaseModel):
    """Source reference for response grounding."""
    document_name: str
    chunk_id: str
    relevance_score: float
    text_preview: str = Field(description="First 200 chars of the source chunk")


class QueryResponse(BaseModel):
    """Standard query response with mandatory source attribution."""
    query: str
    answer: str
    sources: list[SourceAttribution] = []
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Model's confidence in the response groundedness",
    )
    was_kb_fallback: bool = Field(
        default=False,
        description="Whether the KB fallback mechanism was triggered",
    )
    processing_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DebugQueryResponse(QueryResponse):
    """
    Extended response for /query/debug — includes full pipeline trace.
    Restricted to evaluator+ roles for security (exposes internal state).
    """
    retrieval_trace: list[dict[str, Any]] = Field(
        default=[],
        description="Full retrieval pipeline trace including scores",
    )
    hallucination_check: dict[str, Any] = Field(
        default={},
        description="Hallucination detection results",
    )
    kb_fallback_trace: Optional[dict[str, Any]] = Field(
        default=None,
        description="KB fallback details if triggered",
    )
    chunks_used: list[dict[str, Any]] = Field(
        default=[],
        description="Raw chunks used for generation",
    )
    prompt_sent: Optional[str] = Field(
        default=None,
        description="Actual prompt sent to the LLM",
    )
    token_usage: Optional[dict[str, int]] = Field(
        default=None,
        description="Token count breakdown",
    )


# =============================================================================
# Knowledge Base Endpoints
# =============================================================================

class KBFetchRequest(BaseModel):
    """Request payload for fetching data from the Knowledge Base."""
    query: str = Field(
        ...,
        description="The query to search the Knowledge Base",
    )
    scheme_id: Optional[str] = Field(
        default=None,
        description="Specific scheme ID to fetch (bypasses search)",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of KB records to return",
    )


class KBRecord(BaseModel):
    """A single record from the Knowledge Base."""
    scheme_id: str
    scheme_name: str
    category: str
    description: str
    eligibility: Optional[str] = None
    benefits: Optional[str] = None
    interest_rate: Optional[str] = None
    source: str = "knowledge_base"
    last_verified: Optional[str] = None


class KBFetchResponse(BaseModel):
    """Response from the Knowledge Base."""
    records: list[KBRecord]
    total_found: int
    query: str
    token_valid: bool = True


# =============================================================================
# Evaluation Endpoints
# =============================================================================

class EvaluationRequest(BaseModel):
    """Request payload for running evaluation."""
    test_queries: list[str] = Field(
        ...,
        min_length=1,
        description="List of queries to evaluate",
    )
    expected_answers: Optional[list[str]] = Field(
        default=None,
        description="Optional ground truth answers for comparison",
    )
    include_hallucination_analysis: bool = True


class EvaluationMetrics(BaseModel):
    """Evaluation results for a single query."""
    query: str
    generated_answer: str
    groundedness_score: float
    relevance_score: float
    hallucination_detected: bool
    kb_fallback_triggered: bool
    response_time_ms: float


class EvaluationResponse(BaseModel):
    """Aggregated evaluation results."""
    total_queries: int
    avg_groundedness_score: float
    avg_relevance_score: float
    hallucination_rate: float
    kb_fallback_rate: float
    avg_response_time_ms: float
    results: list[EvaluationMetrics]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Chunk Inspection
# =============================================================================

class ChunkInspectRequest(BaseModel):
    """Request to inspect chunking of a document or text."""
    text: Optional[str] = Field(
        default=None,
        description="Raw text to chunk (for analysis)",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="ID of an already-ingested document",
    )


class ChunkInfo(BaseModel):
    """Detailed information about a single chunk."""
    chunk_id: str
    text: str
    char_count: int
    token_count_approx: int
    metadata: dict[str, Any] = {}
    semantic_boundary: Optional[str] = Field(
        default=None,
        description="Detected semantic boundary type (paragraph, section, etc.)",
    )


class ChunkInspectResponse(BaseModel):
    """Response with chunk analysis details."""
    total_chunks: int
    avg_chunk_size: float
    chunking_strategy: str
    chunks: list[ChunkInfo]


# =============================================================================
# Retrieval Logs
# =============================================================================

class RetrievalLogEntry(BaseModel):
    """A single retrieval operation log entry."""
    timestamp: str
    query: str
    top_k: int
    chunks_retrieved: int
    avg_similarity_score: float
    max_similarity_score: float
    min_similarity_score: float
    retrieval_time_ms: float
    hallucination_detected: bool
    kb_fallback_triggered: bool


class RetrievalLogsResponse(BaseModel):
    """Collection of retrieval logs for observability."""
    total_logs: int
    logs: list[RetrievalLogEntry]
    period_start: Optional[str] = None
    period_end: Optional[str] = None


# =============================================================================
# Health Check
# =============================================================================

class ComponentHealth(BaseModel):
    """Health status of a single system component."""
    name: str
    status: str = Field(description="healthy | degraded | unhealthy")
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """System health status with component-level breakdown."""
    status: str = Field(description="healthy | degraded | unhealthy")
    version: str
    uptime_seconds: float
    components: list[ComponentHealth]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
