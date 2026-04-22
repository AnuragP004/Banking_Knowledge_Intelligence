"""
==============================================================================
RAG Pipeline Service — Retrieval-Augmented Generation Orchestrator
==============================================================================

ARCHITECTURAL ROLE:
    This is the central orchestrator of the RAG system. It coordinates:
    
    1. RETRIEVAL:  Query → Vector Store → Top-K relevant chunks
    2. GENERATION: Retrieved context + query → LLM → Draft response
    3. VALIDATION: Draft response → Hallucination Detector → Score
    4. FALLBACK:   If hallucinated → KB Token → KB Fetch → Regenerate
    5. RESPONSE:   Final grounded answer + source attribution
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    RAG Pipeline Flow                            │
    │                                                                │
    │  Query ──▶ Embed ──▶ Vector Search ──▶ Retrieve Top-K         │
    │                                          │                     │
    │                                          ▼                     │
    │                              Build Prompt with Context         │
    │                                          │                     │
    │                                          ▼                     │
    │                              LLM Generation (Gemini Flash)    │
    │                                          │                     │
    │                                          ▼                     │
    │                          Hallucination Detection               │
    │                                  │              │              │
    │                              Grounded       Hallucinated       │
    │                                  │              │              │
    │                                  ▼              ▼              │
    │                          Return Answer    KB Fallback Path     │
    │                                              │                 │
    │                                              ▼                 │
    │                                    /kb/token → /kb/fetch       │
    │                                              │                 │
    │                                              ▼                 │
    │                                    Regenerate with KB Data     │
    │                                              │                 │
    │                                              ▼                 │
    │                                    Return KB-Grounded Answer   │
    └─────────────────────────────────────────────────────────────────┘

PROMPT ENGINEERING:
    The system prompt is carefully designed for the banking domain:
    - Instructs the LLM to ONLY use provided context
    - Forbids fabrication of schemes, rates, or eligibility
    - Requires source attribution in the response
    - Maintains a professional banking advisory tone
    
    WHY Gemini 2.0 Flash?
    - 1M token context window (fits massive retrieval sets)
    - Free tier available (ideal for assessment/development)
    - Excellent instruction-following for RAG tasks
    - Fast (< 2 seconds for typical banking queries)
    - The architecture doesn't rely on LLM quality — KB fallback catches errors
"""

import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

class GroundedResponse(BaseModel):
    answer: str = Field(description="The generated answer to the user's query.")
    claims_total: int = Field(description="Total number of atomic claims in the answer.")
    claims_supported: int = Field(description="Number of claims supported by the context.")
    self_evaluated_groundedness: float = Field(description="Groundedness score from 0.0 to 1.0.")
    unsupported_details: str = Field(description="Details of any unsupported claims, or 'None'.")
    reasoning: str = Field(description="Brief reasoning for the groundedness score.")

from loguru import logger

from app.core.config import get_settings


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class RetrievalResult:
    """A single retrieved chunk with metadata."""
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete RAG pipeline output."""
    query: str
    answer: str
    sources: list[RetrievalResult]
    confidence_score: float
    was_kb_fallback: bool
    hallucination_score: Optional[float] = None
    hallucination_reasoning: Optional[str] = None
    processing_time_ms: float = 0.0
    token_usage: Optional[dict] = None
    prompt_sent: Optional[str] = None
    retrieval_trace: list[dict] = field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# System Prompts
# =============================================================================

BANKING_SYSTEM_PROMPT = """You are a knowledgeable and precise Indian banking advisor AI, part of the Viniyog One Enterprise RAG system.

CRITICAL RULES:
1. You MUST ONLY answer based on the provided CONTEXT below. NEVER fabricate information.
2. If the context does not contain sufficient information to answer, say: "Based on the available knowledge base, I don't have sufficient information to answer this query accurately."
3. NEVER invent scheme names, interest rates, eligibility criteria, or any banking data.
4. Always mention which banking scheme(s) you are referencing in your answer.
5. Use a professional but accessible tone suitable for banking customers.
6. When citing interest rates or monetary amounts, use the exact figures from the context.
7. Structure your response clearly with relevant details about eligibility, benefits, and key features.

FORMAT:
- Begin with a direct answer to the question
- Provide key details (eligibility, benefits, amounts) if relevant
- End with the scheme name and source attribution
"""

KB_FALLBACK_SYSTEM_PROMPT = """You are a knowledgeable Indian banking advisor AI. You are answering from VERIFIED Knowledge Base data that has been manually curated and validated.

CRITICAL RULES:
1. This data comes from the authoritative Knowledge Base — it is the SOURCE OF TRUTH.
2. Answer ONLY using the provided KB records below. Do not add any information not present in the data.
3. Present the information clearly and accurately.
4. Always reference the scheme name and scheme ID.
5. Use the exact figures, eligibility criteria, and benefits from the KB data.

This is a KB-FALLBACK response — the initial RAG retrieval was insufficient, so we are using verified, curated data instead.
"""


# =============================================================================
# RAG Pipeline Service
# =============================================================================

class RAGPipelineService:
    """
    Orchestrates the full RAG pipeline: Retrieve → Generate → Validate → Respond.
    
    Uses Google Gemini as the LLM backend.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None
        self._initialized = False
        # In-memory ring buffer for retrieval logs — capped at 500 entries
        # This provides the data for /retrieval/logs endpoint.
        from collections import deque
        self._retrieval_logs: deque = deque(maxlen=500)

    def initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._initialized:
            return
        
        if self._settings.GEMINI_API_KEY:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._settings.GEMINI_API_KEY)
                logger.info(f"Gemini client initialized | model={self._settings.GEMINI_MODEL}")
            except Exception as e:
                logger.error(f"Gemini initialization failed: {e}")
                self._client = None
        else:
            logger.warning(
                "Gemini API key not configured — LLM generation will use "
                "context-only responses (no hallucination possible)"
            )
        
        self._initialized = True

    async def process_query(
        self,
        query: str,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> PipelineResult:
        """
        Execute the full RAG pipeline.
        
        Pipeline:
        1. Retrieve top-K chunks from vector store
        2. Build prompt with retrieved context
        3. Generate draft response via LLM
        4. Run hallucination detection
        5. If hallucinated: trigger KB fallback
        6. Return final grounded response
        """
        self.initialize()
        start_time = time.time()
        retrieval_trace = []

        # =================================================================
        # Step 1: RETRIEVAL — Embed query and search vector store
        # =================================================================
        retrieval_start = time.time()
        from app.services.vector_store_service import get_vector_store_service
        vector_service = get_vector_store_service()
        
        raw_results = vector_service.search(query=query, top_k=top_k)
        
        sources = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in raw_results
        ]
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        retrieval_trace.append({
            "stage": "vector_search",
            "status": "completed",
            "chunks_found": len(sources),
            "top_score": sources[0].score if sources else 0.0,
            "time_ms": round(retrieval_time, 1),
        })
        
        logger.debug(
            f"Retrieval complete | query='{query[:50]}' | "
            f"chunks={len(sources)} | top_score={sources[0].score if sources else 0:.3f}"
        )

        # =================================================================
        # Step 2: GENERATION — Build prompt and call LLM
        # =================================================================
        if not sources:
            return PipelineResult(
                query=query,
                answer=(
                    "I could not find any relevant information in the banking "
                    "knowledge base for your query. Please try rephrasing or "
                    "ask about a specific Indian banking scheme."
                ),
                sources=sources,
                confidence_score=0.0,
                was_kb_fallback=False,
                processing_time_ms=(time.time() - start_time) * 1000,
                retrieval_trace=retrieval_trace,
            )

        # Build context from retrieved chunks
        context = self._build_context(sources)
        prompt = self._build_user_prompt(query, context)
        
        # Generate response
        gen_start = time.time()
        structured_ans, token_usage = await self._generate(
            system_prompt=BANKING_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        gen_time = (time.time() - gen_start) * 1000
        
        answer = structured_ans.answer if structured_ans else ""
        
        retrieval_trace.append({
            "stage": "generation",
            "status": "completed",
            "model": self._settings.GEMINI_MODEL,
            "time_ms": round(gen_time, 1),
            "token_usage": token_usage,
        })

        # =================================================================
        # Step 3: HALLUCINATION DETECTION
        # =================================================================
        from app.services.hallucination_detector import get_hallucination_detector
        detector = get_hallucination_detector()
        
        detection_start = time.time()
        
        # Pass the pre-computed self-eval metrics directly to the detector
        detection_result = await detector.evaluate(
            query=query,
            answer=answer,
            context_chunks=[s.text for s in sources],
            llm_eval_score=structured_ans.self_evaluated_groundedness if structured_ans else 1.0,
            llm_claims_total=structured_ans.claims_total if structured_ans else 0,
            llm_claims_supported=structured_ans.claims_supported if structured_ans else 0,
            llm_unsupported_details=structured_ans.unsupported_details if structured_ans else "",
            llm_reasoning=structured_ans.reasoning if structured_ans else "",
        )
        detection_time = (time.time() - detection_start) * 1000
        
        retrieval_trace.append({
            "stage": "hallucination_detection",
            "status": "completed",
            "groundedness_score": detection_result.groundedness_score,
            "is_hallucinated": detection_result.is_hallucinated,
            "reasoning": detection_result.reasoning,
            "time_ms": round(detection_time, 1),
        })
        
        logger.info(
            f"Hallucination check | score={detection_result.groundedness_score:.2f} | "
            f"hallucinated={detection_result.is_hallucinated} | "
            f"query='{query[:50]}'"
        )

        # =================================================================
        # Step 4: KB FALLBACK (if hallucination detected)
        # =================================================================
        was_kb_fallback = False
        
        if detection_result.is_hallucinated:
            logger.warning(
                f"Hallucination detected! Triggering KB fallback | "
                f"score={detection_result.groundedness_score:.2f} | "
                f"query='{query[:50]}'"
            )
            
            fallback_start = time.time()
            fallback_answer, fallback_sources = await self._kb_fallback(query)
            fallback_time = (time.time() - fallback_start) * 1000
            
            if fallback_answer:
                answer = fallback_answer
                was_kb_fallback = True
                # Append KB sources
                for ks in fallback_sources:
                    sources.append(ks)
                
                retrieval_trace.append({
                    "stage": "kb_fallback",
                    "status": "triggered",
                    "reason": f"Groundedness score {detection_result.groundedness_score:.2f} < threshold {self._settings.HALLUCINATION_THRESHOLD}",
                    "kb_records_used": len(fallback_sources),
                    "time_ms": round(fallback_time, 1),
                })
            else:
                retrieval_trace.append({
                    "stage": "kb_fallback",
                    "status": "failed",
                    "reason": "KB returned no results for query",
                })

        # =================================================================
        # Step 5: BUILD FINAL RESPONSE
        # =================================================================
        confidence = (
            detection_result.groundedness_score
            if not was_kb_fallback
            else 0.95  # KB data is always high confidence (source of truth)
        )
        
        processing_time = (time.time() - start_time) * 1000

        # =================================================================
        # Step 6: RECORD RETRIEVAL LOG (for /retrieval/logs endpoint)
        # =================================================================
        scores = [s.score for s in sources] if sources else [0.0]
        self._retrieval_logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "top_k": top_k,
            "chunks_retrieved": len(sources),
            "avg_similarity_score": round(sum(scores) / max(len(scores), 1), 4),
            "max_similarity_score": round(max(scores), 4) if scores else 0.0,
            "min_similarity_score": round(min(scores), 4) if scores else 0.0,
            "retrieval_time_ms": round(retrieval_time, 1),
            "hallucination_detected": detection_result.is_hallucinated,
            "kb_fallback_triggered": was_kb_fallback,
        })

        return PipelineResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence_score=round(confidence, 4),
            was_kb_fallback=was_kb_fallback,
            hallucination_score=detection_result.groundedness_score,
            hallucination_reasoning=detection_result.reasoning,
            processing_time_ms=round(processing_time, 1),
            token_usage=token_usage,
            prompt_sent=prompt if include_sources else None,
            retrieval_trace=retrieval_trace,
        )

    # =========================================================================
    # Retrieval Logs (for /retrieval/logs endpoint)
    # =========================================================================

    def get_retrieval_logs(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """
        Return recent retrieval operation logs from the in-memory ring buffer.

        The buffer is capped at 500 entries (most recent). Logs include
        query text, retrieval scores, timing, and hallucination outcomes.
        """
        logs = list(self._retrieval_logs)
        # Reverse so newest first
        logs.reverse()
        return logs[offset:offset + limit]

    # =========================================================================
    # LLM Generation (Gemini)
    # =========================================================================

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[Optional[GroundedResponse], Optional[dict]]:
        """
        Generate a strictly structured and self-evaluated response using Google Gemini.
        """
        if not self._client:
            return self._fallback_no_llm(user_prompt), None
        
        try:
            from google.genai import types
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Use structure output to force the single API call model
            response = self._client.models.generate_content(
                model=self._settings.GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, # Needs to be deterministic for eval accuracy
                    max_output_tokens=self._settings.GEMINI_MAX_TOKENS,
                    response_mime_type="application/json",
                    response_schema=GroundedResponse,
                ),
            )
            
            structured_ans = response.parsed
            
            token_usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
                }
            
            return structured_ans, token_usage
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return self._fallback_no_llm(user_prompt), None

    # =========================================================================
    # KB Fallback
    # =========================================================================

    async def _kb_fallback(
        self, query: str
    ) -> tuple[Optional[str], list[RetrievalResult]]:
        """
        Execute the KB fallback path:
        1. Search the Knowledge Base (FTS5)
        2. Build context from KB records
        3. Regenerate answer using ONLY KB data
        
        WHY SEPARATE FROM VECTOR SEARCH?
        The KB uses deterministic text search (FTS5), not vector similarity.
        If vector search returned hallucination-prone results, using the
        same approach would risk the same failure. FTS5 provides guaranteed,
        keyword-matched results as the source of truth.
        """
        from app.services.kb_service import get_kb_service
        
        kb_service = get_kb_service()
        kb_records = kb_service.search(query, limit=3)
        
        if not kb_records:
            logger.warning(f"KB fallback returned no results for: {query[:50]}")
            return None, []
        
        # Build KB context
        kb_context_parts = []
        kb_sources = []
        
        for record in kb_records:
            kb_context_parts.append(
                f"SCHEME: {record.scheme_name} (ID: {record.scheme_id})\n"
                f"Category: {record.category}\n"
                f"Description: {record.description}\n"
                f"Eligibility: {record.eligibility or 'N/A'}\n"
                f"Benefits: {record.benefits or 'N/A'}\n"
                f"Interest Rate: {record.interest_rate or 'N/A'}\n"
                f"Source: {record.source}\n"
                f"Last Verified: {record.last_verified or 'N/A'}"
            )
            kb_sources.append(RetrievalResult(
                chunk_id=f"kb_{record.scheme_id}",
                text=record.description[:200],
                score=0.95,  # KB data is always high confidence
                metadata={
                    "scheme_name": record.scheme_name,
                    "scheme_id": record.scheme_id,
                    "category": record.category,
                    "source": "knowledge_base_fallback",
                },
            ))
        
        kb_context = "\n\n---\n\n".join(kb_context_parts)
        
        # Regenerate with KB data
        kb_prompt = (
            f"USER QUESTION: {query}\n\n"
            f"VERIFIED KNOWLEDGE BASE DATA:\n\n{kb_context}\n\n"
            f"Provide a comprehensive answer using ONLY the verified KB data above."
        )
        
        structured_kb_ans, _ = await self._generate(
            system_prompt=KB_FALLBACK_SYSTEM_PROMPT,
            user_prompt=kb_prompt,
        )
        
        # Extract the answer string from the GroundedResponse object
        answer = structured_kb_ans.answer if structured_kb_ans else ""
        
        # Tag the answer as KB-sourced
        if "[KB-FALLBACK]" not in answer:
            answer = f"[KB-VERIFIED RESPONSE]\n\n{answer}"
        
        logger.info(
            f"KB fallback successful | query='{query[:50]}' | "
            f"kb_records={len(kb_records)}"
        )
        
        return answer, kb_sources

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _build_context(sources: list[RetrievalResult]) -> str:
        """
        Build structured context from retrieved chunks using XML-style tags.
        
        WHY XML TAGS?
        LLMs parse XML-style tags more reliably than plain text separators.
        Explicit <source> identifiers enable better attribution and allow
        the model to reference specific chunks in its response.
        """
        context_parts = []
        for i, source in enumerate(sources, 1):
            scheme_name = source.metadata.get("scheme_name", "Unknown")
            # Include cross-encoder score if available
            score_info = f"Relevance: {source.score:.3f}"
            context_parts.append(
                f"<source id=\"{i}\" scheme=\"{scheme_name}\" {score_info}>\n"
                f"{source.text}\n"
                f"</source>"
            )
        return "\n\n".join(context_parts)

    @staticmethod
    def _build_user_prompt(query: str, context: str) -> str:
        """Build the user prompt with query and structured context."""
        return (
            f"<context label=\"verified banking knowledge base\">\n"
            f"{context}\n"
            f"</context>\n\n"
            f"<question>\n{query}\n</question>\n\n"
            f"Answer the question using ONLY the data inside <source> tags above. "
            f"Cite the source id(s) you used. "
            f"If the context doesn't contain enough information, say so explicitly."
        )

    @staticmethod
    def _fallback_no_llm(prompt: str) -> GroundedResponse:
        """
        When no LLM is available, extract the context and present it directly.
        """
        ans_str = prompt
        if "<context" in prompt and "<question>" in prompt:
            context_section = prompt.split("<question>")[0]
            context_section = context_section.replace('<context label="verified banking knowledge base">', "")
            context_section = context_section.replace('</context>', "")
            context_section = context_section.strip()
            
            question = prompt.split("<question>")[-1].split("</question>")[0].strip()
            
            ans_str = (
                f"Based on the banking knowledge base, here is the relevant information "
                f"for your query about '{question}':\n\n{context_section}\n\n"
                f"[Note: Configure GEMINI_API_KEY for natural language responses.]"
            )
        elif "CONTEXT" in prompt and "USER QUESTION" in prompt:
            context_section = prompt.split("USER QUESTION")[0]
            context_section = context_section.replace("CONTEXT (from verified banking knowledge base):", "")
            context_section = context_section.strip().strip("-").strip()
            
            question = prompt.split("USER QUESTION:")[-1].split("\n")[0].strip()
            
            ans_str = (
                f"Based on the banking knowledge base, here is the relevant information "
                f"for your query about '{question}':\n\n{context_section}\n\n"
                f"[Note: Configure GEMINI_API_KEY for natural language responses.]"
            )
        
        return GroundedResponse(
            answer=ans_str,
            claims_total=0,
            claims_supported=0,
            self_evaluated_groundedness=1.0, # Deterministic direct extract
            unsupported_details="",
            reasoning="Direct context extraction"
        )


# =============================================================================
# Singleton
# =============================================================================

_rag_pipeline_instance: Optional[RAGPipelineService] = None


def get_rag_pipeline() -> RAGPipelineService:
    """Get or create the RAG pipeline singleton."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipelineService()
    return _rag_pipeline_instance
