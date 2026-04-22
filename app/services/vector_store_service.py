"""
==============================================================================
Embedding & Vector Store Service — ChromaDB + BGE-Large
==============================================================================

ARCHITECTURAL DECISION (Embedding Model):
    We use BAAI/bge-large-en-v1.5 because:
    1. **MTEB benchmark leader** in its size class (335M params)
    2. **1024-dim embeddings** — balance of precision and storage
    3. **Instruction-aware** — supports query/passage prefixes
       ("Represent this sentence: ..." for passages,
        "Represent this sentence for retrieval: ..." for queries)
    4. **Apache 2.0 license** — no commercial restrictions
    
    vs. Instructor-XL:
    - Instructor-XL is larger (1.5B params) but slower
    - BGE-large achieves 95%+ of Instructor-XL quality at 4x speed
    - For a 3-second response time target, BGE-large is optimal

ARCHITECTURAL DECISION (ChromaDB):
    ChromaDB was chosen over FAISS because:
    1. **Persistence** — data survives process restarts
    2. **Metadata filtering** — filter by scheme category, date, etc.
    3. **Simpler API** — add/query/update without manual index management
    4. **Built-in embedding support** — can embed on add (but we use custom)
    
    vs. FAISS:
    - FAISS is faster for pure ANN search
    - But lacks metadata filtering (critical for banking domain)
    - No built-in persistence (requires manual save/load)
    - For 15-50 documents, ChromaDB's overhead is negligible

VECTOR SEARCH STRATEGY:
    We use cosine similarity (default for normalized embeddings).
    
    WHY COSINE (not L2/IP)?
    - Cosine similarity is magnitude-invariant
    - BGE embeddings are already L2-normalized
    - Industry standard for text similarity
    - Interpretable range [0, 1] for thresholding
"""

import time
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from loguru import logger

from app.core.config import get_settings
from app.services.chunking_service import Chunk


# =============================================================================
# Embedding & Vector Store Service
# =============================================================================

class VectorStoreService:
    """
    Manages embedding generation and ChromaDB vector storage.
    
    Lifecycle:
    1. initialize(): Load embedding model + connect to ChromaDB
    2. index_chunks(): Embed and store chunks
    3. search(): Embed query and retrieve top-K similar chunks
    4. health_check(): Verify store connectivity
    """

    def __init__(self):
        self._settings = get_settings()
        self._embedding_model: Optional[SentenceTransformer] = None
        self._cross_encoder: Optional[CrossEncoder] = None
        self._chroma_client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self._initialized = False

    def initialize(self) -> None:
        """
        Initialize the embedding model and ChromaDB collection.
        
        This is called once at application startup and takes
        ~5-10 seconds (model download on first run).
        """
        if self._initialized:
            return

        start = time.time()

        # ------------------------------------------------------------------
        # Load Embedding Model
        # ------------------------------------------------------------------
        logger.info(f"Loading embedding model: {self._settings.EMBEDDING_MODEL}")
        self._embedding_model = SentenceTransformer(
            self._settings.EMBEDDING_MODEL,
        )
        model_load_time = time.time() - start
        logger.info(
            f"Embedding model loaded | time={model_load_time:.2f}s | "
            f"dim={self._settings.EMBEDDING_DIMENSION}"
        )

        # ------------------------------------------------------------------
        # Initialize ChromaDB
        # ------------------------------------------------------------------
        # Persistent storage so vectors survive restarts
        self._chroma_client = chromadb.PersistentClient(
            path=self._settings.VECTOR_STORE_PATH,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # No telemetry for banking
            ),
        )

        # Get or create the collection
        # Using cosine similarity (embeddings are L2-normalized)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._settings.CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",   # Distance metric
                "description": "Banking knowledge embeddings",
            },
        )

        existing_count = self._collection.count()
        logger.info(
            f"ChromaDB initialized | collection={self._settings.CHROMA_COLLECTION_NAME} | "
            f"existing_vectors={existing_count}"
        )

        # ------------------------------------------------------------------
        # Load Cross-Encoder for Re-ranking
        # ------------------------------------------------------------------
        # WHY RE-RANKING? Bi-encoders (BGE) embed query and document
        # separately — fast but approximate. Cross-encoders process them
        # TOGETHER through attention layers, computing deep token-level
        # interactions. This dramatically improves retrieval precision.
        # The trade-off is ~50ms per query — acceptable for banking.
        if self._settings.RERANK_ENABLED:
            rerank_start = time.time()
            logger.info(f"Loading cross-encoder model: {self._settings.RERANK_MODEL}")
            self._cross_encoder = CrossEncoder(
                self._settings.RERANK_MODEL,
            )
            rerank_load_time = time.time() - rerank_start
            logger.info(
                f"Cross-encoder loaded | model={self._settings.RERANK_MODEL} | "
                f"time={rerank_load_time:.2f}s"
            )

        self._initialized = True

    # =========================================================================
    # Indexing
    # =========================================================================

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """
        Embed and index chunks into ChromaDB.
        
        Process:
        1. Extract texts from chunks
        2. Generate embeddings in batches (GPU-friendly)
        3. Upsert into ChromaDB with metadata
        
        WHY UPSERT (not insert)?
        Idempotent — re-running ingestion updates existing vectors
        instead of creating duplicates. Critical for development
        iteration and data refreshes.
        
        Args:
            chunks: List of Chunk objects to index
            batch_size: Embedding batch size (64 balances speed/memory)
        
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        self._ensure_initialized()

        start = time.time()
        total_indexed = 0

        # Process in batches for memory efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare data for ChromaDB
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "char_count": c.char_count,
                    "token_count_approx": c.token_count_approx,
                    "semantic_boundary": c.semantic_boundary,
                    # Flatten metadata (ChromaDB requires scalar values)
                    **{
                        k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                        for k, v in c.metadata.items()
                        if k not in ("chunk_index",)  # Avoid duplicates
                    },
                }
                for c in batch
            ]

            # Generate embeddings
            # Prefix for passage embedding (BGE-specific optimization)
            prefixed_texts = [
                f"Represent this sentence: {t}" for t in texts
            ]
            embeddings = self._embedding_model.encode(
                prefixed_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            ).tolist()

            # Upsert into ChromaDB
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_indexed += len(batch)

        elapsed = time.time() - start
        logger.info(
            f"Indexing complete | chunks={total_indexed} | "
            f"time={elapsed:.2f}s | "
            f"total_vectors={self._collection.count()}"
        )

        return total_indexed

    # =========================================================================
    # Search / Retrieval
    # =========================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Semantic search: embed query and retrieve top-K similar chunks.
        
        Process:
        1. Embed the query with retrieval prefix
        2. Query ChromaDB for nearest neighbors
        3. Filter by similarity threshold
        4. Return results with scores and metadata
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            category_filter: Optional category filter (e.g., "Insurance")
            similarity_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of dicts with keys: chunk_id, text, score, metadata
        """
        self._ensure_initialized()

        threshold = similarity_threshold or self._settings.RETRIEVAL_SIMILARITY_THRESHOLD
        start = time.time()

        # Embed query with retrieval prefix (BGE-specific)
        query_embedding = self._embedding_model.encode(
            f"Represent this sentence for retrieval: {query}",
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Build ChromaDB query
        where_filter = None
        if category_filter:
            where_filter = {"category": {"$eq": category_filter}}

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()) or top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Process results
        # ChromaDB returns cosine distance, we convert to similarity
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance / 2)
                distance = results["distances"][0][i]
                similarity = 1 - (distance / 2)

                # Apply threshold filtering
                if similarity >= threshold:
                    search_results.append({
                        "chunk_id": chunk_id,
                        "text": results["documents"][0][i],
                        "score": round(similarity, 4),
                        "distance": round(distance, 4),
                        "metadata": results["metadatas"][0][i],
                    })

        elapsed = (time.time() - start) * 1000  # Convert to ms
        logger.debug(
            f"Vector search (bi-encoder) | query='{query[:60]}' | top_k={top_k} | "
            f"results={len(search_results)} | time={elapsed:.1f}ms"
        )

        # ------------------------------------------------------------------
        # Stage 2: Cross-Encoder Re-ranking
        # ------------------------------------------------------------------
        # The bi-encoder embeds query and document SEPARATELY. The cross-
        # encoder processes them TOGETHER through attention layers, computing
        # deep token-level interactions for much higher precision.
        if self._cross_encoder and self._settings.RERANK_ENABLED and search_results:
            rerank_start = time.time()
            rerank_top_k = self._settings.RERANK_TOP_K

            # Build query-document pairs for cross-encoder scoring
            pairs = [
                [query, result["text"]] for result in search_results
            ]
            ce_scores = self._cross_encoder.predict(pairs).tolist()

            # Attach cross-encoder scores and sort by them
            for i, result in enumerate(search_results):
                result["bi_encoder_score"] = result["score"]
                result["cross_encoder_score"] = round(float(ce_scores[i]), 4)

            search_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

            # Keep only top RERANK_TOP_K
            search_results = search_results[:rerank_top_k]

            rerank_elapsed = (time.time() - rerank_start) * 1000
            logger.debug(
                f"Cross-encoder re-rank | input={len(pairs)} | output={len(search_results)} | "
                f"top_ce_score={search_results[0]['cross_encoder_score'] if search_results else 0} | "
                f"time={rerank_elapsed:.1f}ms"
            )

        return search_results

    # =========================================================================
    # Management Methods
    # =========================================================================

    def get_collection_stats(self) -> dict:
        """Return statistics about the vector collection."""
        self._ensure_initialized()
        return {
            "collection_name": self._settings.CHROMA_COLLECTION_NAME,
            "total_vectors": self._collection.count(),
            "embedding_model": self._settings.EMBEDDING_MODEL,
            "embedding_dimension": self._settings.EMBEDDING_DIMENSION,
            "distance_metric": "cosine",
        }

    def clear_collection(self) -> None:
        """Clear all vectors from the collection (for re-indexing)."""
        self._ensure_initialized()
        self._chroma_client.delete_collection(
            self._settings.CHROMA_COLLECTION_NAME
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector collection cleared")

    def health_check(self) -> dict:
        """Return health status of the vector store."""
        try:
            self._ensure_initialized()
            count = self._collection.count()
            return {
                "status": "healthy" if count > 0 else "degraded",
                "vector_count": count,
                "collection": self._settings.CHROMA_COLLECTION_NAME,
                "embedding_model": self._settings.EMBEDDING_MODEL,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _ensure_initialized(self):
        """Ensure the service is initialized before use."""
        if not self._initialized:
            self.initialize()


# =============================================================================
# Singleton
# =============================================================================

_vector_store_instance: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    """Get or create the vector store service singleton."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance
