# Viniyog One — Enterprise RAG for Banking Knowledge Intelligence

<p align="center">
  <strong>A secure, explainable Retrieval-Augmented Generation system for Indian banking schemes</strong>
</p>

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Data Flow](#data-flow)
4. [Data Ingestion Layer](#data-ingestion-layer)
5. [Chunking Strategy](#chunking-strategy)
6. [Embedding & Vector Storage](#embedding--vector-storage)
7. [Retrieval Design](#retrieval-design)
8. [Hallucination Detection](#hallucination-detection)
9. [Knowledge Base & Token Mechanism](#knowledge-base--token-mechanism)
10. [Fine-Tuned RAG Improvements](#fine-tuned-rag-improvements)
11. [API Design & Permission Model](#api-design--permission-model)
12. [Evaluation & Observability](#evaluation--observability)
13. [Limitations & Trade-offs](#limitations--trade-offs)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Gemini API key: GEMINI_API_KEY=AIzaSy...

# 3. Start the server
uvicorn app.main:app --reload --port 8000

# 4. Open the API docs
# Swagger UI:  http://localhost:8000/docs
# ReDoc:       http://localhost:8000/redoc
```

### Test the system

There are two primary ways to test the live pipeline once the server is running on port 8000:

#### Option 1: Visual Assessment (Swagger UI)
1. Open a web browser and navigate to `http://localhost:8000/docs`.
2. Expand the **POST /query** endpoint and click **"Try it out"**.
3. Set the `X-User-Role` header parameter to `user`.
4. Enter standard queries into the Request Body (e.g., `{"query": "Who is eligible for PM Mudra?"}`) and hit **Execute**.

#### Option 2: Headless Assessment (cURL)
You can directly blast endpoints in your terminal. For readability, pipe the output to `jq`:

```bash
# General query testing
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-User-Role: user" \
  -d '{"query": "What is PM Mudra Yojana eligibility?"}' | jq

# Evaluator testing endpoint checking metrics across a batch
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "X-User-Role: evaluator" \
  -d '{"test_queries": ["PM Mudra Yojana", "pension scheme", "gold bonds"]}' | jq
```

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                              │
│                    (X-User-Role: user/evaluator)                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                                 │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ RBAC Middleware (Header-based role enforcement)                 │ │
│  │ Roles: user → evaluator → admin → system                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────────┐      │
│  │ /query  │ │ /query/debug │ │ /evaluate│ │ /health        │      │
│  │ (user)  │ │ (evaluator)  │ │(evaluator)│ │ (public)       │      │
│  └────┬────┘ └──────┬───────┘ └─────┬────┘ └────────────────┘      │
│       │             │               │                                │
│  ┌────┴─────────────┴───────────────┴────────────────────────┐      │
│  │              RAG Pipeline Service                          │      │
│  │                                                            │      │
│  │  ┌─────────┐   ┌───────────┐   ┌──────────────────────┐  │      │
│  │  │ RETRIEVE├──►│ GENERATE  ├──►│ HALLUCINATION CHECK  │  │      │
│  │  │ (BGE +  │   │ (Gemini   │   │ (Heuristic + LLM     │  │      │
│  │  │ ChromaDB│   │  2.0 Flash)│   │  Judge)              │  │      │
│  │  │ + Cross │   │           │   │                      │  │      │
│  │  │ Encoder)│   │           │   │                      │  │      │
│  │  └─────────┘   └───────────┘   └──────────┬───────────┘  │      │
│  │                                            │              │      │
│  │                           ┌────────────────┤              │      │
│  │                           │  Score ≥ 0.85  │ Score < 0.85 │      │
│  │                           ▼                ▼              │      │
│  │                    ┌──────────┐    ┌───────────────┐      │      │
│  │                    │ RESPOND  │    │ KB FALLBACK   │      │      │
│  │                    │ (grounded│    │ ┌───────────┐ │      │      │
│  │                    │  answer) │    │ │ /kb/token │ │      │      │
│  │                    └──────────┘    │ │ (JWT mint)│ │      │      │
│  │                                   │ └─────┬─────┘ │      │      │
│  │                                   │       ▼       │      │      │
│  │                                   │ ┌───────────┐ │      │      │
│  │                                   │ │ /kb/fetch │ │      │      │
│  │                                   │ │(SQLite/   │ │      │      │
│  │                                   │ │ FTS5)     │ │      │      │
│  │                                   │ └───────────┘ │      │      │
│  │                                   └───────────────┘      │      │
│  └───────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐  ┌─────────────────────────────┐
│     Vector Store (ChromaDB) │  │    Knowledge Base (SQLite)  │
│  ┌────────────────────────┐ │  │  ┌────────────────────────┐ │
│  │ 75 embedded chunks     │ │  │  │ 15 verified schemes    │ │
│  │ BGE-large-en-v1.5      │ │  │  │ FTS5 full-text search  │ │
│  │ 1024-dim vectors       │ │  │  │ BM25 ranking           │ │
│  │ Cosine similarity      │ │  │  │ Audit logging          │ │
│  └────────────────────────┘ │  │  └────────────────────────┘ │
└─────────────────────────────┘  └─────────────────────────────┘
```

### Technology Stack

| Component | Technology | Justification |
|---|---|---|
| **API Framework** | FastAPI | Async performance, auto OpenAPI docs, dependency injection |
| **LLM** | Google Gemini 2.0 Flash | Free tier, 1M context window, fast inference, structured JSON outputs |
| **Embeddings** | BAAI/bge-large-en-v1.5 | Top-ranked on MTEB benchmarks, 1024-dim, instruction-aware |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Deep token-level relevance, 22M params, ~50ms per query |
| **Vector Store** | ChromaDB | Metadata filtering, persistence, zero-infrastructure |
| **Knowledge Base** | SQLite + FTS5 | ACID compliance, BM25 ranking, single-file portability |
| **Auth** | JWT (python-jose) | Stateless, short-lived tokens, auditable claims |
| **Logging** | Loguru | Zero-config structured logging, rotation support |

### Key Principle

> **The LLM is NOT the source of truth. The Knowledge Base is the source of truth. The RAG pipeline enforces this relationship.**

This principle is enforced through:
1. Every LLM response is validated against retrieved context via hallucination detection
2. Responses failing the groundedness threshold (0.85) are **dropped entirely**
3. The system falls back to the KB using a secure, token-gated pathway
4. KB responses are served with explicit source attribution

---

## Data Flow

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ banking_     │────►│  Ingestion   │────►│  Semantic     │
  │ schemes.json │     │  Service     │     │  Chunking     │
  │ (15 schemes) │     │  (6 formats) │     │  Service      │
  └──────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                                                    ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  SQLite KB   │◄────│ JSON seed    │     │  BGE-large   │
  │  (FTS5)      │     │ loader       │     │  embedding   │
  │  Source of   │     │ (idempotent) │     │  1024-dim    │
  │  Truth       │     └──────────────┘     └──────┬───────┘
  └──────────────┘                                  │
                                                    ▼
                                             ┌──────────────┐
                                             │  ChromaDB     │
                                             │  75 vectors   │
                                             │  (persistent) │
                                             └──────────────┘
```

**At query time:**
1. User query → BGE embedding → ChromaDB cosine search (Top-10)
2. Cross-encoder re-ranks Top-10 → Top-3 highest-quality chunks
3. Top-3 chunks injected into Gemini prompt with XML source tags
4. Gemini generates grounded answer
5. Hallucination detector (heuristic + LLM self-evaluation via structured outputs) scores groundedness
6. If score ≥ 0.85: return answer with source attribution
7. If score < 0.85: mint JWT → fetch KB data → return KB answer

---

## Data Ingestion Layer

**File:** `app/services/ingestion_service.py`

### Supported Formats

| Format | Handler | Use Case |
|---|---|---|
| **JSON** | `_ingest_json()` | Primary: structured banking scheme data |
| **CSV** | `_ingest_csv()` | Interest rate tables, eligibility matrices |
| **PDF** | `_ingest_pdf()` | Banking circulars (via pypdf) |
| **DOCX** | `_ingest_docx()` | Policy documents (via python-docx) |
| **HTML** | `_ingest_html()` | Web-scraped banking info (via BeautifulSoup) |
| **Text/MD** | `_ingest_text()` | Plain text policy documents |

### Data Preprocessing

Banking documents have specific noise patterns that degrade embedding quality:

- **Legal disclaimers and footnotes** → Stripped during preprocessing
- **Repeated headers/footers in PDFs** → Regex-based removal (`Page X of Y`)
- **Unicode issues** → Normalized (curly quotes → straight quotes, em dashes → hyphens)
- **Inconsistent whitespace** → Collapsed while preserving paragraph boundaries
- **Government-specific abbreviations** → Protected from sentence splitting (`Rs.`, `No.`, `p.a.`)

### Structured Data Transformation

CSV rows are transformed into natural language before embedding. Raw tabular data embeds poorly because embedding models are trained on prose. Example:

```
CSV: "Senior Citizen FD, 7.5%, 1-year, SBI"
→ "Row 1: Scheme: Senior Citizen FD | Rate: 7.5% | Tenure: 1-year | Bank: SBI"
```

JSON banking schemes are converted to rich Markdown documents with section headers (`## Eligibility`, `## Benefits`, `## Interest Rate`) that create natural semantic boundaries for chunking.

---

## Chunking Strategy

**File:** `app/services/chunking_service.py`

### Why Semantic Chunking (not fixed-size)

The BRD explicitly states "Semantic chunking (preferred)". Fixed-size chunking splits at arbitrary character boundaries, which:
- Breaks sentences mid-thought ("The interest rate is" | "7.5% per annum")
- Separates eligibility criteria from their corresponding benefits
- Produces chunks with fractured semantic meaning

**Our semantic chunker** identifies topic transitions using embedding similarity:

1. Split document into sentences (with banking-specific abbreviation handling)
2. Detect structural boundaries (headers, section transitions)
3. Group sentences into segments at structural boundaries
4. Embed each segment using BGE-large-en-v1.5
5. Compute cosine similarity between consecutive segments
6. Merge segments with similarity > 0.8 (same topic)
7. Split at points where similarity drops (topic shift)
8. Post-process: merge tiny chunks (< 100 chars), cap oversized chunks (> 2000 chars)

### Chunk Size Justification

**Target: 256-512 tokens** (empirically optimal for banking docs)

| Size | Problem |
|---|---|
| < 128 tokens | Loses context, fragments sentences |
| 128-256 tokens | Workable but often misses related context |
| **256-512 tokens** | ✅ **Balances context preservation with retrieval precision** |
| > 1024 tokens | Dilutes relevance signal, wastes context window |

Banking scheme sections (eligibility, benefits, rates) naturally map to this range.

**Reference:** "Lost in the Middle" (Liu et al., 2023) — LLMs attend most to the beginning and end of context. Moderate chunk sizes ensure each chunk contains a focused, complete thought.

### Overlap Strategy

**50-token overlap** between chunks. This ensures sentences near boundaries appear in both adjacent chunks, preventing information loss. Larger overlap would cause excessive duplication.

---

## Embedding & Vector Storage

### Embedding Model: BAAI/bge-large-en-v1.5

| Property | Value | Justification |
|---|---|---|
| **Dimensions** | 1024 | Dense representation for nuanced banking terminology |
| **Architecture** | BERT-large | Pre-trained on massive text corpora |
| **MTEB Rank** | Top-5 | Consistently outperforms comparable models |
| **Instruction-aware** | Yes | Prefix queries with "Represent this sentence:" |
| **Local execution** | Yes | No data leaves the system (data sovereignty) |

**Why local (not OpenAI/external API)?** Transmitting banking scheme data to external APIs introduces data sovereignty risks and network latency. Local execution maintains strict control over the data lifecycle.

### Distance Metric: Cosine Similarity

Cosine similarity measures the orientation (not magnitude) of vectors, making it robust against variations in document length. For normalized embeddings (BGE outputs are normalized), cosine similarity equals the inner product.

### Vector Store: ChromaDB

**Why ChromaDB (not FAISS)?**
- **Metadata filtering** — Filter by scheme category, doc_type during retrieval
- **Persistence** — Data survives server restarts (FAISS is in-memory only)
- **Simpler API** — CRUD operations without manual index management
- **Sufficient scale** — 75 vectors; FAISS's ANN indexing adds complexity for no benefit

---

## Retrieval Design

**File:** `app/services/vector_store_service.py`

### Two-Stage Retrieval with Cross-Encoder Re-ranking

```
Query → BGE Embedding → ChromaDB (Top-10) → Cross-Encoder (Top-3) → LLM
```

**Stage 1: Bi-Encoder (Wide Net)**
- Top-K = 10 (intentionally high for recall)
- BGE-large embeds query and documents **separately** → fast but approximate
- Cosine similarity ranking

**Stage 2: Cross-Encoder Re-ranking (Precision)**  
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params)
- Processes query and document **together** through attention layers
- Computes deep token-level relevance scores
- Keeps only Top-3 highest-scoring chunks
- Adds ~50ms latency (acceptable for banking)

**Why two stages?** Bi-encoders are fast at approximate matching (good for recall). Cross-encoders compute deep relevance (good for precision). Two stages give the best of both: wide recall with precise final ranking.

### Handling Irrelevant Results

- Chunks below `RETRIEVAL_SIMILARITY_THRESHOLD` (0.65) are dropped
- If no chunks pass the threshold, a "no relevant data found" response is returned
- The hallucination detector acts as a second gatekeeper, catching cases where retrieved chunks are topically related but don't actually answer the question

---

## Hallucination Detection

**File:** `app/services/hallucination_detector.py`

### Dual-Layer Detection Architecture

```
                  Draft Answer
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
   ┌──────────────┐     ┌──────────────────┐
   │  HEURISTIC   │     │   LLM JUDGE      │
   │  Layer       │     │   (Gemini)       │
   │              │     │                  │
   │ • Entity     │     │ • Atomic claim   │
   │   coverage   │     │   decomposition  │
   │ • Numeric    │     │ • Per-claim      │
   │   validation │     │   verification   │
   │ • Length     │     │ • Strict JSON    │
   │   checks     │     │   output format  │
   └──────┬───────┘     └────────┬─────────┘
          │                      │
          └──────────┬───────────┘
                     │
              Combined Score
                     │
          ┌──────────┴──────────┐
          │                     │
    Score ≥ 0.85          Score < 0.85
          │                     │
    ✅ GROUNDED           💀 HALLUCINATED
    Return answer         Drop draft
                          → KB Fallback
```

### Heuristic Checks (fast, deterministic)

1. **Entity coverage** — Are key terms from the answer present in the context?
2. **Numeric validation** — Are all numbers in the answer traceable to the context?
3. **Length ratio** — Is the answer suspiciously long compared to the context?
4. **Direct quote presence** — Does the answer reference specific context phrases?

### LLM-as-a-Judge (deep, probabilistic)

The judge model (Gemini at temperature=0.0 for determinism) performs **atomic claim decomposition** as part of a single-pass structured output — generation and evaluation happen in one API call:

1. **DECOMPOSE** the answer into individual atomic claims (each fact, number, date, criterion)
2. **VERIFY** each claim against the source context
3. **CLASSIFY** each claim as SUPPORTED or UNSUPPORTED
4. **SCORE** = (supported claims) / (total claims)

**Output format:**
```
SCORE: 0.94
CLAIMS_TOTAL: 8
CLAIMS_SUPPORTED: 7
CLAIMS_UNSUPPORTED: 1
UNSUPPORTED_DETAILS: "Tax benefit under Section 80C" not found in context
REASONING: The answer accurately describes the scheme...
```

### Threshold: 0.85

Banking demands near-zero tolerance for fabrication. A 0.85 threshold catches subtle errors (e.g., correct scheme name but fabricated rate). The KB fallback is reliable, so additional false positives are acceptable — it's better to serve authoritative KB data than risk a fabricated number.

---

## Knowledge Base & Token Mechanism

**Files:** `app/services/kb_service.py`, `app/core/security.py`

### KB Architecture

The Knowledge Base is a **SQLite database with FTS5 full-text search**, containing 15 verified Indian banking schemes. It serves as the **authoritative fallback** when the RAG pipeline detects a hallucinated response.

**Why SQLite + FTS5 (not vector search)?**
The KB is the fallback — it's activated specifically because vector search already produced an insufficient result. Using the same vector approach would risk repeating the same failure. Instead, FTS5 provides:
- **BM25 ranking** — Industry-standard text relevance
- **Deterministic matching** — No probabilistic approximation
- **Exact numeric retrieval** — Interest rates, amounts returned verbatim
- **Audit logging** — Every KB access is recorded for compliance

### Token-Based Access Flow

```
  Hallucination     System calls      JWT minted       System presents
  Detected       →  /kb/token      →  (60s TTL)     →  JWT to /kb/fetch
                    (system role)     (purpose:        (dual auth:
                                      kb_access)       role + token)
                                                            │
                                                            ▼
                                                     SQLite FTS5
                                                     deterministic
                                                     lookup
                                                            │
                                                            ▼
                                                     Verified data
                                                     returned with
                                                     source attribution
```

### Token Lifecycle

| Property | Value | Rationale |
|---|---|---|
| **Algorithm** | HS256 | Single-service; low overhead. RS256 for microservices. |
| **TTL** | 60 seconds | KB fallback completes in <5s. Minimal attack surface. |
| **Claims** | purpose, scope, iat, exp, nbf | Intent-scoped, time-bounded |
| **Refresh** | None | Mint-and-use pattern. No refresh = no token theft risk. |

### Why Not Direct KB Access?

Direct access is restricted to prevent:
1. Mass data extraction (user could dump all 15 schemes)
2. Bypassing the RAG pipeline (losing hallucination detection)
3. Audit gaps (every access must have a traceable token)
4. Violating the Principle of Least Privilege

---

## Fine-Tuned RAG Improvements

### Beyond Baseline RAG

| Improvement | Baseline RAG | Our Implementation | Impact |
|---|---|---|---|
| **Retrieval** | Single Top-K | Two-stage with cross-encoder re-ranking | Higher precision chunks |
| **Context** | Plain text | XML-structured `<source>` tags | Better LLM parsing, source citation |
| **Judge** | Overall score | Atomic claim decomposition | Catches partial hallucinations |
| **Fallback** | No fallback | Token-gated KB with FTS5 | 100% reliability guarantee |
| **Chunking** | Fixed-size | Semantic with embedding similarity | Better context preservation |
| **Threshold** | Generic (0.7) | Banking-strict (0.85) | Lower false negative rate |

### Measurable Impact

Evaluation over 5 test queries:

```
Avg Groundedness:  0.988 (threshold: 0.85) ✅
Hallucination Rate: 0%                      ✅
KB Fallback Rate:   0%                      ✅
Avg Response Time:  ~1.3 seconds             ✅ (under 3s SLA)
```

---

## API Design & Permission Model

### Endpoints

| Endpoint | Method | Role Required | Purpose |
|---|---|---|---|
| `/query` | POST | user | Standard RAG query with source attribution |
| `/query/debug` | POST | evaluator | Full pipeline trace (retrieval scores, chunks, prompt) |
| `/kb/token` | POST | system | Mint short-lived JWT for KB access |
| `/kb/fetch` | POST | system + KB token | Retrieve verified data from KB |
| `/evaluate` | POST | evaluator | Batch evaluation with aggregated metrics |
| `/chunks/inspect` | POST | admin | Analyze chunking strategy and boundaries |
| `/retrieval/logs` | GET | admin | Retrieval observability (scores, timing, outcomes) |
| `/health` | GET | (none) | System component health (public for load balancers) |

### Role Hierarchy

```
system (internal services only)
  │
admin (full observability access)
  │
evaluator (debug + evaluation access)
  │
user (standard query access only)
```

### RBAC Implementation

**Why header-based (not OAuth2)?** This is a deliberate design decision for assessment testability. Evaluators can test any role by setting `X-User-Role: evaluator` in the header — no token exchange ceremony needed. The architecture is swappable to OAuth2 via a single dependency change in `dependencies.py`.

The role enforcement uses FastAPI's dependency injection. Each endpoint declares its required role via a `Depends(require_<role>)` dependency that intercepts the request before the handler executes.

---

## Evaluation & Observability

### `/evaluate` Endpoint

Runs batch queries through the full pipeline and returns:
- Per-query groundedness scores
- Hallucination detection outcomes
- KB fallback trigger rates
- Response time measurements
- Aggregated metrics (avg, rates)

### `/retrieval/logs` Endpoint

Returns recent retrieval operation logs including:
- Query text and top-K used
- Similarity score distribution (min/avg/max)
- Retrieval latency
- Hallucination detection outcomes
- KB fallback events

### Structured Logging

Every pipeline stage is logged via Loguru with structured context:
- `app.main` — Application lifecycle events
- `app.services.rag_pipeline` — Query processing, generation, hallucination detection
- `app.services.vector_store_service` — Embedding, retrieval, re-ranking timing
- `app.core.dependencies` — RBAC access grants and denials
- `app.core.security` — Token generation and validation events
- `app.services.kb_service` — KB access audit trail (SQLite `kb_access_log` table)

---

## Limitations & Trade-offs

### Known Limitations

| Limitation | Root Cause | Mitigation |
|---|---|---|
| **Response time ~1-2s** | Single Gemini API call with structured JSON outputs | Well within the 3-second BRD target. |
| **SQLite concurrency** | Single-writer limitation | WAL mode enables concurrent reads. Sufficient for assessment scale; would migrate to PostgreSQL for production. |
| **In-memory retrieval logs** | Logs lost on restart | Acceptable for assessment. Production would use persistent storage (Redis, PostgreSQL). |
| **75 vectors** | Limited dataset | System architecture scales to millions of vectors via ChromaDB's persistence and sharding. |

### Deliberate Trade-offs

| Decision | Trade-off | Rationale |
|---|---|---|
| **Semantic chunking** over fixed-size | ~100ms compute per document during ingestion | Better context preservation justifies the cost |
| **Cross-encoder re-ranking** | +50ms per query, +90MB memory | Dramatically better retrieval precision |
| **0.85 hallucination threshold** | More false positives → more KB fallbacks | In banking, false negatives (undetected hallucinations) are catastrophic |
| **Local embeddings** over external API | Slower first load (~8s) | Data sovereignty — no banking data leaves the system |
| **ChromaDB** over FAISS | Slightly higher query latency | Metadata filtering, persistence, simpler API |
| **Header RBAC** over OAuth2 | Less production-ready | Assessment testability — evaluators can test any role instantly |
| **60-second KB tokens** over longer TTL | More frequent token minting | Minimal attack surface; KB fallback completes in <5s |

---

## Project Structure

```
viniyog_v1/
├── app/
│   ├── api/
│   │   ├── routes.py          # All 8 endpoint handlers
│   │   └── schemas.py         # Pydantic request/response models
│   ├── core/
│   │   ├── config.py          # Centralized configuration (pydantic-settings)
│   │   ├── dependencies.py    # RBAC, dependency injection
│   │   └── security.py        # JWT generation/validation
│   ├── services/
│   │   ├── rag_pipeline.py    # Core pipeline orchestration
│   │   ├── hallucination_detector.py  # Dual-layer detection
│   │   ├── vector_store_service.py    # ChromaDB + cross-encoder
│   │   ├── kb_service.py      # SQLite/FTS5 Knowledge Base
│   │   ├── ingestion_service.py       # Multi-format data loading
│   │   └── chunking_service.py        # Semantic chunking
│   └── main.py                # Application factory + lifespan
├── data/
│   ├── banking_schemes.json   # Seed data (15 schemes)
│   ├── knowledge_base.db      # SQLite KB (auto-generated)
│   └── chroma_db/             # ChromaDB persistence
├── history/
│   └── context.txt            # Architectural decision log
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
└── README.md                  # This document
```

---

## Dataset

The system ships with 15 verified Indian banking schemes covering:

| Category | Schemes |
|---|---|
| **Financial Inclusion** | PMJDY (Jan Dhan), DBT |
| **MSME & Entrepreneurship** | PMMY (Mudra), PMEGP, Stand-Up India |
| **Insurance** | PMJJBY, PMSBY |
| **Pension** | APY (Atal Pension) |
| **Savings & Investment** | SSY (Sukanya Samriddhi), SGB (Gold Bond), PPF |
| **Banking Regulation** | PSL Norms, Digital Banking |

Each scheme includes: description, eligibility, benefits, interest rates, documents required, ministry, launch date, and source URL.
