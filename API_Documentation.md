# API Documentation

## Overview
Viniyog One utilizes FastAPI to provide a robust, asynchronously capable RESTful API for executing Retrieval-Augmented Generation (RAG) processes. All endpoints are fortified behind a hierarchical Role-Based Access Control (RBAC) middleware ensuring banking security constraints.

---

## Permission Hierarchy

All endpoints require the `X-User-Role` header with a valid role. Standard API keys are abstractly mapped to these roles for testing.
1. **user**: Can make standard queries.
2. **evaluator**: Can run trace tools, diagnostics, and evaluations.
3. **admin**: Can view system logs and infrastructure endpoints.
4. **system**: An internal service role, explicitly required to mint and utilize short-lived backend tokens for the Knowledge Base fallback.

---

## Core Endpoints

### 1. `POST /query`
**Description:** Executes a standard user query against the RAG pipeline. Returns an answer grounded in the verified data sources along with exact source attributions.
- **Required Role:** `user`
- **Request Body:**
  ```json
  {
    "query": "What is the PM Mudra Yojana eligibility?"
  }
  ```
- **Response Structure (200 OK):**
  ```json
  {
    "query": "What is the PM Mudra Yojana eligibility?",
    "answer": "The PM Mudra Yojana provides loans to non-corporate, non-farm businesses...",
    "sources": [
      {
        "content": "...relevant context snippet...",
        "metadata": {"scheme_name": "PMMY"}
      }
    ]
  }
  ```

### 2. `POST /query/debug`
**Description:** Returns a full diagnostic trace of the RAG pipeline execution, detailing similarity scores, hallucination evaluation grades, and internal timing.
- **Required Role:** `evaluator`

### 3. `POST /evaluate`
**Description:** Accepts an array of queries to execute the pipeline in batch mode. Aggregates Average Groundedness Scores, Hallucination/Fallback Rates, and Latency metrics.
- **Required Role:** `evaluator`
- **Request Body:**
  ```json
  {
    "test_queries": ["Question 1", "Question 2"]
  }
  ```

---

## Infrastructure Endpoints

### 4. `GET /health`
**Description:** Public endpoint allowing load balancers to poll component availability (Vector Store, Embedding Model, LLM connection, SQLite Knowledge Base).
- **Required Role:** None

### 5. `GET /retrieval/logs`
**Description:** Returns historical query logs including embedding similarity metrics and fallback events. Operates via a ring-buffer to prevent out-of-memory errors.
- **Required Role:** `admin`

---

## Internal Security Endpoints (Token-Gated)

### 6. `POST /kb/token`
**Description:** Mints a short-lived, 1-minute TTL HS256-signed JSON Web Token specifically scoped for Knowledge Base interactions. This prevents direct mass extraction.
- **Required Role:** `system`

### 7. `POST /kb/fetch`
**Description:** Authoritative Knowledge Base retrieval fallback. Triggered only when the RAG Hallucination grading fails the 0.85 threshold limit. Uses deterministic FTS5 SQL lookups.
- **Required Role:** `system` (plus Bearer token from `/kb/token`)
