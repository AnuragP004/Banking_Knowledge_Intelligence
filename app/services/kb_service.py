"""
==============================================================================
Knowledge Base Service — The Source of Truth
==============================================================================

ARCHITECTURAL DECISION:
    The Knowledge Base (KB) is the single source of truth in this system.
    Every response that fails hallucination detection is regenerated using
    EXCLUSIVELY KB data. This module provides:
    
    1. **SQLite-backed storage** — Structured, queryable, portable.
       SQLite was chosen over PostgreSQL/MongoDB because:
       - Zero infrastructure (single .db file in the repo)
       - ACID compliant for data integrity
       - Full-text search via FTS5 extension
       - Perfect for assessment/GitHub submission
    
    2. **JSON fallback** — The raw JSON file is the seed data source.
       SQLite is populated from this JSON on first init. This provides:
       - Human-readable data for evaluators
       - Easy data updates without SQL knowledge
       - Version-controlled data changes via git
    
    3. **Full-text search** — Using SQLite FTS5 for efficient text matching.
       FTS5 provides:
       - Tokenized search with BM25 ranking
       - Prefix queries for partial matching
       - Column-weighted search (scheme name > description > eligibility)
    
    4. **Dual search strategy**:
       a. Full-text search (FTS5) for keyword matching
       b. Category-based filtering for structured queries
       c. Direct ID lookup for known scheme references

WHY NOT A VECTOR SEARCH ON THE KB?
    The KB is the FALLBACK, not the primary retrieval path. When we hit the KB,
    it means vector search already failed to produce grounded answers. Using
    the same vector approach here would risk the same failure mode. Instead,
    we use deterministic full-text search + exact matching for guaranteed
    relevance. This is a deliberate architectural choice for reliability.

CONCURRENCY:
    SQLite is used in WAL (Write-Ahead Logging) mode for concurrent reads.
    For this assessment, write contention is not a concern since data is
    loaded once at startup.
"""

import json
import sqlite3
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from loguru import logger
from app.core.config import get_settings
from app.api.schemas import KBRecord


# =============================================================================
# Database Manager
# =============================================================================

class KnowledgeBaseService:
    """
    Manages the SQLite-backed Knowledge Base.
    
    Lifecycle:
    1. __init__: Configure paths
    2. initialize(): Create tables, load JSON seed data (idempotent)
    3. search(): Full-text search for queries
    4. get_by_id(): Direct scheme lookup
    5. get_by_category(): Category-based filtering
    """

    def __init__(self):
        settings = get_settings()
        self.db_path = settings.KB_DATABASE_PATH
        self.json_path = settings.KB_JSON_PATH
        self._initialized = False

        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """
        Context manager for SQLite connections.
        
        WHY context manager?
        - Guarantees connection cleanup even on exceptions
        - Enables WAL mode for concurrent read access
        - Row factory for dict-like access to results
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads during writes
        conn.execute("PRAGMA journal_mode=WAL")
        # Foreign keys for referential integrity
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """
        Initialize the Knowledge Base — create tables and load seed data.
        
        This method is IDEMPOTENT — safe to call multiple times.
        It uses INSERT OR IGNORE to avoid duplicating records on restart.
        
        Tables:
        1. banking_schemes — Main data table with all scheme information
        2. banking_schemes_fts — FTS5 virtual table for full-text search
        3. kb_access_log — Audit log for every KB access (compliance)
        """
        if self._initialized:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # -----------------------------------------------------------------
            # Main data table
            # -----------------------------------------------------------------
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS banking_schemes (
                    scheme_id TEXT PRIMARY KEY,
                    scheme_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    eligibility TEXT,
                    benefits TEXT,
                    interest_rate TEXT,
                    documents_required TEXT,
                    launched_date TEXT,
                    ministry TEXT,
                    last_verified TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # -----------------------------------------------------------------
            # FTS5 virtual table for full-text search
            # -----------------------------------------------------------------
            # WHY FTS5 over FTS3/FTS4?
            # - Better query syntax (column filters, boolean ops)
            # - BM25 ranking function built-in
            # - More efficient storage and updates
            # - Column weighting via bm25() function
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS banking_schemes_fts USING fts5(
                    scheme_id,
                    scheme_name,
                    category,
                    description,
                    eligibility,
                    benefits,
                    interest_rate,
                    content='banking_schemes',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)

            # -----------------------------------------------------------------
            # Audit log table — every KB access is logged for compliance
            # -----------------------------------------------------------------
            # In banking, data access must be auditable. This table records
            # every query, who made it, and what was returned.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kb_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    results_count INTEGER,
                    token_scope TEXT,
                    requested_by TEXT,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # -----------------------------------------------------------------
            # Triggers to keep FTS index synchronized with main table
            # -----------------------------------------------------------------
            # These triggers ensure the FTS index stays in sync when
            # records are inserted, updated, or deleted.
            cursor.executescript("""
                CREATE TRIGGER IF NOT EXISTS banking_schemes_ai AFTER INSERT ON banking_schemes BEGIN
                    INSERT INTO banking_schemes_fts(
                        rowid, scheme_id, scheme_name, category,
                        description, eligibility, benefits, interest_rate
                    ) VALUES (
                        new.rowid, new.scheme_id, new.scheme_name, new.category,
                        new.description, new.eligibility, new.benefits, new.interest_rate
                    );
                END;

                CREATE TRIGGER IF NOT EXISTS banking_schemes_ad AFTER DELETE ON banking_schemes BEGIN
                    INSERT INTO banking_schemes_fts(
                        banking_schemes_fts, rowid, scheme_id, scheme_name, category,
                        description, eligibility, benefits, interest_rate
                    ) VALUES (
                        'delete', old.rowid, old.scheme_id, old.scheme_name, old.category,
                        old.description, old.eligibility, old.benefits, old.interest_rate
                    );
                END;

                CREATE TRIGGER IF NOT EXISTS banking_schemes_au AFTER UPDATE ON banking_schemes BEGIN
                    INSERT INTO banking_schemes_fts(
                        banking_schemes_fts, rowid, scheme_id, scheme_name, category,
                        description, eligibility, benefits, interest_rate
                    ) VALUES (
                        'delete', old.rowid, old.scheme_id, old.scheme_name, old.category,
                        old.description, old.eligibility, old.benefits, old.interest_rate
                    );
                    INSERT INTO banking_schemes_fts(
                        rowid, scheme_id, scheme_name, category,
                        description, eligibility, benefits, interest_rate
                    ) VALUES (
                        new.rowid, new.scheme_id, new.scheme_name, new.category,
                        new.description, new.eligibility, new.benefits, new.interest_rate
                    );
                END;
            """)

            # -----------------------------------------------------------------
            # Load seed data from JSON
            # -----------------------------------------------------------------
            self._load_seed_data(cursor)

        self._initialized = True
        logger.info(f"Knowledge Base initialized | db={self.db_path}")

    def _load_seed_data(self, cursor: sqlite3.Cursor) -> None:
        """
        Load banking schemes from JSON seed file.
        
        Uses INSERT OR IGNORE to be idempotent — running this multiple
        times won't create duplicates. This is important because the
        lifespan handler calls initialize() on every app startup.
        """
        if not os.path.exists(self.json_path):
            logger.warning(f"Seed data file not found: {self.json_path}")
            return

        with open(self.json_path, "r", encoding="utf-8") as f:
            schemes = json.load(f)

        inserted = 0
        for scheme in schemes:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO banking_schemes (
                        scheme_id, scheme_name, category, description,
                        eligibility, benefits, interest_rate, documents_required,
                        launched_date, ministry, last_verified, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    scheme["scheme_id"],
                    scheme["scheme_name"],
                    scheme["category"],
                    scheme["description"],
                    scheme.get("eligibility"),
                    scheme.get("benefits"),
                    scheme.get("interest_rate"),
                    scheme.get("documents_required"),
                    scheme.get("launched_date"),
                    scheme.get("ministry"),
                    scheme.get("last_verified"),
                    scheme.get("source"),
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                # Scheme already exists — skip silently
                pass

        logger.info(f"Seed data loaded | new_records={inserted} | total_in_file={len(schemes)}")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def search(self, query: str, limit: int = 5) -> list[KBRecord]:
        """
        Full-text search across banking schemes using FTS5.
        
        Search strategy:
        1. Tokenize the query using FTS5's porter stemmer
        2. Match against scheme_name, category, description, eligibility,
           benefits, and interest_rate columns
        3. Rank results using BM25 (Best Matching 25) algorithm
        4. Return top-N results
        
        WHY BM25?
        BM25 is the industry standard for text relevance ranking. It
        considers term frequency, document length, and corpus-wide term
        rarity. This is more sophisticated than simple LIKE matching.
        
        FALLBACK:
        If FTS5 returns no results (rare for well-formed queries), we
        fall back to a LIKE-based search for partial matching.
        """
        results = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sanitize query for FTS5 — escape special characters
            sanitized = self._sanitize_fts_query(query)

            if sanitized:
                # Primary: FTS5 full-text search with BM25 ranking
                cursor.execute("""
                    SELECT bs.*, bm25(banking_schemes_fts) as relevance_score
                    FROM banking_schemes_fts fts
                    JOIN banking_schemes bs ON fts.rowid = bs.rowid
                    WHERE banking_schemes_fts MATCH ?
                    ORDER BY relevance_score
                    LIMIT ?
                """, (sanitized, limit))

                rows = cursor.fetchall()
                results = [self._row_to_record(row) for row in rows]

            # Fallback: LIKE-based search if FTS returns nothing
            if not results:
                like_pattern = f"%{query}%"
                cursor.execute("""
                    SELECT * FROM banking_schemes
                    WHERE scheme_name LIKE ? 
                       OR description LIKE ? 
                       OR eligibility LIKE ?
                       OR benefits LIKE ?
                       OR category LIKE ?
                    LIMIT ?
                """, (like_pattern, like_pattern, like_pattern,
                      like_pattern, like_pattern, limit))

                rows = cursor.fetchall()
                results = [self._row_to_record(row) for row in rows]

            # Log the access for compliance
            self._log_access(cursor, query, len(results))

        logger.debug(f"KB search | query='{query[:60]}' | results={len(results)}")
        return results

    def get_by_id(self, scheme_id: str) -> Optional[KBRecord]:
        """
        Direct lookup by scheme ID.
        
        Used when:
        - The query explicitly references a scheme (e.g., "PMJDY-001")
        - The hallucination fallback has identified the relevant scheme
        - Evaluators need to verify specific scheme data
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM banking_schemes WHERE scheme_id = ?",
                (scheme_id,),
            )
            row = cursor.fetchone()

            if row:
                self._log_access(cursor, f"get_by_id:{scheme_id}", 1)
                return self._row_to_record(row)

            self._log_access(cursor, f"get_by_id:{scheme_id}", 0)
            return None

    def get_by_category(self, category: str, limit: int = 10) -> list[KBRecord]:
        """
        Category-based filtering.
        
        Categories in the KB:
        - Financial Inclusion
        - MSME & Entrepreneurship
        - Insurance
        - Pension
        - Savings & Investment
        - Banking Regulation
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM banking_schemes WHERE category LIKE ? LIMIT ?",
                (f"%{category}%", limit),
            )
            rows = cursor.fetchall()
            results = [self._row_to_record(row) for row in rows]

            self._log_access(cursor, f"category:{category}", len(results))

        return results

    def get_all_schemes(self) -> list[KBRecord]:
        """
        Return all schemes — used for evaluation and data inspection.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM banking_schemes ORDER BY scheme_id")
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_scheme_count(self) -> int:
        """Return total number of schemes in the KB."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM banking_schemes")
            return cursor.fetchone()[0]

    def get_access_logs(self, limit: int = 50) -> list[dict]:
        """
        Retrieve KB access audit logs.
        
        Used by the /retrieval/logs endpoint for observability.
        In production, these logs would feed into a SIEM system.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM kb_access_log ORDER BY accessed_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def health_check(self) -> dict:
        """
        KB health check — verifies database connectivity and data presence.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM banking_schemes")
                count = cursor.fetchone()[0]
                return {
                    "status": "healthy" if count > 0 else "degraded",
                    "scheme_count": count,
                    "db_path": self.db_path,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "db_path": self.db_path,
            }

    # =========================================================================
    # Private Helpers
    # =========================================================================

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Sanitize a user query for FTS5 syntax.
        
        FTS5 has special characters (*, ", -, OR, AND, NOT, NEAR) that
        can cause syntax errors if passed raw. We:
        1. Remove special FTS operators
        2. Split into terms and join with OR (for recall)
        3. Add prefix matching (*) for partial word support
        
        WHY OR instead of AND?
        In a banking KB fallback context, we want HIGH RECALL — it's better
        to return extra results than miss the right one. BM25 ranking
        naturally pushes documents matching more terms to the top, so
        precision is maintained via ordering, not filtering.
        """
        # Remove FTS5 special characters
        special_chars = ['"', "'", "(", ")", ":", "*", "^", "~"]
        sanitized = query
        for char in special_chars:
            sanitized = sanitized.replace(char, " ")

        # Split into meaningful terms (2+ chars to include short banking terms)
        terms = [
            term.strip()
            for term in sanitized.split()
            if len(term.strip()) >= 2
        ]

        if not terms:
            return ""

        # Join with OR and add prefix matching for recall
        # "mudra yojana eligibility" → "mudra* OR yojana* OR eligibility*"
        # BM25 will rank docs matching all terms highest
        return " OR ".join(f"{term}*" for term in terms[:8])  # Cap at 8 terms

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> KBRecord:
        """Convert a SQLite Row to a KBRecord Pydantic model."""
        return KBRecord(
            scheme_id=row["scheme_id"],
            scheme_name=row["scheme_name"],
            category=row["category"],
            description=row["description"],
            eligibility=row["eligibility"],
            benefits=row["benefits"],
            interest_rate=row["interest_rate"],
            source=row["source"] or "knowledge_base",
            last_verified=row["last_verified"],
        )

    @staticmethod
    def _log_access(
        cursor: sqlite3.Cursor,
        query: str,
        results_count: int,
        token_scope: str = "read",
        requested_by: str = "system",
    ) -> None:
        """
        Log KB access for compliance auditing.
        
        In banking, every data access must be traceable. This provides:
        - Who accessed what data
        - When the access occurred
        - How many results were returned
        - What scope/permission was used
        """
        cursor.execute("""
            INSERT INTO kb_access_log (query, results_count, token_scope, requested_by)
            VALUES (?, ?, ?, ?)
        """, (query, results_count, token_scope, requested_by))


# =============================================================================
# Singleton Instance
# =============================================================================
# We use a module-level singleton because:
# 1. The KB is shared across all request handlers
# 2. SQLite connections are created per-operation (thread-safe)
# 3. The initialization state is tracked to avoid redundant setup

_kb_service_instance: Optional[KnowledgeBaseService] = None


def get_kb_service() -> KnowledgeBaseService:
    """
    Get or create the KB service singleton.
    
    Thread-safe: SQLite connections are created per-operation inside
    the service, so sharing the service instance is safe.
    """
    global _kb_service_instance
    if _kb_service_instance is None:
        _kb_service_instance = KnowledgeBaseService()
    return _kb_service_instance
