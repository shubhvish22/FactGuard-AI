
import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional

from config import HASH_DB_PATH

logger = logging.getLogger("FactGuard.HistoryManager")


class HistoryManager:
    """Manages the verification_history SQLite table with thread-safe queries and WAL mode."""

    def __init__(self, db_path: str = HASH_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=20.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create verification_history table if it does not already exist."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_type TEXT NOT NULL,          -- 'text' | 'image'
                    content_preview TEXT NOT NULL,          -- Snippet of text or image filename
                    source_url TEXT DEFAULT '',             -- Original webpage URL
                    page_title TEXT DEFAULT '',             -- Original webpage title
                    direct_image_url TEXT DEFAULT '',       -- Direct image source URL
                    claims_json TEXT DEFAULT '[]',          -- JSON serialized list of claims and verdicts
                    overall_verdict TEXT DEFAULT '',        -- Main verdict (Supported, Contradicted, etc.)
                    ml_prediction TEXT DEFAULT '',          -- ML label (FAKE, REAL, ai_generated, etc.)
                    ml_confidence REAL DEFAULT 0.0,         -- Confidence score
                    evidence_sources_json TEXT DEFAULT '[]',-- Top canonical evidence sources
                    source_discovery_status TEXT DEFAULT '',-- 'sources_found' | 'insufficient_evidence'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_created ON verification_history(created_at DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_type ON verification_history(submission_type);")
            conn.commit()
            logger.info("Verification history table initialized successfully in %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize verification_history table: %s", e)
        finally:
            if conn:
                conn.close()

    def save_entry(
        self,
        submission_type: str = "",
        content_preview: str = "",
        source_url: str = "",
        page_title: str = "",
        direct_image_url: str = "",
        claims: Optional[List[Dict[str, Any]]] = None,
        overall_verdict: str = "",
        ml_prediction: str = "",
        ml_confidence: float = 0.0,
        evidence_sources: Optional[List[Dict[str, Any]]] = None,
        source_discovery_status: str = "",
        **kwargs,
    ) -> Optional[int]:
        """Save a new verification record into history and return its ID."""
        sub_type = submission_type or kwargs.get("item_type", "text")
        preview = content_preview or kwargs.get("submitted_text") or kwargs.get("image_identifier", "")
        title = page_title or kwargs.get("webpage_title", "")
        verdict = overall_verdict or kwargs.get("top_verdict", "")

        claims_str = json.dumps(claims or [], ensure_ascii=False)
        sources_str = json.dumps(evidence_sources or [], ensure_ascii=False)
        preview_clean = (preview or "")[:400].strip()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO verification_history (
                    submission_type, content_preview, source_url, page_title,
                    direct_image_url, claims_json, overall_verdict, ml_prediction,
                    ml_confidence, evidence_sources_json, source_discovery_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sub_type,
                preview_clean,
                (source_url or "").strip(),
                (title or "").strip(),
                (direct_image_url or "").strip(),
                claims_str,
                verdict,
                ml_prediction,
                float(ml_confidence or 0.0),
                sources_str,
                source_discovery_status,
            ))
            conn.commit()
            entry_id = cursor.lastrowid
            logger.info("Saved verification history entry #%s (%s)", entry_id, sub_type)
            return entry_id
        except Exception as e:
            logger.error("Error saving verification history: %s", e)
            return None
        finally:
            if conn:
                conn.close()

    def get_history(
        self,
        page: int = 1,
        limit: int = 10,
        filter_type: str = "all",
        query: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Retrieve paginated verification history with type filtering and search."""
        f_type = kwargs.get("item_type", filter_type)
        lim = kwargs.get("per_page", limit)

        page = max(1, int(page))
        lim = max(1, min(100, int(lim)))
        offset = (page - 1) * lim

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            conditions = []
            params = []

            if f_type in ("text", "image"):
                conditions.append("submission_type = ?")
                params.append(f_type)

            if query and query.strip():
                q_like = f"%{query.strip()}%"
                conditions.append("(content_preview LIKE ? OR source_url LIKE ? OR page_title LIKE ? OR overall_verdict LIKE ?)")
                params.extend([q_like, q_like, q_like, q_like])

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            # Count total matching rows
            count_sql = f"SELECT COUNT(*) FROM verification_history {where_clause}"
            cursor.execute(count_sql, params)
            total_count = cursor.fetchone()[0]

            # Fetch paginated rows
            select_sql = f"""
                SELECT id, submission_type, content_preview, source_url, page_title,
                       direct_image_url, claims_json, overall_verdict, ml_prediction,
                       ml_confidence, evidence_sources_json, source_discovery_status, created_at
                FROM verification_history
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            cursor.execute(select_sql, params + [lim, offset])
            rows = cursor.fetchall()

            entries = []
            for r in rows:
                entries.append(self._row_to_dict(r))

            total_pages = max(1, (total_count + lim - 1) // lim)

            return {
                "entries": entries,
                "items": entries,
                "total": total_count,
                "page": page,
                "limit": lim,
                "per_page": lim,
                "total_pages": total_pages,
            }
        except Exception as e:
            logger.error("Error fetching history: %s", e)
            return {
                "entries": [],
                "items": [],
                "total": 0,
                "page": page,
                "limit": lim,
                "per_page": lim,
                "total_pages": 1,
                "error": str(e),
            }
        finally:
            if conn:
                conn.close()

    def get_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single verification history entry by ID."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM verification_history WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None
        except Exception as e:
            logger.error("Error fetching history entry #%s: %s", entry_id, e)
            return None
        finally:
            if conn:
                conn.close()

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a history entry by ID."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM verification_history WHERE id = ?", (entry_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            logger.info("Deleted history entry #%s: %s", entry_id, deleted)
            return deleted
        except Exception as e:
            logger.error("Error deleting history entry #%s: %s", entry_id, e)
            return False
        finally:
            if conn:
                conn.close()

    def clear_all(self) -> int:
        """Clear all entries in verification_history table."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM verification_history")
            conn.commit()
            deleted_count = cursor.rowcount
            logger.info("Cleared all %s records from verification_history", deleted_count)
            return deleted_count
        except Exception as e:
            logger.error("Error clearing verification history: %s", e)
            return 0
        finally:
            if conn:
                conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row into a clean dictionary with parsed JSON fields."""
        claims = []
        sources = []
        try:
            claims = json.loads(row["claims_json"]) if row["claims_json"] else []
        except Exception:
            claims = []

        try:
            sources = json.loads(row["evidence_sources_json"]) if row["evidence_sources_json"] else []
        except Exception:
            sources = []

        return {
            "id": row["id"],
            "submission_type": row["submission_type"],
            "item_type": row["submission_type"],
            "content_preview": row["content_preview"],
            "submitted_text": row["content_preview"],
            "image_identifier": row["content_preview"],
            "source_url": row["source_url"] or "",
            "page_title": row["page_title"] or "",
            "webpage_title": row["page_title"] or "",
            "direct_image_url": row["direct_image_url"] or "",
            "claims": claims,
            "overall_verdict": row["overall_verdict"] or "",
            "top_verdict": row["overall_verdict"] or "",
            "ml_prediction": row["ml_prediction"] or "",
            "ml_confidence": round(float(row["ml_confidence"] or 0.0), 4),
            "evidence_sources": sources,
            "source_discovery_status": row["source_discovery_status"] or "",
            "created_at": str(row["created_at"]),
        }


# Singleton manager instance
_history_manager_instance = None


def get_history_manager() -> HistoryManager:
    """Retrieve singleton instance of HistoryManager."""
    global _history_manager_instance
    if _history_manager_instance is None:
        _history_manager_instance = HistoryManager()
    return _history_manager_instance
