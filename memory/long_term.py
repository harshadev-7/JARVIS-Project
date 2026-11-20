"""J.A.R.V.I.S. Long-Term Persistent Memory (Phase 8).

Coordinates persistent storage of:
- Full conversation history
- User-approved preferences
- Curated knowledge repository
- Task execution audits

Strict Security:
- Automatically detects sensitive information (passwords, API keys, private tokens)
- Enforces explicit user permission before persisting sensitive data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from .database import DatabaseManager, db_manager


class SensitiveDataError(PermissionError):
    """Raised when sensitive data is submitted without explicit user permission."""
    pass


class LongTermMemory:
    """Manages persistent SQLite memory records with security validation."""

    # Patterns indicating potentially sensitive credentials or private information
    SENSITIVE_PATTERNS = [
        r"(password|passwd|pwd)\s*[:=]\s*\S+",
        r"(sk-[a-zA-Z0-9]{20,})",  # API keys
        r"(bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*)",  # Auth tokens
        r"(ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59})",
        r"\b(?:\d[ -]*?){13,16}\b",  # Credit card numbers
        r"(secret_key|private_key)\s*[:=]\s*\S+",
    ]

    def __init__(self, manager: Optional[DatabaseManager] = None):
        self.db = manager or db_manager

    def is_sensitive(self, text: str) -> bool:
        """Check if content matches sensitive credential or private information patterns."""
        if not text:
            return False
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    # ── Conversation History ───────────────────────────────────────────────

    def store_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        allow_sensitive: bool = False,
    ) -> int:
        """Store conversation message in SQLite database."""
        if self.is_sensitive(content) and not allow_sensitive:
            raise SensitiveDataError(
                "Security policy violation: Sensitive data detected (password/key/token). "
                "Explicit user approval is required to store this information."
            )

        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_history (session_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, meta_json, now),
            )
            conn.commit()
            return cursor.lastrowid

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a specific session."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, role, content, metadata_json, created_at
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    # ── User Preferences ───────────────────────────────────────────────────

    def set_preference(
        self,
        key: str,
        value: str,
        category: str = "general",
        user_approved: bool = True,
        confidence: float = 1.0,
    ) -> bool:
        """Store or update user preference with security validation."""
        if (self.is_sensitive(key) or self.is_sensitive(value)) and not user_approved:
            raise SensitiveDataError(
                f"Cannot save preference '{key}': contains sensitive credentials without user approval."
            )

        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_preferences (key, value, category, confidence, user_approved, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    confidence = excluded.confidence,
                    user_approved = excluded.user_approved,
                    updated_at = excluded.updated_at
                """,
                (key, value, category, confidence, 1 if user_approved else 0, now),
            )
            conn.commit()
            return True

    def get_preference(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a user preference by key."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, key, value, category, confidence, user_approved, updated_at FROM user_preferences WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "key": row["key"],
                "value": row["value"],
                "category": row["category"],
                "confidence": row["confidence"],
                "user_approved": bool(row["user_approved"]),
                "updated_at": row["updated_at"],
            }

    def list_preferences(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute(
                    "SELECT key, value, category, user_approved FROM user_preferences WHERE category = ?",
                    (category,),
                )
            else:
                cursor.execute("SELECT key, value, category, user_approved FROM user_preferences")
            rows = cursor.fetchall()
            return [
                {
                    "key": r["key"],
                    "value": r["value"],
                    "category": r["category"],
                    "user_approved": bool(r["user_approved"]),
                }
                for r in rows
            ]

    # ── Knowledge Storage ──────────────────────────────────────────────────

    def store_knowledge(
        self,
        title: str,
        content: str,
        category: str = "general",
        topic: Optional[str] = None,
        source: str = "user",
        is_verified: bool = True,
    ) -> int:
        """Store knowledge snippet in SQLite database."""
        now = datetime.now(timezone.utc).isoformat()
        cat = category or topic or "general"

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO knowledge_items (title, content, category, source, is_verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (title, content, cat, source, 1 if is_verified else 0, now),
            )
            conn.commit()
            return cursor.lastrowid

    def get_knowledge(self, item_id: int) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_items WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    # ── Task Execution History ─────────────────────────────────────────────

    def record_task(
        self,
        task_id: str,
        goal: str,
        status: str,
        task_name: Optional[str] = None,
        summary: Optional[str] = None,
        steps_count: int = 1,
        duration_sec: float = 0.0,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO task_execution_history (task_id, goal, status, steps_count, duration_sec, error_summary, lessons_learned, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, goal, status, steps_count, duration_sec, summary or "", "", now),
            )
            conn.commit()
            return cursor.lastrowid


# Default long-term memory instance
long_term_memory = LongTermMemory()

__all__ = ["LongTermMemory", "SensitiveDataError", "long_term_memory"]
