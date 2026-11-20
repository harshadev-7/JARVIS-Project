"""J.A.R.V.I.S. Memory Search, Retrieval & Lifecycle Operations (Phase 8).

Provides:
- Keyword & similarity search across conversations, preferences, and knowledge items
- Context synthesis for augmenting neural model prompts
- Safe memory deletion (session wipe, preference deletion, full reset)
- Online database backup triggering
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from .database import DatabaseManager, db_manager
from .long_term import LongTermMemory, long_term_memory
from .short_term import ShortTermMemory, short_term_memory


class MemoryRetrieval:
    """Manages memory search, context augmentation, deletion, and backups."""

    def __init__(
        self,
        long_term: Optional[LongTermMemory] = None,
        short_term: Optional[ShortTermMemory] = None,
        db: Optional[DatabaseManager] = None,
    ):
        self.long_term = long_term or long_term_memory
        self.short_term = short_term or short_term_memory
        self.db = db or db_manager

    def search_memory(
        self,
        query: str,
        search_knowledge: bool = True,
        search_conversations: bool = True,
        search_preferences: bool = True,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search across SQLite tables using keyword frequency matching."""
        terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        if not terms:
            return []

        results: List[Dict[str, Any]] = []

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Search Knowledge Items
            if search_knowledge:
                cursor.execute(
                    "SELECT id, category, title, content, source, created_at FROM knowledge_items ORDER BY id DESC LIMIT 50"
                )
                for row in cursor.fetchall():
                    text_blob = f"{row['category']} {row['title']} {row['content']}".lower()
                    score = sum(text_blob.count(t) for t in terms)
                    if score > 0:
                        results.append({
                            "type": "knowledge",
                            "id": row["id"],
                            "title": row["title"],
                            "content": row["content"],
                            "category": row["category"],
                            "score": score,
                        })

            # 2. Search User Preferences
            if search_preferences:
                cursor.execute("SELECT key, value, category FROM user_preferences")
                for row in cursor.fetchall():
                    text_blob = f"{row['key']} {row['value']} {row['category']}".lower()
                    score = sum(text_blob.count(t) for t in terms)
                    if score > 0:
                        results.append({
                            "type": "preference",
                            "key": row["key"],
                            "value": row["value"],
                            "category": row["category"],
                            "score": score,
                        })

            # 3. Search Past Conversations
            if search_conversations:
                cursor.execute(
                    "SELECT id, session_id, role, content, created_at FROM conversation_history ORDER BY id DESC LIMIT 100"
                )
                for row in cursor.fetchall():
                    content_lower = row["content"].lower()
                    score = sum(content_lower.count(t) for t in terms)
                    if score > 0:
                        results.append({
                            "type": "conversation",
                            "id": row["id"],
                            "session_id": row["session_id"],
                            "role": row["role"],
                            "content": row["content"],
                            "score": score,
                        })

        # Rank by match score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def retrieve_context_for_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_items: int = 3,
    ) -> str:
        """Construct context augmentation text from relevant knowledge and preferences."""
        matches = self.search_memory(
            query=query,
            search_knowledge=True,
            search_preferences=True,
            search_conversations=False,
            limit=max_items,
        )

        if not matches:
            return ""

        context_lines = ["Relevant persistent context from JARVIS memory:"]
        for item in matches:
            if item["type"] == "knowledge":
                context_lines.append(f"- [{item['category']}] {item['title']}: {item['content']}")
            elif item["type"] == "preference":
                context_lines.append(f"- [Preference] {item['key']}: {item['value']}")

        return "\n".join(context_lines)

    # ── Deletion Operations ────────────────────────────────────────────────

    def delete_session_history(self, session_id: str) -> int:
        """Delete all conversation turns for a session from both memory tiers."""
        self.short_term.clear_session(session_id)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count

    def delete_preference(self, key: str) -> bool:
        """Delete a user preference."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_preferences WHERE key = ?", (key,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def delete_knowledge_item(self, item_id: int) -> bool:
        """Delete a specific knowledge record."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (item_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def clear_all_memory(self, confirm: bool = False) -> bool:
        """Wipe all persistent tables upon explicit confirmation."""
        if not confirm:
            raise PermissionError("Explicit confirmation flag (confirm=True) required to wipe memory.")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversation_history;")
            cursor.execute("DELETE FROM user_preferences;")
            cursor.execute("DELETE FROM knowledge_items;")
            cursor.execute("DELETE FROM task_execution_history;")
            conn.commit()
            return True

    # ── Backup Trigger ─────────────────────────────────────────────────────

    def backup(self, backup_dir: str | Path = "backups") -> Path:
        """Trigger an online database backup."""
        return self.db.backup_database(backup_dir=backup_dir)


# Default memory retrieval instance
memory_retrieval = MemoryRetrieval()

__all__ = ["MemoryRetrieval", "memory_retrieval"]
