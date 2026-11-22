"""J.A.R.V.I.S. Knowledge Base Engine.

Provides unified management over offline knowledge stored in:
- Persistent SQLite database (KnowledgeItem table)
- Semantic Vector Store (TF-IDF cosine similarity search)
- Metadata filtering, categorization, and cache
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logger import log
from core.memory.persistent_memory import persistent_memory


@dataclass
class KnowledgeEntry:
    """Structured representation of a knowledge item."""
    id: Optional[int]
    title: str
    content: str
    category: str = "general"
    source: str = "user"
    score: float = 0.0
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        """Format entry as a concise citation string."""
        return f"[{self.category.upper()}] {self.title}: {self.content}"


class KnowledgeBase:
    """Central manager for offline domain knowledge and indexed facts."""

    def __init__(self):
        self._memory = persistent_memory

    async def initialize(self) -> None:
        """Ensure backing storage tables and vector index are initialized."""
        await self._memory.initialize()

    async def add_knowledge(
        self,
        title: str,
        content: str,
        category: str = "general",
        source: str = "user",
    ) -> int:
        """Store verified knowledge item in SQLite and index in vector store."""
        item_id = await self._memory.store_knowledge(
            title=title.strip(),
            content=content.strip(),
            category=category.strip().lower(),
            source=source.strip(),
        )
        log.info(f"KnowledgeBase added entry #{item_id}: '{title}' [{category}]")
        return item_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.05,
    ) -> List[KnowledgeEntry]:
        """Perform semantic vector search across the knowledge base.

        Args:
            query: Natural language query string.
            top_k: Maximum number of entries to return.
            category: Optional category filter (e.g. 'java', 'os', 'algorithms').
            min_score: Minimum cosine similarity threshold (0.0 to 1.0).

        Returns:
            Ranked list of KnowledgeEntry objects.
        """
        raw_results = await self._memory.search_knowledge(query, top_k=top_k * 2)
        entries: List[KnowledgeEntry] = []

        for r in raw_results:
            score = r.get("score", 0.0)
            if score < min_score:
                continue

            meta = r.get("metadata", {})
            entry_cat = meta.get("category", "general")
            if category and entry_cat.lower() != category.lower():
                continue

            text = r.get("text", "")
            title = meta.get("title", "")
            content = text
            if ":" in text and not title:
                parts = text.split(":", 1)
                title = parts[0].strip()
                content = parts[1].strip()

            entries.append(
                KnowledgeEntry(
                    id=meta.get("id"),
                    title=title or "Knowledge Item",
                    content=content,
                    category=entry_cat,
                    source=meta.get("source", "system"),
                    score=round(score, 4),
                    metadata=meta,
                )
            )

            if len(entries) >= top_k:
                break

        return entries

    async def delete(self, item_id: int) -> bool:
        """Delete knowledge item by ID from SQLite and vector store."""
        success = await self._memory.delete_item("knowledge", item_id)
        if success:
            log.info(f"KnowledgeBase deleted item #{item_id}")
        return success


# Global singleton instance
knowledge_base = KnowledgeBase()

__all__ = ["KnowledgeBase", "KnowledgeEntry", "knowledge_base"]
