"""J.A.R.V.I.S. Retrieval-Augmented Generation (RAG) Engine.

Retrieves verified offline facts from the Knowledge Base and augments
prompts before passing them to the custom local Transformer or Central Brain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from knowledge.knowledge_base import KnowledgeEntry, knowledge_base


@dataclass
class RAGContext:
    """Retrieved facts and formatted augmentation context."""
    query: str
    entries: List[KnowledgeEntry] = field(default_factory=list)
    augmented_prompt: str = ""
    has_relevant_knowledge: bool = False
    top_score: float = 0.0

    def get_summary_context(self) -> str:
        """Return formatted bulleted facts for system context injection."""
        if not self.entries:
            return ""
        bullets = [f"• {e.title}: {e.content}" for e in self.entries]
        return "Verified Offline Knowledge:\n" + "\n".join(bullets)


class RAGEngine:
    """Coordinates retrieval-augmented factual generation."""

    def __init__(self, min_confidence: float = 0.08):
        self.min_confidence = min_confidence

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
    ) -> RAGContext:
        """Search knowledge base and assemble factual context.

        Args:
            query: User prompt or research query.
            top_k: Number of knowledge items to retrieve.
            category: Optional category filter.

        Returns:
            RAGContext containing retrieved entries and formatted prompt.
        """
        entries = await knowledge_base.search(
            query=query,
            top_k=top_k,
            category=category,
            min_score=self.min_confidence,
        )

        top_score = entries[0].score if entries else 0.0
        has_knowledge = len(entries) > 0

        # Construct augmented prompt
        if has_knowledge:
            facts_block = "\n".join([f"[{e.category}] {e.content}" for e in entries])
            augmented = (
                f"Context Knowledge:\n{facts_block}\n\n"
                f"User Question: {query}\n"
                f"Response based on facts:"
            )
        else:
            augmented = query

        return RAGContext(
            query=query,
            entries=entries,
            augmented_prompt=augmented,
            has_relevant_knowledge=has_knowledge,
            top_score=top_score,
        )


# Global singleton instance
rag_engine = RAGEngine()

__all__ = ["RAGContext", "RAGEngine", "rag_engine"]
