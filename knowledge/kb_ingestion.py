"""J.A.R.V.I.S. Knowledge Ingestion Pipeline.

Parses text documents, technical notes, and markdown into semantically
coherent chunks and indexes them into the Knowledge Base.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from core.logger import log
from knowledge.knowledge_base import knowledge_base


class KBIngestionPipeline:
    """Document chunker and batch ingestion pipeline."""

    def __init__(self, chunk_size_words: int = 80, overlap_words: int = 15):
        self.chunk_size = chunk_size_words
        self.overlap = overlap_words

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping semantic chunks."""
        # Split on double newlines (paragraphs) first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []

        for p in paragraphs:
            words = p.split()
            if len(words) <= self.chunk_size:
                chunks.append(p)
            else:
                # Sliding window over large paragraph
                start = 0
                while start < len(words):
                    end = start + self.chunk_size
                    chunk = " ".join(words[start:end])
                    chunks.append(chunk)
                    start += self.chunk_size - self.overlap

        return chunks

    async def ingest_document(
        self,
        title: str,
        text: str,
        category: str = "general",
        source: str = "document",
    ) -> List[int]:
        """Chunk and ingest a complete text document into the knowledge base.

        Returns list of created knowledge entry IDs.
        """
        chunks = self.chunk_text(text)
        created_ids: List[int] = []

        for idx, chunk in enumerate(chunks):
            chunk_title = f"{title} (Part {idx + 1}/{len(chunks)})" if len(chunks) > 1 else title
            item_id = await knowledge_base.add_knowledge(
                title=chunk_title,
                content=chunk,
                category=category,
                source=source,
            )
            created_ids.append(item_id)

        log.info(f"Ingested document '{title}' ({len(chunks)} chunk(s)) into category '{category}'.")
        return created_ids

    async def ingest_file(
        self,
        file_path: str | Path,
        category: Optional[str] = None,
    ) -> List[int]:
        """Read a file from disk and ingest into the knowledge base."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found for ingestion: {path.resolve()}")

        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_category = category or path.suffix.replace(".", "") or "general"
        return await self.ingest_document(
            title=path.stem.replace("_", " ").title(),
            text=text,
            category=doc_category,
            source=str(path.name),
        )


# Global singleton instance
kb_ingestion = KBIngestionPipeline()

__all__ = ["KBIngestionPipeline", "kb_ingestion"]
