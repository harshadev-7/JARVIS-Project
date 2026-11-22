"""J.A.R.V.I.S. Knowledge Base & RAG Package.

Provides offline document ingestion, semantic search, and retrieval-augmented generation.
"""

from .knowledge_base import KnowledgeBase, KnowledgeEntry, knowledge_base
from .kb_ingestion import KBIngestionPipeline, kb_ingestion
from .rag_engine import RAGContext, RAGEngine, rag_engine

__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "knowledge_base",
    "KBIngestionPipeline",
    "kb_ingestion",
    "RAGContext",
    "RAGEngine",
    "rag_engine",
]
