"""
JARVIS AI — Semantic Vector Memory Engine
==========================================
Provides vector-based semantic retrieval across knowledge items,
past conversations, and episodic memories.
Implements TF-IDF + cosine similarity vector space model for local zero-dependency
semantic search, designed to integrate seamlessly with ChromaDB/Qdrant.
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class VectorDocument:
    """A document indexed in semantic vector space."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: List[float] = field(default_factory=list)


class SemanticVectorStore:
    """
    Lightweight, fast, in-process semantic vector index with cosine similarity search.
    """

    def __init__(self):
        self._documents: Dict[str, VectorDocument] = {}
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Convert text into lowercase word tokens."""
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency."""
        tf: Dict[str, float] = {}
        total = len(tokens)
        if total == 0:
            return tf
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        for t in tf:
            tf[t] = tf[t] / total
        return tf

    def _rebuild_index(self):
        """Recompute IDF across all indexed documents."""
        n_docs = len(self._documents)
        if n_docs == 0:
            return

        doc_freq: Dict[str, int] = {}
        for doc in self._documents.values():
            tokens = set(self._tokenize(doc.text))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        self._idf = {
            t: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            for t, df in doc_freq.items()
        }

        # Build vocabulary
        self._vocabulary = {t: idx for idx, t in enumerate(self._idf.keys())}

        # Embed all documents
        for doc in self._documents.values():
            doc.embedding = self._vectorize(doc.text)

    def _vectorize(self, text: str) -> List[float]:
        """Compute TF-IDF dense embedding vector."""
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        vec = [0.0] * len(self._vocabulary)

        norm_sq = 0.0
        for t, val in tf.items():
            if t in self._vocabulary:
                idx = self._vocabulary[t]
                weight = val * self._idf.get(t, 1.0)
                vec[idx] = weight
                norm_sq += weight * weight

        # Normalize to unit vector
        norm = math.sqrt(norm_sq)
        if norm > 0.0:
            vec = [v / norm for v in vec]

        return vec

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two unit vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        return sum(a * b for a, b in zip(vec1, vec2))

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Insert or replace a document in the vector store."""
        doc = VectorDocument(
            doc_id=doc_id,
            text=text,
            metadata=metadata or {},
        )
        self._documents[doc_id] = doc
        self._rebuild_index()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        """
        Query vector index for semantically relevant documents.
        Returns list of (document, similarity_score) sorted descending.
        """
        if not self._documents or not query.strip():
            return []

        query_vec = self._vectorize(query)
        results: List[Tuple[VectorDocument, float]] = []

        for doc in self._documents.values():
            sim = self._cosine_similarity(query_vec, doc.embedding)
            if sim > 0.0:
                results.append((doc, round(sim, 4)))

        # Sort descending by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """Remove document and rebuild index."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._rebuild_index()
            return True
        return False

    def clear(self):
        """Wipe vector memory."""
        self._documents.clear()
        self._vocabulary.clear()
        self._idf.clear()

    def count(self) -> int:
        return len(self._documents)


semantic_vector_store = SemanticVectorStore()
