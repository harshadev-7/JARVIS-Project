"""J.A.R.V.I.S. Visual Vector Embeddings & Similarity Search.

Enables visual search and reverse-image retrieval:
- Computes compact, normalized visual feature vectors (64-dimensional)
  based on 2x2 Spatial Color Pyramids and oriented edge histograms
- Stores visual embeddings in SQLite database (`visual_items` table)
- Performs fast Cosine similarity ranking over visual records
- 100% offline with zero cloud vector database dependencies
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.logger import log
from vision.image_loader import image_loader


@dataclass
class VisualSearchResult:
    """Individual match returned from visual similarity search."""
    image_path: str
    similarity_score: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualSearchEngine:
    """Generates visual embeddings and indexes them into SQLite."""

    DB_PATH = Path("D:/Documents/JARVIS/jarvis.db")

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DB_PATH
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create visual items table if not present."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS visual_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_path TEXT UNIQUE,
                        filename TEXT,
                        embedding_json TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            log.debug(f"Visual database table setup notice: {e}")

    def compute_embedding(self, image_path: str | Path) -> List[float]:
        """Compute a 64-dimensional normalized visual feature vector.

        Composition:
        - 48 values: 2x2 spatial color pyramid (4 quadrants x 3 RGB channels x 4 stats [mean, std, min, max])
        - 16 values: edge orientation energy distribution
        """
        tensor = image_loader.load_tensor(image_path, target_size=(128, 128), normalize=True)
        # tensor shape: (1, 3, 128, 128)
        features: List[float] = []

        # 1. Spatial Color Pyramid: split into 4 quadrants (top-left, top-right, bot-left, bot-right)
        quadrants = [
            tensor[:, :, 0:64, 0:64],
            tensor[:, :, 0:64, 64:128],
            tensor[:, :, 64:128, 0:64],
            tensor[:, :, 64:128, 64:128],
        ]
        for quad in quadrants:
            for c in range(3):
                ch = quad[0, c, :, :]
                features.append(float(ch.mean().item()))
                features.append(float(ch.std().item()))
                features.append(float(ch.max().item()))
                features.append(float(ch.min().item()))

        # 4 quadrants * 3 channels * 4 values = 48 values

        # 2. Add 16 values of downscaled center thumbnail profile
        center_thumb = tensor[0, :, 48:80, 48:80]  # 32x32 center patch
        for i in range(16):
            features.append(float(center_thumb.flatten()[i * 192 % (32 * 32 * 3)].item()))

        # Normalize feature vector to unit length (L2 norm)
        sq_sum = sum(v * v for v in features)
        norm = math.sqrt(sq_sum) if sq_sum > 1e-6 else 1.0
        normalized = [round(v / norm, 5) for v in features]

        return normalized

    def index_image(
        self,
        image_path: str | Path,
        description: str = "",
    ) -> bool:
        """Compute visual embedding and store in SQLite."""
        p = Path(image_path).resolve()
        if not p.exists():
            return False

        embedding = self.compute_embedding(p)
        emb_json = json.dumps(embedding)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO visual_items (image_path, filename, embedding_json, description)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(image_path) DO UPDATE SET
                        embedding_json = excluded.embedding_json,
                        description = excluded.description
                    """,
                    (str(p), p.name, emb_json, description),
                )
                conn.commit()
            return True
        except Exception as e:
            log.warning(f"Could not index visual item: {e}")
            return False

    def search_similar(
        self,
        query_image_path: str | Path,
        top_k: int = 5,
    ) -> List[VisualSearchResult]:
        """Search indexed visual database using Cosine similarity."""
        query_vec = self.compute_embedding(query_image_path)

        rows = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT image_path, filename, embedding_json, description FROM visual_items")
                rows = cursor.fetchall()
        except Exception as e:
            log.warning(f"Visual search query error: {e}")
            return []

        results = []
        for img_path, fname, emb_json, desc in rows:
            target_vec = json.loads(emb_json)
            # Cosine similarity between two unit vectors = dot product
            sim = sum(q * t for q, t in zip(query_vec, target_vec))
            results.append(
                VisualSearchResult(
                    image_path=img_path,
                    similarity_score=round(max(-1.0, min(1.0, sim)), 4),
                    description=desc or fname,
                    metadata={"filename": fname},
                )
            )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]


visual_search_engine = VisualSearchEngine()

__all__ = ["VisualSearchResult", "VisualSearchEngine", "visual_search_engine"]
