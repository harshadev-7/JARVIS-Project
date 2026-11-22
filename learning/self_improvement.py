"""J.A.R.V.I.S. Self-Improvement & Continuous Fine-Tuning Engine.

Allows JARVIS to learn incrementally from its own verified experiences:
1. Harvests high-quality dialogue pairs from SQLite persistent memory.
2. Formulates an Experience Replay Buffer blended with core domain corpus
   to prevent catastrophic forgetting.
3. Executes a resource-aware micro-fine-tuning cycle locally on CPU.
4. Validates training loss convergence before updating checkpoints.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from core.logger import log
from custom_ai.dataset import clean_text, split_train_val
from custom_ai.inference_core import LocalModelInference
from custom_ai.model import DecoderOnlyTransformer
from custom_ai.tokenizer import TextTokenizer
from custom_ai.trainer import LanguageModelTrainer


class SelfImprovementEngine:
    """Orchestrates autonomous continuous learning and model fine-tuning."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
        corpus_path: Optional[str | Path] = None,
    ):
        self.db_path = Path(db_path or "D:/Documents/JARVIS/jarvis.db")
        self.checkpoint_path = Path(checkpoint_path or "D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1.pt")
        self.corpus_path = Path(corpus_path or "D:/Documents/JARVIS/datasets/jarvis_corpus.txt")

    def harvest_experience_pairs(self, limit: int = 50) -> List[str]:
        """Extract user-assistant dialogue pairs from SQLite persistent memory."""
        if not self.db_path.exists():
            return []

        samples: List[str] = []
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM conversation_episodes
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit * 2,),
            )
            rows = cursor.fetchall()
            conn.close()

            # Pair consecutive user and assistant turns
            user_msg = ""
            for role, content in rows:
                if role == "user":
                    user_msg = content.strip()
                elif role == "assistant" and user_msg:
                    # Filter out short or error responses
                    if len(content.strip()) > 15 and "error" not in content.lower():
                        sample = f"user: {user_msg}\nassistant: {content.strip()}"
                        samples.append(sample)
                    user_msg = ""
        except Exception as e:
            log.warning(f"Could not harvest conversation episodes from memory: {e}")

        return samples

    def build_replay_buffer(self, max_core_samples: int = 40) -> List[str]:
        """Combine newly harvested dialogue with core domain corpus."""
        harvested = self.harvest_experience_pairs()

        core_samples: List[str] = []
        if self.corpus_path.exists():
            text = self.corpus_path.read_text(encoding="utf-8")
            core_samples = [clean_text(b) for b in text.split("\n\n") if clean_text(b)]

        # Blend: Prioritize core anchors to retain stable language structure
        buffer = core_samples[:max_core_samples] + harvested
        log.info(f"Experience Replay Buffer built: {len(harvested)} harvested + {len(core_samples[:max_core_samples])} core anchors.")
        return buffer

    def run_fine_tuning_cycle(
        self,
        epochs: int = 2,
        batch_size: int = 2,
        lr: float = 5e-4,
    ) -> Dict[str, Any]:
        """Execute a safe, low-learning-rate fine-tuning step on CPU.

        Returns:
            Dictionary with training telemetry and metrics.
        """
        if not self.checkpoint_path.exists():
            return {"status": "skipped", "reason": "Base checkpoint missing"}

        buffer = self.build_replay_buffer()
        if len(buffer) < 4:
            return {"status": "skipped", "reason": "Insufficient dialogue experience in replay buffer"}

        # 1. Load active inference model and tokenizer
        engine = LocalModelInference(checkpoint_path=self.checkpoint_path)
        model = engine.model
        tokenizer = engine.tokenizer

        if model is None or tokenizer is None:
            return {"status": "failed", "reason": "Failed to load model from checkpoint"}

        # 2. Split train/val
        train_texts, val_texts = split_train_val(buffer, val_ratio=0.15, seed=42)

        # 3. Train
        trainer = LanguageModelTrainer(
            model=model,
            tokenizer=tokenizer,
            lr=lr,
            seq_len=model.max_seq_len,
            device="cpu",
        )

        log.info(f"SelfImprovementEngine starting {epochs}-epoch micro fine-tuning cycle...")
        result = trainer.train_with_validation(
            train_texts=train_texts,
            val_texts=val_texts,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path=str(self.checkpoint_path),
            sample_prompt="hello jarvis",
        )

        return {
            "status": "success",
            "samples_trained": len(train_texts),
            "samples_validated": len(val_texts),
            "final_train_loss": result.get("train_loss"),
            "final_val_loss": result.get("val_loss"),
            "epochs": epochs,
            "checkpoint": str(self.checkpoint_path),
        }


self_improvement_engine = SelfImprovementEngine()

__all__ = ["SelfImprovementEngine", "self_improvement_engine"]
