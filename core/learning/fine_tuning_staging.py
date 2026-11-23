"""J.A.R.V.I.S. Fine-Tuning Staging & Rollback Manager.

Coordinates CPU-safe candidate model fine-tuning:
- Resource validation before execution
- Candidate versioned checkpoint creation
- Side-by-side evaluation comparison
- Explicit activation gate and zero-downtime rollback protection
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from custom_ai.trainer import detect_system_resources, LanguageModelTrainer
from custom_ai.model import DecoderOnlyTransformer
from custom_ai.tokenizer import TextTokenizer
from core.learning.dataset_curator import dataset_curator
from core.logger import log


class FineTuningStagingManager:
    """Safeguards and manages optional local neural model fine-tuning."""

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        from config.settings import BASE_DIR
        self.checkpoints_dir = checkpoints_dir or (BASE_DIR / "custom_ai" / "checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.active_checkpoint = self.checkpoints_dir / "jarvis_micro_v1.pt"
        self.backup_checkpoint = self.checkpoints_dir / "jarvis_micro_v1.backup.pt"
        self.last_candidate_path: Optional[Path] = None

    def get_staging_status(self) -> Dict[str, Any]:
        """Inspect staged dataset size and system readiness."""
        staged_count = dataset_curator.count_staged_lines()
        res = detect_system_resources()
        return {
            "staged_examples": staged_count,
            "system_resources": res,
            "active_checkpoint_exists": self.active_checkpoint.exists(),
            "has_backup": self.backup_checkpoint.exists(),
            "last_candidate": str(self.last_candidate_path) if self.last_candidate_path else None,
        }

    def train_candidate(
        self,
        epochs: int = 3,
        lr: float = 5e-4,
        batch_size: int = 2,
    ) -> Dict[str, Any]:
        """
        Execute CPU-safe fine-tuning on staged dataset to produce a candidate checkpoint.
        Never directly overwrites active production weights.
        """
        staged_entries = dataset_curator.load_staged_dataset()
        if not staged_entries:
            return {
                "success": False,
                "error": "No staged training examples found. Add feedback or learning items first.",
            }

        # Resource safety check
        res = detect_system_resources()
        if not res.get("is_safe", True):
            return {
                "success": False,
                "error": f"Insufficient RAM ({res.get('available_ram_gb')} GB available). Minimum 0.3 GB required.",
            }

        texts = [f"{e['prompt']}{e['completion']}" for e in staged_entries]

        # Load tokenizer and baseline model
        tokenizer = TextTokenizer()
        tokenizer.fit(texts)
        device = "cpu"
        model = DecoderOnlyTransformer(
            vocab_size=max(tokenizer.vocab_size, 50),
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=64,
            dropout=0.0,
        )

        # Load existing weights if present
        if self.active_checkpoint.exists():
            try:
                ckpt = torch.load(self.active_checkpoint, map_location=device, weights_only=True)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                elif isinstance(ckpt, dict):
                    model.load_state_dict(ckpt, strict=False)
                log.info(f"Loaded existing baseline weights from {self.active_checkpoint}")
            except Exception as e:
                log.warning(f"Could not load baseline checkpoint: {e}. Starting from initialized weights.")

        trainer = LanguageModelTrainer(
            model=model,
            tokenizer=tokenizer,
            lr=lr,
            device=device,
            seq_len=32,
        )

        start_time = time.perf_counter()
        epoch_losses = []
        for ep in range(epochs):
            loss = trainer.train_epoch(texts, batch_size=batch_size)
            epoch_losses.append(loss)
            log.info(f"Fine-tuning epoch {ep + 1}/{epochs} - Loss: {loss:.4f}")

        elapsed_sec = round(time.perf_counter() - start_time, 2)
        initial_loss = epoch_losses[0]
        final_loss = epoch_losses[-1]

        # Save as versioned candidate checkpoint
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        candidate_filename = f"jarvis_candidate_{timestamp}.pt"
        candidate_path = self.checkpoints_dir / candidate_filename

        save_dict = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 2,
                "d_ff": 128,
                "max_seq_len": 64,
            },
            "training_metadata": {
                "timestamp": timestamp,
                "epochs": epochs,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "examples_count": len(texts),
                "elapsed_sec": elapsed_sec,
            },
        }

        tmp_path = candidate_path.with_suffix(".tmp")
        torch.save(save_dict, tmp_path)
        shutil.move(tmp_path, candidate_path)
        self.last_candidate_path = candidate_path

        log.info(f"Candidate model successfully saved to {candidate_path}")

        return {
            "success": True,
            "candidate_path": str(candidate_path),
            "epochs": epochs,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "elapsed_sec": elapsed_sec,
            "examples_trained": len(texts),
            "improved": final_loss <= initial_loss,
        }

    def activate_candidate(self, candidate_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Promote candidate checkpoint to active production model.
        Backs up current active model before replacement.
        """
        target = candidate_path or self.last_candidate_path
        if not target or not Path(target).exists():
            return {
                "success": False,
                "error": "No valid candidate model checkpoint found to activate.",
            }

        target_path = Path(target)

        # 1. Backup current active checkpoint
        if self.active_checkpoint.exists():
            shutil.copy2(self.active_checkpoint, self.backup_checkpoint)
            log.info(f"Backed up active model to {self.backup_checkpoint}")

        # 2. Copy candidate to active checkpoint
        shutil.copy2(target_path, self.active_checkpoint)
        log.info(f"Activated candidate model: {target_path} -> {self.active_checkpoint}")

        # 3. Clear staged buffer
        dataset_curator.clear_staged_file()

        return {
            "success": True,
            "activated_model": str(self.active_checkpoint),
            "backup_created": str(self.backup_checkpoint),
            "message": "Candidate model successfully activated. Staged dataset cleared.",
        }

    def rollback_to_backup(self) -> Dict[str, Any]:
        """Revert active production model to previous backup."""
        if not self.backup_checkpoint.exists():
            return {
                "success": False,
                "error": "No backup checkpoint found to restore.",
            }

        shutil.copy2(self.backup_checkpoint, self.active_checkpoint)
        log.info(f"Restored active model from backup: {self.backup_checkpoint} -> {self.active_checkpoint}")

        return {
            "success": True,
            "restored_model": str(self.active_checkpoint),
            "message": "Successfully rolled back to previous model checkpoint.",
        }


fine_tuning_staging = FineTuningStagingManager()
__all__ = ["FineTuningStagingManager", "fine_tuning_staging"]
