"""J.A.R.V.I.S. Resource-Aware Training Pipeline.

Optimized for local CPU execution (Intel Core i3 / 8 GB RAM):
- Hardware detection (CPU threads & RAM check) before training
- PyTorch DataLoader with small batching and memory efficiency
- Next-token prediction with Cross-Entropy loss (ignoring padding)
- AdamW optimizer with gradient clipping and optional gradient accumulation
- Validation loss monitoring with best checkpoint auto-saving
- Resume training support from saved checkpoints
- Post-epoch text generation sampling to monitor learning progress
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import TextDataset
from .generation import TextGenerator


def detect_system_resources() -> Dict[str, Any]:
    """Inspect available CPU and RAM resources before training."""
    mem = psutil.virtual_memory()
    total_ram_gb = round(mem.total / (1024**3), 2)
    available_ram_gb = round(mem.available / (1024**3), 2)
    cpu_logical = psutil.cpu_count(logical=True) or 4
    cpu_physical = psutil.cpu_count(logical=False) or 2

    # Recommended batch size based on available RAM
    recommended_batch = 2 if available_ram_gb < 2.0 else 4

    return {
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "cpu_logical": cpu_logical,
        "cpu_physical": cpu_physical,
        "recommended_batch_size": recommended_batch,
        "is_safe": available_ram_gb >= 0.3,
    }


class LanguageModelTrainer:
    """End-to-end training manager for the custom J.A.R.V.I.S. Transformer."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        device: str = "cpu",
        ignore_index: int = -100,
        seq_len: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.ignore_index = ignore_index
        self.grad_accum_steps = max(1, gradient_accumulation_steps)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.training_history: List[Dict[str, Any]] = []
        self.best_val_loss = float("inf")

    def _make_loader(
        self, texts: Iterable[str], batch_size: int = 2, shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch DataLoader from text corpus."""
        dataset = TextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            ignore_index=self.ignore_index,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass and Cross-Entropy loss computation."""
        # inputs: (B, T), targets: (B, T)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(inputs)  # (B, T, vocab_size)

        # Reshape to (B * T, vocab_size) and (B * T)
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return loss

    def train_epoch(self, texts_or_loader, batch_size: int = 2) -> float:
        """Train for a single epoch."""
        if isinstance(texts_or_loader, DataLoader):
            loader = texts_or_loader
        else:
            loader = self._make_loader(texts_or_loader, batch_size=batch_size, shuffle=True)

        self.model.train()
        total_loss = 0.0
        step_count = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            inputs = batch["input_ids"]
            targets = batch["labels"]

            loss = self._compute_loss(inputs, targets)
            loss_scaled = loss / self.grad_accum_steps
            loss_scaled.backward()

            total_loss += loss.item()
            step_count += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / max(step_count, 1)

    def evaluate(self, texts_or_loader, batch_size: int = 2) -> float:
        """Evaluate validation loss on held-out text."""
        if isinstance(texts_or_loader, DataLoader):
            loader = texts_or_loader
        else:
            loader = self._make_loader(texts_or_loader, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        step_count = 0

        with torch.no_grad():
            for batch in loader:
                inputs = batch["input_ids"]
                targets = batch["labels"]
                loss = self._compute_loss(inputs, targets)
                total_loss += loss.item()
                step_count += 1

        return total_loss / max(step_count, 1)

    def train_with_validation(
        self,
        train_texts: Iterable[str],
        val_texts: Iterable[str],
        epochs: int = 3,
        batch_size: int = 2,
        checkpoint_path: Optional[str] = None,
        sample_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full training loop with validation, checkpointing, and generation checks."""
        resources = detect_system_resources()

        history = []
        train_loader = self._make_loader(train_texts, batch_size=batch_size, shuffle=True)
        val_loader = self._make_loader(val_texts, batch_size=batch_size, shuffle=False)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
            }

            # Generation sample check
            if sample_prompt:
                generator = TextGenerator(
                    model=self.model, tokenizer=self.tokenizer, device=self.device
                )
                sample = generator.generate(sample_prompt, max_new_tokens=8, temperature=0.7)
                epoch_record["sample_generation"] = sample

            history.append(epoch_record)

            # Auto-save best checkpoint
            if val_loss < self.best_val_loss and checkpoint_path:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    extra={"val_loss": val_loss, "best": True},
                )

        self.training_history.extend(history)
        latest = history[-1] if history else {"train_loss": 0.0, "val_loss": 0.0, "epoch": 0}

        return {
            "epoch": latest["epoch"],
            "train_loss": latest["train_loss"],
            "val_loss": latest["val_loss"],
            "best_val_loss": round(self.best_val_loss, 4),
            "history": history,
            "resources": resources,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save complete training state to file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "tokenizer_state": (
                self.tokenizer.to_dict() if hasattr(self.tokenizer, "to_dict") else {}
            ),
            "model_config": {
                "vocab_size": getattr(self.model, "vocab_size", None),
                "embed_dim": getattr(self.model, "embed_dim", None),
                "num_layers": getattr(self.model, "num_layers", None),
                "num_heads": getattr(self.model, "num_heads", None),
                "max_seq_len": getattr(self.model, "max_seq_len", None),
            },
            "extra": extra or {},
        }
        torch.save(payload, target_path)
        return str(target_path)

    def load_checkpoint(
        self,
        path: str | Path,
        model: Optional[nn.Module] = None,
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint and resume model/optimizer states."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(target_path, map_location=self.device, weights_only=False)

        target_model = model or self.model
        target_model.load_state_dict(checkpoint["model_state"])

        if load_optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]

        return checkpoint


# Backward compatibility alias
TinyLanguageModelTrainer = LanguageModelTrainer

__all__ = [
    "LanguageModelTrainer",
    "TinyLanguageModelTrainer",
    "detect_system_resources",
]
