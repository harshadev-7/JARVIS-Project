"""Training utilities for the custom JARVIS transformer.

Provides a lightweight ``Trainer`` class that orchestrates:
* epoch loops
* optional validation
* loss calculation (cross‑entropy)
* optimizer step with gradient accumulation support
* checkpoint saving and optional resume

The implementation is deliberately CPU‑friendly and avoids heavy
dependencies – only PyTorch and the project's own modules are used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from ..model import DecoderOnlyTransformer
from ..tokenizer import TextTokenizer


class Trainer:
    """Simple trainer for ``DecoderOnlyTransformer``.

    Parameters
    ----------
    model: DecoderOnlyTransformer
        The model to train.
    tokenizer: TextTokenizer
        Tokenizer used for the dataset (required for checkpoint saving).
    train_loader: DataLoader
        DataLoader yielding ``{"input_ids": Tensor, "labels": Tensor}``.
    val_loader: Optional[DataLoader]
        Validation DataLoader (same format as ``train_loader``).
    optimizer: torch.optim.Optimizer
        Optimiser for parameter updates.
    device: str, optional
        ``"cpu"`` (default) or ``"cuda"`` if available.
    lr_scheduler: Optional[Callable[[int], float]]
        Callable receiving current epoch and returning a learning‑rate factor.
    max_epochs: int, default 10
        Number of training epochs.
    grad_accum_steps: int, default 1
        Gradient accumulation steps to simulate larger batch sizes.
    checkpoint_dir: str | Path, default "./checkpoints"
        Directory where ``.pt`` checkpoint files are stored.
    checkpoint_prefix: str, default "jarvis_checkpoint"
        Prefix for checkpoint filenames.
    resume_from: Optional[str | Path]
        Path to a checkpoint to resume training from.
    """

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        tokenizer: TextTokenizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu",
        lr_scheduler: Optional[Callable[[int], float]] = None,
        max_epochs: int = 10,
        grad_accum_steps: int = 1,
        checkpoint_dir: str | Path = "./checkpoints",
        checkpoint_prefix: str = "jarvis_checkpoint",
        resume_from: Optional[str | Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.lr_scheduler = lr_scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_prefix = checkpoint_prefix
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.start_epoch = 0
        self.best_val_loss: Optional[float] = None

        if resume_from:
            self._load_checkpoint(resume_from)

        # Simple cross‑entropy loss (ignore padding token)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    # ---------------------------------------------------------------------
    # Checkpoint handling
    # ---------------------------------------------------------------------
    def _checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"{self.checkpoint_prefix}_epoch{epoch}.pt"

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model, optimiser and tokenizer state.

        The checkpoint is a ``dict`` with keys:
        * ``model_state`` – ``model.state_dict()``
        * ``optimizer_state`` – ``optimizer.state_dict()``
        * ``tokenizer_state`` – ``tokenizer.to_dict()``
        * ``epoch`` – current epoch (int)
        * ``model_config`` – hyper‑parameters required to recreate the model
        * ``training_args`` – a shallow copy of trainer init args for reproducibility
        """
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "tokenizer_state": self.tokenizer.to_dict(),
            "epoch": epoch,
            "model_config": {
                "vocab_size": self.model.vocab_size,
                "embed_dim": self.model.embed_dim,
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "ff_dim": self.model.ff_dim,
                "max_seq_len": self.model.max_seq_len,
                "dropout": getattr(self.model, "dropout", 0.0),
                "tie_weights": getattr(self.model, "tie_weights", False),
            },
            "training_args": {
                "max_epochs": self.max_epochs,
                "grad_accum_steps": self.grad_accum_steps,
                "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else None,
            },
        }
        path = self._checkpoint_path(epoch)
        torch.save(ckpt, path)
        if is_best:
            best_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt"
            torch.save(ckpt, best_path)
        print(f"[Trainer] Checkpoint saved: {path}")

    def _load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=self.device)
        # Load model state – re‑instantiate if config differs
        if "model_config" in payload:
            cfg = payload["model_config"]
            # Re‑create model only if shape mismatch (rare when resuming same model)
            if (
                cfg["vocab_size"] != self.model.vocab_size
                or cfg["embed_dim"] != self.model.embed_dim
                or cfg["num_layers"] != self.model.num_layers
                or cfg["num_heads"] != self.model.num_heads
                or cfg["max_seq_len"] != self.model.max_seq_len
            ):
                self.model = DecoderOnlyTransformer(
                    vocab_size=cfg["vocab_size"],
                    embed_dim=cfg["embed_dim"],
                    num_layers=cfg["num_layers"],
                    num_heads=cfg["num_heads"],
                    ff_dim=cfg.get("ff_dim"),
                    max_seq_len=cfg["max_seq_len"],
                    dropout=cfg.get("dropout", 0.0),
                    tie_weights=cfg.get("tie_weights", False),
                ).to(self.device)
        self.model.load_state_dict(payload["model_state"], strict=False)
        self.model.eval()
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.tokenizer = TextTokenizer.from_dict(payload["tokenizer_state"])
        self.start_epoch = payload.get("epoch", 0) + 1
        self.best_val_loss = payload.get("best_val_loss")
        print(f"[Trainer] Loaded checkpoint {ckpt_path} (starting at epoch {self.start_epoch})")

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training schedule.

        For each epoch the trainer:
        1. Iterates over ``train_loader`` computing loss and performing optimizer
           steps with optional gradient accumulation.
        2. Evaluates on ``val_loader`` (if provided) and records the validation loss.
        3. Saves a checkpoint (and a *best* checkpoint when validation improves).
        """
        for epoch in range(self.start_epoch, self.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            for step, batch in enumerate(self.train_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass – model returns logits of shape (B, T, vocab)
                logits = self.model(input_ids)
                # Shift logits & labels for next‑token prediction
                loss = self.criterion(
                    logits.view(-1, self.model.vocab_size), labels.view(-1)
                ) / self.grad_accum_steps
                loss.backward()
                epoch_loss += loss.item() * self.grad_accum_steps

                if step % self.grad_accum_steps == 0:
                    # Optional LR scheduler hook per epoch boundary
                    if self.lr_scheduler:
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.lr_scheduler(epoch)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            avg_train_loss = epoch_loss / len(self.train_loader)
            print(f"[Trainer] Epoch {epoch} – Train loss: {avg_train_loss:.4f}")

            # Validation (if any)
            val_loss: Optional[float] = None
            if self.val_loader:
                val_loss = self._validate()
                print(f"[Trainer] Epoch {epoch} – Val   loss: {val_loss:.4f}")
                is_best = self.best_val_loss is None or val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                self.save_checkpoint(epoch)

    def _validate(self) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, self.model.vocab_size), labels.view(-1)
                )
                total += loss.item()
                count += 1
        return total / max(count, 1)

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def set_device(self, device: str) -> None:
        """Move model and internal state to a new device (e.g., after detecting GPU)."""
        self.device = device
        self.model.to(device)
        # Optimizer parameters stay on the correct device automatically.

    def get_state_dict(self) -> Dict[str, Any]:
        """Return a shallow snapshot of trainer state (useful for debugging)."""
        return {
            "epoch": self.start_epoch,
            "best_val_loss": self.best_val_loss,
            "optimizer_state": self.optimizer.state_dict(),
        }

__all__ = ["Trainer"]
