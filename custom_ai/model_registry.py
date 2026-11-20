"""J.A.R.V.I.S. Local Model Registry (Phase 6).

Coordinates model checkpoints, active model selection, and registry state:
- Auto-discovers checkpoints in checkpoints/local_models
- Safely handles corrupt or missing checkpoints with custom error boundaries
- Provides unified access to LocalModelInference
- Tracks active model selection with full status reporting
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .inference import (
    CheckpointError,
    CheckpointNotFoundError,
    InferenceResponse,
    InvalidCheckpointError,
    LocalModelInference,
)
from .model import DecoderOnlyTransformer
from .tokenizer import TextTokenizer


class LocalModelRegistry:
    """Central registry and life-cycle manager for local J.A.R.V.I.S. models."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints/local_models"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Dict[str, Any]] = {}
        self._inference_engines: Dict[str, LocalModelInference] = {}
        self._default_name: Optional[str] = None

    def discover_checkpoints(self) -> List[Path]:
        """Find all .pt checkpoint files in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return []
        return sorted(self.checkpoint_dir.glob("*.pt"))

    def register_model(
        self,
        name: str,
        model: torch.nn.Module,
        tokenizer: TextTokenizer,
    ) -> Dict[str, Any]:
        """Register an in-memory model and tokenizer."""
        inference = LocalModelInference(model=model, tokenizer=tokenizer, model_name=name)
        self._models[name] = {
            "model": model,
            "tokenizer": tokenizer,
            "status": "ready",
            "checkpoint": None,
            "metadata": {
                "vocab_size": getattr(model, "vocab_size", tokenizer.vocab_size),
                "embed_dim": getattr(model, "embed_dim", None),
                "parameters": getattr(model, "get_num_params", lambda: 0)(),
            },
        }
        self._inference_engines[name] = inference
        if self._default_name is None:
            self._default_name = name
        return self._models[name]

    def register_from_checkpoint(
        self,
        name: str,
        checkpoint_path: str | Path,
        vocab_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Safely load and register a model from a saved checkpoint."""
        try:
            inference = LocalModelInference(
                checkpoint_path=checkpoint_path,
                vocab_path=vocab_path,
                model_name=name,
            )
            self._models[name] = {
                "model": inference.model,
                "tokenizer": inference.tokenizer,
                "status": "ready",
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "metadata": {
                    "vocab_size": inference.tokenizer.vocab_size if inference.tokenizer else None,
                    "embed_dim": getattr(inference.model, "embed_dim", None),
                    "parameters": inference.model.get_num_params() if inference.model else 0,
                },
            }
            self._inference_engines[name] = inference
            if self._default_name is None:
                self._default_name = name
            return self._models[name]
        except CheckpointError as e:
            self._models[name] = {
                "status": "error",
                "error": str(e),
                "checkpoint": str(checkpoint_path),
            }
            raise

    def set_active_model(self, model_name: str) -> None:
        """Set the default model for inference."""
        if model_name not in self._models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(self._models.keys())}")
        self._default_name = model_name

    def get_inference_engine(self, model_name: Optional[str] = None) -> LocalModelInference:
        """Retrieve the inference engine for the requested or default model."""
        name = model_name or self._default_name
        if name is None or name not in self._inference_engines:
            raise ValueError(f"No active inference engine found for model: {name}")
        return self._inference_engines[name]

    def _checkpoint_path_for(self, model_name: str) -> Path:
        return self.checkpoint_dir / f"{model_name}.pt"

    def save_current_model(
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save registered model state to checkpoint file."""
        entry = self._models.get(model_name)
        if entry is None or "model" not in entry:
            raise ValueError(f"Unknown or uninitialized local model '{model_name}'")

        path = self._checkpoint_path_for(model_name)
        payload = {
            "model_state": entry["model"].state_dict(),
            "metadata": metadata or entry["metadata"],
            "model_config": {
                "vocab_size": getattr(entry["model"], "vocab_size", entry["tokenizer"].vocab_size),
                "embed_dim": getattr(entry["model"], "embed_dim", 64),
                "num_layers": getattr(entry["model"], "num_layers", 2),
                "num_heads": getattr(entry["model"], "num_heads", 4),
                "max_seq_len": getattr(entry["model"], "max_seq_len", 32),
            },
            "tokenizer_state": entry["tokenizer"].to_dict(),
        }
        torch.save(payload, path)
        entry["checkpoint"] = str(path)
        entry["status"] = "trained"
        return str(path)

    def load_model(
        self, model_name: str, checkpoint_path: Optional[str] = None
    ) -> torch.nn.Module:
        """Load checkpoint weights into a registered model."""
        entry = self._models.get(model_name)
        if entry is None:
            raise ValueError(f"Unknown local model '{model_name}'")

        path = checkpoint_path or entry.get("checkpoint") or self._checkpoint_path_for(model_name)
        if not Path(path).exists():
            return entry["model"]

        payload = torch.load(path, map_location="cpu")
        entry["model"].load_state_dict(payload["model_state"])
        entry["status"] = "ready"
        entry["checkpoint"] = str(path)
        return entry["model"]

    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_new_tokens: int = 8,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> str:
        """Generate string output using the specified or default model."""
        engine = self.get_inference_engine(model_name)
        return engine.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Return comprehensive status for all models in the registry."""
        report: Dict[str, Dict[str, Any]] = {}
        for name, entry in self._models.items():
            metadata = entry.get("metadata", {})
            report[name] = {
                "status": entry.get("status", "ready"),
                "checkpoint": entry.get("checkpoint"),
                "vocab_size": metadata.get("vocab_size"),
                "embed_dim": metadata.get("embed_dim"),
                "parameters": metadata.get("parameters"),
                "ready": entry.get("status", "ready") in {"ready", "trained"},
                "is_default": name == self._default_name,
            }
        return report


__all__ = ["LocalModelRegistry"]
