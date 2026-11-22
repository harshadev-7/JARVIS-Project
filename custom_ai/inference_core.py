"""J.A.R.V.I.S. Local Model Inference Engine (Phase 6).

Provides a local inference interface for custom Transformer models:
- Safe checkpoint loading with corruption/absence validation
- Autoregressive text generation with temperature, top-k, and repetition penalty
- Structured generation metadata (latency ms, token counts, model type)
- Detailed model status reporting
- Zero external cloud or API dependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

import torch

from .generation import TextGenerator
from .model import DecoderOnlyTransformer
from .tokenizer import TextTokenizer


class CheckpointError(Exception):
    """Base exception for model checkpoint operations."""
    pass


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    """Raised when the specified checkpoint file does not exist on disk."""
    pass


class InvalidCheckpointError(CheckpointError, ValueError):
    """Raised when a checkpoint file is corrupt or has an invalid structure."""
    pass


@dataclass
class InferenceResponse:
    """Structured response from the local inference engine."""
    text: str
    prompt: str
    tokens_generated: int
    model_name: str = "custom_transformer"
    model_type: str = "custom_transformer"
    duration_ms: float = 0.0
    status: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "prompt": self.prompt,
            "tokens_generated": self.tokens_generated,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
        }


class LocalModelInference:
    """High-reliability local inference runner for custom J.A.R.V.I.S. models."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[TextTokenizer] = None,
        checkpoint_path: Optional[str | Path] = None,
        vocab_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        model_name: str = "jarvis_custom",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.checkpoint_path: Optional[str] = None
        self.model = model
        self.tokenizer = tokenizer

        if checkpoint_path is not None:
            self.load_from_checkpoint(checkpoint_path, vocab_path=vocab_path)
        elif self.model is not None and self.tokenizer is not None:
            self.model.to(self.device)
            self.model.eval()
            self.generator = TextGenerator(self.model, self.tokenizer, device=self.device)
        else:
            self.generator = None

    def load_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        vocab_path: Optional[str | Path] = None,
    ) -> None:
        """Safely load and validate a model checkpoint from disk."""
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise CheckpointNotFoundError(f"Checkpoint file does not exist: {ckpt_file.resolve()}")

        # Attempt to load torch payload safely
        try:
            payload = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        except Exception as e:
            raise InvalidCheckpointError(
                f"Failed to read checkpoint '{ckpt_file.name}'. File may be corrupt or not a valid PyTorch file: {e}"
            )

        if not isinstance(payload, dict):
            raise InvalidCheckpointError(
                f"Invalid checkpoint format in '{ckpt_file.name}'. Expected dict, got {type(payload).__name__}."
            )

        if "model_state" not in payload:
            raise InvalidCheckpointError(
                f"Checkpoint '{ckpt_file.name}' is missing required 'model_state' key."
            )

        # 1. Restore Tokenizer
        if vocab_path and Path(vocab_path).exists():
            vocab_data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            self.tokenizer = TextTokenizer.from_dict(vocab_data)
        elif "tokenizer_state" in payload and payload["tokenizer_state"].get("vocab"):
            self.tokenizer = TextTokenizer.from_dict(payload["tokenizer_state"])
        elif self.tokenizer is None:
            # Look for sibling vocab file with same base name
            sibling_vocab = ckpt_file.parent / f"{ckpt_file.stem}_vocab.json"
            if sibling_vocab.exists():
                vocab_data = json.loads(sibling_vocab.read_text(encoding="utf-8"))
                self.tokenizer = TextTokenizer.from_dict(vocab_data)
            else:
                raise InvalidCheckpointError(
                    f"No tokenizer state found in checkpoint and vocab file missing at: {sibling_vocab}"
                )

        # 2. Instantiate Model Architecture
        model_config = payload.get("model_config") or payload.get("config", {})
        vocab_size = model_config.get("vocab_size", self.tokenizer.vocab_size)
        embed_dim = model_config.get("embed_dim", 64)
        num_layers = model_config.get("num_layers", 2)
        num_heads = model_config.get("num_heads", 4)
        max_seq_len = model_config.get("max_seq_len", 32)

        try:
            self.model = DecoderOnlyTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
            )
            self.model.load_state_dict(payload["model_state"])
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise InvalidCheckpointError(
                f"Model configuration mismatch or corrupt state dict in '{ckpt_file.name}': {e}"
            )

        self.checkpoint_path = str(ckpt_file.resolve())
        self.generator = TextGenerator(self.model, self.tokenizer, device=self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Autoregressively generate string response (backward-compatible signature)."""
        resp = self.generate_detailed(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        return resp.text

    def generate_detailed(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
    ) -> InferenceResponse:
        """Execute text generation with timing telemetry and token metrics."""
        if self.generator is None or self.model is None or self.tokenizer is None:
            raise RuntimeError("Inference engine is not initialized with a valid model and tokenizer.")

        start_time = time.perf_counter()
        raw_output = self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Estimate tokens generated
        gen_tokens = len(self.tokenizer.encode(raw_output)) - len(self.tokenizer.encode(prompt))
        tokens_generated = max(1, gen_tokens)

        return InferenceResponse(
            text=raw_output,
            prompt=prompt,
            tokens_generated=tokens_generated,
            model_name=self.model_name,
            model_type="custom_transformer",
            duration_ms=duration_ms,
            status="success",
        )

    def status(self) -> Dict[str, Any]:
        """Return system status and architecture parameters of the local model."""
        if self.model is None:
            return {
                "ready": False,
                "status": "uninitialized",
                "checkpoint": None,
                "model_type": "custom_transformer",
            }

        return {
            "ready": True,
            "status": "operational",
            "model_name": self.model_name,
            "model_type": "custom_transformer",
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "parameters": self.model.get_num_params(),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "embed_dim": getattr(self.model, "embed_dim", None),
            "num_layers": getattr(self.model, "num_layers", None),
            "num_heads": getattr(self.model, "num_heads", None),
            "max_seq_len": getattr(self.model, "max_seq_len", None),
        }


__all__ = [
    "CheckpointError",
    "CheckpointNotFoundError",
    "InferenceResponse",
    "InvalidCheckpointError",
    "LocalModelInference",
]
