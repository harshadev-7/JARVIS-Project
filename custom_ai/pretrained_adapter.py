"""J.A.R.V.I.S. Local Model Adapter Framework (Section 7).

Provides independent adapters for:
1. Custom Transformer Model (our scratch-built neural network)
2. Optional Pretrained Local Model (e.g. direct GGUF / local weights without Ollama or cloud APIs)

Strict Rules:
- NO Ollama, NO Gemini, NO OpenAI dependencies.
- Custom Transformer and Pretrained Local Model are maintained as strictly separate components.
- Responses clearly and unambiguously identify the exact model that generated them.
- NEVER silently switch between model providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional

from .inference import InferenceResponse, LocalModelInference


class BaseLocalModelAdapter(ABC):
    """Abstract base class for independent local models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> InferenceResponse:
        """Generate response and return structured response with model identity."""
        pass

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Return operational state and model specifications."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model weights and execution libraries are ready."""
        pass


class CustomTransformerAdapter(BaseLocalModelAdapter):
    """Adapter for our custom GPT-style Transformer built from scratch."""

    def __init__(self, inference_engine: LocalModelInference):
        self.engine = inference_engine
        self.model_name = self.engine.model_name

    def is_available(self) -> bool:
        return self.engine.model is not None and self.engine.generator is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
    ) -> InferenceResponse:
        resp = self.engine.generate_detailed(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        resp.model_type = "custom_transformer"
        return resp

    def status(self) -> Dict[str, Any]:
        s = self.engine.status()
        s["adapter_type"] = "custom_transformer"
        s["provider"] = "local_scratch"
        return s


class PretrainedLocalModelAdapter(BaseLocalModelAdapter):
    """Adapter for optional open-weight models executed directly via local binaries (e.g. GGUF).

    Strictly offline. Does NOT use Ollama, OpenAI, or Gemini.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        model_name: str = "local_pretrained_gguf",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.model_name = model_name
        self._llm = None
        self._initialize_if_available()

    def _initialize_if_available(self) -> None:
        """Attempt to load local weights using direct Python bindings if installed."""
        if self.model_path and self.model_path.exists():
            try:
                # Direct local llama-cpp without daemon / server
                from llama_cpp import Llama  # Optional dependency
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=512,
                    n_threads=4,  # Tuned for Core i3 4 cores
                    verbose=False,
                )
            except ImportError:
                self._llm = None
            except Exception:
                self._llm = None

    def is_available(self) -> bool:
        return self._llm is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> InferenceResponse:
        if not self.is_available():
            raise RuntimeError(
                f"Pretrained local model '{self.model_name}' is not loaded. "
                "Ensure local weight file exists and direct local inference bindings are installed."
            )

        start = time.perf_counter()
        result = self._llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0

        generated_text = result["choices"][0]["text"].strip()
        usage = result.get("usage", {})
        tokens_gen = usage.get("completion_tokens", len(generated_text.split()))

        return InferenceResponse(
            text=generated_text,
            prompt=prompt,
            tokens_generated=tokens_gen,
            model_name=self.model_name,
            model_type="pretrained_local",
            duration_ms=duration_ms,
            status="success",
        )

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.is_available(),
            "status": "operational" if self.is_available() else "not_configured",
            "model_name": self.model_name,
            "model_type": "pretrained_local",
            "model_path": str(self.model_path) if self.model_path else None,
            "adapter_type": "pretrained_local",
            "provider": "direct_local_weights",
            "requires_cloud": False,
            "requires_ollama": False,
        }


__all__ = [
    "BaseLocalModelAdapter",
    "CustomTransformerAdapter",
    "PretrainedLocalModelAdapter",
]
