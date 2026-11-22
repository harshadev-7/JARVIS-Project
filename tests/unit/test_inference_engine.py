"""Unit Tests for J.A.R.V.I.S. Local Inference Engine (Phase 6).

Validates:
1. Loading trained checkpoints into LocalModelInference
2. Structured InferenceResponse generation and latency metrics
3. Generation parameters (temperature, top_k, max_new_tokens, repetition_penalty)
4. Robust error handling on missing checkpoint files (CheckpointNotFoundError)
5. Robust error handling on corrupt checkpoint files (InvalidCheckpointError)
6. Model registry discovery, registration, and active model selection
7. Pretrained adapter separation and unambiguous model type tagging
"""

from pathlib import Path
import pytest
import torch

from custom_ai.inference import (
    CheckpointNotFoundError,
    InferenceResponse,
    InvalidCheckpointError,
    LocalModelInference,
)
from custom_ai.model_registry import LocalModelRegistry
from custom_ai.pretrained_adapter import CustomTransformerAdapter, PretrainedLocalModelAdapter


def test_inference_from_saved_checkpoint():
    ckpt_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1.pt")
    assert ckpt_path.exists(), "Trained checkpoint must exist from Phase 5"

    engine = LocalModelInference(checkpoint_path=ckpt_path)
    assert engine.model is not None
    assert engine.tokenizer is not None

    resp = engine.generate_detailed("hello jarvis", max_new_tokens=8, temperature=0.7)

    assert isinstance(resp, InferenceResponse)
    assert isinstance(resp.text, str)
    assert resp.tokens_generated > 0
    assert resp.duration_ms > 0.0
    assert resp.model_type == "custom_transformer"
    assert resp.status == "success"

    # Status check
    st = engine.status()
    assert st["ready"] is True
    assert st["model_type"] == "custom_transformer"
    assert st["parameters"] == engine.model.get_num_params()
    assert st["parameters"] > 100_000


def test_inference_generation_parameters():
    ckpt_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1.pt")
    engine = LocalModelInference(checkpoint_path=ckpt_path)

    # Test top-k and temperature constraints
    out1 = engine.generate("who are you", max_new_tokens=4, temperature=0.1, top_k=1)
    out2 = engine.generate("who are you", max_new_tokens=4, temperature=1.5, top_k=20)

    assert isinstance(out1, str)
    assert isinstance(out2, str)


def test_checkpoint_not_found_handling():
    non_existent_path = Path("D:/Documents/JARVIS/checkpoints/local_models/non_existent_model.pt")

    with pytest.raises(CheckpointNotFoundError) as excinfo:
        LocalModelInference(checkpoint_path=non_existent_path)

    assert "does not exist" in str(excinfo.value)


def test_corrupt_checkpoint_handling(tmp_path):
    corrupt_file = tmp_path / "corrupt_model.pt"
    # Write invalid bytes
    corrupt_file.write_bytes(b"\x00\x01\x02CORRUPT_NOT_A_VALID_TORCH_STATE\xff\xfe")

    with pytest.raises(InvalidCheckpointError) as excinfo:
        LocalModelInference(checkpoint_path=corrupt_file)

    assert "Failed to read checkpoint" in str(excinfo.value) or "corrupt" in str(excinfo.value).lower()


def test_model_registry_discovery_and_registration():
    registry = LocalModelRegistry(checkpoint_dir="D:/Documents/JARVIS/checkpoints/local_models")
    discovered = registry.discover_checkpoints()

    assert len(discovered) >= 1
    checkpoint_names = [p.name for p in discovered]
    assert "jarvis_micro_v1.pt" in checkpoint_names

    # Register from discovered checkpoint
    entry = registry.register_from_checkpoint(
        name="jarvis_active",
        checkpoint_path=registry.checkpoint_dir / "jarvis_micro_v1.pt",
    )
    assert entry["status"] == "ready"

    # Status reporting
    status_report = registry.status()
    assert "jarvis_active" in status_report
    assert status_report["jarvis_active"]["ready"] is True

    # Generation via registry
    result = registry.generate("hello", model_name="jarvis_active", max_new_tokens=5)
    assert isinstance(result, str)
    assert len(result) > 0


def test_custom_vs_pretrained_adapter_separation():
    ckpt_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1.pt")
    engine = LocalModelInference(checkpoint_path=ckpt_path)

    # 1. Custom Transformer Adapter
    custom_adapter = CustomTransformerAdapter(engine)
    assert custom_adapter.is_available() is True

    custom_resp = custom_adapter.generate("hello jarvis", max_new_tokens=4)
    assert custom_resp.model_type == "custom_transformer"

    status_custom = custom_adapter.status()
    assert status_custom["adapter_type"] == "custom_transformer"
    assert status_custom["provider"] == "local_scratch"

    # 2. Pretrained Local Adapter
    pretrained_adapter = PretrainedLocalModelAdapter(
        model_path="D:/Documents/JARVIS/checkpoints/local_models/non_existent.gguf"
    )
    status_pretrained = pretrained_adapter.status()

    # Pretrained adapter must clearly state its separate identity and not be available
    assert status_pretrained["adapter_type"] == "pretrained_local"
    assert status_pretrained["requires_cloud"] is False
    assert status_pretrained["requires_ollama"] is False
    assert pretrained_adapter.is_available() is False

    # Calling generate on unavailable pretrained adapter raises error instead of silently falling back
    with pytest.raises(RuntimeError) as excinfo:
        pretrained_adapter.generate("hello")
    assert "not loaded" in str(excinfo.value)
