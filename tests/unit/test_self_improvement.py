"""Unit tests for J.A.R.V.I.S. Self-Improvement & Continuous Fine-Tuning Engine (Phase 20)."""

import pytest
from pathlib import Path
from learning.self_improvement import SelfImprovementEngine, self_improvement_engine


def test_harvest_and_replay_buffer():
    """Verify that harvested conversations and core anchors build a replay buffer."""
    engine = SelfImprovementEngine()
    buffer = engine.build_replay_buffer(max_core_samples=10)

    assert isinstance(buffer, list)
    assert len(buffer) > 0
    # Core samples should be non-empty strings
    assert all(isinstance(s, str) and len(s) > 5 for s in buffer)


def test_micro_fine_tuning_execution(tmp_path):
    """Verify a 1-epoch micro fine-tuning step runs on CPU without errors."""
    engine = SelfImprovementEngine()
    if not engine.checkpoint_path.exists():
        pytest.skip("Base checkpoint not yet trained.")

    result = engine.run_fine_tuning_cycle(epochs=1, batch_size=2, lr=1e-4)
    assert result["status"] == "success"
    assert result["samples_trained"] > 0
    assert result["final_train_loss"] is not None
