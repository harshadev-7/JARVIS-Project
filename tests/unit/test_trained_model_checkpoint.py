"""Unit Tests for J.A.R.V.I.S. Trained Checkpoint Verification (Phase 5).

Validates:
1. Saved checkpoint file and vocabulary JSON existence
2. Checkpoint structure (model_state, optimizer_state, model_config, best_val_loss)
3. Model weight restoration fidelity
4. Autoregressive inference on the restored model
"""

import json
from pathlib import Path
import torch

from custom_ai.generation import TextGenerator
from custom_ai.model import DecoderOnlyTransformer
from custom_ai.tokenizer import TextTokenizer


def test_trained_checkpoint_and_vocab_exist():
    ckpt_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1.pt")
    vocab_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1_vocab.json")

    assert ckpt_path.exists(), f"Checkpoint must exist at {ckpt_path}"
    assert vocab_path.exists(), f"Vocab file must exist at {vocab_path}"
    assert ckpt_path.stat().st_size > 10_000, "Checkpoint file size must be non-trivial"


def test_checkpoint_payload_structure():
    ckpt_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1.pt")
    payload = torch.load(ckpt_path, map_location="cpu")

    assert "model_state" in payload
    assert "optimizer_state" in payload
    assert "model_config" in payload
    assert "epoch" in payload
    assert "best_val_loss" in payload

    cfg = payload["model_config"]
    assert cfg["embed_dim"] == 64
    assert cfg["num_layers"] == 2
    assert cfg["num_heads"] == 4


def test_checkpoint_loading_and_inference():
    ckpt_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1.pt")
    vocab_path = Path("D:/Documents/JARVIS/checkpoints/local_models/jarvis_micro_v1_vocab.json")

    vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
    tokenizer = TextTokenizer.from_dict(vocab_data)

    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = payload["model_config"]

    model = DecoderOnlyTransformer(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    generator = TextGenerator(model, tokenizer, device="cpu")
    output = generator.generate("hello jarvis", max_new_tokens=8, temperature=0.7, top_k=5)

    assert isinstance(output, str)
    assert len(output) > 0
    assert "hello" in output.lower() or "jarvis" in output.lower()
