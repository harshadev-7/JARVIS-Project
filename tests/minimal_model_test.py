import os, sys
import pathlib

# Ensure project root is on PYTHONPATH
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    import torch
except ImportError:
    print("torch not installed – minimal model test skipped")
    sys.exit(0)

from custom_ai.model import TransformerModel
from custom_ai.tokenizer import Tokenizer

def test_model_forward_and_checkpoint(tmp_path):
    # Build a tiny vocab
    tokenizer = Tokenizer()
    corpus = ["hello world", "test sentence"]
    tokenizer.fit(corpus)
    vocab_size = len(tokenizer.vocab)

    # Minimal model configuration
    config = {
        "vocab_size": vocab_size,
        "embed_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ff_dim": 64,
        "max_seq_len": 8,
    }
    model = TransformerModel(**config)
    model.train()

    # Encode a sample sentence
    tokens = torch.tensor([tokenizer.encode("hello world")], dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (1, tokens.size(1), vocab_size)

    # Save checkpoint
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    assert ckpt_path.is_file()

    # Load checkpoint into a fresh model
    new_model = TransformerModel(**config)
    new_model.load_state_dict(torch.load(ckpt_path)["model_state"])
    new_logits = new_model(tokens)
    assert torch.allclose(logits, new_logits)
    print("minimal model forward & checkpoint test passed")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_model_forward_and_checkpoint(pathlib.Path(td))
