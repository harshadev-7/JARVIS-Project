import torch
import os
from pathlib import Path
from .model import DecoderOnlyTransformer
from .trainer import LanguageModelTrainer
from .dataset import TinyTextDataset, clean_text, load_corpus_file, split_train_val
from .tokenizer import TextTokenizer

def save_checkpoint(
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: TextTokenizer,
    epoch: int,
    checkpoint_path: str | os.PathLike,
) -> None:
    """Save model, optimizer and tokenizer state.

    The checkpoint is a simple ``torch.save`` of a dict.  It can be loaded with
    ``load_checkpoint`` or via :class:`custom_ai.inference.LocalModelInference`.
    """
    model_cfg = {
        "vocab_size": model.vocab_size,
        "embed_dim": model.embed_dim,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "ff_dim": model.ff_dim,
        "max_seq_len": model.max_seq_len,
    }
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer_state": tokenizer.to_dict(),
        "epoch": epoch,
        "config": model_cfg,
        "model_config": model_cfg,
    }
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, checkpoint_path)

def load_checkpoint(
    checkpoint_path: str | os.PathLike,
    device: str = "cpu",
) -> tuple[DecoderOnlyTransformer, torch.optim.Optimizer, TextTokenizer, int]:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns ``(model, optimizer, tokenizer, epoch)``.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("model_config") or ckpt.get("config", {})
    model = DecoderOnlyTransformer(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer_state"])

    tokenizer = TextTokenizer.from_dict(ckpt["tokenizer_state"])
    epoch = ckpt.get("epoch", 0)
    return model, optimizer, tokenizer, epoch
