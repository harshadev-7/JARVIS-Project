import argparse
import sys
from pathlib import Path

import torch

from .trainer import LanguageModelTrainer, detect_system_resources
from .dataset import load_corpus_file, split_train_val
from .tokenizer import TextTokenizer
from .model import DecoderOnlyTransformer
from .checkpoint_utils import save_checkpoint

def quick_train(
    corpus_path: str,
    epochs: int = 3,
    output_dir: str = "checkpoints",
    seq_len: int = 16,
    sample_prompt: str | None = None,
) -> str:
    """Programmatic wrapper for a full training run.

    Returns the path to the final checkpoint file.
    """
    # Load corpus and split
    texts = load_corpus_file(corpus_path)
    train_texts, val_texts = split_train_val(texts, train_ratio=0.9)

    # Tokenizer
    tokenizer = TextTokenizer()
    tokenizer.fit(train_texts)

    # Model configuration – CPU‑friendly defaults
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ff_dim": 64,
        "max_seq_len": seq_len,
        "dropout": 0.0,
        "tie_weights": False,
    }
    model = DecoderOnlyTransformer(**config)

    # Determine batch size
    resources = detect_system_resources()
    batch_size = resources.get("recommended_batch_size", 2)

    trainer = LanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        lr=5e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        device="cpu",
        seq_len=seq_len,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "quick_train_checkpoint.pth"

    # Run training with validation and checkpointing
    trainer.train_with_validation(
        train_texts=train_texts,
        val_texts=val_texts,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=str(checkpoint_path),
        sample_prompt=sample_prompt,
    )

    # Save final checkpoint
    save_checkpoint(model, trainer.optimizer, tokenizer, epoch=epochs, checkpoint_path=str(checkpoint_path))
    return str(checkpoint_path)

# Optional CLI entry point for quick manual runs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick‑train helper (programmatic).")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--sample-prompt", type=str, default=None)
    args = parser.parse_args()
    path = quick_train(
        corpus_path=args.corpus,
        epochs=args.epochs,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        sample_prompt=args.sample_prompt,
    )
    print(f"Quick‑train completed. Checkpoint saved to {path}")
