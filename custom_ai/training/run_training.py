import argparse
import os
from pathlib import Path

import torch

from ..trainer import LanguageModelTrainer, detect_system_resources
from ..dataset import load_corpus_file, split_train_val
from ..tokenizer import TextTokenizer
from ..model import DecoderOnlyTransformer
from ..checkpoint_utils import save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train a tiny GPT‑style model for J.A.R.V.I.S.")
    parser.add_argument("--corpus", type=str, required=True, help="Path to a plain‑text corpus file.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None, help="Batch size (auto‑scaled if omitted).")
    parser.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=16, help="Context length for the model.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="checkpoints", help="Directory to store checkpoints.")
    parser.add_argument("--sample-prompt", "--sample_prompt", dest="sample_prompt", type=str, default=None, help="Prompt used for generation sampling each epoch.")
    args = parser.parse_args()

    # Load and split corpus
    texts = load_corpus_file(args.corpus)
    train_texts, val_texts = split_train_val(texts, train_ratio=0.9)

    # Tokenizer
    tokenizer = TextTokenizer()
    tokenizer.fit(train_texts)

    # Model config (CPU‑friendly defaults)
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ff_dim": 64,
        "max_seq_len": args.seq_len,
        "dropout": 0.0,
        "tie_weights": False,
    }
    model = DecoderOnlyTransformer(**config)

    # Detect resources for batch size if not provided
    resources = detect_system_resources()
    batch_size = args.batch_size or resources.get("recommended_batch_size", 2)

    trainer = LanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        lr=5e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        device="cpu",
        seq_len=args.seq_len,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.pth"

    history = trainer.train_with_validation(
        train_texts=train_texts,
        val_texts=val_texts,
        epochs=args.epochs,
        batch_size=batch_size,
        checkpoint_path=str(checkpoint_path),
        sample_prompt=args.sample_prompt,
    )

    # Save final checkpoint (epoch after loop)
    save_checkpoint(model, trainer.optimizer, tokenizer, epoch=args.epochs, checkpoint_path=str(checkpoint_path))
    print("Training complete. History summary:")
    for rec in history.get("history", [history]):
        print(rec)
    print(f"Best checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
