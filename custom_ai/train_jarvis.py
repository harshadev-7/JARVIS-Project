"""J.A.R.V.I.S. Local Model Training Script (Phase 5).

Trains the custom decoder-only Transformer ("JARVIS-Micro") locally on CPU:
- Uses datasets/jarvis_corpus.txt
- Fits TextTokenizer
- Builds DecoderOnlyTransformer (d_model=64, 2 layers, 4 heads)
- Evaluates train/validation loss across epochs
- Saves best model checkpoint to checkpoints/local_models/jarvis_micro_v1.pt
- Tests autoregressive text generation post-training
"""

import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from custom_ai.dataset import clean_text, load_corpus_file, split_train_val
from custom_ai.generation import TextGenerator
from custom_ai.model import DecoderOnlyTransformer
from custom_ai.tokenizer import TextTokenizer
from custom_ai.trainer import LanguageModelTrainer, detect_system_resources


def train_jarvis_model(
    corpus_path: str = "datasets/jarvis_corpus.txt",
    checkpoint_dir: str = "checkpoints/local_models",
    model_name: str = "jarvis_micro_v1",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 2e-3,
    seq_len: int = 32,
    embed_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    ff_dim: int = 256,
):
    print("=" * 65)
    print("      J.A.R.V.I.S. LOCAL AI TRAINING PIPELINE — PHASE 5")
    print("=" * 65)

    # 1. Hardware Inspection
    res = detect_system_resources()
    print(f"[*] CPU Cores: {res['cpu_physical']} physical, {res['cpu_logical']} logical")
    print(f"[*] System RAM: {res['available_ram_gb']} GB available / {res['total_ram_gb']} GB total")
    print(f"[*] Recommended Batch Size: {res['recommended_batch_size']}")
    print(f"[*] Execution Mode: 100% Local CPU Execution (PyTorch CPU)")
    print("-" * 65)

    # 2. Dataset Loading & Preprocessing
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found at: {corpus_file.resolve()}")

    samples = load_corpus_file(corpus_file)
    print(f"[*] Loaded {len(samples)} dialogue and instruction samples from {corpus_file.name}")

    train_texts, val_texts = split_train_val(samples, val_ratio=0.15, seed=42)
    print(f"[*] Training Samples: {len(train_texts)} | Validation Samples: {len(val_texts)}")

    # 3. Tokenizer Construction
    tokenizer = TextTokenizer(min_frequency=1)
    tokenizer.fit(samples)
    print(f"[*] Vocabulary Size: {tokenizer.vocab_size} tokens")

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = out_dir / f"{model_name}_vocab.json"
    vocab_file.write_text(json.dumps(tokenizer.to_dict(), indent=2), encoding="utf-8")
    print(f"[*] Serialized vocabulary saved to: {vocab_file}")

    # 4. Model Instantiation
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_len=seq_len,
        dropout=0.1,
    )
    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"[*] Architecture: {num_layers} Decoder Blocks, {num_heads} Heads, d_model={embed_dim}, d_ff={ff_dim}")
    print(f"[*] Total Parameters: {total_params:,} (Non-embedding: {non_emb_params:,})")
    print(f"[*] Estimated Model FP32 Footprint: ~{total_params * 4 / (1024**2):.2f} MB")
    print("-" * 65)

    # Baseline Untrained Generation Sample
    generator_init = TextGenerator(model, tokenizer, device="cpu")
    print("[*] Baseline (Untrained) Output for 'hello jarvis':")
    baseline_out = generator_init.generate("hello jarvis", max_new_tokens=6, temperature=0.8)
    print(f"    --> \"{baseline_out}\"")
    print("-" * 65)

    # 5. Training Loop
    trainer = LanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        lr=lr,
        seq_len=seq_len,
        device="cpu",
    )

    checkpoint_file = out_dir / f"{model_name}.pt"
    print(f"[*] Commencing training for {epochs} epochs...")
    start_time = time.time()

    train_summary = trainer.train_with_validation(
        train_texts=train_texts,
        val_texts=val_texts,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=str(checkpoint_file),
        sample_prompt="hello jarvis",
    )
    duration = time.time() - start_time

    print(f"[*] Training finished in {duration:.2f} seconds ({duration / epochs:.2f}s/epoch)")
    print("-" * 65)
    print(" Epoch | Train Loss | Val Loss | Sample Output")
    print("-" * 65)
    for rec in train_summary["history"]:
        sample_snippet = rec.get("sample_generation", "")[:35]
        print(f"  {rec['epoch']:2d}   |   {rec['train_loss']:.4f}   |  {rec['val_loss']:.4f}  | {sample_snippet}...")
    print("-" * 65)

    # 6. Checkpoint Verification & Final Generation
    print(f"[*] Loading best saved checkpoint: {checkpoint_file}")
    best_model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_len=seq_len,
    )
    trainer.load_checkpoint(checkpoint_file, model=best_model)
    generator_final = TextGenerator(best_model, tokenizer, device="cpu")

    test_prompts = [
        "hello jarvis",
        "who are you",
        "check system diagnostics",
        "what can you do",
    ]

    print("\n[*] Final Checkpoint Autoregressive Generation Test:")
    for prompt in test_prompts:
        output = generator_final.generate(
            prompt, max_new_tokens=10, temperature=0.7, top_k=10
        )
        print(f"  Prompt: \"{prompt}\"")
        print(f"  Output: \"{output}\"\n")

    print(f"[*] Model successfully trained and saved to: {checkpoint_file}")
    return train_summary


if __name__ == "__main__":
    train_jarvis_model()
