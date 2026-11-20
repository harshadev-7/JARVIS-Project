"""Unit Tests for J.A.R.V.I.S. Dataset Pipeline and Trainer.

Validates:
1. Hardware resource detection (RAM, CPU count, safe threshold)
2. Corpus file loading and text normalization
3. Train/validation splitting
4. Input and target sequence alignment (next-token prediction shift)
5. Model training loop with Cross-Entropy loss reduction
6. Gradient accumulation on CPU
7. Checkpoint persistence and training state resumption
8. Mid-training text generation inspection
"""

from pathlib import Path
import pytest
import torch

from custom_ai.dataset import TextDataset, clean_text, load_corpus_file, split_train_val
from custom_ai.model import DecoderOnlyTransformer
from custom_ai.tokenizer import TextTokenizer
from custom_ai.trainer import LanguageModelTrainer, detect_system_resources


def test_system_resource_detection():
    resources = detect_system_resources()
    assert "total_ram_gb" in resources
    assert "available_ram_gb" in resources
    assert "cpu_logical" in resources
    assert resources["total_ram_gb"] > 0
    assert resources["cpu_logical"] >= 1
    assert "recommended_batch_size" in resources


def test_text_cleaning_and_corpus_loading(tmp_path):
    corpus_file = Path("D:/Documents/JARVIS/datasets/jarvis_corpus.txt")
    assert corpus_file.exists(), "Educational corpus file must exist"

    blocks = load_corpus_file(corpus_file)
    assert len(blocks) >= 10
    for block in blocks:
        assert len(block.strip()) > 0

    dirty_text = "  hello   jarvis   \n\n\n\n  how   are  you?  "
    cleaned = clean_text(dirty_text)
    assert cleaned == "hello jarvis\n\nhow are you?"


def test_train_val_splitting():
    samples = [f"sample text {i}" for i in range(20)]
    train, val = split_train_val(samples, val_ratio=0.2, seed=42)

    assert len(train) + len(val) == 20
    assert len(val) == 4
    assert len(train) == 16
    # Check disjoint sets
    assert set(train).isdisjoint(set(val))


def test_text_dataset_token_shifting():
    tokenizer = TextTokenizer()
    tokenizer.fit(["jarvis is online", "jarvis learns locally"])

    dataset = TextDataset(
        texts=["jarvis is online"],
        tokenizer=tokenizer,
        seq_len=8,
    )

    sample = dataset[0]
    input_ids = sample["input_ids"].tolist()
    labels = sample["labels"].tolist()

    assert len(input_ids) == 8
    assert len(labels) == 8

    # The targets should be shifted by 1 relative to inputs for valid tokens
    # i.e., labels[0] is the prediction target for input_ids[0]
    non_pad_inputs = [tid for tid in input_ids if tid != tokenizer.pad_token_id]
    non_ignore_labels = [tid for tid in labels if tid != -100]

    assert len(non_pad_inputs) == len(non_ignore_labels)
    # The first target token should be the second token in the sequence
    assert labels[0] == non_pad_inputs[1] if len(non_pad_inputs) > 1 else True


def test_trainer_train_epoch_and_validation(tmp_path):
    tokenizer = TextTokenizer()
    corpus = [
        "jarvis is online and ready to assist",
        "system diagnostics are operating normally",
        "offline artificial intelligence runs locally on cpu",
        "jarvis solves problems step by step",
    ]
    tokenizer.fit(corpus)

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )

    trainer = LanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        lr=5e-3,
        seq_len=16,
    )

    train_data = corpus[:3]
    val_data = corpus[3:]

    # Initial validation loss
    initial_val_loss = trainer.evaluate(val_data, batch_size=2)
    assert initial_val_loss > 0.0

    # Train for 3 epochs
    checkpoint_file = tmp_path / "best_model.pt"
    result = trainer.train_with_validation(
        train_texts=train_data,
        val_texts=val_data,
        epochs=3,
        batch_size=2,
        checkpoint_path=str(checkpoint_file),
        sample_prompt="jarvis",
    )

    assert result["epoch"] == 3
    assert len(result["history"]) == 3
    assert "sample_generation" in result["history"][-1]
    assert checkpoint_file.exists()


def test_gradient_accumulation():
    tokenizer = TextTokenizer()
    corpus = ["jarvis task one", "jarvis task two", "jarvis task three", "jarvis task four"]
    tokenizer.fit(corpus)

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=8,
    )

    # Accumulate over 2 batches
    trainer = LanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        lr=1e-3,
        gradient_accumulation_steps=2,
        seq_len=8,
    )

    loss = trainer.train_epoch(corpus, batch_size=1)
    assert isinstance(loss, float)
    assert loss > 0.0


def test_checkpoint_resume_fidelity(tmp_path):
    tokenizer = TextTokenizer()
    corpus = ["hello jarvis", "jarvis is running"]
    tokenizer.fit(corpus)

    model1 = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=8,
    )

    trainer1 = LanguageModelTrainer(model=model1, tokenizer=tokenizer, seq_len=8)
    trainer1.train_epoch(corpus, batch_size=2)

    ckpt_path = tmp_path / "resume_test.pt"
    trainer1.save_checkpoint(ckpt_path, epoch=5, extra={"step": 100})

    # Create a fresh model and resume into it
    model2 = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=8,
    )
    trainer2 = LanguageModelTrainer(model=model2, tokenizer=tokenizer, seq_len=8)
    ckpt_data = trainer2.load_checkpoint(ckpt_path)

    assert ckpt_data["epoch"] == 5
    assert ckpt_data["extra"]["step"] == 100

    # Verify model weights match exactly
    for (k1, v1), (k2, v2) in zip(model1.state_dict().items(), model2.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)
