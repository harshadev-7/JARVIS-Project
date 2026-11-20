import torch

from custom_ai.dataset import TinyTextDataset
from custom_ai.inference import LocalModelInference
from custom_ai.model import DecoderOnlyTransformer
from custom_ai.tokenizer import TextTokenizer


def test_tokenizer_round_trip_and_vocab_limits():
    tokenizer = TextTokenizer(min_frequency=1)
    text = "hello jarvis hello world"
    tokenizer.fit([text])

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert tokenizer.vocab_size >= 5
    assert decoded.strip().lower() == text.lower()


def test_transformer_forward_shapes():
    tokenizer = TextTokenizer(min_frequency=1)
    tokenizer.fit(["hello jarvis hello world", "jarvis is ready"])

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        max_seq_len=32,
    )

    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    logits = model(tokens)

    assert logits.shape == (1, 5, tokenizer.vocab_size)


def test_tiny_dataset_creates_sequence_batches():
    dataset = TinyTextDataset(["hello jarvis", "jarvis hello world"], seq_len=4)
    sample = dataset[0]

    assert isinstance(sample, dict)
    assert "input_ids" in sample
    assert "labels" in sample
    assert len(sample["input_ids"]) == 4
    assert len(sample["labels"]) == 4


def test_local_inference_generates_text():
    tokenizer = TextTokenizer(min_frequency=1)
    tokenizer.fit(["hello jarvis", "jarvis is ready", "hello world"])

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        max_seq_len=32,
    )
    inference = LocalModelInference(model=model, tokenizer=tokenizer)

    result = inference.generate("jarvis", max_new_tokens=6, temperature=0.8)

    assert isinstance(result, str)
    assert len(result) > 0


def test_training_pipeline_runs_and_checkpoint_round_trip(tmp_path):
    tokenizer = TextTokenizer(min_frequency=1)
    tokenizer.fit([
        "jarvis is ready to help",
        "jarvis learns from local data",
        "hello from jarvis",
    ])

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        max_seq_len=32,
    )

    trainer = __import__('custom_ai.trainer', fromlist=['TinyLanguageModelTrainer']).TinyLanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        seq_len=8,
        lr=1e-3,
    )

    loss = trainer.train_epoch([
        "jarvis is ready to help",
        "jarvis learns from local data",
        "hello from jarvis",
    ], batch_size=2)

    assert isinstance(loss, float)
    assert loss >= 0.0

    checkpoint_path = tmp_path / "jarvis_phase2.pt"
    trainer.save_checkpoint(str(checkpoint_path), epoch=1)
    assert checkpoint_path.exists()

    restored = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        max_seq_len=32,
    )
    trainer.load_checkpoint(str(checkpoint_path), restored)

    assert restored.vocab_size == tokenizer.vocab_size


def test_local_model_registry_tracks_status_and_training_state(tmp_path):
    tokenizer = TextTokenizer(min_frequency=1)
    tokenizer.fit([
        "jarvis is ready",
        "jarvis helps locally",
        "local model training",
    ])

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        max_seq_len=32,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    from custom_ai.model_registry import LocalModelRegistry

    registry = LocalModelRegistry(checkpoint_dir=str(checkpoint_dir))
    registry.register_model("jarvis-local", model=model, tokenizer=tokenizer)
    registry.save_current_model("jarvis-local")

    status = registry.status()
    assert "jarvis-local" in status
    assert status["jarvis-local"]["status"] in {"ready", "trained"}

    generated = registry.generate("jarvis", model_name="jarvis-local", max_new_tokens=5)
    assert isinstance(generated, str)
    assert len(generated) > 0


def test_training_loop_tracks_validation_loss_and_resume():
    tokenizer = TextTokenizer(min_frequency=1)
    tokenizer.fit([
        "jarvis is ready to help",
        "jarvis learns from local data",
        "jarvis writes code locally",
        "local model learns quietly",
        "jarvis never stops improving",
    ])

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        max_seq_len=32,
    )

    trainer = __import__('custom_ai.trainer', fromlist=['TinyLanguageModelTrainer']).TinyLanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        seq_len=8,
        lr=1e-3,
    )

    train_texts = [
        "jarvis is ready to help",
        "jarvis learns from local data",
        "jarvis writes code locally",
    ]
    val_texts = [
        "jarvis never stops improving",
        "local model learns quietly",
    ]

    result = trainer.train_with_validation(train_texts, val_texts, epochs=2, batch_size=2)

    assert "train_loss" in result
    assert "val_loss" in result
    assert isinstance(result["train_loss"], float)
    assert isinstance(result["val_loss"], float)
    assert result["val_loss"] >= 0.0

    checkpoint_path = "jarvis_phase4_resume.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=2)
    resumed = trainer.load_checkpoint(checkpoint_path)
    assert "epoch" in resumed
    assert resumed["epoch"] == 2
