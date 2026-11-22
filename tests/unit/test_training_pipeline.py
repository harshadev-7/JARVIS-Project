import torch
from pathlib import Path
from custom_ai.tokenizer import TextTokenizer
from custom_ai.training.dataset_loader import TextDataset
from custom_ai.training.trainer import Trainer
from torch.utils.data import DataLoader

def test_training_loop(tmp_path: Path):
    # tiny corpus
    corpus = "hello jarvis. this is a tiny training set for unit testing."
    # write to temp file
    data_file = tmp_path / "corpus.txt"
    data_file.write_text(corpus, encoding="utf-8")

    # tokenizer
    tokenizer = TextTokenizer(min_frequency=1)
    tokenizer.fit([corpus])

    # dataset & loader
    ds = TextDataset(str(data_file), tokenizer, seq_len=8)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    # model (small config)
    from custom_ai.model import DecoderOnlyTransformer
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        ff_dim=64,
        max_seq_len=8,
    )

    # trainer (1 epoch, no val)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=loader,
        max_epochs=1,
        grad_accum_steps=1,
        checkpoint_dir=tmp_path,
    )
    trainer.train()

    # checkpoint should exist
    ckpt_files = list(tmp_path.glob("jarvis_checkpoint_epoch0.pt"))
    assert ckpt_files, "Checkpoint was not saved"
