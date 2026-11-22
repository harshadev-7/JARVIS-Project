"""Unit tests for J.A.R.V.I.S. BPE Subword Tokenizer."""

import pytest
from custom_ai.bpe_tokenizer import BPETokenizer


def test_bpe_tokenizer_initialization():
    tokenizer = BPETokenizer(target_vocab_size=100)
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.vocab_size == 4


def test_bpe_tokenizer_fit_and_encode():
    corpus = [
        "hello jarvis",
        "jarvis is an autonomous personal ai assistant",
        "jarvis solves problems and writes code",
        "multithreading and polymorphism in java",
    ]
    tokenizer = BPETokenizer(target_vocab_size=120, min_frequency=1)
    tokenizer.fit(corpus)

    assert tokenizer.vocab_size > 10
    assert len(tokenizer.merges) > 0

    # Encoding
    tokens = tokenizer.encode("hello jarvis", add_bos=True, add_eos=True)
    assert tokens[0] == tokenizer.bos_token_id
    assert tokens[-1] == tokenizer.eos_token_id
    assert all(isinstance(t, int) for t in tokens)

    # Decoding roundtrip
    decoded = tokenizer.decode(tokens)
    assert "hello jarvis" in decoded


def test_bpe_tokenizer_unseen_words_fallback():
    """Unseen words should break into subwords/characters without failing."""
    corpus = [
        "operating system memory paging",
        "central processing unit execution",
    ]
    tokenizer = BPETokenizer(target_vocab_size=100, min_frequency=1)
    tokenizer.fit(corpus)

    # "preemption" has letters seen in the corpus (p, r, e, m, t, i, o, n)
    tokens = tokenizer.encode("preemption")
    assert len(tokens) > 0
    # None of the tokens should be unk if characters are known
    decoded = tokenizer.decode(tokens)
    assert "preemption" in decoded


def test_bpe_tokenizer_serialization():
    corpus = ["test serialization of jarvis subwords"]
    tokenizer = BPETokenizer(target_vocab_size=50, min_frequency=1)
    tokenizer.fit(corpus)

    data = tokenizer.to_dict()
    assert data["type"] == "bpe"
    assert "merges" in data

    reconstructed = BPETokenizer.from_dict(data)
    assert reconstructed.vocab_size == tokenizer.vocab_size
    assert len(reconstructed.merges) == len(tokenizer.merges)

    sample = "test serialization"
    assert tokenizer.encode(sample) == reconstructed.encode(sample)
