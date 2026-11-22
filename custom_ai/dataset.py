"""J.A.R.V.I.S. Dataset Pipeline for Autoregressive Language Modeling.

Provides:
- Text cleaning and normalization
- File loading and dataset parsing
- Train / Validation deterministic splitting
- Tokenization with <bos> and <eos> wrapping
- Sliding-window sequence chunking
- Next-token prediction target alignment:
    inputs = [x_0, x_1, ..., x_{T-1}]
    labels = [x_1, x_2, ..., x_T]
- Padding and ignore_index support for standard PyTorch loss functions
"""

from __future__ import annotations

from pathlib import Path
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-printable artifacts."""
    if not text:
        return ""
    # Strip whitespace line by line and collapse internal spaces
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines()]
    text = "\n".join(lines)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_corpus_file(file_path: str | Path, delimiter: str = "\n\n") -> List[str]:
    """Load and parse a text dataset file into sample blocks."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    content = path.read_text(encoding="utf-8")
    blocks = content.split(delimiter)
    cleaned = [clean_text(b) for b in blocks if clean_text(b)]
    return cleaned


def split_train_val(
    samples: List[str],
    val_ratio: float = 0.15,
    train_ratio: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split samples into train and validation sets.

    Either `val_ratio` (fraction of data for validation) or `train_ratio`
    (fraction for training) can be supplied. If `train_ratio` is provided it
    overrides `val_ratio` via `val_ratio = 1 - train_ratio`.
    """
    if train_ratio is not None:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
        val_ratio = 1 - train_ratio
    if not samples:
        return [], []

    shuffled = samples[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]

    # Fallback if corpus is tiny (ensure train set has at least 1 sample)
    if not train_samples:
        train_samples = val_samples

    return train_samples, val_samples


class TextDataset(Dataset):
    """PyTorch Dataset for next-token prediction language modeling."""

    def __init__(
        self,
        texts: Iterable[str],
        seq_len: int = 32,
        tokenizer=None,
        pad_token_id: int = 0,
        stride: int | None = None,
        ignore_index: int = -100,
    ):
        texts_list = list(texts)

        # Fallback if tokenizer is not passed: instantiate and fit a TextTokenizer
        if tokenizer is None:
            from .tokenizer import TextTokenizer
            self.tokenizer = TextTokenizer()
            self.tokenizer.fit(texts_list)
        else:
            self.tokenizer = tokenizer

        self.seq_len = seq_len
        self.stride = stride or (seq_len // 2 if seq_len > 4 else seq_len)
        self.ignore_index = ignore_index
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", pad_token_id)
        self.samples: List[Dict[str, torch.Tensor]] = []

        self._build_samples(texts_list)

    def _build_samples(self, texts: Iterable[str]) -> None:
        """Tokenize text and construct input/label pairs with shifted targets."""
        for text in texts:
            cleaned = clean_text(text)
            if not cleaned:
                continue

            # Encode text with beginning and end of sequence tokens
            tokens = self.tokenizer.encode(cleaned, add_bos=True, add_eos=True)
            if len(tokens) < 2:
                continue

            # Need at least seq_len + 1 tokens to form an input and shifted target
            if len(tokens) <= self.seq_len:
                input_chunk = tokens[:-1]
                target_chunk = tokens[1:]

                # Pad to seq_len
                pad_len = self.seq_len - len(input_chunk)
                padded_inputs = input_chunk + [self.pad_token_id] * pad_len
                padded_targets = target_chunk + [self.ignore_index] * pad_len

                self.samples.append(
                    {
                        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
                        "labels": torch.tensor(padded_targets, dtype=torch.long),
                    }
                )
            else:
                # Sliding window chunking
                for start in range(0, len(tokens) - self.seq_len, self.stride):
                    chunk = tokens[start : start + self.seq_len + 1]
                    input_chunk = chunk[:-1]
                    target_chunk = chunk[1:]

                    self.samples.append(
                        {
                            "input_ids": torch.tensor(input_chunk, dtype=torch.long),
                            "labels": torch.tensor(target_chunk, dtype=torch.long),
                        }
                    )

        # Fallback empty sample if nothing qualified
        if not self.samples:
            self.samples.append(
                {
                    "input_ids": torch.tensor([self.pad_token_id] * self.seq_len, dtype=torch.long),
                    "labels": torch.tensor([self.ignore_index] * self.seq_len, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# Backward compatibility alias
TinyTextDataset = TextDataset

__all__ = [
    "TextDataset",
    "TinyTextDataset",
    "clean_text",
    "load_corpus_file",
    "split_train_val",
]
