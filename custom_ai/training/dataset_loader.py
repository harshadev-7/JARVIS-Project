# -*- coding: utf-8 -*-
"""
Dataset loader for the custom JARVIS transformer.
Provides a lightweight TextDataset class that reads a plain‑text file,
cleans the data, tokenises it using the project's TextTokenizer, and
yields fixed‑length sequences suitable for next‑token prediction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset

from ..tokenizer import TextTokenizer


class TextDataset(Dataset):
    """In‑memory dataset for language modeling.

    Args:
        file_path: Path to a UTF‑8 encoded text file (or a directory of files).
        tokenizer: An instantiated ``TextTokenizer`` already fitted to the
            training corpus.
        seq_len: The length of input sequences (including the target token).
        add_bos: Whether to prepend ``<bos>`` to each sequence.
        add_eos: Whether to append ``<eos>`` to each sequence.
    """

    def __init__(
        self,
        file_path: str | os.PathLike,
        tokenizer: TextTokenizer,
        seq_len: int = 32,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Load raw text (support a single file or a directory of files)
        path = Path(file_path)
        if path.is_dir():
            texts = []
            for txt_file in path.rglob("*.txt"):
                texts.append(txt_file.read_text(encoding="utf-8"))
            raw = "\n".join(texts)
        else:
            raw = path.read_text(encoding="utf-8")

        # Basic cleaning – collapse whitespace, strip leading/trailing spaces
        cleaned = " ".join(raw.split())
        # Tokenise the whole corpus once
        self.tokens: List[int] = self.tokenizer.encode(
            cleaned, add_bos=self.add_bos, add_eos=self.add_eos
        )

        # Ensure we have at least one full block
        if len(self.tokens) < self.seq_len + 1:
            raise ValueError(
                f"Corpus too small for seq_len={self.seq_len}. "
                f"Got {len(self.tokens)} tokens."
            )

    def __len__(self) -> int:
        # Number of sliding windows we can take
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int) -> dict:
        # Input = tokens[idx : idx + seq_len]
        # Target = tokens[idx + 1 : idx + seq_len + 1]
        inp = torch.tensor(self.tokens[idx : idx + self.seq_len], dtype=torch.long)
        tgt = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return {"input_ids": inp, "labels": tgt}

__all__ = ["TextDataset"]
