"""J.A.R.V.I.S. Custom Text Tokenizer.

Implements rule-based word and subword-ready tokenization:
- Vocabulary building with minimum frequency thresholds
- Special tokens: <pad> (0), <unk> (1), <bos> (2), <eos> (3)
- Encoding and decoding with round-trip fidelity
- Vocabulary serialization and deserialization (to_dict / from_dict)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


class TextTokenizer:
    """Lightweight, deterministic text tokenizer for local J.A.R.V.I.S. models."""

    def __init__(self, min_frequency: int = 1):
        self.min_frequency = min_frequency
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize_text(self, text: str) -> List[str]:
        """Normalize and split text into punctuation-aware tokens."""
        normalized = text.lower().replace("\n", " ")
        symbols = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", '"', "'", "-", "/"]
        for symbol in symbols:
            normalized = normalized.replace(symbol, f" {symbol} ")
        return [token for token in normalized.split() if token.strip()]

    def fit(self, texts: Iterable[str]) -> None:
        """Build vocabulary from a corpus of texts."""
        counts: Counter[str] = Counter()
        for text in texts:
            for token in self._tokenize_text(text):
                counts[token] += 1

        tokens = []
        for token, count in counts.items():
            if count >= self.min_frequency:
                tokens.append(token)

        # Sort tokens by frequency descending, then alphabetically for determinism
        tokens = sorted(set(tokens), key=lambda t: (-counts[t], t))

        # Assign special token IDs first
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert a text string into a list of token integer IDs."""
        tokens: List[str] = []
        if add_bos:
            tokens.append("<bos>")

        for token in self._tokenize_text(text):
            tokens.append(token if token in self.vocab else "<unk>")

        if add_eos:
            tokens.append("<eos>")

        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token integer IDs back into readable text."""
        tokens = [self.reverse_vocab.get(int(tid), "<unk>") for tid in token_ids]
        filtered = [t for t in tokens if t not in {"<pad>", "<bos>", "<eos>"}]
        if not filtered:
            return ""
        return " ".join(filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tokenizer state to dictionary."""
        return {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "min_frequency": self.min_frequency,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TextTokenizer:
        """Reconstruct tokenizer from dictionary."""
        tokenizer = cls(min_frequency=data.get("min_frequency", 1))
        tokenizer.special_tokens = data.get("special_tokens", ["<pad>", "<unk>", "<bos>", "<eos>"])
        tokenizer.vocab = data.get("vocab", {})
        tokenizer.reverse_vocab = {int(idx): token for token, idx in tokenizer.vocab.items()}
        tokenizer.pad_token_id = data.get("pad_token_id", 0)
        tokenizer.unk_token_id = data.get("unk_token_id", 1)
        tokenizer.bos_token_id = data.get("bos_token_id", 2)
        tokenizer.eos_token_id = data.get("eos_token_id", 3)
        return tokenizer


__all__ = ["TextTokenizer"]
