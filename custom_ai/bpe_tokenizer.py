"""J.A.R.V.I.S. Byte-Pair Encoding (BPE) Subword Tokenizer.

Implements pure-Python subword tokenization from first principles:
- Learns merge rules iteratively from text corpus
- Preserves whole words when frequent, falls back to subwords or characters when rare
- Eliminates out-of-vocabulary (OOV) errors for unseen domain terminology
- Fully serializable to/from JSON dictionary
- Drop-in API compatible with TextTokenizer
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class BPETokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer for J.A.R.V.I.S. custom Transformer.

    Words are segmented into character sequences terminated by '</w>'.
    Frequent adjacent character pairs are iteratively merged into subwords.
    """

    END_OF_WORD = "</w>"

    def __init__(
        self,
        target_vocab_size: int = 1000,
        min_frequency: int = 2,
    ):
        self.target_vocab_size = target_vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initialize special token IDs."""
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.reverse_vocab = {idx: tok for idx, tok in enumerate(self.special_tokens)}
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _pre_tokenize(self, text: str) -> List[str]:
        """Normalize text and split into words and punctuation tokens."""
        normalized = text.lower().replace("\n", " ")
        # Separate punctuation with whitespace
        symbols = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", '"', "'", "-", "/", "+", "=", "*"]
        for sym in symbols:
            normalized = normalized.replace(sym, f" {sym} ")
        return [w for w in normalized.split() if w.strip()]

    def _get_pair_stats(self, vocab_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Compute frequency of adjacent symbol pairs across all words."""
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word_tuple, freq in vocab_freqs.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_pair_in_vocab(
        self,
        pair: Tuple[str, str],
        vocab_freqs: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """Replace all occurrences of pair in words with the merged subword."""
        new_vocab_freqs: Dict[Tuple[str, ...], int] = {}
        first, second = pair
        merged = first + second

        for word_tuple, freq in vocab_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab_freqs[tuple(new_word)] = freq

        return new_vocab_freqs

    def fit(self, texts: Iterable[str]) -> None:
        """Learn BPE merge rules and construct subword vocabulary from training texts."""
        self._init_special_tokens()
        self.merges = []
        self.bpe_ranks = {}

        # Step 1: Count word frequencies
        word_counts: Counter[str] = Counter()
        for text in texts:
            words = self._pre_tokenize(text)
            word_counts.update(words)

        # Step 2: Initialize words as character tuples + end-of-word symbol
        vocab_freqs: Dict[Tuple[str, ...], int] = {}
        initial_chars: Set[str] = set()

        for word, count in word_counts.items():
            char_tuple = tuple(list(word) + [self.END_OF_WORD])
            vocab_freqs[char_tuple] = count
            initial_chars.update(char_tuple)

        # Step 3: Add base characters to vocabulary
        sorted_chars = sorted(initial_chars)
        for char in sorted_chars:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.reverse_vocab[idx] = char

        # Step 4: Iteratively find and merge most frequent pairs
        num_merges = max(0, self.target_vocab_size - len(self.vocab))

        for merge_idx in range(num_merges):
            pairs = self._get_pair_stats(vocab_freqs)
            if not pairs:
                break

            # Find best pair by frequency
            best_pair, best_freq = max(pairs.items(), key=lambda item: item[1])
            if best_freq < self.min_frequency:
                break

            vocab_freqs = self._merge_pair_in_vocab(best_pair, vocab_freqs)
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_idx

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.reverse_vocab[idx] = merged_token

    def _tokenize_word(self, word: str) -> List[str]:
        """Segment a single word into subwords using learned BPE merges."""
        word_tuple = tuple(list(word) + [self.END_OF_WORD])

        if not self.bpe_ranks:
            return list(word_tuple)

        while len(word_tuple) > 1:
            # Find all adjacent pairs in current word
            pairs = [(word_tuple[i], word_tuple[i + 1]) for i in range(len(word_tuple) - 1)]
            # Find the pair with the lowest rank (earliest learned merge)
            candidate_pairs = [p for p in pairs if p in self.bpe_ranks]
            if not candidate_pairs:
                break

            best_pair = min(candidate_pairs, key=lambda p: self.bpe_ranks[p])
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            word_tuple = tuple(new_word)

        return list(word_tuple)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert a text string into a list of token integer IDs."""
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)

        words = self._pre_tokenize(text)
        for word in words:
            subwords = self._tokenize_word(word)
            for subword in subwords:
                token_id = self.vocab.get(subword, self.unk_token_id)
                tokens.append(token_id)

        if add_eos:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token integer IDs back into readable text."""
        subwords: List[str] = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                continue
            subword = self.reverse_vocab.get(tid_int, "<unk>")
            subwords.append(subword)

        # Concatenate and replace END_OF_WORD with spaces
        full_text = "".join(subwords).replace(self.END_OF_WORD, " ").strip()
        # Clean up spaces before punctuation
        full_text = re.sub(r'\s+([.,!?:;])', r'\1', full_text)
        return full_text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize BPE tokenizer state to dictionary."""
        return {
            "type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BPETokenizer:
        """Reconstruct BPETokenizer from serialized dictionary."""
        tokenizer = cls(
            target_vocab_size=data.get("target_vocab_size", 1000),
            min_frequency=data.get("min_frequency", 2),
        )
        tokenizer.special_tokens = data.get("special_tokens", ["<pad>", "<unk>", "<bos>", "<eos>"])
        tokenizer.vocab = data.get("vocab", {})
        tokenizer.reverse_vocab = {int(idx): tok for tok, idx in tokenizer.vocab.items()}
        raw_merges = data.get("merges", [])
        tokenizer.merges = [tuple(pair) for pair in raw_merges]
        tokenizer.bpe_ranks = {tuple(pair): idx for idx, pair in enumerate(tokenizer.merges)}

        tokenizer.pad_token_id = data.get("pad_token_id", 0)
        tokenizer.unk_token_id = data.get("unk_token_id", 1)
        tokenizer.bos_token_id = data.get("bos_token_id", 2)
        tokenizer.eos_token_id = data.get("eos_token_id", 3)
        return tokenizer


__all__ = ["BPETokenizer"]
