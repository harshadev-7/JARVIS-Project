"""J.A.R.V.I.S. Neural Sampling & Logit Modulation Engine (Phase 13).

Provides mathematically grounded token sampling strategies for local Transformers:
1. Temperature Scaling: Controls peakiness/entropy of probability distribution.
2. Repetition Penalty: Penalizes previously generated tokens to prevent degeneration loops.
3. Top-k Truncation: Restricts candidates to the top-k highest probability tokens.
4. Top-p (Nucleus) Sampling: Restricts candidates to the dynamic minimal nucleus
   whose cumulative probability meets or exceeds threshold p (Holtzman et al.).
5. Combined Sampler: Composes temperature, repetition penalty, top-k, and top-p.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set
import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Iterable[int],
    penalty: float = 1.1,
) -> torch.Tensor:
    """Apply multiplicative penalty to previously generated token logits.

    Args:
        logits: 1D tensor of raw unnormalized logits of shape (vocab_size,)
        generated_tokens: Iterable of token IDs already emitted.
        penalty: Penalty multiplier (1.0 = no penalty, >1.0 = penalize repetition).

    Returns:
        Modulated logits tensor.
    """
    if penalty == 1.0 or not generated_tokens:
        return logits

    modulated = logits.clone()
    seen_tokens: Set[int] = set(generated_tokens)

    for token_id in seen_tokens:
        if token_id < modulated.size(0):
            if modulated[token_id] > 0:
                modulated[token_id] /= penalty
            else:
                modulated[token_id] *= penalty

    return modulated


def apply_temperature(logits: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
    """Scale logits by temperature.

    Args:
        logits: Unnormalized logits tensor.
        temperature: Temperature factor (> 0.0). Lower = deterministic; Higher = diverse.
    """
    temp = max(temperature, 1e-5)
    return logits / temp


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits keeping only the top-k candidates, masking the rest with -inf."""
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits

    k_val = min(top_k, logits.size(-1))
    threshold, _ = torch.topk(logits, k_val)
    min_val = threshold[..., -1, None]
    return torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """Nucleus (Top-p) Sampling: keep dynamic subset of tokens with cumulative mass >= p.

    Args:
        logits: Logits tensor of shape (..., vocab_size).
        top_p: Cumulative probability threshold (0.0 < top_p <= 1.0).
    """
    if top_p >= 1.0 or top_p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices to include the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: Optional[int] = 20,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    history: Optional[Iterable[int]] = None,
) -> int:
    """Compose logit modulations and draw the next token ID from categorical distribution."""
    # Ensure logits are 1D vector (vocab_size,)
    if logits.dim() > 1:
        logits = logits.squeeze()

    # 1. Repetition Penalty
    if history:
        logits = apply_repetition_penalty(logits, history, penalty=repetition_penalty)

    # 2. Temperature scaling
    logits = apply_temperature(logits, temperature=temperature)

    # 3. Top-k filtering
    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k=top_k)

    # 4. Top-p (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        logits = apply_top_p(logits, top_p=top_p)

    # 5. Softmax & Categorical multinomial draw
    probs = F.softmax(logits, dim=-1)

    # Handle numerical edge cases (NaN / Inf protection)
    if torch.isnan(probs).any() or (probs.sum() == 0):
        probs = torch.ones_like(probs) / probs.size(-1)

    next_token_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_token_id


__all__ = [
    "apply_repetition_penalty",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "sample_next_token",
]
