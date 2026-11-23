"""J.A.R.V.I.S. Modern Transformer Building Blocks (Phase 25).

Implemented from mathematical first principles using PyTorch tensor operations:
1. RMSNorm (Root Mean Square Layer Normalization) - Zhang & Sennrich (2019)
   - Normalizes by root-mean-square without mean-centering
   - Reduces compute time on CPU by 15-20% while maintaining training stability
2. SwiGLU (Swish Gated Linear Unit) - Shazeer (2020)
   - Replaces standard 2-layer FFN with gated activation: (Swish(xW_gate) * xW_up) W_down
   - Standard feed-forward architecture used in modern foundation models (LLaMA, Mistral)
3. RoPE (Rotary Position Embeddings) - Su et al. (2021)
   - Encodes absolute position with a rotation matrix that naturally incorporates relative distance
   - Invariant to sequence length expansion and avoids learnable positional parameters
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Formula:
        RMSNorm(x) = (x / RMS(x)) * gamma
        where RMS(x) = sqrt( mean(x^2) + eps )
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (SwiGLU) Feed-Forward Network.

    Formula:
        SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        # Standard LLaMA heuristic: hidden_dim = (4 * embed_dim * 2) // 3 rounded to multiple of 8
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * embed_dim) / 3)
            hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated computation: SiLU(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        out = self.w_down(hidden)
        return self.dropout(out)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for relative position attention.

    Applies a complex rotation to query and key vector pairs based on token index:
        x_rotated = (x * cos) + (rotate_half(x) * sin)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension ({dim}) must be even.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: theta_i = base ^ (-2(i-1)/dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cosine and sine tables up to max_seq_len
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=device or self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        # Duplicate freqs across real and imaginary parts: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dimensions of the input tensor."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors.

        Args:
            q: Query tensor of shape (B, num_heads, T, head_dim)
            k: Key tensor of shape (B, num_kv_heads, T, head_dim)
            offset: Starting sequence position (used for single-step KV caching)

        Returns:
            Tuple of rotated (q_rot, k_rot) with identical shapes.
        """
        seq_len = q.shape[2]
        end_pos = offset + seq_len

        # Expand cache if needed
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(max(end_pos, self.max_seq_len * 2), q.device)

        cos = self.cos_cached[offset:end_pos].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[offset:end_pos].to(dtype=q.dtype, device=q.device)

        # Reshape for broadcasting: (1, 1, T, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot
