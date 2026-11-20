"""J.A.R.V.I.S. Custom Masked Multi-Head Self-Attention.

Implemented from mathematical first principles using PyTorch tensor operations:
- Query, Key, Value linear projections (W_q, W_k, W_v)
- Multi-head partitioning: d_k = d_model / h
- Scaled Dot-Product Attention: Softmax( (Q K^T) / sqrt(d_k) + M ) V
- Strictly causal autoregressive triangular masking (tokens cannot see future tokens)
- Head concatenation and output projection (W_o)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiHeadAttention(nn.Module):
    """Handcrafted Masked Multi-Head Self-Attention for decoder-only Transformers.

    No high-level black-box wrappers (e.g. nn.MultiheadAttention) are used.
    The forward pass explicitly tracks:
      X (B, T, d_model) -> Q, K, V -> Multi-Head Split -> Scaled Dot-Product -> Masking -> Softmax -> Output Proj
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Individual linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection combining concatenated head outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for masked self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            causal_mask: Optional boolean or additive mask. If None, an autoregressive
                         lower-triangular mask is constructed dynamically.

        Returns:
            Projected attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Step 1: Project inputs to Query, Key, and Value
        # Shape: (B, T, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Split into multi-head format
        # Reshape to (B, T, num_heads, head_dim) then transpose to (B, num_heads, T, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Compute raw attention scores: S = (Q * K^T) / sqrt(d_k)
        # q: (B, h, T, d_k), k^T: (B, h, d_k, T) -> scores: (B, h, T, T)
        scaling_factor = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling_factor

        # Step 4: Apply Autoregressive Causal Masking
        # Elements where j > i (future tokens) receive -infinity so their softmax probability is 0
        if causal_mask is None:
            # Create lower-triangular boolean mask: True for allowed positions (j <= i)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )

        if causal_mask.dtype == torch.bool:
            # Mask out upper triangle: replace False with -1e9
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        else:
            # Additive mask
            scores = scores + causal_mask

        # Step 5: Softmax to obtain attention weights (probabilities summing to 1 across key dimension)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Context aggregation: Weighted sum of Value vectors
        # (B, h, T, T) x (B, h, T, d_k) -> (B, h, T, d_k)
        context = torch.matmul(attn_weights, v)

        # Step 7: Concatenate heads back to (B, T, embed_dim)
        # Transpose back to (B, T, h, d_k) and contiguous flatten to (B, T, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Step 8: Final linear projection
        out = self.out_proj(context)
        out = self.resid_dropout(out)
        return out


__all__ = ["MaskedMultiHeadAttention"]
