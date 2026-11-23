"""J.A.R.V.I.S. Attention Suite (Phase 25).

Implements modern attention paradigms from first principles:
1. Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) - Ainslie et al. (2023)
   - Groups query heads over shared key/value head projections
   - Seamlessly interpolates between MHA (num_kv_heads = num_heads) and MQA (num_kv_heads = 1)
   - Reduces memory bandwidth bottleneck during autoregressive token generation
2. Dynamic Key-Value Cache (KVCache)
   - Caches past Key/Value tensors across steps to achieve O(1) step complexity
3. Attention Weight Extraction & Entropy Analysis
   - Extracts layer-by-layer attention maps (B, H, T, T)
   - Computes attention entropy to quantify attention dispersion vs sharp focus
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modern_blocks import RotaryPositionalEmbedding


class KVCache:
    """Dynamic Key-Value cache for autoregressive decoding."""

    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append newly computed keys and values to the cache.

        Args:
            k_new: Tensor of shape (B, num_kv_heads, T_new, head_dim)
            v_new: Tensor of shape (B, num_kv_heads, T_new, head_dim)

        Returns:
            Tuple of concatenated full (keys, values) along sequence dimension.
        """
        if self.k_cache is None or self.v_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
        else:
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)

        return self.k_cache, self.v_cache

    @property
    def seq_len(self) -> int:
        return 0 if self.k_cache is None else self.k_cache.shape[2]

    def reset(self) -> None:
        self.k_cache = None
        self.v_cache = None


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with optional RoPE and KV-caching.

    Supports:
    - Multi-Head Attention (MHA): num_kv_heads == num_heads
    - Grouped-Query Attention (GQA): 1 < num_kv_heads < num_heads
    - Multi-Query Attention (MQA): num_kv_heads == 1
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads ({self.num_kv_heads})."
            )

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope = (
            RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
            if use_rope
            else None
        )

        # Storage for last computed attention weights (for analysis)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Forward pass for Grouped-Query Attention.

        Args:
            x: Input tensor of shape (B, T, embed_dim)
            causal_mask: Optional attention mask (B, 1, T, total_seq_len)
            kv_cache: Optional KVCache for autoregressive generation
            output_attentions: Whether to store attention weights in self.last_attn_weights

        Returns:
            Projected attention tensor of shape (B, T, embed_dim)
        """
        B, T, _ = x.shape
        offset = kv_cache.seq_len if kv_cache is not None else 0

        # 1. Linear projections
        # q: (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # k, v: (B, num_kv_heads, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Apply RoPE if enabled
        if self.rope is not None:
            q, k = self.rope(q, k, offset=offset)

        # 3. Update KV-cache if active
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Total key/value sequence length
        T_total = k.shape[2]

        # 4. Repeat KV heads to match query heads (Grouped expansion)
        if self.num_queries_per_kv > 1:
            # Expand (B, num_kv_heads, 1, T_total, head_dim) -> (B, num_kv_heads, num_queries_per_kv, T_total, head_dim)
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.num_queries_per_kv, T_total, self.head_dim)
            k = k.reshape(B, self.num_heads, T_total, self.head_dim)

            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.num_queries_per_kv, T_total, self.head_dim)
            v = v.reshape(B, self.num_heads, T_total, self.head_dim)

        # 5. Scaled Dot-Product Attention: Softmax((Q @ K.T) / sqrt(d_k) + Mask)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 6. Apply causal mask
        if causal_mask is not None:
            scores = scores + causal_mask
        elif T > 1 and kv_cache is None:
            # Construct standard causal lower-triangular mask
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask.unsqueeze(0).unsqueeze(1)
        elif kv_cache is not None and T == 1:
            # Autoregressive single step can attend to all past tokens in cache
            pass

        attn_weights = F.softmax(scores, dim=-1)
        if output_attentions:
            self.last_attn_weights = attn_weights.detach()

        attn_weights = self.dropout(attn_weights)

        # 7. Weighted sum over values: (B, num_heads, T, head_dim)
        context = torch.matmul(attn_weights, v)

        # 8. Concatenate heads and project output: (B, T, embed_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(context)


class AttentionAnalyzer:
    """Analytical tool for measuring attention patterns, entropy, and head dispersion."""

    @staticmethod
    def calculate_entropy(attn_weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Calculate Shannon entropy for each attention head.

        Args:
            attn_weights: Tensor of shape (B, num_heads, T, T_total)

        Returns:
            Entropy tensor of shape (B, num_heads, T), measuring dispersion.
            Lower entropy = sharp focus on specific tokens.
            Higher entropy = uniform / dispersed attention across context.
        """
        # H = - sum( p * log(p) )
        log_p = torch.log(attn_weights + eps)
        entropy = -torch.sum(attn_weights * log_p, dim=-1)
        return entropy

    @staticmethod
    def summarize_head_behavior(attn_weights: torch.Tensor) -> Dict[str, float]:
        """Produce summary metrics across all attention heads."""
        entropy = AttentionAnalyzer.calculate_entropy(attn_weights)
        mean_entropy = float(entropy.mean().item())
        max_entropy = float(entropy.max().item())
        min_entropy = float(entropy.min().item())

        # Check diagonal focus (self-attention onto current token)
        B, H, T, T_total = attn_weights.shape
        if T == T_total:
            diag_focus = float(torch.diagonal(attn_weights, dim1=-2, dim2=-1).mean().item())
        else:
            diag_focus = 0.0

        return {
            "mean_entropy": round(mean_entropy, 4),
            "min_entropy": round(min_entropy, 4),
            "max_entropy": round(max_entropy, 4),
            "diagonal_focus": round(diag_focus, 4),
        }
