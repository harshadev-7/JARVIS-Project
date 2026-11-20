"""J.A.R.V.I.S. Transformer Decoder Block & Feed-Forward Sublayers.

Implements the core decoder layer for autoregressive language modeling:
- Pre-Layer Normalization (Pre-LN) for gradient stability
- Position-wise Feed-Forward Network (MLP) with GELU non-linearity and 4x expansion
- Residual connections (skip connections) around attention and MLP sub-layers
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MaskedMultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (MLP).

    Applies two linear transformations with a GELU non-linearity in between:
      FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    where W_1 expands dimensionality by 4x, and W_2 projects back to embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        # Standard GPT ratio is 4 * embed_dim
        hidden_dim = ff_dim if ff_dim is not None else 4 * embed_dim

        self.linear1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, embed_dim)

        Returns:
            Output tensor of shape (B, T, embed_dim)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerDecoderBlock(nn.Module):
    """Single Transformer Decoder Block with Pre-Layer Normalization.

    Architecture:
      1. Residual Sub-layer 1:
         h' = x + Dropout( MaskedMultiHeadAttention( LayerNorm_1(x) ) )
      2. Residual Sub-layer 2:
         out = h' + Dropout( FeedForward( LayerNorm_2(h') ) )

    Pre-LN ensures that gradients pass directly through the identity residual
    connection without undergoing unnormalized transformations, preventing
    vanishing gradients during CPU training.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.self_attn = MaskedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            causal_mask: Optional causal attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Sub-layer 1: Masked Multi-Head Attention with Pre-LN & Residual
        normed_x1 = self.norm1(x)
        attn_out = self.self_attn(normed_x1, causal_mask=causal_mask)
        x = x + attn_out

        # Sub-layer 2: Feed-Forward MLP with Pre-LN & Residual
        normed_x2 = self.norm2(x)
        ffn_out = self.ffn(normed_x2)
        x = x + ffn_out

        return x


__all__ = ["FeedForward", "TransformerDecoderBlock"]
