"""J.A.R.V.I.S. Custom Decoder-Only Transformer Model.

A modular, lightweight GPT-style language model built from first principles:
- Token and learnable positional embeddings
- Stack of handcrafted TransformerDecoderBlocks
- Final Layer Normalization
- Language model output projection head (lm_head)
- Full parameter counting and resource inspection
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MaskedMultiHeadAttention
from .transformer import FeedForward, TransformerDecoderBlock


class DecoderOnlyTransformer(nn.Module):
    """Compact decoder-only Transformer language model for J.A.R.V.I.S.

    Designed from scratch for CPU execution on Windows laptops (Intel Core i3 / 8 GB RAM).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int | None = None,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.ff_dim = ff_dim if ff_dim is not None else 4 * embed_dim

        # 1. Embedding sublayers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 2. Stack of decoder blocks
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=self.ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final layer normalization
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

        # 4. Language modeling output head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Optional weight tying between token embedding and lm_head
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Weight initialization following standard GPT initialization scheme
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with standard normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, positional and token embedding parameters are excluded.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for next-token prediction.

        Args:
            input_ids: Tensor of token IDs with shape (batch_size, seq_len)
            causal_mask: Optional mask override

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )

        # Generate positional indices: [0, 1, ..., seq_len - 1]
        device = input_ids.device
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Token + positional embedding
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)

        # Forward pass through decoder layers
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)

        # Apply final LayerNorm and project to vocabulary logits
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


# Backward compatibility alias
DecoderLayer = TransformerDecoderBlock


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a lower-triangular causal mask where True denotes allowed positions."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


__all__ = [
    "DecoderLayer",
    "DecoderOnlyTransformer",
    "FeedForward",
    "MaskedMultiHeadAttention",
    "TransformerDecoderBlock",
    "build_causal_mask",
]
