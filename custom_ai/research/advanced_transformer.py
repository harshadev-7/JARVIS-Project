"""J.A.R.V.I.S. Advanced Transformer Engine (Phase 25).

A modular, state-of-the-art decoder-only language model:
- Configurable attention: Multi-Head (MHA), Grouped-Query (GQA), Multi-Query (MQA)
- Configurable normalization: RMSNorm vs LayerNorm
- Configurable feed-forward: SwiGLU vs standard GELU/ReLU
- Configurable positional encoding: RoPE vs Learned Positional Embeddings
- Native KV-caching for O(1) single-step autoregressive token generation
- Weight tying support (token_embedding <-> lm_head)
- Parameter inspection and FLOPs estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_variants import GroupedQueryAttention, KVCache
from .modern_blocks import RMSNorm, RotaryPositionalEmbedding, SwiGLU


@dataclass
class AdvancedModelConfig:
    """Hyperparameter and architectural specification for AdvancedTransformer."""
    vocab_size: int
    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: Optional[int] = 2  # Default GQA: 2 KV heads for 4 query heads
    ff_dim: Optional[int] = None
    max_seq_len: int = 256
    dropout: float = 0.0
    norm_type: str = "rms_norm"       # "rms_norm" or "layer_norm"
    activation_type: str = "swiglu"   # "swiglu", "gelu", or "relu"
    pos_encoding: str = "rope"        # "rope", "learned", or "sinusoidal"
    tie_weights: bool = True
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "norm_type": self.norm_type,
            "activation_type": self.activation_type,
            "pos_encoding": self.pos_encoding,
            "tie_weights": self.tie_weights,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdvancedModelConfig:
        return cls(**data)


class StandardFFN(nn.Module):
    """Classic 2-layer Feed-Forward Network with GELU or ReLU."""

    def __init__(self, embed_dim: int, ff_dim: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act_fn(self.fc1(x))))


class AdvancedDecoderBlock(nn.Module):
    """Single Transformer Decoder block with Pre-LayerNorm topology."""

    def __init__(self, cfg: AdvancedModelConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Normalization 1 (Pre-Attention)
        if cfg.norm_type == "rms_norm":
            self.norm1 = RMSNorm(cfg.embed_dim, eps=cfg.eps)
        else:
            self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)

        # 2. Attention Sublayer
        use_rope = (cfg.pos_encoding == "rope")
        self.attn = GroupedQueryAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            dropout=cfg.dropout,
            use_rope=use_rope,
            max_seq_len=cfg.max_seq_len,
        )

        # 3. Normalization 2 (Pre-FFN)
        if cfg.norm_type == "rms_norm":
            self.norm2 = RMSNorm(cfg.embed_dim, eps=cfg.eps)
        else:
            self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)

        # 4. Feed-Forward Sublayer
        if cfg.activation_type == "swiglu":
            self.ffn = SwiGLU(
                embed_dim=cfg.embed_dim,
                hidden_dim=cfg.ff_dim,
                dropout=cfg.dropout,
            )
        else:
            ff_dim = cfg.ff_dim or (4 * cfg.embed_dim)
            self.ffn = StandardFFN(
                embed_dim=cfg.embed_dim,
                ff_dim=ff_dim,
                activation=cfg.activation_type,
                dropout=cfg.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Pre-LN attention with residual
        normed1 = self.norm1(x)
        attn_out = self.attn(
            normed1,
            causal_mask=causal_mask,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
        )
        x = x + attn_out

        # Pre-LN FFN with residual
        normed2 = self.norm2(x)
        ffn_out = self.ffn(normed2)
        x = x + ffn_out
        return x


class AdvancedTransformer(nn.Module):
    """Complete Advanced Decoder-Only Transformer Language Model."""

    def __init__(self, cfg: AdvancedModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.embed_dim = cfg.embed_dim
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len

        # 1. Token Embeddings
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)

        # 2. Positional Embeddings (if not using RoPE)
        if cfg.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        elif cfg.pos_encoding == "sinusoidal":
            self.register_buffer("position_embedding", self._build_sinusoidal(cfg.max_seq_len, cfg.embed_dim), persistent=False)
        else:
            # RoPE handles positions inside each attention head
            self.position_embedding = None

        self.emb_dropout = nn.Dropout(cfg.dropout)

        # 3. Stack of Decoder Blocks
        self.layers = nn.ModuleList([AdvancedDecoderBlock(cfg) for _ in range(cfg.num_layers)])

        # 4. Final Normalization
        if cfg.norm_type == "rms_norm":
            self.final_norm = RMSNorm(cfg.embed_dim, eps=cfg.eps)
        else:
            self.final_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)

        # 5. Output Projection Head
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # 6. Optional Weight Tying
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small standard deviation for training stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def _build_sinusoidal(self, max_seq_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_seq_len, dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            input_ids: Long tensor of token IDs (B, T)
            causal_mask: Optional attention mask
            kv_caches: List of KVCache objects (one per decoder layer) for generation
            output_attentions: Whether to capture attention weight maps

        Returns:
            Logits tensor of shape (B, T, vocab_size)
        """
        B, T = input_ids.shape
        offset = kv_caches[0].seq_len if kv_caches is not None and len(kv_caches) > 0 else 0

        # Token embedding
        x = self.token_embedding(input_ids)

        # Add learned or sinusoidal position embedding if not RoPE
        if self.position_embedding is not None:
            if isinstance(self.position_embedding, nn.Embedding):
                positions = torch.arange(offset, offset + T, dtype=torch.long, device=input_ids.device)
                x = x + self.position_embedding(positions)
            else:
                x = x + self.position_embedding[:, offset : offset + T, :]

        x = self.emb_dropout(x)

        # Pass through decoder blocks
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x = layer(
                x,
                causal_mask=causal_mask,
                kv_cache=cache_i,
                output_attentions=output_attentions,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def create_kv_caches(self) -> List[KVCache]:
        """Create a fresh list of KVCache buffers for each layer."""
        return [KVCache() for _ in range(self.num_layers)]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with optional KV-caching.

        Args:
            input_ids: Initial prompt tensor (B, T)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling randomness
            top_k: Top-k filtering threshold
            use_cache: Whether to use KV caching for O(1) step latency
            eos_token_id: Stop generation when encountered

        Returns:
            Generated sequence tensor of shape (B, T + generated_tokens)
        """
        self.eval()
        curr_ids = input_ids.clone()
        kv_caches = self.create_kv_caches() if use_cache else None

        for step in range(max_new_tokens):
            if use_cache and step > 0:
                # In cached mode, forward only the single latest token
                input_step = curr_ids[:, -1:]
            else:
                # In first step (or uncached mode), forward full prefix
                input_step = curr_ids if not use_cache else curr_ids

            logits = self.forward(input_step, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                min_value = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return curr_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculate total number of model parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def get_model_summary(self) -> Dict[str, Any]:
        """Produce technical architecture profile for telemetry."""
        return {
            "model_type": "AdvancedDecoderOnlyTransformer",
            "vocab_size": self.cfg.vocab_size,
            "embed_dim": self.cfg.embed_dim,
            "num_layers": self.cfg.num_layers,
            "num_heads": self.cfg.num_heads,
            "num_kv_heads": self.cfg.num_kv_heads,
            "attention_type": "GQA" if self.cfg.num_kv_heads not in (1, self.cfg.num_heads) else ("MQA" if self.cfg.num_kv_heads == 1 else "MHA"),
            "norm_type": self.cfg.norm_type,
            "activation_type": self.cfg.activation_type,
            "pos_encoding": self.cfg.pos_encoding,
            "tie_weights": self.cfg.tie_weights,
            "total_parameters": self.get_num_params(),
            "non_embedding_parameters": self.get_num_params(non_embedding=True),
        }
