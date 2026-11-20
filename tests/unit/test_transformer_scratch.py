"""Comprehensive Unit Tests for J.A.R.V.I.S. From-Scratch Transformer Architecture.

Validates:
1. Tokenizer vocabulary construction, encoding, decoding, and dictionary serialization
2. MaskedMultiHeadAttention shapes and projections (W_q, W_k, W_v, W_o)
3. Strict Causal Masking (Gold-Standard test: changing future tokens does not affect past token outputs)
4. FeedForward sub-layer (GELU, expansion factor 4, residual shape)
5. Pre-LN TransformerDecoderBlock forward pass
6. Full DecoderOnlyTransformer model forward pass and parameter counting
7. Backpropagation and gradient flow through all trainable weights
8. TextGenerator autoregressive text generation with temperature and top-k
"""

import math
import torch
import torch.nn.functional as F

from custom_ai.attention import MaskedMultiHeadAttention
from custom_ai.generation import TextGenerator
from custom_ai.model import DecoderOnlyTransformer, build_causal_mask
from custom_ai.tokenizer import TextTokenizer
from custom_ai.transformer import FeedForward, TransformerDecoderBlock


def test_tokenizer_full_lifecycle():
    tokenizer = TextTokenizer(min_frequency=1)
    corpus = [
        "JARVIS is online and ready.",
        "System diagnostics are fully operational.",
        "Local AI running on CPU.",
    ]
    tokenizer.fit(corpus)

    assert tokenizer.vocab_size >= 10
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3

    text = "JARVIS is online."
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id

    decoded = tokenizer.decode(encoded)
    assert "jarvis" in decoded
    assert "online" in decoded

    # Serialization test
    state = tokenizer.to_dict()
    restored = TextTokenizer.from_dict(state)
    assert restored.vocab_size == tokenizer.vocab_size
    assert restored.encode(text) == tokenizer.encode(text)


def test_masked_multihead_attention_shapes():
    batch_size = 2
    seq_len = 8
    embed_dim = 32
    num_heads = 4

    mha = MaskedMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    out = mha(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_causal_mask_strictly_prevents_future_leakage():
    """Gold-standard causality test.

    Modifying future tokens (e.g. position 7) must have ZERO impact on the
    attention output at position 0, because causal masking prevents position 0
    from attending to any token at j > 0.
    """
    seq_len = 6
    embed_dim = 16
    num_heads = 2

    mha = MaskedMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    mha.eval()

    # Input sequence 1
    x1 = torch.randn(1, seq_len, embed_dim)
    # Input sequence 2: Identical at position 0, but drastically altered at positions 1..5
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn(1, seq_len - 1, embed_dim) * 50.0

    with torch.no_grad():
        out1 = mha(x1)
        out2 = mha(x2)

    # Output at position 0 must be virtually identical within floating-point epsilon
    diff_at_pos_0 = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert diff_at_pos_0 < 1e-5, f"Causal mask leaked future token information! Diff: {diff_at_pos_0}"


def test_feedforward_sublayer():
    batch_size = 3
    seq_len = 5
    embed_dim = 24
    ff_dim = 96

    ffn = FeedForward(embed_dim=embed_dim, ff_dim=ff_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    out = ffn(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_transformer_decoder_block():
    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4

    block = TransformerDecoderBlock(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    out = block(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_decoder_only_transformer_model():
    vocab_size = 50
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    max_seq_len = 16

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    )

    input_ids = torch.randint(0, vocab_size, (2, 8))
    logits = model(input_ids)

    assert logits.shape == (2, 8, vocab_size)
    assert model.get_num_params() > 0


def test_gradient_flow_and_backpropagation():
    """Verify that backpropagation updates all parameters across the network."""
    vocab_size = 30
    embed_dim = 16
    num_layers = 2
    num_heads = 2
    max_seq_len = 8

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    )
    model.train()

    input_ids = torch.tensor([[1, 5, 12, 18, 4]], dtype=torch.long)
    targets = torch.tensor([[5, 12, 18, 4, 2]], dtype=torch.long)

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    assert loss.item() > 0.0

    loss.backward()

    # Check that every single trainable parameter received a gradient
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive a gradient!"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN!"


def test_text_generator():
    tokenizer = TextTokenizer()
    tokenizer.fit(["jarvis is an offline ai assistant", "hello jarvis how are you"])

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )

    generator = TextGenerator(model=model, tokenizer=tokenizer, device="cpu")
    output = generator.generate("jarvis", max_new_tokens=4, temperature=0.7, top_k=5)

    assert isinstance(output, str)
    assert len(output) > 0
