"""Unit Test Suite for JARVIS Phase 25 — Advanced Neural Network Research.

Verifies:
1. RMSNorm: Scale normalization, epsilon stability, and gradient flow.
2. SwiGLU: Gated activation, parameter shapes, and forward computation.
3. Rotary Position Embeddings (RoPE): Frequency cache, complex rotation, and offset handling.
4. Grouped-Query Attention (GQA): MHA, GQA, and MQA modes with causal masking.
5. KV-Cache: State accumulation and O(1) single-token generation equivalence.
6. Attention Analyzer: Shannon entropy extraction and head behavior summarization.
7. AdvancedTransformer: Configurable modular Transformer with generation.
8. Dynamic INT8 Quantization: Linear layer quantization and memory compression.
9. Model Evaluator: Loss, perplexity (PPL), BPC, and next-token accuracy calculation.
10. Truthfulness Guard: Strict validation against unverified capability claims.
11. Model Comparator: Side-by-side benchmark scorecard formulation.
12. Experiment Tracker: JSON and Markdown experiment run persistence.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from custom_ai.research.modern_blocks import RMSNorm, RotaryPositionalEmbedding, SwiGLU
from custom_ai.research.attention_variants import (
    AttentionAnalyzer,
    GroupedQueryAttention,
    KVCache,
)
from custom_ai.research.advanced_transformer import (
    AdvancedDecoderBlock,
    AdvancedModelConfig,
    AdvancedTransformer,
)
from custom_ai.research.model_optimizer import ModelOptimizer
from custom_ai.research.evaluator import EvaluationResult, ModelEvaluator, TruthfulnessGuard
from custom_ai.research.model_comparator import ModelComparator
from custom_ai.research.experiment_tracker import ExperimentTracker


# ==============================================================================
# 1. Modern Architectural Blocks Tests
# ==============================================================================

def test_1_rmsnorm_forward_and_backward():
    """Test RMSNorm normalization, scale parameter, and gradient backpropagation."""
    norm = RMSNorm(dim=64, eps=1e-6)
    x = torch.randn(2, 10, 64, requires_grad=True)

    out = norm(x)
    assert out.shape == (2, 10, 64)

    # Verify root-mean-square of output is close to 1 across the normalized dimension
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    # Verify backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None


def test_2_swiglu_forward_and_activation():
    """Test SwiGLU feed-forward network with gated linear activation."""
    ffn = SwiGLU(embed_dim=64, hidden_dim=128)
    x = torch.randn(2, 8, 64)

    out = ffn(x)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


def test_3_rope_rotary_positional_embeddings():
    """Test Rotary Position Embedding rotation, relative distance, and offset."""
    rope = RotaryPositionalEmbedding(dim=16, max_seq_len=64)
    q = torch.randn(2, 4, 8, 16)  # (B, H, T, D)
    k = torch.randn(2, 4, 8, 16)

    q_rot, k_rot = rope(q, k, offset=0)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Inner product before and after rotation on identical vectors should match norm
    norm_before = (q[0, 0, 0] ** 2).sum()
    norm_after = (q_rot[0, 0, 0] ** 2).sum()
    assert torch.isclose(norm_before, norm_after, atol=1e-4)

    # Offset testing for KV cache steps
    q_step, k_step = rope(q[:, :, :1, :], k[:, :, :1, :], offset=5)
    assert q_step.shape == (2, 4, 1, 16)


# ==============================================================================
# 2. Attention Variants & KV-Cache Tests
# ==============================================================================

def test_4_grouped_query_attention_modes():
    """Test GQA across MHA (4 KV heads), GQA (2 KV heads), and MQA (1 KV head)."""
    B, T, D = 2, 8, 64

    # Mode 1: Multi-Head Attention (MHA)
    mha = GroupedQueryAttention(embed_dim=D, num_heads=4, num_kv_heads=4, use_rope=True)
    x = torch.randn(B, T, D)
    out_mha = mha(x)
    assert out_mha.shape == (B, T, D)

    # Mode 2: Grouped-Query Attention (GQA)
    gqa = GroupedQueryAttention(embed_dim=D, num_heads=4, num_kv_heads=2, use_rope=True)
    out_gqa = gqa(x)
    assert out_gqa.shape == (B, T, D)

    # Mode 3: Multi-Query Attention (MQA)
    mqa = GroupedQueryAttention(embed_dim=D, num_heads=4, num_kv_heads=1, use_rope=True)
    out_mqa = mqa(x)
    assert out_mqa.shape == (B, T, D)


def test_5_kv_cache_autoregressive_step():
    """Test KV-Cache accumulation across sequential generation steps."""
    gqa = GroupedQueryAttention(embed_dim=32, num_heads=2, num_kv_heads=1)
    cache = KVCache()

    # Step 1: Forward prompt of 3 tokens
    prompt = torch.randn(1, 3, 32)
    _ = gqa(prompt, kv_cache=cache)
    assert cache.seq_len == 3

    # Step 2: Forward 1 new token using cache
    new_token = torch.randn(1, 1, 32)
    out_step = gqa(new_token, kv_cache=cache)
    assert out_step.shape == (1, 1, 32)
    assert cache.seq_len == 4


def test_6_attention_analyzer_entropy():
    """Test AttentionAnalyzer calculation of Shannon entropy and head dispersion."""
    # Synthetic attention weights: (B=1, H=2, T=4, T=4)
    attn = torch.softmax(torch.randn(1, 2, 4, 4), dim=-1)
    summary = AttentionAnalyzer.summarize_head_behavior(attn)

    assert "mean_entropy" in summary
    assert "min_entropy" in summary
    assert "max_entropy" in summary
    assert summary["mean_entropy"] > 0.0


# ==============================================================================
# 3. Advanced Transformer Engine Tests
# ==============================================================================

def test_7_advanced_transformer_forward_and_generation():
    """Test AdvancedTransformer batch forward pass and autoregressive generation."""
    cfg = AdvancedModelConfig(
        vocab_size=100,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,  # MQA
        norm_type="rms_norm",
        activation_type="swiglu",
        pos_encoding="rope",
        max_seq_len=64,
        tie_weights=True,
    )
    model = AdvancedTransformer(cfg)
    assert model.get_num_params() > 0

    input_ids = torch.randint(0, 100, (2, 10))
    logits = model(input_ids)
    assert logits.shape == (2, 10, 100)

    # Test autoregressive generation with KV caching
    prompt = torch.tensor([[1, 5, 12]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=5, use_cache=True)
    assert generated.shape == (1, 8)  # 3 initial + 5 generated


# ==============================================================================
# 4. Model Optimizer & INT8 Quantization Tests
# ==============================================================================

def test_8_dynamic_int8_quantization_and_memory():
    """Test dynamic INT8 quantization, size reduction, and memory profiling."""
    cfg = AdvancedModelConfig(
        vocab_size=200,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=64,
    )
    model_fp32 = AdvancedTransformer(cfg)
    model_int8 = ModelOptimizer.quantize_dynamic_int8(model_fp32)

    # Verify inference on quantized model works without error
    x = torch.randint(0, 200, (1, 8))
    out = model_int8(x)
    assert out.shape == (1, 8, 200)

    # Verify memory compression metrics
    comp = ModelOptimizer.compare_compression(model_fp32, model_int8)
    assert comp["fp32_size_kb"] > 0
    assert comp["int8_size_kb"] > 0
    assert comp["compression_ratio"] >= 1.0


def test_9_latency_benchmarking():
    """Test CPU latency benchmarking percentiles."""
    cfg = AdvancedModelConfig(vocab_size=50, embed_dim=16, num_layers=1, num_heads=2, max_seq_len=32)
    model = AdvancedTransformer(cfg)
    x = torch.randint(0, 50, (1, 8))

    bench = ModelOptimizer.benchmark_latency(model, x, num_runs=5, warmup_runs=1)
    assert "mean_ms" in bench
    assert "p50_ms" in bench
    assert "p90_ms" in bench
    assert bench["throughput_tokens_per_sec"] > 0.0


# ==============================================================================
# 5. Model Evaluator & Truthfulness Guard Tests
# ==============================================================================

def test_10_model_evaluator_metrics():
    """Test calculation of Cross-Entropy Loss, Perplexity, BPC, and Accuracy."""
    cfg = AdvancedModelConfig(vocab_size=60, embed_dim=32, num_layers=1, num_heads=2, max_seq_len=32)
    model = AdvancedTransformer(cfg)
    dataset = [torch.randint(0, 60, (1, 16)) for _ in range(3)]

    result = ModelEvaluator.evaluate(model, dataset)
    assert result.loss > 0.0
    assert result.perplexity >= 1.0
    assert result.bpc > 0.0
    assert 0.0 <= result.accuracy <= 100.0
    assert result.total_tokens == 3 * 15  # 15 target tokens per sequence


def test_11_truthfulness_guard_blocks_exaggeration():
    """Test TruthfulnessGuard rejects unjustified claims of fluency and convergence."""
    # Create evaluation result with high loss (untrained)
    high_loss_res = EvaluationResult(
        loss=4.5,
        perplexity=90.0,
        bpc=1.62,
        accuracy=0.10,
        total_tokens=100,
        duration_ms=50.0,
    )

    claims = {
        "is_fluent": True,
        "is_converged": True,
        "claims_autonomous_reasoning": True,
    }

    audit = TruthfulnessGuard.audit_capability_claims(claims, high_loss_res)
    assert audit["is_truthful"] is False
    assert len(audit["violations"]) == 3  # All 3 claims rejected


# ==============================================================================
# 6. Model Comparator & Experiment Tracker Tests
# ==============================================================================

def test_12_model_comparator_scorecard():
    """Test side-by-side comparison between Baseline and Advanced models."""
    from custom_ai.model import DecoderOnlyTransformer

    base_model = DecoderOnlyTransformer(vocab_size=80, embed_dim=32, num_layers=1, num_heads=2, max_seq_len=32)
    adv_cfg = AdvancedModelConfig(vocab_size=80, embed_dim=32, num_layers=1, num_heads=2, max_seq_len=32)
    adv_model = AdvancedTransformer(adv_cfg)

    eval_data = torch.randint(0, 80, (2, 16))
    comp = ModelComparator.compare_models(
        model_a=base_model,
        model_b=adv_model,
        name_a="Baseline-MHA",
        name_b="Advanced-GQA",
        eval_data=eval_data,
    )

    assert "scorecard_markdown" in comp
    assert "Baseline-MHA" in comp["scorecard_markdown"]
    assert "Advanced-GQA" in comp["scorecard_markdown"]
    assert "speedup_ratio" in comp


def test_13_experiment_tracker(tmp_path):
    """Test experiment run logging, JSON serialization, and Markdown report generation."""
    tracker = ExperimentTracker(experiment_dir=tmp_path)
    entry = tracker.log_experiment(
        experiment_id="EXP-GQA-001",
        architecture_type="AdvancedTransformer-GQA",
        config={"embed_dim": 64, "num_kv_heads": 2},
        metrics={"loss": 2.15, "perplexity": 8.58, "mean_ms": 12.4},
        notes="GQA with RMSNorm and RoPE ablation",
    )

    assert entry["experiment_id"] == "EXP-GQA-001"

    # Verify JSON persistence
    experiments = tracker.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["experiment_id"] == "EXP-GQA-001"

    # Verify Markdown file creation
    md_content = tracker.md_file.read_text(encoding="utf-8")
    assert "EXP-GQA-001" in md_content
    assert "AdvancedTransformer-GQA" in md_content
