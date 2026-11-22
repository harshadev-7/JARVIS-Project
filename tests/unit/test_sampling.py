"""Unit tests for J.A.R.V.I.S. Neural Sampling & Logit Modulation (Phase 13)."""

import pytest
import torch

from custom_ai.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)


def test_repetition_penalty():
    logits = torch.tensor([2.0, 4.0, 1.0])
    penalized = apply_repetition_penalty(logits, generated_tokens=[1], penalty=2.0)
    # Logit at index 1 should be halved from 4.0 to 2.0
    assert penalized[1].item() == 2.0
    assert penalized[0].item() == 2.0
    assert penalized[2].item() == 1.0


def test_temperature_scaling():
    logits = torch.tensor([1.0, 2.0, 3.0])
    hot = apply_temperature(logits, temperature=2.0)
    assert hot[2].item() == 1.5

    cold = apply_temperature(logits, temperature=0.5)
    assert cold[2].item() == 6.0


def test_top_k_filtering():
    logits = torch.tensor([1.0, 5.0, 2.0, 8.0, 0.5])
    filtered = apply_top_k(logits, top_k=2)
    # Only index 1 (5.0) and index 3 (8.0) should remain, rest -inf
    assert filtered[3].item() == 8.0
    assert filtered[1].item() == 5.0
    assert filtered[0].item() == float("-inf")
    assert filtered[2].item() == float("-inf")
    assert filtered[4].item() == float("-inf")


def test_top_p_nucleus_filtering():
    # Construct a distribution where top 2 tokens dominate mass
    logits = torch.tensor([10.0, 8.0, 0.1, 0.1, 0.1])
    filtered = apply_top_p(logits, top_p=0.8)
    assert filtered[0].item() == 10.0
    # Smallest tail tokens should be masked
    assert filtered[-1].item() == float("-inf")


def test_sample_next_token_bounds():
    logits = torch.randn(50)
    token = sample_next_token(logits, temperature=0.8, top_k=10, top_p=0.9, repetition_penalty=1.1)
    assert isinstance(token, int)
    assert 0 <= token < 50
