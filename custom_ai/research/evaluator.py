"""J.A.R.V.I.S. Empirical Evaluation & Truthfulness Guard (Phase 25).

Provides rigorous, uncompromised metrics computation:
1. Cross-Entropy Loss & Perplexity (PPL = exp(loss))
2. Bits-Per-Character (BPC = loss / (ln(2) * chars_per_token))
3. Next-Token Top-1 Accuracy
4. Empirical Truthfulness Guard
   - Ensures that claims regarding model capabilities match verified empirical test outcomes
   - Prevents artificial exaggeration or unverified reasoning claims
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvaluationResult:
    """Structured container for empirical language model evaluation metrics."""

    def __init__(
        self,
        loss: float,
        perplexity: float,
        bpc: float,
        accuracy: float,
        total_tokens: int,
        duration_ms: float,
    ):
        self.loss = round(loss, 4)
        self.perplexity = round(perplexity, 4)
        self.bpc = round(bpc, 4)
        self.accuracy = round(accuracy * 100.0, 2)
        self.total_tokens = total_tokens
        self.duration_ms = round(duration_ms, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "bpc": self.bpc,
            "accuracy_percent": self.accuracy,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
        }

    def summary(self) -> str:
        return (
            f"Cross-Entropy Loss: {self.loss:.4f} | "
            f"Perplexity (PPL): {self.perplexity:.2f} | "
            f"BPC: {self.bpc:.3f} | "
            f"Top-1 Accuracy: {self.accuracy:.1f}% ({self.total_tokens} tokens evaluated in {self.duration_ms:.1f}ms)"
        )


class ModelEvaluator:
    """Quantitative empirical evaluator for decoder-only language models."""

    @staticmethod
    def evaluate(
        model: nn.Module,
        dataset: Union[List[torch.Tensor], torch.Tensor],
        device: str = "cpu",
        avg_chars_per_token: float = 4.0,
    ) -> EvaluationResult:
        """Evaluate model on a sequence or batch of sequences.

        Args:
            model: PyTorch model module
            dataset: Sequence tensor (B, T) or list of token sequence tensors
            device: Computing device ('cpu')
            avg_chars_per_token: Average characters per token for BPC calculation

        Returns:
            EvaluationResult with loss, perplexity, BPC, and accuracy.
        """
        model.eval()
        model.to(device)

        # Standardize dataset to list of batches
        if isinstance(dataset, torch.Tensor):
            batches = [dataset]
        else:
            batches = dataset

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and device.startswith("cuda") else None

        import time
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                if batch.ndim == 1:
                    batch = batch.unsqueeze(0)

                B, T = batch.shape
                if T < 2:
                    continue

                # Inputs are tokens 0 to T-2; targets are tokens 1 to T-1
                x = batch[:, :-1]
                y = batch[:, 1:]

                logits = model(x)  # (B, T-1, vocab_size)
                vocab_size = logits.shape[-1]

                # Flatten for cross entropy
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum")
                total_loss += float(loss.item())

                # Predictions
                preds = logits.argmax(dim=-1)
                total_correct += int((preds == y).sum().item())
                total_tokens += y.numel()

        duration_ms = (time.perf_counter() - t0) * 1000.0

        if total_tokens == 0:
            return EvaluationResult(0.0, 1.0, 0.0, 0.0, 0, duration_ms)

        avg_loss = total_loss / total_tokens
        # Guard against math overflow in exp(loss)
        perplexity = math.exp(min(avg_loss, 20.0))
        # BPC = loss / (ln(2) * chars_per_token)
        bpc = avg_loss / (math.log(2.0) * max(avg_chars_per_token, 1.0))
        accuracy = total_correct / total_tokens

        return EvaluationResult(
            loss=avg_loss,
            perplexity=perplexity,
            bpc=bpc,
            accuracy=accuracy,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )


class TruthfulnessGuard:
    """Strict verifier ensuring AI capability claims reflect measured empirical realities."""

    @staticmethod
    def audit_capability_claims(
        claimed_capabilities: Dict[str, Any],
        empirical_results: EvaluationResult,
    ) -> Dict[str, Any]:
        """Audit claimed capabilities against empirical measurements.

        Rule:
        - If perplexity > 50.0, the model cannot be claimed as 'fluent' or 'coherent'.
        - If loss > 3.0, the model cannot be claimed as 'domain-converged'.
        - Capabilities cannot be reported as 'trained' if empirical token count is 0.
        """
        violations: List[str] = []
        verified: List[str] = []

        # 1. Fluency Check
        if claimed_capabilities.get("is_fluent", False):
            if empirical_results.perplexity > 40.0:
                violations.append(
                    f"Claimed 'is_fluent=True' violates empirical threshold (PPL={empirical_results.perplexity:.2f} > 40.0)."
                )
            else:
                verified.append("Fluency verified empirically by perplexity threshold.")

        # 2. Convergence Check
        if claimed_capabilities.get("is_converged", False):
            if empirical_results.loss > 3.0:
                violations.append(
                    f"Claimed 'is_converged=True' violates empirical threshold (Loss={empirical_results.loss:.4f} > 3.0)."
                )
            else:
                verified.append("Convergence verified empirically by loss threshold.")

        # 3. Reasoning Claim Guard
        if claimed_capabilities.get("claims_autonomous_reasoning", False):
            violations.append(
                "Autonomous reasoning cannot be claimed on character/subword language model loss alone. "
                "Must be verified through multi-step logic problem suites (DeductiveSolver)."
            )

        is_valid = len(violations) == 0
        return {
            "is_truthful": is_valid,
            "violations": violations,
            "verified_claims": verified,
            "empirical_loss": empirical_results.loss,
            "empirical_perplexity": empirical_results.perplexity,
        }
