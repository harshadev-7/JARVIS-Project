"""J.A.R.V.I.S. Side-by-Side Model Comparator (Phase 25).

Compares two model architectures or checkpoints (e.g. Baseline MHA vs Advanced GQA+RoPE):
- Parameter efficiency and parameter count
- Memory size (KB)
- Inference latency percentiles (P50, P90)
- Validation loss and Perplexity (PPL)
- Comparative generation outputs on benchmark prompts
- Exports structured markdown comparison scorecards
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from .evaluator import ModelEvaluator, EvaluationResult
from .model_optimizer import ModelOptimizer


class ModelComparator:
    """Benchmark comparator evaluating two models on identical datasets and prompts."""

    @staticmethod
    def compare_models(
        model_a: nn.Module,
        model_b: nn.Module,
        name_a: str,
        name_b: str,
        eval_data: torch.Tensor,
        benchmark_prompt: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive side-by-side evaluation between model_a and model_b.

        Args:
            model_a: First model (e.g. Baseline)
            model_b: Second model (e.g. Advanced)
            name_a: Display name for Model A
            name_b: Display name for Model B
            eval_data: Sequence tensor (B, T) for validation evaluation
            benchmark_prompt: Optional prompt tensor for text generation comparison
            tokenizer: Optional tokenizer to decode generated token outputs

        Returns:
            Dictionary containing metrics comparison, winner per category, and formatted scorecard.
        """
        # 1. Parameter Counts
        params_a = sum(p.numel() for p in model_a.parameters())
        params_b = sum(p.numel() for p in model_b.parameters())

        # 2. Checkpoint Sizes
        size_a_kb = ModelOptimizer.get_serialized_size_bytes(model_a) / 1024.0
        size_b_kb = ModelOptimizer.get_serialized_size_bytes(model_b) / 1024.0

        # 3. Latency Benchmarks
        lat_a = ModelOptimizer.benchmark_latency(model_a, eval_data[:1, :min(eval_data.shape[1], 32)])
        lat_b = ModelOptimizer.benchmark_latency(model_b, eval_data[:1, :min(eval_data.shape[1], 32)])

        # 4. Empirical Evaluation (Loss & Perplexity)
        eval_a: EvaluationResult = ModelEvaluator.evaluate(model_a, eval_data)
        eval_b: EvaluationResult = ModelEvaluator.evaluate(model_b, eval_data)

        # 5. Optional Generation Comparison
        gen_a_text = ""
        gen_b_text = ""
        if benchmark_prompt is not None and tokenizer is not None:
            with torch.no_grad():
                out_a = model_a.generate(benchmark_prompt, max_new_tokens=16) if hasattr(model_a, "generate") else benchmark_prompt
                out_b = model_b.generate(benchmark_prompt, max_new_tokens=16) if hasattr(model_b, "generate") else benchmark_prompt
                gen_a_text = tokenizer.decode(out_a[0].tolist()) if hasattr(tokenizer, "decode") else str(out_a[0].tolist())
                gen_b_text = tokenizer.decode(out_b[0].tolist()) if hasattr(tokenizer, "decode") else str(out_b[0].tolist())

        # Determine advantages
        ppl_diff = eval_b.perplexity - eval_a.perplexity
        speedup = lat_a["mean_ms"] / max(lat_b["mean_ms"], 1e-4)

        scorecard = (
            f"### **J.A.R.V.I.S. Neural Architecture Benchmark: {name_a} vs {name_b}**\n\n"
            f"| Metric | {name_a} | {name_b} | Advantage |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Parameters** | {params_a:,} | {params_b:,} | {name_a if params_a < params_b else name_b} ({abs(params_a - params_b):,} delta) |\n"
            f"| **Checkpoint Size** | {size_a_kb:.1f} KB | {size_b_kb:.1f} KB | {name_a if size_a_kb < size_b_kb else name_b} |\n"
            f"| **Inference Mean Latency** | {lat_a['mean_ms']:.2f} ms | {lat_b['mean_ms']:.2f} ms | {name_a if lat_a['mean_ms'] < lat_b['mean_ms'] else name_b} ({speedup:.2f}x speed ratio) |\n"
            f"| **Inference P90 Latency** | {lat_a['p90_ms']:.2f} ms | {lat_b['p90_ms']:.2f} ms | {name_a if lat_a['p90_ms'] < lat_b['p90_ms'] else name_b} |\n"
            f"| **Throughput** | {lat_a['throughput_tokens_per_sec']:.1f} tok/s | {lat_b['throughput_tokens_per_sec']:.1f} tok/s | {name_a if lat_a['throughput_tokens_per_sec'] > lat_b['throughput_tokens_per_sec'] else name_b} |\n"
            f"| **Cross-Entropy Loss** | {eval_a.loss:.4f} | {eval_b.loss:.4f} | {name_a if eval_a.loss < eval_b.loss else name_b} |\n"
            f"| **Perplexity (PPL)** | {eval_a.perplexity:.2f} | {eval_b.perplexity:.2f} | {name_a if eval_a.perplexity < eval_b.perplexity else name_b} |\n"
            f"| **Next-Token Accuracy** | {eval_a.accuracy:.1f}% | {eval_b.accuracy:.1f}% | {name_a if eval_a.accuracy > eval_b.accuracy else name_b} |\n"
        )

        return {
            "name_a": name_a,
            "name_b": name_b,
            "params_a": params_a,
            "params_b": params_b,
            "size_a_kb": size_a_kb,
            "size_b_kb": size_b_kb,
            "latency_a": lat_a,
            "latency_b": lat_b,
            "eval_a": eval_a.to_dict(),
            "eval_b": eval_b.to_dict(),
            "ppl_delta": round(ppl_diff, 4),
            "speedup_ratio": round(speedup, 2),
            "generation_a": gen_a_text,
            "generation_b": gen_b_text,
            "scorecard_markdown": scorecard,
        }
