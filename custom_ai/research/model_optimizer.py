"""J.A.R.V.I.S. Model Optimization & Dynamic Quantization Suite (Phase 25).

Provides lightweight, hardware-aware optimization tools for CPU execution:
1. Dynamic INT8 Quantization (PyTorch Dynamic Post-Training Quantization)
   - Converts Linear layer weights from 32-bit floating point (FP32) to 8-bit signed integer (qint8)
   - Reduces model memory footprint by 50-70% on local RAM
   - Accelerates CPU matrix multiplication through vectorized integer kernels
2. Memory Footprint Profiling
   - Measures exact byte sizes of model parameters and buffers
3. Latency & Throughput Benchmarking
   - Measures P50, P90, P99 token generation latency and throughput (tokens/sec)
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class ModelOptimizer:
    """Optimization and profiling utility for J.A.R.V.I.S. local Transformer models."""

    @staticmethod
    def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization to linear layers of the model.

        Args:
            model: PyTorch model module (AdvancedTransformer or DecoderOnlyTransformer)

        Returns:
            Quantized PyTorch model with INT8 dynamic linear layers.
        """
        # Ensure model is in eval mode before quantization
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model

    @staticmethod
    def get_model_size_bytes(model: nn.Module) -> int:
        """Calculate total in-memory size of model parameters and buffers in bytes."""
        total_bytes = 0
        for p in model.parameters():
            if hasattr(p, "element_size") and hasattr(p, "nelement"):
                total_bytes += p.element_size() * p.nelement()
        for b in model.buffers():
            if hasattr(b, "element_size") and hasattr(b, "nelement"):
                total_bytes += b.element_size() * b.nelement()
        return total_bytes

    @staticmethod
    def get_serialized_size_bytes(model: nn.Module) -> int:
        """Measure exact serialized checkpoint file size in bytes via in-memory buffer."""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getbuffer().nbytes

    @staticmethod
    def benchmark_latency(
        model: nn.Module,
        input_ids: torch.Tensor,
        num_runs: int = 10,
        warmup_runs: int = 2,
    ) -> Dict[str, float]:
        """Benchmark forward pass latency and throughput on the active CPU.

        Args:
            model: PyTorch model to benchmark
            input_ids: Input tensor (B, T)
            num_runs: Number of timed runs
            warmup_runs: Initial untimed runs for cache warming

        Returns:
            Dictionary with min, median (p50), p90, p99 latency in ms and tokens/sec throughput.
        """
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(warmup_runs):
                _ = model(input_ids)

            latencies: List[float] = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_ids)
                dur = (time.perf_counter() - start) * 1000.0
                latencies.append(dur)

        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(n * 0.50)]
        p90 = latencies[min(int(n * 0.90), n - 1)]
        p99 = latencies[min(int(n * 0.99), n - 1)]
        min_lat = latencies[0]
        mean_lat = sum(latencies) / n

        # Throughput: tokens processed per second
        total_tokens = input_ids.numel()
        throughput = (total_tokens / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

        return {
            "min_ms": round(min_lat, 3),
            "mean_ms": round(mean_lat, 3),
            "p50_ms": round(p50, 3),
            "p90_ms": round(p90, 3),
            "p99_ms": round(p99, 3),
            "throughput_tokens_per_sec": round(throughput, 2),
        }

    @staticmethod
    def compare_compression(
        fp32_model: nn.Module,
        int8_model: nn.Module,
    ) -> Dict[str, Any]:
        """Compare FP32 baseline against INT8 quantized model."""
        fp32_size = ModelOptimizer.get_serialized_size_bytes(fp32_model)
        int8_size = ModelOptimizer.get_serialized_size_bytes(int8_model)

        compression_ratio = (fp32_size / int8_size) if int8_size > 0 else 1.0
        reduction_pct = ((fp32_size - int8_size) / fp32_size * 100.0) if fp32_size > 0 else 0.0

        return {
            "fp32_size_kb": round(fp32_size / 1024.0, 2),
            "int8_size_kb": round(int8_size / 1024.0, 2),
            "compression_ratio": round(compression_ratio, 2),
            "reduction_percent": round(reduction_pct, 2),
        }
