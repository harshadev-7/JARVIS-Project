"""J.A.R.V.I.S. Advanced Neural Network Research Laboratory (Phase 25).

Exposes high-level interfaces for architectural experimentation, optimization,
empirical evaluation, and comparative benchmarking.
"""

from .modern_blocks import RMSNorm, RotaryPositionalEmbedding, SwiGLU
from .attention_variants import GroupedQueryAttention, KVCache, AttentionAnalyzer
from .advanced_transformer import (
    AdvancedDecoderBlock,
    AdvancedModelConfig,
    AdvancedTransformer,
)
from .model_optimizer import ModelOptimizer
from .evaluator import EvaluationResult, ModelEvaluator, TruthfulnessGuard
from .model_comparator import ModelComparator
from .experiment_tracker import ExperimentTracker, experiment_tracker

__all__ = [
    # Modern Blocks
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    # Attention Variants
    "GroupedQueryAttention",
    "KVCache",
    "AttentionAnalyzer",
    # Advanced Model
    "AdvancedModelConfig",
    "AdvancedDecoderBlock",
    "AdvancedTransformer",
    # Optimization
    "ModelOptimizer",
    # Evaluation
    "EvaluationResult",
    "ModelEvaluator",
    "TruthfulnessGuard",
    # Model Comparison
    "ModelComparator",
    # Experiment Tracking
    "ExperimentTracker",
    "experiment_tracker",
]
