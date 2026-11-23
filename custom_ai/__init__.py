"""Custom local AI components for J.A.R.V.I.S.

This module provides the package-level public API by re-exporting commonly used
classes and helpers from the implementation modules below.

This package contains the complete local, CPU-first neural architecture:
- Masked Multi-Head Self-Attention from first principles
- Transformer Decoder blocks with Pre-LN & GELU FeedForward
- Full Decoder-Only Transformer language model
- Vocabulary constructor & Tokenizer
- Autoregressive Text Generation engine
- Dataset builders & Training loops
- Local Model Registry & Pretrained Model Adapters
- Safe Inference Engine with checkpoint validation

The ``__all__`` list defines the supported names exposed by this package.
"""

from .training.run_training import main as run_training
from .quick_train import quick_train
try:
    from .attention import MaskedMultiHeadAttention
except Exception:
    # Defer heavy imports; components that need torch will raise when used
    MaskedMultiHeadAttention = None

from .dataset import TextDataset, TinyTextDataset, clean_text, load_corpus_file, split_train_val
from .generation import TextGenerator
from .sampling import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)
from .checkpoint_utils import save_checkpoint, load_checkpoint
from .inference_core import (
    CheckpointError,
    CheckpointNotFoundError,
    InferenceResponse,
    InvalidCheckpointError,
    LocalModelInference,
)
from .model import DecoderLayer, DecoderOnlyTransformer, build_causal_mask
from .model_registry import LocalModelRegistry
from .pretrained_adapter import (
    BaseLocalModelAdapter,
    CustomTransformerAdapter,
    PretrainedLocalModelAdapter,
)
from .tokenizer import TextTokenizer
from .bpe_tokenizer import BPETokenizer
from .trainer import LanguageModelTrainer, TinyLanguageModelTrainer, detect_system_resources
from .transformer import FeedForward, TransformerDecoderBlock
from .research import (
    AdvancedDecoderBlock,
    AdvancedModelConfig,
    AdvancedTransformer,
    AttentionAnalyzer,
    EvaluationResult,
    ExperimentTracker,
    GroupedQueryAttention,
    KVCache,
    ModelComparator,
    ModelEvaluator,
    ModelOptimizer,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TruthfulnessGuard,
    experiment_tracker,
)

def run_inference(*args, **kwargs):
    from .inference_cli.run_inference import main as _main
    return _main(*args, **kwargs)

__all__ = [
    "run_training",
    "run_inference",
    "quick_train",
    "LocalModelInference",
    "LanguageModelTrainer",
    "TinyLanguageModelTrainer",
    "TextTokenizer",
    "BPETokenizer",
    "DecoderOnlyTransformer",
    "MaskedMultiHeadAttention",
    "save_checkpoint",
    "load_checkpoint",
    "BaseLocalModelAdapter",
    "CustomTransformerAdapter",
    "PretrainedLocalModelAdapter",
    "DecoderLayer",
    "FeedForward",
    "InferenceResponse",
    "CheckpointError",
    "CheckpointNotFoundError",
    "InvalidCheckpointError",
    "LocalModelRegistry",
    "TextDataset",
    "TinyTextDataset",
    "TextGenerator",
    "TransformerDecoderBlock",
    "build_causal_mask",
    "clean_text",
    "detect_system_resources",
    "load_corpus_file",
    "split_train_val",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "GroupedQueryAttention",
    "KVCache",
    "AttentionAnalyzer",
    "AdvancedModelConfig",
    "AdvancedDecoderBlock",
    "AdvancedTransformer",
    "ModelOptimizer",
    "EvaluationResult",
    "ModelEvaluator",
    "TruthfulnessGuard",
    "ModelComparator",
    "ExperimentTracker",
    "experiment_tracker",
]
