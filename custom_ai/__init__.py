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

from .attention import MaskedMultiHeadAttention
from .dataset import TextDataset, TinyTextDataset, clean_text, load_corpus_file, split_train_val
from .generation import TextGenerator
from .inference import (
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
from .trainer import LanguageModelTrainer, TinyLanguageModelTrainer, detect_system_resources
from .transformer import FeedForward, TransformerDecoderBlock

__all__ = [
    "BaseLocalModelAdapter",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CustomTransformerAdapter",
    "DecoderLayer",
    "DecoderOnlyTransformer",
    "FeedForward",
    "InferenceResponse",
    "InvalidCheckpointError",
    "LanguageModelTrainer",
    "LocalModelInference",
    "LocalModelRegistry",
    "MaskedMultiHeadAttention",
    "PretrainedLocalModelAdapter",
    "TextDataset",
    "TextGenerator",
    "TextTokenizer",
    "TinyLanguageModelTrainer",
    "TinyTextDataset",
    "TransformerDecoderBlock",
    "build_causal_mask",
    "clean_text",
    "detect_system_resources",
    "load_corpus_file",
    "split_train_val",
]
