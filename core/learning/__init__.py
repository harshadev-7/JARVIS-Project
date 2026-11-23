"""J.A.R.V.I.S. Adaptive Learning Engine.

Controlled feedback-driven self-improvement pipeline:
- Feedback analysis & root-cause diagnostics
- Persistent learning registry
- Consistency & regression safety guard
- Training dataset curation
- Fine-tuning staging & rollback manager
"""

from .adaptive_engine import AdaptiveLearningEngine, adaptive_learning_engine
from .feedback_analyzer import FeedbackAnalyzer, feedback_analyzer, DiagnosticResult
from .learning_registry import LearningRegistry, learning_registry
from .regression_guard import RegressionGuard, regression_guard
from .dataset_curator import DatasetCurator, dataset_curator
from .fine_tuning_staging import FineTuningStagingManager, fine_tuning_staging

__all__ = [
    "AdaptiveLearningEngine",
    "adaptive_learning_engine",
    "FeedbackAnalyzer",
    "feedback_analyzer",
    "DiagnosticResult",
    "LearningRegistry",
    "learning_registry",
    "RegressionGuard",
    "regression_guard",
    "DatasetCurator",
    "dataset_curator",
    "FineTuningStagingManager",
    "fine_tuning_staging",
]
