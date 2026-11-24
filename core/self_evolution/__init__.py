"""J.A.R.V.I.S. Self-Evolution System.
======================================
Architectural implementation of the controlled self-improvement pipeline:
- Phase 29: Self-Learning Engine (Feedback harvesting, mistake indexing, knowledge updates, candidate training curation)
- Phase 30: Self-Reasoning Improvement
- Phase 31: Self-Code Analysis
- Phase 32: Self-Code Generation
- Phase 33: Automated Testing & Regression Guard
- Phase 34: Self-Version Management
- Phase 35: Backup and Recovery
- Phase 36: Controlled Self-Update System
- Phase 37: Continuous Performance Evaluation
"""

from core.self_evolution.self_learning_engine import (
    SelfLearningEngine,
    self_learning_engine,
    MistakeCorrectionRecord,
    TaskAnalysisResult,
    CandidateTrainingExample,
    LearningProgressReport,
)

__all__ = [
    "SelfLearningEngine",
    "self_learning_engine",
    "MistakeCorrectionRecord",
    "TaskAnalysisResult",
    "CandidateTrainingExample",
    "LearningProgressReport",
]
