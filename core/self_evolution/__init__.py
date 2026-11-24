"""J.A.R.V.I.S. Self-Evolution System.
======================================
Architectural implementation of the controlled self-improvement pipeline:
- Phase 29: Self-Learning Engine (Feedback harvesting, mistake indexing, knowledge updates, candidate training curation)
- Phase 30: Self-Reasoning Improvement (Strategy evaluation, approach comparison, decomposition scoring, benchmarks)
- Phase 31: Self-Code Analysis (AST source inspection, bug detection, duplicate code detection, security linter)
- Phase 32: Self-Code Generation (Targeted refactoring, unified diff patch generation, isolated dev sandbox)
- Phase 33: Automated Testing & Regression Guard (Test runner, regression detection, performance comparison, rejection gate)
- Phase 34: Self-Version Management (Semantic versioning, changelog generator, release history, rollback pointers)
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

from core.self_evolution.self_reasoning_engine import (
    SelfReasoningEngine,
    self_reasoning_engine,
    ReasoningStrategyType,
    StrategyProfile,
    BenchmarkProblem,
    BenchmarkEvaluationResult,
    DecompositionQualityScore,
)

from core.self_evolution.self_code_analyzer import (
    SelfCodeAnalyzer,
    self_code_analyzer,
    CodeIssue,
    CodeIssueType,
    IssueSeverity,
    DuplicateBlock,
    DependencyAnalysis,
    CodeQualityReport,
)

from core.self_evolution.self_code_generator import (
    SelfCodeGenerator,
    self_code_generator,
    CodeImprovementProposal,
    SandboxEnvironmentInfo,
)

from core.self_evolution.self_test_runner import (
    SelfTestRunner,
    self_test_runner,
    TestOutcome,
    SingleTestCaseResult,
    TestSuiteExecutionResult,
    RegressionComparison,
    PerformanceComparison,
    CandidateTestEvaluation,
)

from core.self_evolution.self_version_manager import (
    SelfVersionManager,
    self_version_manager,
    SemanticVersion,
    ReleaseRecord,
    VersionDiff,
)

__all__ = [
    # Phase 29
    "SelfLearningEngine",
    "self_learning_engine",
    "MistakeCorrectionRecord",
    "TaskAnalysisResult",
    "CandidateTrainingExample",
    "LearningProgressReport",
    # Phase 30
    "SelfReasoningEngine",
    "self_reasoning_engine",
    "ReasoningStrategyType",
    "StrategyProfile",
    "BenchmarkProblem",
    "BenchmarkEvaluationResult",
    "DecompositionQualityScore",
    # Phase 31
    "SelfCodeAnalyzer",
    "self_code_analyzer",
    "CodeIssue",
    "CodeIssueType",
    "IssueSeverity",
    "DuplicateBlock",
    "DependencyAnalysis",
    "CodeQualityReport",
    # Phase 32
    "SelfCodeGenerator",
    "self_code_generator",
    "CodeImprovementProposal",
    "SandboxEnvironmentInfo",
    # Phase 33
    "SelfTestRunner",
    "self_test_runner",
    "TestOutcome",
    "SingleTestCaseResult",
    "TestSuiteExecutionResult",
    "RegressionComparison",
    "PerformanceComparison",
    "CandidateTestEvaluation",
    # Phase 34
    "SelfVersionManager",
    "self_version_manager",
    "SemanticVersion",
    "ReleaseRecord",
    "VersionDiff",
]
