"""Unit tests for J.A.R.V.I.S. Phase 30 — Self-Reasoning Improvement.
===================================================================
Tests Dimension 2 of the Self-Improvement System:
1. Evaluating alternative reasoning strategies
2. Comparing candidate solution approaches
3. Task decomposition DAG verification, cycle breaking, and quality scoring
4. Running repeatable deterministic reasoning benchmarks
5. Measuring empirical reasoning accuracy and latency
6. Dynamic strategy selection based on measured performance
"""

import json
import pytest
from pathlib import Path

from core.self_evolution.self_reasoning_engine import (
    SelfReasoningEngine,
    ReasoningStrategyType,
    StrategyProfile,
    BenchmarkProblem,
    BenchmarkEvaluationResult,
    DecompositionQualityScore,
)


@pytest.fixture
def temp_reasoning_engine(tmp_path):
    """Provides a clean SelfReasoningEngine instance with temp paths."""
    evo_dir = tmp_path / "evolution"
    engine = SelfReasoningEngine(evolution_dir=evo_dir)
    engine.initialize()
    return engine


def test_self_reasoning_initialization(temp_reasoning_engine):
    """Verifies that strategy profiles and default benchmarks are properly populated."""
    assert len(temp_reasoning_engine.strategy_profiles) >= 5
    assert len(temp_reasoning_engine.benchmark_suite) >= 9
    assert temp_reasoning_engine._initialized is True

    # Check presence of canonical strategies
    assert ReasoningStrategyType.DEDUCTIVE_FIRST_PRINCIPLES in temp_reasoning_engine.strategy_profiles
    assert ReasoningStrategyType.DECOMPOSED_DIVIDE_AND_CONQUER in temp_reasoning_engine.strategy_profiles
    assert ReasoningStrategyType.HEURISTIC_SCHEMA_MATCHING in temp_reasoning_engine.strategy_profiles
    assert ReasoningStrategyType.CONSTRAINT_SATISFACTION in temp_reasoning_engine.strategy_profiles
    assert ReasoningStrategyType.HYBRID_MULTI_PERSPECTIVE in temp_reasoning_engine.strategy_profiles


def test_evaluate_reasoning_strategies(temp_reasoning_engine):
    """Verifies strategy evaluation correctly ranks strategies by query category."""
    # 1. Arithmetic query
    arithmetic_query = "Calculate 25 * 18"
    ranked_math = temp_reasoning_engine.evaluate_reasoning_strategies(arithmetic_query)
    assert len(ranked_math) >= 5
    top_math = ranked_math[0]
    assert top_math["strategy_type"] == ReasoningStrategyType.DEDUCTIVE_FIRST_PRINCIPLES.value
    assert top_math["suitability_score"] > 0.8

    # 2. Planning query
    plan_query = "Plan a complete database migration"
    ranked_plan = temp_reasoning_engine.evaluate_reasoning_strategies(plan_query)
    top_plan = ranked_plan[0]
    assert top_plan["strategy_type"] in [
        ReasoningStrategyType.DECOMPOSED_DIVIDE_AND_CONQUER.value,
        ReasoningStrategyType.HYBRID_MULTI_PERSPECTIVE.value,
    ]


def test_compare_alternative_approaches(temp_reasoning_engine):
    """Verifies comparison of multiple approaches across safety, speed, and cost."""
    comparison = temp_reasoning_engine.compare_alternative_approaches("What is 25 multiplied by 48?")
    assert "ranked_strategies" in comparison
    assert "evaluated_approaches" in comparison
    assert len(comparison["evaluated_approaches"]) == 3
    assert comparison["recommended_strategy"] == ReasoningStrategyType.DEDUCTIVE_FIRST_PRINCIPLES.value
    assert len(comparison["selection_rationale"]) > 0


def test_improve_task_decomposition_cycle_breaking(temp_reasoning_engine):
    """Verifies that circular dependencies in task decompositions are detected and linearized."""
    # Plan with intentional circular dependency: step-1 -> step-2 -> step-1
    cyclic_steps = [
        {"id": "step-1", "title": "Initialize Config", "description": "Set up config", "dependencies": ["step-2"]},
        {"id": "step-2", "title": "Load DB", "description": "Load database", "dependencies": ["step-1"]},
        {"id": "step-3", "title": "Run Server", "description": "Start listener", "dependencies": ["step-2"]},
    ]

    plan, quality = temp_reasoning_engine.improve_task_decomposition(
        goal="Launch backend service",
        initial_steps=cyclic_steps,
    )

    assert isinstance(quality, DecompositionQualityScore)
    # The cycle was detected
    assert quality.is_dag is False
    assert any("Cycle detected" in s for s in quality.optimization_suggestions)

    # Subtasks were enhanced and cycle was broken
    assert len(plan.subtasks) == 3
    # Step-2 no longer depends circularly on Step-1 in a cycle; dependencies are linearized
    assert plan.subtasks[1].dependencies == ["step-1"]
    assert plan.subtasks[0].verification_check != ""


def test_improve_task_decomposition_monolithic_scoring(temp_reasoning_engine):
    """Verifies granularity scoring penalizes monolithic single-step plans."""
    monolithic_steps = [
        {"id": "step-1", "title": "Do Everything", "description": "Execute whole workflow in one block"}
    ]

    plan, quality = temp_reasoning_engine.improve_task_decomposition(
        goal="Build and deploy system",
        initial_steps=monolithic_steps,
    )

    assert quality.step_count == 1
    assert quality.granularity_score == 0.5  # Penalized
    assert any("monolithic" in s.lower() for s in quality.optimization_suggestions)
    # Verification check and fallback injected
    assert plan.subtasks[0].verification_check != ""
    assert plan.subtasks[0].fallback_plan is not None


def test_run_reasoning_benchmarks_and_accuracy(temp_reasoning_engine):
    """Verifies empirical benchmark execution across math, riddles, code, and facts."""
    result = temp_reasoning_engine.run_reasoning_benchmarks()

    assert isinstance(result, BenchmarkEvaluationResult)
    assert result.total_benchmarks >= 9
    assert result.passed_count >= 7  # High accuracy on deterministic solvers
    assert result.overall_accuracy >= 75.0
    assert result.average_latency_ms > 0.0

    # Check category scores
    assert "math_arithmetic" in result.category_scores
    assert result.category_scores["math_arithmetic"] == 100.0  # Arithmetic math is deterministic
    assert "deductive_riddle" in result.category_scores
    assert "factual_deduction" in result.category_scores

    # Verify results persisted to disk
    assert temp_reasoning_engine.benchmarks_file.exists()
    with open(temp_reasoning_engine.benchmarks_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) >= 1


def test_select_optimal_strategy(temp_reasoning_engine):
    """Verifies dynamic selection of optimal reasoning strategy based on problem type."""
    # Math problem
    strat_math, rat_math = temp_reasoning_engine.select_optimal_strategy("Calculate 25 * 18")
    assert strat_math == ReasoningStrategyType.DEDUCTIVE_FIRST_PRINCIPLES
    assert "math_arithmetic" in rat_math

    # Planning problem
    strat_plan, rat_plan = temp_reasoning_engine.select_optimal_strategy("Plan a full backup and recovery test")
    assert strat_plan in [
        ReasoningStrategyType.DECOMPOSED_DIVIDE_AND_CONQUER,
        ReasoningStrategyType.HYBRID_MULTI_PERSPECTIVE,
    ]


def test_generate_reasoning_report_markdown(temp_reasoning_engine):
    """Verifies markdown report generation for system diagnostics and HUD."""
    md = temp_reasoning_engine.generate_reasoning_report_markdown()
    assert "J.A.R.V.I.S. Self-Reasoning Engine Status (Phase 30)" in md
    assert "Overall Reasoning Accuracy" in md
    assert "Benchmark Problems Evaluated" in md
