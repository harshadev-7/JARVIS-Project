"""Unit Tests for Phase 37: Continuous Performance Evaluator
===========================================================
Tests hardware utilization measurement, response quality assessment,
reasoning accuracy benchmarks, execution reliability, regression detection,
and automated improvement recommendations.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.self_evolution.performance_evaluator import (
    ContinuousPerformanceEvaluator,
    HardwareResourceSnapshot,
    ResponseQualityMetric,
    ReasoningAccuracyMetric,
    ExecutionReliabilityMetric,
    ImprovementRecommendation,
    VersionPerformanceComparison,
    PerformanceEvaluationReport,
)
from core.self_evolution.self_reasoning_engine import SelfReasoningEngine, BenchmarkEvaluationResult
from core.self_evolution.self_version_manager import SelfVersionManager


@pytest.fixture
def mock_evaluator_env():
    """Sets up an isolated environment for testing continuous performance evaluation."""
    temp_dir = tempfile.mkdtemp(prefix="jarvis_eval_test_")
    repo_path = Path(temp_dir) / "repo"
    repo_path.mkdir(parents=True)

    # Mock reasoning engine
    reasoning_mock = MagicMock(spec=SelfReasoningEngine)
    reasoning_mock.run_reasoning_benchmarks.return_value = BenchmarkEvaluationResult(
        run_id="bench_mock",
        timestamp=1000.0,
        total_benchmarks=9,
        passed_count=9,
        failed_count=0,
        overall_accuracy=100.0,
        strategy_accuracies={"Deductive": 100.0, "Heuristic": 100.0},
        category_scores={"math": 100.0, "logic": 100.0, "code": 100.0},
        average_latency_ms=2.5,
        recommendations=[],
    )

    # Mock version manager
    version_mock = MagicMock(spec=SelfVersionManager)
    version_mock.get_current_version.return_value = "1.0.0"

    evaluator = ContinuousPerformanceEvaluator(
        repo_root=repo_path,
        reasoning_engine=reasoning_mock,
        version_mgr=version_mock,
    )

    yield {
        "temp_dir": temp_dir,
        "repo_path": repo_path,
        "evaluator": evaluator,
        "reasoning_mock": reasoning_mock,
        "version_mock": version_mock,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_capture_hardware_snapshot(mock_evaluator_env):
    """Verifies sampling of local CPU and RAM."""
    evaluator = mock_evaluator_env["evaluator"]

    snapshot = evaluator.capture_hardware_snapshot()
    assert isinstance(snapshot, HardwareResourceSnapshot)
    assert snapshot.ram_total_mb > 1000.0
    assert snapshot.ram_used_mb > 100.0
    assert snapshot.process_rss_mb > 0.0
    assert snapshot.process_threads >= 1
    assert snapshot.cpu_percent >= 0.0


def test_evaluate_response_quality_offline_clean(mock_evaluator_env):
    """Verifies response quality scoring on standard offline responses."""
    evaluator = mock_evaluator_env["evaluator"]

    clean_responses = [
        {"query": "Calculate 15 * 3", "response": "15 * 3 = 45. Evaluated deterministically."},
        {"query": "Hello", "response": "Greetings. System operational and ready."},
    ]

    metric = evaluator.evaluate_response_quality(clean_responses)
    assert metric.sample_count == 2
    assert metric.offline_compliance_score == 100.0
    assert metric.coherence_score == 100.0
    assert metric.conciseness_score == 100.0
    assert metric.overall_quality_score == 100.0


def test_evaluate_response_quality_forbidden_cloud_detection(mock_evaluator_env):
    """Verifies that any response referencing external cloud APIs or keys is severely penalized."""
    evaluator = mock_evaluator_env["evaluator"]

    leaked_responses = [
        {"query": "Help me", "response": "Please call api.openai.com with sk-1234567890abcdefghijklmnop to proceed."},
        {"query": "Status", "response": "Connecting to generativelanguage.googleapis.com for cloud data."},
    ]

    metric = evaluator.evaluate_response_quality(leaked_responses)
    assert metric.offline_compliance_score == 0.0
    assert metric.overall_quality_score < 60.0


def test_evaluate_reasoning_accuracy(mock_evaluator_env):
    """Verifies integration with reasoning benchmark evaluation."""
    evaluator = mock_evaluator_env["evaluator"]

    reasoning_metric = evaluator.evaluate_reasoning_accuracy()
    assert reasoning_metric.total_benchmarks == 9
    assert reasoning_metric.passed_count == 9
    assert reasoning_metric.failed_count == 0
    assert reasoning_metric.accuracy_pct == 100.0
    assert reasoning_metric.average_latency_ms == 2.5


def test_evaluate_execution_reliability(mock_evaluator_env):
    """Verifies measurement of subsystem availability."""
    evaluator = mock_evaluator_env["evaluator"]

    reliability_metric = evaluator.evaluate_execution_reliability()
    assert reliability_metric.total_subsystems >= 28
    assert reliability_metric.healthy_count >= 25
    assert reliability_metric.reliability_pct >= 90.0


def test_generate_recommendations_thresholds(mock_evaluator_env):
    """Verifies generation of actionable improvement recommendations when metrics exceed thresholds."""
    evaluator = mock_evaluator_env["evaluator"]

    # Simulated degraded metrics
    high_mem_hardware = HardwareResourceSnapshot(
        timestamp="2026-11-24T00:00:00Z",
        cpu_percent=15.0,
        ram_used_mb=6000.0,
        ram_total_mb=8000.0,
        ram_percent=75.0,
        process_rss_mb=420.0,  # Exceeds 350 MB limit
        process_threads=12,
    )

    low_acc_reasoning = ReasoningAccuracyMetric(
        total_benchmarks=10,
        passed_count=8,
        failed_count=2,
        accuracy_pct=80.0,  # Below 95% threshold
        average_latency_ms=65.0,  # Above 50ms latency target
    )

    degraded_reliability = ExecutionReliabilityMetric(
        total_subsystems=28,
        healthy_count=26,
        degraded_count=2,
        offline_count=0,
        reliability_pct=92.8,
    )

    clean_quality = ResponseQualityMetric(3, 100.0, 100.0, 100.0, 100.0)

    recs = evaluator.generate_recommendations(
        hardware=high_mem_hardware,
        quality=clean_quality,
        reasoning=low_acc_reasoning,
        reliability=degraded_reliability,
    )

    categories = [r.category for r in recs]
    assert "memory" in categories
    assert "reasoning" in categories
    assert "reliability" in categories


def test_run_full_evaluation_and_persistence(mock_evaluator_env):
    """Verifies complete evaluation execution, report generation, and persistence."""
    evaluator = mock_evaluator_env["evaluator"]

    report = evaluator.run_full_evaluation(version="1.0.0")
    assert report.report_id.startswith("eval_")
    assert report.version == "1.0.0"
    assert report.overall_score >= 80.0
    assert len(report.recommendations) >= 1

    # Check persistence
    saved_reports = evaluator.list_reports()
    assert len(saved_reports) == 1
    assert saved_reports[0].report_id == report.report_id

    latest = evaluator.get_latest_report()
    assert latest is not None
    assert latest.report_id == report.report_id


def test_compare_performance_regression_detection(mock_evaluator_env):
    """Verifies regression detection when comparing baseline and candidate reports."""
    evaluator = mock_evaluator_env["evaluator"]

    # Baseline report (1.0.0)
    base_report = evaluator.run_full_evaluation(version="1.0.0")

    # Degraded candidate report (1.0.1)
    cand_hardware = HardwareResourceSnapshot(
        timestamp="2026-11-24T00:00:00Z",
        cpu_percent=10.0,
        ram_used_mb=4000.0,
        ram_total_mb=8000.0,
        ram_percent=50.0,
        process_rss_mb=base_report.hardware.process_rss_mb + 80.0,  # +80 MB (>50 MB limit)
        process_threads=8,
    )
    cand_reasoning = ReasoningAccuracyMetric(
        total_benchmarks=9,
        passed_count=7,
        failed_count=2,
        accuracy_pct=77.8,  # Regressed
        average_latency_ms=base_report.reasoning_accuracy.average_latency_ms * 2.0,  # +100% latency (>25%)
    )
    cand_report = PerformanceEvaluationReport(
        report_id="eval_candidate",
        version="1.0.1",
        timestamp="2026-11-24T01:00:00Z",
        overall_score=70.0,
        hardware=cand_hardware,
        response_quality=base_report.response_quality,
        reasoning_accuracy=cand_reasoning,
        execution_reliability=base_report.execution_reliability,
        recommendations=[],
    )

    comparison = evaluator.compare_performance(base_report, cand_report)
    assert comparison.has_regression is True
    assert len(comparison.regressions) >= 2
    assert any("accuracy regressed" in r for r in comparison.regressions)
    assert any("Latency slowed down" in r for r in comparison.regressions)
    assert any("Memory RSS consumption increased" in r for r in comparison.regressions)
