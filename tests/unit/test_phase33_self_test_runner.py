"""Unit tests for J.A.R.V.I.S. Phase 33 — Automated Testing & Regression Guard.
=============================================================================
Tests Dimension 4 of the Self-Improvement System:
1. Programmatic test execution and outcome collection
2. Regression analysis comparing baseline vs. candidate test runs
3. Detection of regressed tests (tests previously passing that now fail)
4. Detection of performance regressions and execution latency deltas
5. AST module compatibility checks with Python 3.14
6. Automated candidate admission and rejection gates
"""

import json
import pytest
from pathlib import Path

from core.self_evolution.self_test_runner import (
    SelfTestRunner,
    TestOutcome,
    SingleTestCaseResult,
    TestSuiteExecutionResult,
    RegressionComparison,
    PerformanceComparison,
    CandidateTestEvaluation,
)


@pytest.fixture
def temp_test_runner(tmp_path):
    """Provides an isolated SelfTestRunner with dedicated workspace and evolution paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evolution = tmp_path / "evolution"
    runner = SelfTestRunner(workspace_dir=workspace, evolution_dir=evolution)
    runner.initialize()
    return runner


def test_self_test_runner_initialization(temp_test_runner):
    """Verifies test runner initialization and directories."""
    assert temp_test_runner.workspace_dir.exists()
    assert temp_test_runner.evolution_dir.exists()
    assert temp_test_runner._initialized is True


def test_parse_pytest_output(temp_test_runner):
    """Verifies parsing of verbose pytest output lines into structured result objects."""
    sample_output = (
        "tests/unit/test_math.py::test_addition PASSED [ 25%]\n"
        "tests/unit/test_math.py::test_subtraction PASSED [ 50%]\n"
        "tests/unit/test_math.py::test_division FAILED [ 75%]\n"
        "tests/unit/test_math.py::test_special SKIPPED [100%]\n"
    )

    cases = temp_test_runner._parse_pytest_output(sample_output)

    assert len(cases) == 4
    assert cases[0].test_id == "tests/unit/test_math.py::test_addition"
    assert cases[0].outcome == TestOutcome.PASSED.value

    assert cases[2].test_id == "tests/unit/test_math.py::test_division"
    assert cases[2].outcome == TestOutcome.FAILED.value
    assert cases[2].error_message is not None

    assert cases[3].test_id == "tests/unit/test_math.py::test_special"
    assert cases[3].outcome == TestOutcome.SKIPPED.value


def test_compare_test_runs_clean(temp_test_runner):
    """Verifies zero regressions when candidate passes all baseline tests."""
    case_1 = SingleTestCaseResult("test_a", "f.py", TestOutcome.PASSED.value)
    case_2 = SingleTestCaseResult("test_b", "f.py", TestOutcome.PASSED.value)

    baseline = TestSuiteExecutionResult(
        run_id="run_base",
        test_type="unit",
        total_tests=2,
        passed_count=2,
        failed_count=0,
        skipped_count=0,
        duration_sec=1.2,
        test_cases=[case_1, case_2],
        exit_code=0,
        success=True,
        summary="All passed",
    )

    candidate = TestSuiteExecutionResult(
        run_id="run_cand",
        test_type="unit",
        total_tests=2,
        passed_count=2,
        failed_count=0,
        skipped_count=0,
        duration_sec=1.1,
        test_cases=[case_1, case_2],
        exit_code=0,
        success=True,
        summary="All passed",
    )

    comp = temp_test_runner.compare_test_runs(baseline, candidate)

    assert isinstance(comp, RegressionComparison)
    assert comp.has_regression is False
    assert len(comp.regressed_tests) == 0
    assert "CLEAN" in comp.verdict


def test_compare_test_runs_detects_regression(temp_test_runner):
    """Verifies detection of regressed test cases when a test previously passed now fails."""
    case_1 = SingleTestCaseResult("test_alpha", "f.py", TestOutcome.PASSED.value)
    case_2_pass = SingleTestCaseResult("test_beta", "f.py", TestOutcome.PASSED.value)
    case_2_fail = SingleTestCaseResult("test_beta", "f.py", TestOutcome.FAILED.value, error_message="AssertionError")

    baseline = TestSuiteExecutionResult(
        run_id="run_base_2",
        test_type="unit",
        total_tests=2,
        passed_count=2,
        failed_count=0,
        skipped_count=0,
        duration_sec=1.0,
        test_cases=[case_1, case_2_pass],
        exit_code=0,
        success=True,
        summary="Baseline pass",
    )

    candidate = TestSuiteExecutionResult(
        run_id="run_cand_2",
        test_type="unit",
        total_tests=2,
        passed_count=1,
        failed_count=1,
        skipped_count=0,
        duration_sec=1.0,
        test_cases=[case_1, case_2_fail],
        exit_code=1,
        success=False,
        summary="Candidate failed",
    )

    comp = temp_test_runner.compare_test_runs(baseline, candidate)

    assert comp.has_regression is True
    assert "test_beta" in comp.regressed_tests
    assert "REGRESSION DETECTED" in comp.verdict


def test_compare_performance(temp_test_runner):
    """Verifies execution latency delta computation and degradation threshold flagging."""
    # Case A: Acceptable delta (+10%)
    perf_ok = temp_test_runner.compare_performance(1.0, 1.10, max_allowed_degradation_pct=25.0)
    assert perf_ok.has_performance_degradation is False
    assert perf_ok.duration_delta_pct == 10.0

    # Case B: Significant degradation (+50%)
    perf_bad = temp_test_runner.compare_performance(1.0, 1.50, max_allowed_degradation_pct=25.0)
    assert perf_bad.has_performance_degradation is True
    assert perf_bad.duration_delta_pct == 50.0
    assert "degraded" in perf_bad.recommendation.lower()


def test_evaluate_candidate_release_admission_rejection_gates(temp_test_runner):
    """
    Verifies the automated rejection gate:
    1. Clean candidate -> ACCEPTED_FOR_RELEASE
    2. Candidate with test failure -> REJECTED_TEST_FAILURE
    3. Candidate with performance regression -> REJECTED_PERFORMANCE_REGRESSION
    """
    case_p = SingleTestCaseResult("test_1", "f.py", TestOutcome.PASSED.value)
    case_f = SingleTestCaseResult("test_1", "f.py", TestOutcome.FAILED.value)

    baseline = TestSuiteExecutionResult("b1", "unit", 1, 1, 0, 0, 1.0, [case_p], 0, True, "OK")

    # 1. Clean Candidate
    clean_candidate = TestSuiteExecutionResult("c1", "unit", 1, 1, 0, 0, 1.05, [case_p], 0, True, "OK")
    eval_clean = temp_test_runner.evaluate_candidate_release("cand_v1", baseline, clean_candidate)
    assert eval_clean.decision == "ACCEPTED_FOR_RELEASE"
    assert eval_clean.passed_all_tests is True
    assert eval_clean.has_zero_regressions is True

    # 2. Failing Candidate
    failing_candidate = TestSuiteExecutionResult("c2", "unit", 1, 0, 1, 0, 1.0, [case_f], 1, False, "Failed")
    eval_fail = temp_test_runner.evaluate_candidate_release("cand_v2", baseline, failing_candidate)
    assert eval_fail.decision in ["REJECTED_TEST_FAILURE", "REJECTED_REGRESSION"]
    assert eval_fail.passed_all_tests is False

    # 3. Slow Candidate (>25% degradation)
    slow_candidate = TestSuiteExecutionResult("c3", "unit", 1, 1, 0, 0, 2.5, [case_p], 0, True, "OK")
    eval_slow = temp_test_runner.evaluate_candidate_release("cand_v3", baseline, slow_candidate, max_degradation_pct=20.0)
    assert eval_slow.decision == "REJECTED_PERFORMANCE_REGRESSION"
    assert eval_slow.performance_acceptable is False


def test_check_module_compatibility(temp_test_runner):
    """Verifies AST compatibility verification with Python 3.14."""
    # Valid Python file
    valid_file = temp_test_runner.workspace_dir / "valid_code.py"
    valid_file.write_text("def add(x: int, y: int) -> int:\n    return x + y\n", encoding="utf-8")
    compat = temp_test_runner.check_module_compatibility(valid_file)
    assert compat["compatible"] is True
    assert compat["python_version_target"] == "3.14"

    # Invalid Python file
    broken_file = temp_test_runner.workspace_dir / "broken_code.py"
    broken_file.write_text("def broken_func(:\n", encoding="utf-8")
    incompat = temp_test_runner.check_module_compatibility(broken_file)
    assert incompat["compatible"] is False
    assert "syntax" in incompat["message"].lower()


def test_generate_test_report_markdown(temp_test_runner):
    """Verifies markdown report generation for test evaluation gate."""
    eval_result = CandidateTestEvaluation(
        candidate_id="cand_release_001",
        decision="ACCEPTED_FOR_RELEASE",
        passed_all_tests=True,
        has_zero_regressions=True,
        performance_acceptable=True,
        summary="All tests passed cleanly.",
    )

    md = temp_test_runner.generate_test_report_markdown(eval_result)
    assert "Candidate Test Evaluation: `cand_release_001`" in md
    assert "ACCEPTED_FOR_RELEASE" in md
    assert "Gate Decision" in md
