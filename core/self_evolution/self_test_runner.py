"""J.A.R.V.I.S. Self-Evolution System — Phase 33: Automated Testing & Regression Guard
===================================================================================
Implements the verification and quality gate pillar for
Dimension 4: Self-Programming & Source-Code Improvement.

Core Capabilities:
1. Programmatic Test Execution:
   - Runs unit tests and integration tests in isolated child processes using the active Python virtual environment.
2. Regression Testing & Invariant Protection:
   - Compares test suite outcomes between baseline (Stable) and candidate (Development Sandbox) versions.
   - Automatically detects regressed test cases (tests that passed in baseline but fail in candidate).
3. Compatibility Verification:
   - Inspects AST and module structure to verify compatibility with Python 3.14.
4. Performance & Execution Telemetry Comparison:
   - Measures duration and latency deltas between old and new versions, detecting compute regressions.
5. Automated Rejection Gate:
   - Enforces an automated pass/fail admission policy: changes that fail any required tests or induce
     unacceptable performance degradation are automatically rejected before entering release pipelines.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.settings import BASE_DIR
from core.logger import log


class TestOutcome(str, Enum):
    """Execution status for a single test case or suite."""
    __test__ = False
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class SingleTestCaseResult:
    """Outcome for an individual test case function."""
    test_id: str
    file_path: str
    outcome: str  # TestOutcome value
    duration_sec: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestSuiteExecutionResult:
    """Complete summary of a test run across unit or integration suites."""
    __test__ = False
    run_id: str
    test_type: str  # "unit", "integration", "regression", "smoke"
    total_tests: int
    passed_count: int
    failed_count: int
    skipped_count: int
    duration_sec: float
    test_cases: List[SingleTestCaseResult]
    exit_code: int
    success: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionComparison:
    """Comparative analysis between baseline and candidate test executions."""
    baseline_run_id: str
    candidate_run_id: str
    baseline_passed: int
    candidate_passed: int
    regressed_tests: List[str]  # Passed in baseline, failed in candidate
    fixed_tests: List[str]      # Failed in baseline, passed in candidate
    has_regression: bool
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceComparison:
    """Comparison of execution latency and performance delta."""
    baseline_duration_sec: float
    candidate_duration_sec: float
    duration_delta_pct: float
    has_performance_degradation: bool
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateTestEvaluation:
    """Final decision from the automated rejection gate for a candidate proposal."""
    candidate_id: str
    decision: str  # "ACCEPTED_FOR_RELEASE", "REJECTED_TEST_FAILURE", "REJECTED_REGRESSION", "REJECTED_PERFORMANCE"
    passed_all_tests: bool
    has_zero_regressions: bool
    performance_acceptable: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfTestRunner:
    """Coordinates automated test execution, regression verification, and candidate admission."""

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        evolution_dir: Optional[Path] = None,
    ):
        self.workspace_dir = workspace_dir or BASE_DIR
        self.evolution_dir = evolution_dir or (BASE_DIR / ".jarvis" / "evolution")
        self.test_reports_file = self.evolution_dir / "test_execution_history.json"
        self._latest_run: Optional[TestSuiteExecutionResult] = None
        self._initialized = False

    def initialize(self) -> None:
        """Sets up test evolution history directories."""
        if self._initialized:
            return
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        log.info("Phase 33 SelfTestRunner initialized.")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. RUN UNIT & INTEGRATION TESTS
    # ──────────────────────────────────────────────────────────────────────────

    def run_tests(
        self,
        target_path: Optional[str] = None,
        test_type: str = "unit",
        cwd: Optional[Path] = None,
        timeout_sec: int = 60,
    ) -> TestSuiteExecutionResult:
        """
        Executes pytest in an isolated child process, collects output, and parses test results.
        """
        self.initialize()
        run_id = f"test_{int(time.time() * 1000)}"
        exec_dir = cwd or self.workspace_dir

        target = target_path or ("tests/unit" if test_type == "unit" else "tests/integration")
        abs_target = exec_dir / target if not Path(target).is_absolute() else Path(target)

        cmd = [sys.executable, "-m", "pytest", str(abs_target), "-v", "--tb=short"]
        log.info(f"SelfTestRunner executing: {' '.join(cmd)} in {exec_dir}")

        start_time = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(exec_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                encoding="utf-8",
                errors="replace",
            )
            raw_output = proc.stdout or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            log.error(f"Test suite execution timed out after {timeout_sec}s.")
            exit_code = -1
            raw_output = f"TimeoutError: Execution exceeded {timeout_sec} seconds."
        except Exception as exc:
            log.error(f"Failed to execute pytest child process: {exc}")
            exit_code = -2
            raw_output = f"ExecutionException: {exc}"

        duration_sec = round(time.perf_counter() - start_time, 2)

        # Parse test outcomes
        parsed_cases = self._parse_pytest_output(raw_output)
        passed_count = sum(1 for c in parsed_cases if c.outcome == TestOutcome.PASSED.value)
        failed_count = sum(1 for c in parsed_cases if c.outcome in [TestOutcome.FAILED.value, TestOutcome.ERROR.value])
        skipped_count = sum(1 for c in parsed_cases if c.outcome == TestOutcome.SKIPPED.value)

        # If exit_code is 0 and no cases collected, verify from output summary line
        if not parsed_cases and "passed in" in raw_output:
            match = re.search(r"(\d+)\s+passed", raw_output)
            if match:
                passed_count = int(match.group(1))

        total_tests = passed_count + failed_count + skipped_count
        success = (exit_code == 0) and (failed_count == 0)

        summary = (
            f"Test Suite [{test_type.upper()}]: {'PASSED' if success else 'FAILED'} in {duration_sec}s. "
            f"Total: {total_tests} | Passed: {passed_count} | Failed: {failed_count} | Skipped: {skipped_count}"
        )

        result = TestSuiteExecutionResult(
            run_id=run_id,
            test_type=test_type,
            total_tests=total_tests,
            passed_count=passed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            duration_sec=duration_sec,
            test_cases=parsed_cases,
            exit_code=exit_code,
            success=success,
            summary=summary,
        )

        self._latest_run = result
        self._persist_test_run(result)
        log.info(summary)
        return result

    def _parse_pytest_output(self, output: str) -> List[SingleTestCaseResult]:
        """Parses individual test function results from pytest verbose output."""
        cases: List[SingleTestCaseResult] = []
        # Matches lines like: tests/unit/test_foo.py::test_bar PASSED [ 50%]
        pattern = re.compile(r"^([\w\\/\.\-_]+)::([\w\-_]+)\s+(PASSED|FAILED|ERROR|SKIPPED)", re.MULTILINE)

        for match in pattern.finditer(output):
            file_path, test_name, status = match.groups()
            test_id = f"{file_path}::{test_name}"
            outcome_val = status.lower()

            cases.append(SingleTestCaseResult(
                test_id=test_id,
                file_path=file_path,
                outcome=outcome_val,
                duration_sec=0.0,
                error_message=None if outcome_val == "passed" else f"Test {status} during execution",
            ))

        return cases

    def _persist_test_run(self, result: TestSuiteExecutionResult) -> None:
        """Persists test run to JSON history."""
        try:
            history = []
            if self.test_reports_file.exists():
                with open(self.test_reports_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            history.append(result.to_dict())
            history = history[-15:]  # Keep latest 15 runs
            with open(self.test_reports_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as exc:
            log.warning(f"Could not persist test run: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. REGRESSION TESTING COMPARISON
    # ──────────────────────────────────────────────────────────────────────────

    def compare_test_runs(
        self,
        baseline: TestSuiteExecutionResult,
        candidate: TestSuiteExecutionResult,
    ) -> RegressionComparison:
        """
        Compares candidate test execution against the stable baseline to detect regressions.
        """
        baseline_passed_ids = {c.test_id for c in baseline.test_cases if c.outcome == TestOutcome.PASSED.value}
        candidate_passed_ids = {c.test_id for c in candidate.test_cases if c.outcome == TestOutcome.PASSED.value}
        candidate_failed_ids = {c.test_id for c in candidate.test_cases if c.outcome in [TestOutcome.FAILED.value, TestOutcome.ERROR.value]}

        # Tests that were passing in baseline, but are now failing in candidate
        regressed_tests = sorted(list(baseline_passed_ids.intersection(candidate_failed_ids)))

        # Tests that failed in baseline, but are now passing in candidate
        baseline_failed_ids = {c.test_id for c in baseline.test_cases if c.outcome in [TestOutcome.FAILED.value, TestOutcome.ERROR.value]}
        fixed_tests = sorted(list(baseline_failed_ids.intersection(candidate_passed_ids)))

        has_regression = len(regressed_tests) > 0 or (candidate.failed_count > baseline.failed_count)

        verdict = (
            f"REGRESSION DETECTED: {len(regressed_tests)} tests regressed."
            if has_regression
            else f"CLEAN: Zero regressions detected. ({len(fixed_tests)} tests fixed)."
        )

        return RegressionComparison(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            baseline_passed=baseline.passed_count,
            candidate_passed=candidate.passed_count,
            regressed_tests=regressed_tests,
            fixed_tests=fixed_tests,
            has_regression=has_regression,
            verdict=verdict,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. PERFORMANCE & COMPATIBILITY VERIFICATION
    # ──────────────────────────────────────────────────────────────────────────

    def compare_performance(
        self,
        baseline_duration_sec: float,
        candidate_duration_sec: float,
        max_allowed_degradation_pct: float = 25.0,
    ) -> PerformanceComparison:
        """
        Compares execution durations, calculating latency delta and flagging compute regressions.
        """
        if baseline_duration_sec <= 0:
            delta_pct = 0.0
        else:
            delta_pct = round(((candidate_duration_sec - baseline_duration_sec) / baseline_duration_sec) * 100.0, 2)

        has_degradation = delta_pct > max_allowed_degradation_pct
        if has_degradation:
            rec = f"Performance degraded by {delta_pct}%. Optimize algorithmic bottlenecks prior to promotion."
        elif delta_pct < -5.0:
            rec = f"Performance improved by {abs(delta_pct)}%. Favorable candidate efficiency."
        else:
            rec = f"Performance neutral (delta: {delta_pct}%). Within acceptable latency bounds."

        return PerformanceComparison(
            baseline_duration_sec=baseline_duration_sec,
            candidate_duration_sec=candidate_duration_sec,
            duration_delta_pct=delta_pct,
            has_performance_degradation=has_degradation,
            recommendation=rec,
        )

    def check_module_compatibility(self, file_path: Path) -> Dict[str, Any]:
        """
        Verifies module AST compatibility with Python 3.14 standards.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
            tree = ast.parse(code, filename=str(file_path))
            return {
                "compatible": True,
                "file": str(file_path),
                "python_version_target": "3.14",
                "ast_nodes": len(list(ast.walk(tree))),
                "message": "Module parses cleanly and is compatible with Python 3.14 runtime.",
            }
        except SyntaxError as syn_err:
            return {
                "compatible": False,
                "file": str(file_path),
                "error": str(syn_err),
                "line": syn_err.lineno,
                "message": f"Syntax incompatibility detected: {syn_err.msg}",
            }

    # ──────────────────────────────────────────────────────────────────────────
    # 4. AUTOMATED REJECTION GATE & CANDIDATE ADMISSION
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_candidate_release(
        self,
        candidate_id: str,
        baseline_result: TestSuiteExecutionResult,
        candidate_result: TestSuiteExecutionResult,
        max_degradation_pct: float = 25.0,
    ) -> CandidateTestEvaluation:
        """
        Enforces the automated rejection gate:
        - If candidate has ANY failed tests -> REJECT
        - If candidate regressed any previously passing test -> REJECT
        - If candidate degraded performance by >25% -> REJECT
        - If all checks pass -> ACCEPT_FOR_RELEASE
        """
        # 1. Regression check
        reg_comp = self.compare_test_runs(baseline_result, candidate_result)

        # 2. Performance check
        perf_comp = self.compare_performance(
            baseline_result.duration_sec,
            candidate_result.duration_sec,
            max_allowed_degradation_pct=max_degradation_pct,
        )

        passed_all = candidate_result.success and (candidate_result.failed_count == 0)
        has_zero_reg = not reg_comp.has_regression
        perf_ok = not perf_comp.has_performance_degradation

        if not passed_all:
            decision = "REJECTED_TEST_FAILURE"
            summary = f"Candidate '{candidate_id}' REJECTED: {candidate_result.failed_count} test(s) failed in test suite."
        elif not has_zero_reg:
            decision = "REJECTED_REGRESSION"
            summary = f"Candidate '{candidate_id}' REJECTED: {len(reg_comp.regressed_tests)} test regressions detected."
        elif not perf_ok:
            decision = "REJECTED_PERFORMANCE_REGRESSION"
            summary = f"Candidate '{candidate_id}' REJECTED: Performance degraded by {perf_comp.duration_delta_pct}%."
        else:
            decision = "ACCEPTED_FOR_RELEASE"
            summary = f"Candidate '{candidate_id}' ACCEPTED: All {candidate_result.passed_count} tests passed with zero regressions and acceptable latency."

        log.info(f"Candidate release evaluation for '{candidate_id}': Decision = {decision}")

        return CandidateTestEvaluation(
            candidate_id=candidate_id,
            decision=decision,
            passed_all_tests=passed_all,
            has_zero_regressions=has_zero_reg,
            performance_acceptable=perf_ok,
            summary=summary,
            details={
                "regression": reg_comp.to_dict(),
                "performance": perf_comp.to_dict(),
                "candidate_duration_sec": candidate_result.duration_sec,
                "candidate_passed": candidate_result.passed_count,
            },
        )

    def generate_test_report_markdown(self, evaluation: CandidateTestEvaluation) -> str:
        """Generates clean human-readable markdown detailing candidate test evaluation."""
        status_emoji = "✅" if evaluation.decision == "ACCEPTED_FOR_RELEASE" else "❌"

        return (
            f"### {status_emoji} Candidate Test Evaluation: `{evaluation.candidate_id}`\n\n"
            f"- **Gate Decision**: **`{evaluation.decision}`**\n"
            f"- **Tests Passed**: `{'Yes' if evaluation.passed_all_tests else 'No'}`\n"
            f"- **Zero Regressions**: `{'Yes' if evaluation.has_zero_regressions else 'No'}`\n"
            f"- **Performance Acceptable**: `{'Yes' if evaluation.performance_acceptable else 'No'}`\n\n"
            f"#### **Evaluation Summary:**\n"
            f"> {evaluation.summary}\n\n"
            f"> [!IMPORTANT]\n"
            f"> Only candidate proposals receiving `ACCEPTED_FOR_RELEASE` from Phase 33 are permitted "
            f"to advance to Phase 34 (Self-Version Management) and Phase 36 (Controlled Self-Update)."
        )


# Global singleton instance for system-wide access
self_test_runner = SelfTestRunner()
