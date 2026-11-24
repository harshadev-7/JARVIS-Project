"""Unit Tests for Phase 36: Controlled Self-Update Controller
============================================================
Tests end-to-end update packaging, test gating, explicit user approval,
atomic promotion, post-update smoke verification, and auto-rollback.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.self_evolution.self_update_controller import (
    SelfUpdateController,
    UpdatePackage,
    UpdateStatus,
    UpdateExecutionResult,
)
from core.self_evolution.self_code_generator import SelfCodeGenerator
from core.self_evolution.self_test_runner import (
    SelfTestRunner,
    CandidateTestEvaluation,
    TestSuiteExecutionResult,
    TestOutcome,
)
from core.self_evolution.self_version_manager import SelfVersionManager, SemanticVersion
from core.self_evolution.self_backup_recovery import SelfBackupRecoveryManager


def _make_suite_result(total: int = 5, passed: int = 5, failed: int = 0) -> TestSuiteExecutionResult:
    return TestSuiteExecutionResult(
        run_id="run_test",
        test_type="unit",
        total_tests=total,
        passed_count=passed,
        failed_count=failed,
        skipped_count=0,
        duration_sec=0.1,
        test_cases=[],
        exit_code=0 if failed == 0 else 1,
        success=(failed == 0),
        summary=f"{passed}/{total} passed",
    )


@pytest.fixture
def mock_update_env():
    """Sets up an isolated environment with mocked sub-managers for testing the controller."""
    temp_dir = tempfile.mkdtemp(prefix="jarvis_update_test_")
    repo_path = Path(temp_dir) / "repo"
    sandbox_path = Path(temp_dir) / "sandbox"
    backup_path = Path(temp_dir) / "backups"

    repo_path.mkdir(parents=True)
    sandbox_path.mkdir(parents=True)
    backup_path.mkdir(parents=True)

    # Base file in repo
    src_file = repo_path / "core" / "sample.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text("def run():\n    return 1\n", encoding="utf-8")

    # Modified file in sandbox
    sb_file = sandbox_path / "core" / "sample.py"
    sb_file.parent.mkdir(parents=True)
    sb_file.write_text("def run():\n    return 2\n", encoding="utf-8")

    # Mock Code Generator
    code_gen = MagicMock(spec=SelfCodeGenerator)
    code_gen.create_isolated_sandbox.return_value = MagicMock(sandbox_path=str(sandbox_path))
    code_gen.apply_patch_to_sandbox.return_value = {
        "success": True,
        "modified_files": ["core/sample.py"],
    }

    # Mock Test Runner
    test_runner = MagicMock(spec=SelfTestRunner)
    test_runner.run_tests.return_value = _make_suite_result(5, 5, 0)
    test_runner.evaluate_candidate_release.return_value = CandidateTestEvaluation(
        candidate_id="pkg_test",
        decision="ACCEPTED_FOR_RELEASE",
        passed_all_tests=True,
        has_zero_regressions=True,
        performance_acceptable=True,
        summary="Candidate ACCEPTED: All 5 tests passed",
        details={},
    )

    # Real Version Manager & Backup Manager on isolated folders
    version_mgr = SelfVersionManager(workspace_dir=repo_path, evolution_dir=repo_path / "evolution")
    backup_mgr = SelfBackupRecoveryManager(repo_root=repo_path, backup_dir=backup_path)

    # Default healthy startup verifier
    startup_fn = MagicMock(return_value=True)

    controller = SelfUpdateController(
        repo_root=repo_path,
        code_gen=code_gen,
        test_runner=test_runner,
        version_mgr=version_mgr,
        backup_mgr=backup_mgr,
        startup_verifier_fn=startup_fn,
    )

    yield {
        "temp_dir": temp_dir,
        "repo_path": repo_path,
        "sandbox_path": sandbox_path,
        "controller": controller,
        "test_runner": test_runner,
        "version_mgr": version_mgr,
        "backup_mgr": backup_mgr,
        "startup_fn": startup_fn,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_prepare_update_success(mock_update_env):
    """Verifies that an update package with passing tests moves to PENDING_APPROVAL."""
    controller = mock_update_env["controller"]

    pkg = controller.prepare_update(
        patch_ids=["patch_001"],
        bump_type="patch",
        title="Refactor Math Kernel",
        summary="Optimized math evaluation routines.",
    )

    assert pkg.status == UpdateStatus.PENDING_APPROVAL
    assert pkg.target_version == "1.0.1"
    assert pkg.test_passed_count == 5
    assert "core/sample.py" in pkg.modified_files


def test_prepare_update_rejection_on_test_failure(mock_update_env):
    """Verifies that test regressions reject the candidate package immediately."""
    controller = mock_update_env["controller"]
    test_runner = mock_update_env["test_runner"]

    test_runner.run_tests.return_value = _make_suite_result(5, 3, 2)
    test_runner.evaluate_candidate_release.return_value = CandidateTestEvaluation(
        candidate_id="pkg_broken",
        decision="REJECTED_TEST_FAILURE",
        passed_all_tests=False,
        has_zero_regressions=False,
        performance_acceptable=True,
        summary="Candidate REJECTED: 2 test(s) failed in test suite.",
        details={},
    )

    pkg = controller.prepare_update(
        patch_ids=["patch_broken"],
        bump_type="patch",
    )

    assert pkg.status == UpdateStatus.TESTS_FAILED
    assert "Test gate rejected" in pkg.error_message


def test_user_approval_workflow(mock_update_env):
    """Verifies the explicit human authorization gate."""
    controller = mock_update_env["controller"]

    pkg = controller.prepare_update(patch_ids=["patch_002"])
    assert pkg.status == UpdateStatus.PENDING_APPROVAL

    # Approve
    approved = controller.approve_update(pkg.package_id, approver="LeadDeveloper")
    assert approved is True

    updated_pkg = next(p for p in controller.list_packages() if p.package_id == pkg.package_id)
    assert updated_pkg.status == UpdateStatus.APPROVED
    assert updated_pkg.approved_by == "LeadDeveloper"
    assert updated_pkg.approved_at is not None


def test_user_rejection_workflow(mock_update_env):
    """Verifies rejecting an update package."""
    controller = mock_update_env["controller"]

    pkg = controller.prepare_update(patch_ids=["patch_003"])
    rejected = controller.reject_update(pkg.package_id, reason="Performance not convincing")
    assert rejected is True

    updated_pkg = next(p for p in controller.list_packages() if p.package_id == pkg.package_id)
    assert updated_pkg.status == UpdateStatus.REJECTED
    assert updated_pkg.rejection_reason == "Performance not convincing"


def test_apply_unapproved_package_rejected(mock_update_env):
    """Verifies that applying an unapproved package is safely refused."""
    controller = mock_update_env["controller"]

    pkg = controller.prepare_update(patch_ids=["patch_unapproved"])
    res = controller.apply_update(pkg.package_id)

    assert res.success is False
    assert "Cannot apply unapproved package" in res.error


def test_apply_approved_update_success(mock_update_env):
    """Verifies complete end-to-end promotion to active workspace."""
    controller = mock_update_env["controller"]
    repo_path = mock_update_env["repo_path"]
    version_mgr = mock_update_env["version_mgr"]

    pkg = controller.prepare_update(patch_ids=["patch_004"])
    controller.approve_update(pkg.package_id)

    res = controller.apply_update(pkg.package_id)

    assert res.success is True
    assert res.status == UpdateStatus.COMPLETED
    assert res.files_applied == 1
    assert res.safety_backup_id is not None
    assert version_mgr.get_current_version() == "1.0.1"

    # Workspace file should now contain sandbox content (return 2)
    sample_file = repo_path / "core" / "sample.py"
    assert "return 2" in sample_file.read_text(encoding="utf-8")


def test_apply_update_startup_failure_triggers_auto_rollback(mock_update_env):
    """Verifies that if post-update startup smoke checks fail, automatic rollback is executed."""
    controller = mock_update_env["controller"]
    repo_path = mock_update_env["repo_path"]
    startup_fn = mock_update_env["startup_fn"]

    # Configure startup smoke verifier to simulate startup crash
    startup_fn.return_value = False

    pkg = controller.prepare_update(patch_ids=["patch_crash"])
    controller.approve_update(pkg.package_id)

    res = controller.apply_update(pkg.package_id)

    assert res.success is False
    assert res.rollback_triggered is True
    assert res.status == UpdateStatus.ROLLED_BACK

    # Production file must have been rolled back to original content (return 1)
    sample_file = repo_path / "core" / "sample.py"
    assert "return 1" in sample_file.read_text(encoding="utf-8")


def test_manual_rollback(mock_update_env):
    """Verifies operator manual rollback of an applied update."""
    controller = mock_update_env["controller"]
    repo_path = mock_update_env["repo_path"]
    version_mgr = mock_update_env["version_mgr"]

    pkg = controller.prepare_update(patch_ids=["patch_manual_rb"])
    controller.approve_update(pkg.package_id)
    controller.apply_update(pkg.package_id)

    assert "return 2" in (repo_path / "core" / "sample.py").read_text(encoding="utf-8")

    # Now execute manual rollback
    rb_res = controller.manual_rollback(pkg.package_id, reason="Unexpected edge case in prod")
    assert rb_res.success is True
    assert rb_res.rollback_triggered is True
    assert version_mgr.get_current_version() == "1.0.0"

    # Restored to original
    assert "return 1" in (repo_path / "core" / "sample.py").read_text(encoding="utf-8")
