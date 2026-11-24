"""Unit tests for J.A.R.V.I.S. Phase 34 — Self-Version Management.
================================================================
Tests Dimension 4 of the Self-Improvement System:
1. Semantic version parsing, comparison, and automated bumping
2. Baseline release v1.0.0 initialization and ledger persistence
3. Creating candidate releases with structured changelogs
4. Activating releases and promoting development staging to stable
5. Rollback target pointer tracking for safe reversions
6. Version comparison and architectural file deltas
7. Human-readable markdown version reporting
"""

import json
import pytest
from pathlib import Path

from core.self_evolution.self_version_manager import (
    SelfVersionManager,
    SemanticVersion,
    ReleaseRecord,
    VersionDiff,
)


@pytest.fixture
def temp_version_manager(tmp_path):
    """Provides an isolated SelfVersionManager with dedicated workspace and evolution paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evolution = tmp_path / "evolution"
    manager = SelfVersionManager(workspace_dir=workspace, evolution_dir=evolution)
    manager.initialize()
    return manager


def test_semantic_version_parsing_and_bumping():
    """Verifies SemVer parsing, patch/minor/major bumping, and comparison operations."""
    v1 = SemanticVersion.from_string("1.0.0")
    assert str(v1) == "1.0.0"
    assert v1.major == 1 and v1.minor == 0 and v1.patch == 0

    # Tag parsing
    v_rc = SemanticVersion.from_string("v2.1.4-rc2")
    assert str(v_rc) == "2.1.4-rc2"
    assert v_rc.major == 2 and v_rc.tag == "rc2"

    # Bumping
    v_patch = v1.bump_patch()
    assert str(v_patch) == "1.0.1"

    v_minor = v1.bump_minor()
    assert str(v_minor) == "1.1.0"

    v_major = v1.bump_major()
    assert str(v_major) == "2.0.0"

    # Comparison
    assert v1 < v_patch
    assert v_patch < v_minor
    assert v_minor < v_major
    assert v1 == SemanticVersion(1, 0, 0)


def test_self_version_manager_initialization(temp_version_manager):
    """Verifies that baseline v1.0.0 release is initialized and set as active."""
    assert temp_version_manager.get_current_version() == "1.0.0"
    releases = temp_version_manager.list_releases()
    assert len(releases) == 1
    base = releases[0]
    assert base.version == "1.0.0"
    assert base.is_active is True
    assert base.environment == "stable"
    assert temp_version_manager.releases_file.exists()


def test_create_candidate_release(temp_version_manager):
    """
    Verifies that creating a candidate release computes the next version,
    generates a structured changelog, and sets a rollback pointer.
    """
    candidate = temp_version_manager.create_candidate_release(
        bump_type="patch",
        title="Automated Bugfix Patch for Async Sleep",
        modified_files=["core/automation/desktop.py"],
        patch_ids=["patch_async_fix_001"],
        test_run_id="test_suite_run_42",
        test_pass_rate=100.0,
        environment="development_staging",
    )

    assert isinstance(candidate, ReleaseRecord)
    assert candidate.version == "1.0.1"
    assert candidate.environment == "development_staging"
    assert candidate.is_active is False  # Candidate is not yet active stable
    assert candidate.rollback_target_version == "1.0.0"
    assert "patch_async_fix_001" in candidate.changelog_markdown
    assert "100.0%" in candidate.changelog_markdown
    assert "core/automation/desktop.py" in candidate.changelog_markdown

    # Active version should still remain 1.0.0 until activated
    assert temp_version_manager.get_current_version() == "1.0.0"


def test_activate_release(temp_version_manager):
    """
    Verifies promoting a candidate release to the active stable release.
    """
    temp_version_manager.create_candidate_release(
        bump_type="patch",
        title="Performance Optimization",
        modified_files=["core/planner/deductive_solver.py"],
        patch_ids=["patch_perf_01"],
    )

    # Activate 1.0.1
    success = temp_version_manager.activate_release("1.0.1")
    assert success is True
    assert temp_version_manager.get_current_version() == "1.0.1"

    rel_101 = temp_version_manager.get_release("1.0.1")
    assert rel_101.is_active is True
    assert rel_101.environment == "stable"

    # Previous baseline should no longer be marked active
    rel_100 = temp_version_manager.get_release("1.0.0")
    assert rel_100.is_active is False


def test_get_rollback_target(temp_version_manager):
    """
    Verifies that get_rollback_target returns the previous stable version.
    """
    temp_version_manager.create_candidate_release(bump_type="patch", title="Candidate 1.0.1")
    temp_version_manager.activate_release("1.0.1")

    rollback = temp_version_manager.get_rollback_target("1.0.1")
    assert rollback is not None
    assert rollback.version == "1.0.0"


def test_compare_versions(temp_version_manager):
    """
    Verifies version diff comparison across files, patches, and test pass rates.
    """
    temp_version_manager.create_candidate_release(
        bump_type="minor",
        title="Neural Architecture Upgrades",
        modified_files=["custom_ai/model.py", "custom_ai/research/attention.py"],
        patch_ids=["patch_rope_01", "patch_gqa_02"],
        test_pass_rate=100.0,
    )

    diff = temp_version_manager.compare_versions("1.0.0", "1.1.0")
    assert isinstance(diff, VersionDiff)
    assert diff.from_version == "1.0.0"
    assert diff.to_version == "1.1.0"
    assert "custom_ai/model.py" in diff.files_modified
    assert "patch_rope_01" in diff.patches_applied
    assert "Comparison v1.0.0 -> v1.1.0" in diff.summary


def test_generate_version_report_markdown(temp_version_manager):
    """Verifies markdown report generation for CLI and HUD display."""
    md = temp_version_manager.generate_version_report_markdown()
    assert "J.A.R.V.I.S. Self-Version Management Status (Phase 34)" in md
    assert "Active Stable Release" in md
    assert "Rollback Target" in md
    assert "Recent Release History" in md
