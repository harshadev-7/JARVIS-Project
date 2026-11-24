"""Unit tests for J.A.R.V.I.S. Phase 32 — Self-Code Generation.
============================================================
Tests Dimension 4 of the Self-Improvement System:
1. Synthesizing targeted code improvements based on detected issues
2. Emitting unified diff (.patch) files without modifying active source code
3. Generating beginner-friendly modification rationales
4. Managing isolated Development Sandboxes (Environment B)
5. Verifying syntax validity inside sandbox before admitting proposals
6. Strictly preserving active production files in Environment A (Stable)
"""

import json
import pytest
from pathlib import Path

from core.self_evolution.self_code_generator import (
    SelfCodeGenerator,
    CodeImprovementProposal,
    SandboxEnvironmentInfo,
)
from core.self_evolution.self_code_analyzer import CodeIssue, CodeIssueType, IssueSeverity


@pytest.fixture
def temp_code_generator(tmp_path):
    """Provides an isolated SelfCodeGenerator with temp workspace, evolution, and sandbox paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evolution = tmp_path / "evolution"
    sandbox = tmp_path / "sandbox"
    generator = SelfCodeGenerator(workspace_dir=workspace, evolution_dir=evolution, sandbox_root=sandbox)
    generator.initialize()
    return generator


def test_self_code_generator_initialization(temp_code_generator):
    """Verifies that patch and sandbox directories are initialized."""
    assert temp_code_generator.workspace_dir.exists()
    assert temp_code_generator.evolution_dir.exists()
    assert temp_code_generator.patches_dir.exists()
    assert temp_code_generator.sandbox_root.exists()


def test_generate_improvement_proposal_bare_except(temp_code_generator):
    """
    Verifies that a bare except handler is refactored into an explicit handler,
    a unified diff patch is generated, and the active production file is left untouched.
    """
    original_code = (
        "def query_database(record_id: int):\n"
        "    try:\n"
        "        data = fetch_record(record_id)\n"
        "    except:\n"
        "        data = None\n"
        "    return data\n"
    )
    target_rel = "db_service.py"
    target_file = temp_code_generator.workspace_dir / target_rel
    target_file.write_text(original_code, encoding="utf-8")

    issue = CodeIssue(
        issue_id="iss_bare_except_1",
        file_path=target_rel,
        line_number=4,
        issue_type=CodeIssueType.BUG_RISK.value,
        severity=IssueSeverity.MEDIUM.value,
        title="Bare Exception Handler",
        description="Bare except catches all signals.",
        recommendation="Use explicit except Exception as exc:",
    )

    proposal = temp_code_generator.generate_improvement_proposal(target_rel, issue)

    assert isinstance(proposal, CodeImprovementProposal)
    assert proposal.status == "staged"
    assert "except Exception as exc:" in proposal.after_snippet
    assert "except:" in proposal.before_snippet
    assert "KeyboardInterrupt" in proposal.rationale
    assert "--- a/db_service.py" in proposal.patch_content
    assert "+++ b/db_service.py" in proposal.patch_content

    # Verify patch file on disk
    patch_path = temp_code_generator.patches_dir / f"{proposal.proposal_id}.patch"
    assert patch_path.exists()

    # STRICT SAFETY CHECK: Original active file is 100% UNTOUCHED
    current_content = target_file.read_text(encoding="utf-8")
    assert current_content == original_code


def test_generate_improvement_proposal_mutable_default(temp_code_generator):
    """
    Verifies that a mutable default argument is safely refactored to None,
    with an educational explanation of shared state in Python.
    """
    original_code = (
        "def append_log(message: str, buffer: list = []):\n"
        "    buffer.append(message)\n"
        "    return buffer\n"
    )
    target_rel = "logger_service.py"
    target_file = temp_code_generator.workspace_dir / target_rel
    target_file.write_text(original_code, encoding="utf-8")

    issue = CodeIssue(
        issue_id="iss_mutable_default_1",
        file_path=target_rel,
        line_number=1,
        issue_type=CodeIssueType.BUG_RISK.value,
        severity=IssueSeverity.HIGH.value,
        title="Mutable Default Argument",
        description="Mutable default argument buffer shares state.",
        recommendation="Use None as default.",
    )

    proposal = temp_code_generator.generate_improvement_proposal(target_rel, issue)

    assert isinstance(proposal, CodeImprovementProposal)
    assert "buffer=None" in proposal.after_snippet
    assert "evaluated only once" in proposal.rationale.lower()

    # Original file must remain untouched
    assert target_file.read_text(encoding="utf-8") == original_code


def test_create_isolated_sandbox(temp_code_generator):
    """Verifies that an isolated development sandbox environment is created."""
    sandbox_info = temp_code_generator.create_isolated_sandbox("test_dev_sandbox")

    assert isinstance(sandbox_info, SandboxEnvironmentInfo)
    assert sandbox_info.is_isolated is True
    assert Path(sandbox_info.sandbox_path).exists()
    assert sandbox_info.stable_path == str(temp_code_generator.workspace_dir)


def test_apply_patch_to_sandbox_syntax_verification(temp_code_generator):
    """
    Verifies that a proposed patch applies cleanly inside the development sandbox,
    passes AST syntax verification, and leaves production code unchanged.
    """
    original_code = (
        "def compute_score(values: list):\n"
        "    try:\n"
        "        return sum(values)\n"
        "    except:\n"
        "        return 0\n"
    )
    target_rel = "analytics.py"
    target_file = temp_code_generator.workspace_dir / target_rel
    target_file.write_text(original_code, encoding="utf-8")

    issue = CodeIssue(
        issue_id="iss_bare_except_2",
        file_path=target_rel,
        line_number=4,
        issue_type=CodeIssueType.BUG_RISK.value,
        severity=IssueSeverity.MEDIUM.value,
        title="Bare Exception Handler",
        description="Bare except catches all signals.",
        recommendation="Use explicit except Exception as exc:",
    )

    proposal = temp_code_generator.generate_improvement_proposal(target_rel, issue)
    apply_result = temp_code_generator.apply_patch_to_sandbox(proposal, "test_sandbox")

    assert apply_result["success"] is True
    assert apply_result["syntax_valid"] is True
    assert proposal.status == "applied_in_sandbox"

    # Verify patched file exists in sandbox with replacement
    sandbox_file = Path(apply_result["sandbox_file"])
    assert sandbox_file.exists()
    sandbox_content = sandbox_file.read_text(encoding="utf-8")
    assert "except Exception as exc:" in sandbox_content

    # STRICT SAFETY: Original workspace file must still have original code!
    assert target_file.read_text(encoding="utf-8") == original_code


def test_apply_invalid_patch_rejection(temp_code_generator):
    """Verifies that a patch that would cause a syntax error is rejected."""
    code = "def valid_function():\n    return 42\n"
    target_rel = "valid_module.py"
    target_file = temp_code_generator.workspace_dir / target_rel
    target_file.write_text(code, encoding="utf-8")

    # Manually crafted proposal with broken syntax in after_snippet
    proposal = CodeImprovementProposal(
        proposal_id="prop_broken_syntax",
        target_file=target_rel,
        issue_id=None,
        title="Broken Syntax Test",
        rationale="Testing syntax error rejection gate.",
        before_snippet="return 42",
        after_snippet="return def broken syntax (((",
        patch_content="",
    )

    apply_result = temp_code_generator.apply_patch_to_sandbox(proposal, "broken_sandbox")

    assert apply_result["success"] is False
    assert apply_result["syntax_valid"] is False
    assert proposal.status == "rejected"
    assert "syntax" in apply_result["error"].lower()


def test_revert_sandbox(temp_code_generator):
    """Verifies clearing and resetting the development sandbox."""
    temp_code_generator.create_isolated_sandbox("sandbox_to_clear")
    cleared = temp_code_generator.revert_sandbox("sandbox_to_clear")
    assert cleared is True
    assert not (temp_code_generator.sandbox_root / "sandbox_to_clear").exists()


def test_generate_proposal_markdown(temp_code_generator):
    """Verifies human-readable markdown generation for proposal reviews."""
    proposal = CodeImprovementProposal(
        proposal_id="prop_preview_01",
        target_file="core/test.py",
        issue_id="iss_01",
        title="Refactor time.sleep to asyncio.sleep",
        rationale="Prevents event loop blocking in async coroutines.",
        before_snippet="time.sleep(1)",
        after_snippet="await asyncio.sleep(1)",
        patch_content="--- a/core/test.py\n+++ b/core/test.py\n@@ -1,1 +1,1 @@\n-time.sleep(1)\n+await asyncio.sleep(1)",
    )

    md = temp_code_generator.generate_proposal_markdown(proposal)
    assert "Code Improvement Proposal" in md
    assert "core/test.py" in md
    assert "Modification Rationale" in md
    assert "Unified Diff" in md
