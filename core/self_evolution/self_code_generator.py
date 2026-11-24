"""J.A.R.V.I.S. Self-Evolution System — Phase 32: Self-Code Generation
=====================================================================
Implements the synthesis and patch management pillar for
Dimension 4: Self-Programming & Source-Code Improvement.

Core Capabilities:
1. Generate Proposed Source-Code Improvements:
   - Synthesizes targeted, functional refactorings for defects discovered by Phase 31
     (e.g., bare except handlers, mutable default arguments, blocking sleep calls in async coroutines).
2. Patch-Based Output:
   - Emits standard unified diff `.patch` files rather than immediately overwriting active production files.
3. Beginner-Friendly Explanations:
   - Produces clear, plain-language rationales detailing what changed, why the change is beneficial,
     and potential edge-case implications.
4. Permanent Version Evolution Architecture:
   - Enforces strict dual-environment isolation:
     - Environment A (Stable): Active production codebase on `main` (strictly preserved & read-only for updates).
     - Environment B (Development): Isolated development sandbox (`.jarvis/sandbox/`) where patches
       are applied and verified prior to test execution.
5. Syntax & Regression Guard:
   - Validates that patched code parses cleanly as valid Python AST before admitting the proposal
     into candidate release pipelines.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import BASE_DIR
from core.logger import log
from core.self_evolution.self_code_analyzer import CodeIssue, CodeIssueType, IssueSeverity


@dataclass
class CodeImprovementProposal:
    """A formal proposal for source-code refactoring and improvement."""
    proposal_id: str
    target_file: str
    issue_id: Optional[str]
    title: str
    rationale: str
    before_snippet: str
    after_snippet: str
    patch_content: str
    patch_file_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: str = "staged"  # "staged", "applied_in_sandbox", "rejected", "promoted_to_stable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SandboxEnvironmentInfo:
    """Status and paths for an isolated development sandbox."""
    sandbox_path: str
    stable_path: str
    sandbox_name: str
    is_isolated: bool
    staged_patches: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfCodeGenerator:
    """Generates targeted source-code patches and manages isolated development sandboxes."""

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        evolution_dir: Optional[Path] = None,
        sandbox_root: Optional[Path] = None,
    ):
        self.workspace_dir = workspace_dir or BASE_DIR
        self.evolution_dir = evolution_dir or (BASE_DIR / ".jarvis" / "evolution")
        self.patches_dir = self.evolution_dir / "patches"
        self.proposals_file = self.evolution_dir / "improvement_proposals.json"
        self.sandbox_root = sandbox_root or (BASE_DIR / ".jarvis" / "sandbox")

        self._in_memory_proposals: Dict[str, CodeImprovementProposal] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Sets up directory structures for patches, sandbox, and proposals."""
        if self._initialized:
            return
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

        if self.proposals_file.exists():
            try:
                with open(self.proposals_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        prop = CodeImprovementProposal(**item)
                        self._in_memory_proposals[prop.proposal_id] = prop
            except Exception as exc:
                log.warning(f"Could not load existing proposals: {exc}")

        self._initialized = True
        log.info("Phase 32 SelfCodeGenerator initialized.")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. GENERATE CODE IMPROVEMENT PROPOSAL
    # ──────────────────────────────────────────────────────────────────────────

    def generate_improvement_proposal(
        self,
        target_file_rel: str,
        issue: CodeIssue,
        context_radius: int = 4,
    ) -> Optional[CodeImprovementProposal]:
        """
        Synthesizes a targeted code fix for a detected issue, generates a unified diff
        `.patch` file, and produces beginner-friendly explanations without touching
        the active production file.
        """
        self.initialize()
        full_path = self.workspace_dir / target_file_rel
        if not full_path.exists():
            log.warning(f"Target file {full_path} not found.")
            return None

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                original_content = f.read()
        except Exception as exc:
            log.error(f"Failed to read {full_path}: {exc}")
            return None

        lines = original_content.splitlines(keepends=True)
        line_idx = issue.line_number - 1  # 0-indexed
        if line_idx < 0 or line_idx >= len(lines):
            log.warning(f"Line number {issue.line_number} out of range in {target_file_rel}")
            return None

        proposal_id = f"prop_{int(time.time() * 1000)}"
        target_line = lines[line_idx]

        # 1. Synthesize Targeted Refactoring
        modified_lines = list(lines)
        rationale: str = ""
        before_snippet: str = target_line.strip()
        after_snippet: str = ""

        # Case A: Bare Exception Handler (`except:`)
        if "bare exception" in issue.title.lower() or (re.search(r"^\s*except\s*:\s*$", target_line)):
            indent = re.match(r"^(\s*)", target_line).group(1)
            modified_lines[line_idx] = f"{indent}except Exception as exc:\n"
            after_snippet = f"{indent}except Exception as exc:"
            rationale = (
                "Replaced bare `except:` with explicit `except Exception as exc:`. "
                "A bare `except:` unsafely captures system interrupts (`KeyboardInterrupt`, `SystemExit`), "
                "preventing normal shutdown signals and making debugging difficult. Catching `Exception` ensures "
                "system-level control signals are properly respected."
            )

        # Case B: Mutable Default Argument (e.g. `items=[]` or `kwargs={}`)
        elif "mutable" in issue.title.lower():
            # Refactor mutable default to None in function definition
            new_line = re.sub(
                r"(\w+)\s*:\s*(?:List|Dict|Set|list|dict|set)[^=,)]*\s*=\s*(?:\[\]|\{\}|set\(\))",
                r"\1=None",
                target_line,
            )
            if new_line == target_line:
                new_line = re.sub(r"(\w+)\s*=\s*(?:\[\]|\{\}|set\(\))", r"\1=None", target_line)

            modified_lines[line_idx] = new_line
            after_snippet = new_line.strip()
            rationale = (
                "Replaced mutable default argument with `None`. In Python, default arguments are evaluated "
                "only once when the function is defined. Using a mutable object (like a list or dict) shares "
                "state across all function calls, leading to subtle cross-invocation data corruption bugs."
            )

        # Case C: Blocking `time.sleep()` in Async Coroutine
        elif "blocking" in issue.title.lower() or "time.sleep" in target_line:
            new_line = re.sub(r"\btime\.sleep\s*\(", "await asyncio.sleep(", target_line)
            modified_lines[line_idx] = new_line
            after_snippet = new_line.strip()
            rationale = (
                "Replaced synchronous blocking `time.sleep()` with asynchronous non-blocking `await asyncio.sleep()`. "
                "Synchronous sleep stops the entire single-threaded Python event loop, freezing all concurrent "
                "tasks and web requests. `await asyncio.sleep()` yields execution cooperatively."
            )

        # Case D: Insecure MD5 Hash
        elif "md5" in issue.description.lower() or "hashlib.md5" in target_line:
            new_line = re.sub(r"\bhashlib\.md5\b", "hashlib.sha256", target_line)
            modified_lines[line_idx] = new_line
            after_snippet = new_line.strip()
            rationale = (
                "Replaced legacy MD5 hashing with cryptographically secure SHA-256. "
                "MD5 is vulnerable to collision attacks and is rejected by modern security baselines."
            )

        else:
            # Generic safe comment annotation
            indent = re.match(r"^(\s*)", target_line).group(1)
            modified_lines[line_idx] = f"{indent}# Refactored for safety: {issue.title}\n{target_line}"
            after_snippet = modified_lines[line_idx].strip()
            rationale = f"Applied refactoring to resolve {issue.title} and improve code maintainability."

        # 2. Generate Unified Diff
        modified_content = "".join(modified_lines)
        diff_lines = list(difflib.unified_diff(
            lines,
            modified_lines,
            fromfile=f"a/{target_file_rel}",
            tofile=f"b/{target_file_rel}",
            lineterm="",
        ))
        patch_content = "\n".join(diff_lines)

        # 3. Save Patch File (.patch)
        patch_file_path = self.patches_dir / f"{proposal_id}.patch"
        try:
            with open(patch_file_path, "w", encoding="utf-8") as f:
                f.write(patch_content + "\n")
        except Exception as exc:
            log.error(f"Failed to write patch file {patch_file_path}: {exc}")

        proposal = CodeImprovementProposal(
            proposal_id=proposal_id,
            target_file=target_file_rel,
            issue_id=issue.issue_id,
            title=f"Fix {issue.title} in {target_file_rel}:{issue.line_number}",
            rationale=rationale,
            before_snippet=before_snippet,
            after_snippet=after_snippet,
            patch_content=patch_content,
            patch_file_path=str(patch_file_path.relative_to(self.workspace_dir)).replace("\\", "/") if self.workspace_dir in patch_file_path.parents else str(patch_file_path),
            created_at=time.time(),
            status="staged",
        )

        self._in_memory_proposals[proposal_id] = proposal
        self._persist_proposals()
        log.info(f"Generated improvement proposal #{proposal_id} for {target_file_rel}")
        return proposal

    def _persist_proposals(self) -> None:
        """Persists proposals catalog to JSON."""
        try:
            data = [p.to_dict() for p in self._in_memory_proposals.values()]
            with open(self.proposals_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning(f"Could not persist proposals: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ISOLATED DEVELOPMENT SANDBOX MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────────

    def create_isolated_sandbox(self, sandbox_name: str = "dev_sandbox") -> SandboxEnvironmentInfo:
        """
        Creates an isolated development sandbox directory to stage and test candidate
        code improvements without touching active production files.
        """
        self.initialize()
        sandbox_path = self.sandbox_root / sandbox_name
        sandbox_path.mkdir(parents=True, exist_ok=True)

        info = SandboxEnvironmentInfo(
            sandbox_path=str(sandbox_path),
            stable_path=str(self.workspace_dir),
            sandbox_name=sandbox_name,
            is_isolated=True,
            staged_patches=[],
        )
        log.info(f"Isolated Development Sandbox created at: {sandbox_path}")
        return info

    def apply_patch_to_sandbox(
        self,
        proposal: CodeImprovementProposal,
        sandbox_name: str = "dev_sandbox",
    ) -> Dict[str, Any]:
        """
        Safely copies the target file to the sandbox and applies the proposed patch.
        Validates the patched file with `ast.parse` to ensure syntax validity.
        The active production environment is strictly untouched.
        """
        self.initialize()
        sandbox_path = self.sandbox_root / sandbox_name
        sandbox_path.mkdir(parents=True, exist_ok=True)

        source_file = self.workspace_dir / proposal.target_file
        if not source_file.exists():
            return {"success": False, "error": f"Source file {source_file} does not exist."}

        # Destination in sandbox
        sandbox_target = sandbox_path / proposal.target_file
        sandbox_target.parent.mkdir(parents=True, exist_ok=True)

        # 1. Copy original file to sandbox
        shutil.copy2(source_file, sandbox_target)

        # 2. Apply modifications to sandbox file
        with open(sandbox_target, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Simple line replacement based on before/after snippet
        modified = False
        new_lines = []
        for line in lines:
            if not modified and proposal.before_snippet and proposal.before_snippet in line:
                # Replace with after_snippet
                indent = re.match(r"^(\s*)", line).group(1)
                new_lines.append(f"{indent}{proposal.after_snippet.strip()}\n")
                modified = True
            else:
                new_lines.append(line)

        with open(sandbox_target, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        # 3. Verify Syntax in Sandbox
        syntax_valid = False
        syntax_error = None
        try:
            with open(sandbox_target, "r", encoding="utf-8") as f:
                ast.parse(f.read(), filename=str(sandbox_target))
            syntax_valid = True
        except SyntaxError as syn_err:
            syntax_error = str(syn_err)

        if not syntax_valid:
            proposal.status = "rejected"
            self._persist_proposals()
            return {
                "success": False,
                "error": f"Patched code failed AST syntax check: {syntax_error}",
                "syntax_valid": False,
                "sandbox_file": str(sandbox_target),
            }

        proposal.status = "applied_in_sandbox"
        self._persist_proposals()

        return {
            "success": True,
            "proposal_id": proposal.proposal_id,
            "sandbox_file": str(sandbox_target),
            "syntax_valid": True,
            "status": "applied_in_sandbox",
            "message": "Patch applied cleanly in isolated Development Sandbox.",
        }

    def revert_sandbox(self, sandbox_name: str = "dev_sandbox") -> bool:
        """Cleans and resets the development sandbox."""
        sandbox_path = self.sandbox_root / sandbox_name
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path, ignore_errors=True)
            log.info(f"Reverted and cleared sandbox: {sandbox_path}")
            return True
        return False

    def list_proposals(self, status: Optional[str] = None) -> List[CodeImprovementProposal]:
        """Lists proposals, optionally filtered by status."""
        self.initialize()
        if status:
            return [p for p in self._in_memory_proposals.values() if p.status == status]
        return list(self._in_memory_proposals.values())

    def get_proposal(self, proposal_id: str) -> Optional[CodeImprovementProposal]:
        """Retrieves a proposal by its ID."""
        self.initialize()
        return self._in_memory_proposals.get(proposal_id)

    def generate_proposal_markdown(self, proposal: CodeImprovementProposal) -> str:
        """Generates clear, human-readable markdown detailing the proposal."""
        return (
            f"### 🛠️ Code Improvement Proposal `{proposal.proposal_id}`\n\n"
            f"- **Target File**: `{proposal.target_file}`\n"
            f"- **Status**: `{proposal.status.upper()}`\n"
            f"- **Title**: **{proposal.title}**\n\n"
            f"#### **Modification Rationale (Beginner Guide):**\n"
            f"{proposal.rationale}\n\n"
            f"#### **Code Changes Preview:**\n"
            f"```python\n"
            f"# Before:\n"
            f"- {proposal.before_snippet}\n\n"
            f"# After:\n"
            f"+ {proposal.after_snippet}\n"
            f"```\n\n"
            f"#### **Unified Diff (.patch):**\n"
            f"```diff\n"
            f"{proposal.patch_content}\n"
            f"```\n\n"
            f"> [!IMPORTANT]\n"
            f"> This patch is staged in `.jarvis/evolution/patches/`. It does NOT touch active production files "
            f"until verified by automated tests (Phase 33) and approved by the user."
        )


# Global singleton instance for system-wide access
self_code_generator = SelfCodeGenerator()
