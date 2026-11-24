"""J.A.R.V.I.S. Self-Evolution System — Phase 36: Controlled Self-Update Controller
================================================================================
Orchestrates the end-to-end self-improvement lifecycle:
1. Preparation: Validates patches, applies them to an isolated sandbox, and runs regression gates.
2. Reporting: Compiles transparent change summaries and test benchmarks.
3. User Approval Gate: Requires explicit user authorization before active workspace modification.
4. Atomic Activation: Creates a pre-update safety backup, copies verified code, and updates release records.
5. Post-Update Verification: Executes startup smoke tests; automatically triggers rollback if startup fails.

Strict Safety Directives:
- Never silently modify production/active code.
- User authorization is mandatory before promotion.
- Automated instant rollback on any post-update failure.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.self_evolution.self_code_generator import SelfCodeGenerator, self_code_generator
from core.self_evolution.self_test_runner import SelfTestRunner, self_test_runner, CandidateTestEvaluation
from core.self_evolution.self_version_manager import SelfVersionManager, self_version_manager, ReleaseRecord, SemanticVersion
from core.self_evolution.self_backup_recovery import SelfBackupRecoveryManager, self_backup_recovery, BackupType

logger = logging.getLogger("JARVIS.SelfUpdateController")


class UpdateStatus(str, enum.Enum):
    """Lifecycle statuses of an update package."""
    PREPARING = "preparing"
    TESTS_FAILED = "tests_failed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLYING = "applying"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class UpdatePackage:
    """Encapsulates a staged self-update candidate package."""
    package_id: str
    created_at: str
    previous_version: str
    target_version: str
    title: str
    summary: str
    patch_ids: List[str]
    modified_files: List[str]
    status: UpdateStatus
    test_passed_count: int = 0
    test_failed_count: int = 0
    safety_backup_id: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    applied_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UpdatePackage:
        data_copy = dict(data)
        data_copy["status"] = UpdateStatus(data_copy["status"])
        return cls(**data_copy)


@dataclass
class UpdateExecutionResult:
    """Outcome of an update application or rollback."""
    package_id: str
    success: bool
    status: UpdateStatus
    target_version: str
    files_applied: int
    safety_backup_id: Optional[str] = None
    rollback_triggered: bool = False
    details: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class SelfUpdateController:
    """Master controller managing the preparation, gating, activation, and recovery of self-updates."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        code_gen: Optional[SelfCodeGenerator] = None,
        test_runner: Optional[SelfTestRunner] = None,
        version_mgr: Optional[SelfVersionManager] = None,
        backup_mgr: Optional[SelfBackupRecoveryManager] = None,
        startup_verifier_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.repo_root = (
            repo_root.resolve()
            if repo_root
            else Path(__file__).resolve().parent.parent.parent
        )
        self.code_gen = code_gen or self_code_generator
        self.test_runner = test_runner or self_test_runner
        self.version_mgr = version_mgr or self_version_manager
        self.backup_mgr = backup_mgr or self_backup_recovery
        self.startup_verifier_fn = startup_verifier_fn or self._default_startup_verifier

        self.updates_dir = self.repo_root / ".jarvis" / "evolution" / "updates"
        self.updates_dir.mkdir(parents=True, exist_ok=True)
        self.packages_file = self.updates_dir / "update_packages.json"
        self._ensure_packages_file()

    def _ensure_packages_file(self) -> None:
        if not self.packages_file.exists():
            with open(self.packages_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _load_packages(self) -> List[UpdatePackage]:
        try:
            with open(self.packages_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [UpdatePackage.from_dict(item) for item in data]
        except Exception as exc:
            logger.warning("Could not read update packages file: %s", exc)
            return []

    def _save_packages(self, packages: List[UpdatePackage]) -> None:
        with open(self.packages_file, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in packages], f, indent=2)

    def _default_startup_verifier(self) -> bool:
        """Lightweight smoke verification checking that core imports and registries initialize cleanly."""
        try:
            # Check importability of core subsystems
            from core.integration.subsystem_registry import subsystem_registry
            from core.brain.router import IntentRouter
            router = IntentRouter()
            subsystems = subsystem_registry.list_subsystems()
            return len(subsystems) >= 28 and router is not None
        except Exception as exc:
            logger.error("Startup smoke verification failed: %s", exc)
            return False

    def prepare_update(
        self,
        patch_ids: List[str],
        bump_type: str = "patch",
        title: str = "",
        summary: str = "",
        test_file_paths: Optional[List[str]] = None,
    ) -> UpdatePackage:
        """Prepares an update package by applying patches to a sandbox and validating with tests."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        package_id = f"pkg_{timestamp_str}_{bump_type}"

        current_ver_str = self.version_mgr.get_current_version()
        current_semver = SemanticVersion.from_string(current_ver_str)
        if bump_type == "minor":
            target_semver = current_semver.bump_minor()
        elif bump_type == "major":
            target_semver = current_semver.bump_major()
        else:
            target_semver = current_semver.bump_patch()
        target_ver_str = str(target_semver)

        # 1. Setup isolated sandbox and apply patches
        sandbox_info = self.code_gen.create_isolated_sandbox()
        sandbox_path = Path(sandbox_info.sandbox_path)

        all_modified_files: Set[str] = set()
        apply_success = True

        for pid in patch_ids:
            res = self.code_gen.apply_patch_to_sandbox(pid)
            if not res["success"]:
                apply_success = False
                logger.error("Patch '%s' failed to apply in sandbox: %s", pid, res.get("error"))
                break
            for mf in res.get("modified_files", []):
                all_modified_files.add(mf)

        if not apply_success:
            pkg = UpdatePackage(
                package_id=package_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                previous_version=current_ver_str,
                target_version=target_ver_str,
                title=title or f"Self-Update {target_ver_str}",
                summary=summary or "Preparation failed during sandbox patching.",
                patch_ids=patch_ids,
                modified_files=list(all_modified_files),
                status=UpdateStatus.TESTS_FAILED,
                error_message="One or more patches failed to apply cleanly to sandbox.",
            )
            packages = self._load_packages()
            packages.append(pkg)
            self._save_packages(packages)
            return pkg

        # 2. Run isolated test suite in sandbox and evaluate release gate
        baseline_res = self.test_runner.run_tests(test_type="unit", cwd=self.repo_root)
        candidate_res = self.test_runner.run_tests(test_type="unit", cwd=sandbox_path)

        eval_result = self.test_runner.evaluate_candidate_release(
            candidate_id=package_id,
            baseline_result=baseline_res,
            candidate_result=candidate_res,
            max_degradation_pct=25.0,
        )

        passed_count = candidate_res.passed_count
        failed_count = candidate_res.failed_count

        if eval_result.decision != "ACCEPTED_FOR_RELEASE":
            pkg = UpdatePackage(
                package_id=package_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                previous_version=current_ver_str,
                target_version=target_ver_str,
                title=title or f"Self-Update {target_ver_str}",
                summary=summary or "Update rejected by automated regression testing.",
                patch_ids=patch_ids,
                modified_files=list(all_modified_files),
                status=UpdateStatus.TESTS_FAILED,
                test_passed_count=passed_count,
                test_failed_count=failed_count,
                error_message=f"Test gate rejected: {eval_result.summary}",
                metadata={"test_gate_decision": eval_result.decision, "summary": eval_result.summary},
            )
        else:
            pkg = UpdatePackage(
                package_id=package_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                previous_version=current_ver_str,
                target_version=target_ver_str,
                title=title or f"Self-Update {target_ver_str}",
                summary=summary or f"Validated candidate passing {passed_count} tests.",
                patch_ids=patch_ids,
                modified_files=list(all_modified_files),
                status=UpdateStatus.PENDING_APPROVAL,
                test_passed_count=passed_count,
                test_failed_count=failed_count,
                metadata={"sandbox_path": str(sandbox_path), "test_gate_decision": eval_result.decision},
            )

        packages = self._load_packages()
        packages.append(pkg)
        self._save_packages(packages)
        return pkg

    def approve_update(self, package_id: str, approver: str = "user") -> bool:
        """Records explicit user approval for a pending update package."""
        packages = self._load_packages()
        pkg = next((p for p in packages if p.package_id == package_id), None)
        if not pkg:
            logger.warning("Update package '%s' not found.", package_id)
            return False

        if pkg.status != UpdateStatus.PENDING_APPROVAL:
            logger.warning("Cannot approve package '%s' in status '%s'.", package_id, pkg.status.value)
            return False

        pkg.status = UpdateStatus.APPROVED
        pkg.approved_by = approver
        pkg.approved_at = datetime.now(timezone.utc).isoformat()
        self._save_packages(packages)
        logger.info("Package '%s' approved by '%s'.", package_id, approver)
        return True

    def reject_update(self, package_id: str, reason: str = "User rejected update.") -> bool:
        """Records user rejection of an update package."""
        packages = self._load_packages()
        pkg = next((p for p in packages if p.package_id == package_id), None)
        if not pkg:
            return False

        pkg.status = UpdateStatus.REJECTED
        pkg.rejection_reason = reason
        self._save_packages(packages)
        logger.info("Package '%s' rejected: %s", package_id, reason)
        return True

    def apply_update(self, package_id: str) -> UpdateExecutionResult:
        """Promotes an approved update package to the active production workspace."""
        packages = self._load_packages()
        pkg = next((p for p in packages if p.package_id == package_id), None)
        if not pkg:
            return UpdateExecutionResult(
                package_id=package_id,
                success=False,
                status=UpdateStatus.REJECTED,
                target_version="unknown",
                files_applied=0,
                error="Package not found.",
            )

        # Strict Safety Directive: Must be approved prior to promotion
        if pkg.status != UpdateStatus.APPROVED:
            return UpdateExecutionResult(
                package_id=package_id,
                success=False,
                status=pkg.status,
                target_version=pkg.target_version,
                files_applied=0,
                error=f"Cannot apply unapproved package. Current status: {pkg.status.value}",
            )

        pkg.status = UpdateStatus.APPLYING
        self._save_packages(packages)

        # 1. Mandatory Safety Pre-Update Backup
        safety_manifest = self.backup_mgr.create_backup(
            backup_type=BackupType.SAFETY_PRE_RESTORE,
            description=f"Automatic pre-update safety backup prior to applying {package_id}",
            version=pkg.previous_version,
            specific_files=pkg.modified_files if pkg.modified_files else None,
        )
        pkg.safety_backup_id = safety_manifest.backup_id

        # 2. Copy verified files from sandbox into production repo
        sandbox_path = Path(pkg.metadata.get("sandbox_path", self.repo_root / ".jarvis" / "sandbox"))
        applied_count = 0

        try:
            for rel_file in pkg.modified_files:
                src_file = sandbox_path / rel_file
                dst_file = self.repo_root / rel_file

                if not src_file.exists():
                    continue

                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                applied_count += 1

            # 3. Post-Update Startup Smoke Verification
            startup_ok = self.startup_verifier_fn()
            if not startup_ok:
                logger.error("Startup verification failed after applying '%s'! Triggering immediate rollback.", package_id)
                # Roll back instantly
                self.backup_mgr.restore_backup(
                    backup_id=safety_manifest.backup_id,
                    destination_dir=self.repo_root,
                    create_safety_snapshot=False,
                )
                pkg.status = UpdateStatus.ROLLED_BACK
                pkg.error_message = "Startup smoke verification failed post-update. Auto-rollback triggered."
                self._save_packages(packages)

                return UpdateExecutionResult(
                    package_id=package_id,
                    success=False,
                    status=UpdateStatus.ROLLED_BACK,
                    target_version=pkg.target_version,
                    files_applied=applied_count,
                    safety_backup_id=safety_manifest.backup_id,
                    rollback_triggered=True,
                    error="Startup smoke verification failed. System rolled back to safety backup.",
                )

            # 4. Successful activation: Record release in Version Manager
            self.version_mgr.create_candidate_release(
                bump_type="patch",
                title=pkg.title,
                modified_files=pkg.modified_files,
                patch_ids=pkg.patch_ids,
            )
            self.version_mgr.activate_release(pkg.target_version)

            pkg.status = UpdateStatus.COMPLETED
            pkg.applied_at = datetime.now(timezone.utc).isoformat()
            self._save_packages(packages)

            logger.info("Update '%s' applied and verified successfully. New active version: %s", package_id, pkg.target_version)
            return UpdateExecutionResult(
                package_id=package_id,
                success=True,
                status=UpdateStatus.COMPLETED,
                target_version=pkg.target_version,
                files_applied=applied_count,
                safety_backup_id=safety_manifest.backup_id,
                rollback_triggered=False,
                details=f"Successfully applied {applied_count} files. Activated version {pkg.target_version}.",
            )

        except Exception as exc:
            logger.error("Unexpected error applying update '%s': %s. Rolling back.", package_id, exc)
            self.backup_mgr.restore_backup(
                backup_id=safety_manifest.backup_id,
                destination_dir=self.repo_root,
                create_safety_snapshot=False,
            )
            pkg.status = UpdateStatus.ROLLED_BACK
            pkg.error_message = str(exc)
            self._save_packages(packages)

            return UpdateExecutionResult(
                package_id=package_id,
                success=False,
                status=UpdateStatus.ROLLED_BACK,
                target_version=pkg.target_version,
                files_applied=applied_count,
                safety_backup_id=safety_manifest.backup_id,
                rollback_triggered=True,
                error=str(exc),
            )

    def manual_rollback(self, package_id: str, reason: str = "Manual user rollback.") -> UpdateExecutionResult:
        """Rolls back an applied update back to its pre-update safety state."""
        packages = self._load_packages()
        pkg = next((p for p in packages if p.package_id == package_id), None)
        if not pkg:
            return UpdateExecutionResult(
                package_id=package_id,
                success=False,
                status=UpdateStatus.ROLLED_BACK,
                target_version="unknown",
                files_applied=0,
                error="Package not found.",
            )

        if not pkg.safety_backup_id:
            return UpdateExecutionResult(
                package_id=package_id,
                success=False,
                status=pkg.status,
                target_version=pkg.target_version,
                files_applied=0,
                error="No safety backup associated with this package.",
            )

        restore_res = self.backup_mgr.restore_backup(
            backup_id=pkg.safety_backup_id,
            destination_dir=self.repo_root,
            create_safety_snapshot=False,
        )

        if restore_res.success:
            self.version_mgr.activate_release(pkg.previous_version)
            pkg.status = UpdateStatus.ROLLED_BACK
            pkg.error_message = f"Rolled back: {reason}"
            self._save_packages(packages)

            return UpdateExecutionResult(
                package_id=package_id,
                success=True,
                status=UpdateStatus.ROLLED_BACK,
                target_version=pkg.previous_version,
                files_applied=restore_res.files_restored,
                safety_backup_id=pkg.safety_backup_id,
                rollback_triggered=True,
                details=f"Rollback successful. Restored version {pkg.previous_version}.",
            )
        else:
            return UpdateExecutionResult(
                package_id=package_id,
                success=False,
                status=pkg.status,
                target_version=pkg.target_version,
                files_applied=0,
                safety_backup_id=pkg.safety_backup_id,
                rollback_triggered=False,
                error=f"Restore failed: {restore_res.error}",
            )

    def list_packages(self) -> List[UpdatePackage]:
        """Lists all update packages sorted descending by creation timestamp."""
        packages = self._load_packages()
        packages.sort(key=lambda p: p.created_at, reverse=True)
        return packages


# Global singleton instance
self_update_controller = SelfUpdateController()
