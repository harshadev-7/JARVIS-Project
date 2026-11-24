"""J.A.R.V.I.S. Self-Evolution System — Phase 35: Backup & Recovery Engine
========================================================================
Provides automated snapshot creation, model checkpoint protection, configuration
preservation, point-in-time state restoration, and cryptographic SHA-256 integrity
verification for J.A.R.V.I.S.

Strict Safety Directives:
1. Never overwrite an active working copy without an automatic pre-restore safety snapshot.
2. Verify SHA-256 cryptographic hashes for every file prior to executing any restoration.
3. Protect neural model checkpoints and knowledge databases from accidental deletion.
4. Keep the latest working version protected during backup pruning.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("JARVIS.SelfBackupRecovery")


class BackupType(str, enum.Enum):
    """Categorized backup scope types."""
    FULL = "full"
    CONFIG_ONLY = "config_only"
    MODEL_CHECKPOINTS = "model_checkpoints"
    KNOWLEDGE_BASE = "knowledge_base"
    SOURCE_CODE = "source_code"
    SAFETY_PRE_RESTORE = "safety_pre_restore"


@dataclass
class BackupFileEntry:
    """Metadata describing a single backed up file."""
    relative_path: str
    sha256_hash: str
    size_bytes: int
    modified_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackupFileEntry:
        return cls(
            relative_path=data["relative_path"],
            sha256_hash=data["sha256_hash"],
            size_bytes=data["size_bytes"],
            modified_timestamp=data.get("modified_timestamp", 0.0),
        )


@dataclass
class BackupManifest:
    """Manifest describing an entire backup archive."""
    backup_id: str
    created_at: str
    backup_type: str
    version: str
    total_files: int
    total_size_bytes: int
    files: Dict[str, BackupFileEntry] = field(default_factory=dict)
    description: str = ""
    status: str = "READY"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at,
            "backup_type": self.backup_type,
            "version": self.version,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "description": self.description,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackupManifest:
        files = {
            k: BackupFileEntry.from_dict(v) for k, v in data.get("files", {}).items()
        }
        return cls(
            backup_id=data["backup_id"],
            created_at=data["created_at"],
            backup_type=data["backup_type"],
            version=data.get("version", "unknown"),
            total_files=data.get("total_files", len(files)),
            total_size_bytes=data.get("total_size_bytes", 0),
            files=files,
            description=data.get("description", ""),
            status=data.get("status", "READY"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IntegrityReport:
    """Outcome of an integrity verification scan."""
    backup_id: str
    is_valid: bool
    total_checked: int
    corrupted_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RestoreResult:
    """Outcome of a backup restoration execution."""
    success: bool
    backup_id: str
    target_dir: str
    files_restored: int
    safety_backup_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfBackupRecoveryManager:
    """Engine orchestrating autonomous backups, integrity checks, and point-in-time recovery."""

    # Default file and folder exclusions
    EXCLUDED_DIRS = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".jarvis/backups",
        ".jarvis/sandbox",
        "htmlcov",
    }

    EXCLUDED_EXTENSIONS = {
        ".pyc",
        ".pyo",
        ".pyd",
        ".tmp",
        ".temp",
        ".log",
    }

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
    ) -> None:
        self.repo_root = (
            repo_root.resolve()
            if repo_root
            else Path(__file__).resolve().parent.parent.parent
        )
        self.backup_dir = (
            backup_dir.resolve()
            if backup_dir
            else self.repo_root / ".jarvis" / "backups"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash in 64KB chunks to maintain minimal memory consumption."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _should_include_file(self, rel_path: str, backup_type: BackupType) -> bool:
        """Determines if a relative path belongs in the specified backup category."""
        path_obj = Path(rel_path)

        # 1. Global exclusions
        for part in path_obj.parts:
            if part in self.EXCLUDED_DIRS:
                return False
        if path_obj.suffix in self.EXCLUDED_EXTENSIONS:
            return False

        # Exclude temporary databases
        if path_obj.name.startswith("test_temp") or path_obj.name.endswith(".tmp"):
            return False

        # 2. Category filtering
        if backup_type in (BackupType.FULL, BackupType.SAFETY_PRE_RESTORE):
            # Include all core project files, checkpoints, databases, configs
            return True

        if backup_type == BackupType.CONFIG_ONLY:
            return (
                rel_path.startswith("config/")
                or path_obj.name in ("pytest.ini", "pyproject.toml", "settings.py")
            )

        if backup_type == BackupType.MODEL_CHECKPOINTS:
            return (
                rel_path.startswith("custom_ai/checkpoints/")
                or rel_path.startswith("custom_ai/experiments/")
                or path_obj.suffix in (".pth", ".pt", ".bin", ".onnx", ".safetensors")
            )

        if backup_type == BackupType.KNOWLEDGE_BASE:
            return (
                path_obj.suffix == ".db"
                or rel_path.startswith(".jarvis/evolution/")
                or rel_path.startswith(".jarvis/staging/")
                or rel_path.startswith("data/")
            )

        if backup_type == BackupType.SOURCE_CODE:
            # Exclude large binaries and checkpoints for lightweight code backup
            if path_obj.suffix in (".pth", ".pt", ".bin", ".onnx", ".safetensors", ".mp4", ".mov"):
                return False
            allowed_prefixes = (
                "core/",
                "custom_ai/",
                "vision/",
                "video_editor/",
                "devices/",
                "services/",
                "config/",
                "tests/",
            )
            return (
                any(rel_path.startswith(prefix) for prefix in allowed_prefixes)
                or path_obj.name in ("main.py", "requirements.txt", "pytest.ini", "README.md")
            )

        return False

    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        description: str = "",
        version: str = "current",
        specific_files: Optional[List[str]] = None,
    ) -> BackupManifest:
        """Creates an atomic backup snapshot with a SHA-256 validated manifest."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_id = f"backup_{timestamp_str}_{backup_type.value}"
        target_backup_path = self.backup_dir / backup_id
        target_data_path = target_backup_path / "data"
        target_data_path.mkdir(parents=True, exist_ok=True)

        files_manifest: Dict[str, BackupFileEntry] = {}
        total_size = 0

        # Collect eligible files
        paths_to_evaluate: List[Path] = []
        if specific_files:
            for sf in specific_files:
                candidate = (self.repo_root / sf).resolve()
                if candidate.exists() and candidate.is_file():
                    paths_to_evaluate.append(candidate)
        else:
            for root, dirs, files in os.walk(self.repo_root):
                # Filter directories in-place to avoid unnecessary walks
                dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRS]
                for file_name in files:
                    paths_to_evaluate.append(Path(root) / file_name)

        for file_path in paths_to_evaluate:
            try:
                rel_path = file_path.relative_to(self.repo_root).as_posix()
            except ValueError:
                continue

            if not specific_files and not self._should_include_file(rel_path, backup_type):
                continue

            try:
                stat = file_path.stat()
                file_hash = self.compute_sha256(file_path)

                dest_file_path = target_data_path / rel_path
                dest_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_file_path)

                entry = BackupFileEntry(
                    relative_path=rel_path,
                    sha256_hash=file_hash,
                    size_bytes=stat.st_size,
                    modified_timestamp=stat.st_mtime,
                )
                files_manifest[rel_path] = entry
                total_size += stat.st_size
            except Exception as exc:
                logger.warning("Failed to backup file '%s': %s", rel_path, exc)

        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            backup_type=backup_type.value,
            version=version,
            total_files=len(files_manifest),
            total_size_bytes=total_size,
            files=files_manifest,
            description=description or f"Autonomous {backup_type.value} snapshot",
            status="READY",
        )

        manifest_file = target_backup_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(
            "Backup '%s' created successfully with %d files (%d bytes).",
            backup_id,
            len(files_manifest),
            total_size,
        )
        return manifest

    def verify_backup_integrity(self, backup_id: str) -> IntegrityReport:
        """Cryptographically verifies every file in a backup against its manifest."""
        backup_path = self.backup_dir / backup_id
        manifest_file = backup_path / "manifest.json"
        data_path = backup_path / "data"

        if not backup_path.exists() or not manifest_file.exists():
            return IntegrityReport(
                backup_id=backup_id,
                is_valid=False,
                total_checked=0,
                details=f"Backup manifest or folder missing for ID '{backup_id}'.",
            )

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = BackupManifest.from_dict(data)

        corrupted: List[str] = []
        missing: List[str] = []
        checked = 0

        for rel_path, entry in manifest.files.items():
            stored_file = data_path / rel_path
            checked += 1
            if not stored_file.exists():
                missing.append(rel_path)
                continue

            current_hash = self.compute_sha256(stored_file)
            if current_hash != entry.sha256_hash:
                corrupted.append(rel_path)

        is_valid = len(corrupted) == 0 and len(missing) == 0
        details = (
            f"Integrity verified: {checked} files checked. All hashes match."
            if is_valid
            else f"Integrity failure: {len(corrupted)} corrupted, {len(missing)} missing."
        )

        return IntegrityReport(
            backup_id=backup_id,
            is_valid=is_valid,
            total_checked=checked,
            corrupted_files=corrupted,
            missing_files=missing,
            details=details,
        )

    def list_backups(self) -> List[BackupManifest]:
        """Lists all existing backups, sorted chronologically descending."""
        if not self.backup_dir.exists():
            return []

        backups: List[BackupManifest] = []
        for item in self.backup_dir.iterdir():
            if item.is_dir():
                manifest_file = item / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        backups.append(BackupManifest.from_dict(data))
                    except Exception as exc:
                        logger.warning("Could not read backup manifest at %s: %s", manifest_file, exc)

        backups.sort(key=lambda b: b.created_at, reverse=True)
        return backups

    def get_backup_manifest(self, backup_id: str) -> Optional[BackupManifest]:
        """Retrieves the manifest for a specific backup ID."""
        manifest_file = self.backup_dir / backup_id / "manifest.json"
        if not manifest_file.exists():
            return None
        with open(manifest_file, "r", encoding="utf-8") as f:
            return BackupManifest.from_dict(json.load(f))

    def restore_backup(
        self,
        backup_id: str,
        destination_dir: Optional[Path] = None,
        create_safety_snapshot: bool = True,
    ) -> RestoreResult:
        """Restores a previous backup snapshot after verifying cryptographic integrity."""
        # 1. Pre-restore integrity check
        integrity = self.verify_backup_integrity(backup_id)
        if not integrity.is_valid:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_dir=str(destination_dir or self.repo_root),
                files_restored=0,
                error=f"Integrity check failed: {integrity.details}",
            )

        manifest = self.get_backup_manifest(backup_id)
        if not manifest:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_dir=str(destination_dir or self.repo_root),
                files_restored=0,
                error="Backup manifest not found.",
            )

        target_root = (destination_dir or self.repo_root).resolve()
        safety_backup_id: Optional[str] = None

        # 2. Strict Safety Directive: Automatic snapshot before restoring to active repo
        if create_safety_snapshot and target_root == self.repo_root:
            affected_rel_paths = list(manifest.files.keys())
            safety_manifest = self.create_backup(
                backup_type=BackupType.SAFETY_PRE_RESTORE,
                description=f"Automated pre-restore safety snapshot before restoring {backup_id}",
                version=manifest.version,
                specific_files=affected_rel_paths,
            )
            safety_backup_id = safety_manifest.backup_id

        # 3. Perform restoration
        data_path = self.backup_dir / backup_id / "data"
        restored_count = 0

        try:
            for rel_path in manifest.files.keys():
                src_file = data_path / rel_path
                dst_file = target_root / rel_path

                if not src_file.exists():
                    continue

                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                restored_count += 1

            logger.info("Restored %d files from backup '%s' to '%s'.", restored_count, backup_id, target_root)
            return RestoreResult(
                success=True,
                backup_id=backup_id,
                target_dir=str(target_root),
                files_restored=restored_count,
                safety_backup_id=safety_backup_id,
            )
        except Exception as exc:
            logger.error("Restore failed unexpectedly: %s", exc)
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_dir=str(target_root),
                files_restored=restored_count,
                safety_backup_id=safety_backup_id,
                error=str(exc),
            )

    def prune_old_backups(self, keep_count: int = 5) -> int:
        """Prunes historical backups while preserving recent snapshots and safety backups."""
        backups = self.list_backups()
        if len(backups) <= keep_count:
            return 0

        # Separate safety backups from regular backups
        regular_backups = [b for b in backups if b.backup_type != BackupType.SAFETY_PRE_RESTORE.value]
        to_delete = regular_backups[keep_count:]
        deleted_count = 0

        for b in to_delete:
            folder = self.backup_dir / b.backup_id
            if folder.exists() and folder.is_dir():
                try:
                    shutil.rmtree(folder)
                    deleted_count += 1
                except Exception as exc:
                    logger.warning("Could not prune backup '%s': %s", b.backup_id, exc)

        return deleted_count


# Global singleton instance
self_backup_recovery = SelfBackupRecoveryManager()
