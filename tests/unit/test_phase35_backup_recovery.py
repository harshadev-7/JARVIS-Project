"""Unit Tests for Phase 35: Backup & Recovery Engine
===================================================
Tests automated snapshot creation, SHA-256 integrity verification,
checkpoint protection, and point-in-time state recovery.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from core.self_evolution.self_backup_recovery import (
    SelfBackupRecoveryManager,
    BackupType,
    BackupFileEntry,
    BackupManifest,
    IntegrityReport,
    RestoreResult,
)


@pytest.fixture
def mock_repo_env():
    """Sets up a temporary simulated repository and backup directory."""
    temp_dir = tempfile.mkdtemp(prefix="jarvis_backup_test_")
    repo_path = Path(temp_dir) / "repo"
    backup_path = Path(temp_dir) / "backups"
    repo_path.mkdir(parents=True)
    backup_path.mkdir(parents=True)

    # Populate simulated project structure
    config_dir = repo_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.py").write_text("API_PORT = 8000\nAUTONOMY = 'HIGH'\n", encoding="utf-8")

    core_dir = repo_path / "core"
    core_dir.mkdir(parents=True)
    (core_dir / "main_brain.py").write_text("class Brain:\n    pass\n", encoding="utf-8")

    checkpoints_dir = repo_path / "custom_ai" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "model_weights.pth").write_bytes(b"\x00\x01\x02\x03\x04\x05WeightsBinary")

    db_file = repo_path / "jarvis.db"
    db_file.write_bytes(b"SQLite format 3\x00DummyHeader")

    manager = SelfBackupRecoveryManager(repo_root=repo_path, backup_dir=backup_path)

    yield {
        "temp_dir": temp_dir,
        "repo_path": repo_path,
        "backup_path": backup_path,
        "manager": manager,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_sha256_computation(mock_repo_env):
    """Verifies chunked SHA-256 calculation."""
    manager = mock_repo_env["manager"]
    repo_path = mock_repo_env["repo_path"]
    test_file = repo_path / "config" / "settings.py"

    h1 = manager.compute_sha256(test_file)
    assert len(h1) == 64
    assert isinstance(h1, str)

    # Identical content yields identical hash
    h2 = manager.compute_sha256(test_file)
    assert h1 == h2


def test_create_full_backup(mock_repo_env):
    """Verifies creating a full project backup and inspecting its manifest."""
    manager = mock_repo_env["manager"]
    repo_path = mock_repo_env["repo_path"]

    manifest = manager.create_backup(
        backup_type=BackupType.FULL,
        description="Test Full System Snapshot",
        version="1.0.0",
    )

    assert manifest.backup_id.startswith("backup_")
    assert manifest.backup_type == "full"
    assert manifest.version == "1.0.0"
    assert manifest.total_files >= 4
    assert "config/settings.py" in manifest.files
    assert "core/main_brain.py" in manifest.files
    assert "custom_ai/checkpoints/model_weights.pth" in manifest.files
    assert "jarvis.db" in manifest.files

    # Verify manifest file exists on disk
    manifest_on_disk = mock_repo_env["backup_path"] / manifest.backup_id / "manifest.json"
    assert manifest_on_disk.exists()


def test_backup_type_filtering(mock_repo_env):
    """Verifies that category-specific backups only copy relevant files."""
    manager = mock_repo_env["manager"]

    # 1. Config only
    config_manifest = manager.create_backup(backup_type=BackupType.CONFIG_ONLY)
    assert all("config" in p or p.endswith(".py") for p in config_manifest.files.keys())
    assert "custom_ai/checkpoints/model_weights.pth" not in config_manifest.files

    # 2. Checkpoints only
    weights_manifest = manager.create_backup(backup_type=BackupType.MODEL_CHECKPOINTS)
    assert "custom_ai/checkpoints/model_weights.pth" in weights_manifest.files
    assert "config/settings.py" not in weights_manifest.files

    # 3. Knowledge base only
    kb_manifest = manager.create_backup(backup_type=BackupType.KNOWLEDGE_BASE)
    assert "jarvis.db" in kb_manifest.files
    assert "core/main_brain.py" not in kb_manifest.files


def test_integrity_verification_pass(mock_repo_env):
    """Verifies that an untouched backup passes cryptographic integrity verification."""
    manager = mock_repo_env["manager"]

    manifest = manager.create_backup(backup_type=BackupType.FULL)
    report = manager.verify_backup_integrity(manifest.backup_id)

    assert report.is_valid is True
    assert report.total_checked == manifest.total_files
    assert len(report.corrupted_files) == 0
    assert len(report.missing_files) == 0
    assert "All hashes match" in report.details


def test_integrity_verification_tampering_and_missing(mock_repo_env):
    """Verifies that modified or missing files are immediately flagged."""
    manager = mock_repo_env["manager"]
    backup_path = mock_repo_env["backup_path"]

    manifest = manager.create_backup(backup_type=BackupType.FULL)
    backup_id = manifest.backup_id

    # Tamper with one file
    tampered_file = backup_path / backup_id / "data" / "config" / "settings.py"
    tampered_file.write_text("TAMPERED_INJECTED_DATA = True\n", encoding="utf-8")

    # Delete another file
    deleted_file = backup_path / backup_id / "data" / "jarvis.db"
    deleted_file.unlink()

    report = manager.verify_backup_integrity(backup_id)
    assert report.is_valid is False
    assert "config/settings.py" in report.corrupted_files
    assert "jarvis.db" in report.missing_files


def test_restore_backup_safe(mock_repo_env):
    """Verifies restoring a backup into a target destination directory."""
    manager = mock_repo_env["manager"]
    repo_path = mock_repo_env["repo_path"]

    # 1. Create baseline backup
    manifest = manager.create_backup(backup_type=BackupType.FULL, version="0.9.0")
    backup_id = manifest.backup_id

    # 2. Modify active repo files (simulate a regression or accidental corruption)
    (repo_path / "config" / "settings.py").write_text("CORRUPTED_SETTINGS = True\n", encoding="utf-8")

    # 3. Restore with safety snapshot
    restore_res = manager.restore_backup(
        backup_id=backup_id,
        destination_dir=repo_path,
        create_safety_snapshot=True,
    )

    assert restore_res.success is True
    assert restore_res.files_restored >= 4
    assert restore_res.safety_backup_id is not None

    # Check restored file matches original state
    restored_content = (repo_path / "config" / "settings.py").read_text(encoding="utf-8")
    assert "API_PORT = 8000" in restored_content
    assert "CORRUPTED_SETTINGS" not in restored_content


def test_restore_rejected_if_corrupted(mock_repo_env):
    """Verifies that restore is safely refused if backup integrity check fails."""
    manager = mock_repo_env["manager"]
    backup_path = mock_repo_env["backup_path"]

    manifest = manager.create_backup(backup_type=BackupType.FULL)
    backup_id = manifest.backup_id

    # Tamper with file
    tampered_file = backup_path / backup_id / "data" / "config" / "settings.py"
    tampered_file.write_text("TAMPERED = True\n", encoding="utf-8")

    restore_res = manager.restore_backup(backup_id=backup_id)
    assert restore_res.success is False
    assert restore_res.files_restored == 0
    assert "Integrity check failed" in restore_res.error


def test_prune_old_backups(mock_repo_env):
    """Verifies retention policy when pruning historical backups."""
    manager = mock_repo_env["manager"]

    # Create 6 regular backups
    for i in range(6):
        manager.create_backup(
            backup_type=BackupType.CONFIG_ONLY,
            description=f"Snapshot {i}",
        )

    assert len(manager.list_backups()) == 6

    # Prune keeping 3
    deleted = manager.prune_old_backups(keep_count=3)
    assert deleted == 3
    assert len(manager.list_backups()) == 3
