"""J.A.R.V.I.S. Database Backup Utility.

Provides automated, safe backup of the SQLite jarvis.db database:
- WAL checkpoint flush (forces all uncommitted changes to main DB file)
- Timestamped backup copy to backups/ subdirectory
- Automatic rotation (keeps last N backups, deletes older ones)
- Integrity check before and after backup
- Zero external dependencies — uses Python stdlib only

Usage:
    from memory.backup import backup_database
    path = backup_database()   # returns path to backup file
    print(f"Backup saved to: {path}")

CLI Usage:
    python -m memory.backup
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path("D:/Documents/JARVIS/jarvis.db")
DEFAULT_BACKUP_DIR = Path("D:/Documents/JARVIS/backups")
MAX_BACKUPS = 10          # Keep last 10 backups; delete older ones


# ── Core Functions ─────────────────────────────────────────────────────────────

def _integrity_check(db_path: Path) -> bool:
    """Run SQLite integrity_check pragma. Returns True if OK."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()
        return result is not None and result[0] == "ok"
    except Exception:
        return False


def _wal_checkpoint(db_path: Path) -> None:
    """Force a WAL checkpoint to flush all changes into the main DB file."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()
    except Exception:
        pass  # Non-fatal; backup will still capture WAL if present


def backup_database(
    db_path: Optional[Path] = None,
    backup_dir: Optional[Path] = None,
    max_backups: int = MAX_BACKUPS,
) -> str:
    """
    Create a timestamped backup of the JARVIS SQLite database.

    Steps:
    1. Verify source DB exists and passes integrity check.
    2. Flush WAL (Write-Ahead Log) to main DB file.
    3. Copy DB file to backups/<timestamp>.db.
    4. Verify backup integrity.
    5. Rotate old backups if count exceeds max_backups.

    Args:
        db_path:    Path to jarvis.db (defaults to D:/Documents/JARVIS/jarvis.db)
        backup_dir: Directory for backup files (defaults to D:/Documents/JARVIS/backups/)
        max_backups: Maximum number of backups to retain.

    Returns:
        Absolute path to the newly created backup file.

    Raises:
        FileNotFoundError: If the source database does not exist.
        RuntimeError: If the source integrity check fails.
        IOError: If the backup file integrity check fails.
    """
    source = db_path or DEFAULT_DB_PATH
    dest_dir = backup_dir or DEFAULT_BACKUP_DIR

    # 1. Verify source
    if not source.exists():
        raise FileNotFoundError(f"Source database not found: {source}")

    if not _integrity_check(source):
        raise RuntimeError(
            f"Source database failed integrity check: {source}. "
            "Do NOT overwrite backups — investigate the database first."
        )

    # 2. WAL checkpoint
    _wal_checkpoint(source)

    # 3. Copy to timestamped backup file
    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = dest_dir / f"jarvis_backup_{timestamp}.db"
    shutil.copy2(str(source), str(backup_file))

    # 4. Verify backup
    if not _integrity_check(backup_file):
        backup_file.unlink(missing_ok=True)
        raise IOError(f"Backup file failed integrity check and was removed: {backup_file}")

    # 5. Rotate old backups
    _rotate_backups(dest_dir, max_backups)

    return str(backup_file.resolve())


def _rotate_backups(backup_dir: Path, max_backups: int) -> None:
    """Delete oldest backup files if count exceeds max_backups."""
    backups = sorted(backup_dir.glob("jarvis_backup_*.db"))
    excess = len(backups) - max_backups
    for old in backups[:excess]:
        try:
            old.unlink()
        except Exception:
            pass


def list_backups(backup_dir: Optional[Path] = None) -> list[dict]:
    """
    Return a list of all backup files with metadata.

    Returns:
        List of dicts: [{"file": str, "size_kb": float, "created": str}, ...]
    """
    dest_dir = backup_dir or DEFAULT_BACKUP_DIR
    if not dest_dir.exists():
        return []

    result = []
    for f in sorted(dest_dir.glob("jarvis_backup_*.db"), reverse=True):
        stat = f.stat()
        result.append({
            "file": str(f.resolve()),
            "size_kb": round(stat.st_size / 1024, 1),
            "created": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return result


def restore_backup(backup_file: Path, db_path: Optional[Path] = None) -> str:
    """
    Restore a backup file over the active database.

    Args:
        backup_file: Path to the .db backup file to restore.
        db_path: Target DB path (defaults to jarvis.db).

    Returns:
        Path to the restored database.

    Raises:
        FileNotFoundError: If backup_file does not exist.
        RuntimeError: If backup integrity check fails.
    """
    target = db_path or DEFAULT_DB_PATH

    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    if not _integrity_check(backup_file):
        raise RuntimeError(f"Backup file failed integrity check: {backup_file}. Restore aborted.")

    # Keep a safety copy of the current DB before restoring
    if target.exists():
        safety = target.with_suffix(".db.pre_restore")
        shutil.copy2(str(target), str(safety))

    shutil.copy2(str(backup_file), str(target))
    return str(target.resolve())


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("J.A.R.V.I.S. Database Backup Utility")
    print("=" * 40)

    if "--list" in sys.argv:
        backups = list_backups()
        if not backups:
            print("No backups found.")
        for b in backups:
            print(f"  {b['created'][:19]}  {b['size_kb']:>8.1f} KB  {b['file']}")
        sys.exit(0)

    try:
        path = backup_database()
        print(f"✅ Backup successful: {path}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        sys.exit(1)


__all__ = [
    "backup_database",
    "list_backups",
    "restore_backup",
]
