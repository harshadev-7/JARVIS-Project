"""
JARVIS AI — Authorized Folder Sandbox & Path Traversal Shield
=============================================================
Enforces strict filesystem sandboxing for multi-device operations.
Guarantees that connected devices can only read or write to explicitly
whitelisted directories approved by the user.

Security Mandates:
- Resolves canonical paths (os.path.realpath) to eliminate symlink traversal.
- Detects and rejects directory traversal sequences ('..', embedded nulls).
- Blocks critical OS system directories (C:\\Windows, C:\\Program Files, /etc, /sys).
- Requires explicit read/write permission scopes.
"""

import os
import re
import uuid
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from core.logger import log


# System directories that must NEVER be authorized
DISALLOWED_SYSTEM_ROOTS = [
    # Windows
    r"^[a-zA-Z]:\\windows",
    r"^[a-zA-Z]:\\program files",
    r"^[a-zA-Z]:\\program files \(x86\)",
    r"^[a-zA-Z]:\\system32",
    r"^[a-zA-Z]:\\recovery",
    r"^[a-zA-Z]:\\boot",
    # Unix / Linux
    r"^/etc",
    r"^/sys",
    r"^/proc",
    r"^/bin",
    r"^/sbin",
    r"^/usr/bin",
    r"^/boot",
    r"^/dev",
    r"^/root",
]


class AuthorizedFolderSandbox:
    """Manages whitelisted directory sandboxes with path traversal prevention."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from config.settings import BASE_DIR
            self.db_path = str(BASE_DIR / "jarvis.db")
        else:
            self.db_path = db_path
        
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._init_db()

    def _init_db(self):
        """Initialize SQLite storage for authorized folders."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS authorized_folders (
                        folder_id TEXT PRIMARY KEY,
                        path TEXT UNIQUE NOT NULL,
                        access_level TEXT NOT NULL DEFAULT 'read_only',
                        authorized_at TEXT NOT NULL
                    )
                """)
                conn.commit()
                # Load existing folders into cache
                cursor.execute("SELECT folder_id, path, access_level, authorized_at FROM authorized_folders")
                for fid, fpath, access, created in cursor.fetchall():
                    self._memory_cache[fid] = {
                        "folder_id": fid,
                        "path": os.path.realpath(fpath),
                        "access_level": access,
                        "authorized_at": created,
                    }
        except Exception as e:
            log.warning(f"AuthorizedFolderSandbox: SQLite initialization error ({e}). Using in-memory store.")

    def _is_disallowed_system_path(self, canonical_path: str) -> bool:
        """Check if path targets protected operating system files."""
        normalized = canonical_path.lower().replace("/", "\\")
        
        # Check root of drive (e.g., C:\ or D:\ or \)
        if re.match(r"^[a-zA-Z]:\\?$", normalized) or normalized in ("\\", "/"):
            return True

        # Path with forward slashes and without drive letter for cross-platform matching
        path_without_drive = re.sub(r"^[a-zA-Z]:", "", normalized).replace("\\", "/")
        if not path_without_drive.startswith("/"):
            path_without_drive = "/" + path_without_drive

        for pattern in DISALLOWED_SYSTEM_ROOTS:
            if re.search(pattern, normalized, re.IGNORECASE) or re.search(pattern, path_without_drive, re.IGNORECASE):
                return True
        return False

    def authorize_folder(self, folder_path: str, access_level: str = "read_only") -> Dict[str, Any]:
        """
        Explicitly whitelist a directory for device access.
        
        Args:
            folder_path: Path to authorize.
            access_level: 'read_only' or 'read_write'.
            
        Returns:
            Dict containing folder_id, canonical path, and access_level.
            
        Raises:
            ValueError: If path contains invalid characters or targets system directories.
        """
        if "\0" in folder_path:
            raise ValueError("Path traversal attempt detected: embedded null character.")

        access_level = access_level.lower()
        if access_level not in ("read_only", "read_write"):
            raise ValueError(f"Invalid access_level: {access_level}. Must be 'read_only' or 'read_write'.")

        canonical = os.path.realpath(os.path.abspath(folder_path))

        if self._is_disallowed_system_path(canonical):
            raise PermissionError(f"Refusing to authorize protected system directory: {canonical}")

        # Check if already authorized
        for entry in self._memory_cache.values():
            if entry["path"].lower() == canonical.lower():
                entry["access_level"] = access_level
                self._update_db(entry)
                log.info(f"Updated authorized folder permission: {canonical} -> {access_level}")
                return entry

        folder_id = f"fld-{uuid.uuid4().hex[:8]}"
        record = {
            "folder_id": folder_id,
            "path": canonical,
            "access_level": access_level,
            "authorized_at": datetime.now(timezone.utc).isoformat(),
        }

        self._memory_cache[folder_id] = record
        self._insert_db(record)
        log.info(f"Authorized directory for device access: {canonical} ({access_level}) [ID: {folder_id}]")
        return record

    def revoke_folder(self, folder_id_or_path: str) -> bool:
        """Revoke authorization for a directory."""
        target_id = None
        canonical = os.path.realpath(os.path.abspath(folder_id_or_path))

        for fid, entry in self._memory_cache.items():
            if fid == folder_id_or_path or entry["path"].lower() == canonical.lower():
                target_id = fid
                break

        if not target_id:
            return False

        del self._memory_cache[target_id]
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM authorized_folders WHERE folder_id = ?", (target_id,))
                conn.commit()
        except Exception as e:
            log.warning(f"Error deleting folder from DB: {e}")

        log.info(f"Revoked folder authorization: {folder_id_or_path}")
        return True

    def list_authorized_folders(self) -> List[Dict[str, Any]]:
        """List all currently authorized directories."""
        return list(self._memory_cache.values())

    def is_path_authorized(self, file_path: str, require_write: bool = False) -> bool:
        """Check if file_path resides within an authorized folder with adequate access level."""
        try:
            self.validate_path_access(file_path, require_write=require_write)
            return True
        except (PermissionError, ValueError):
            return False

    def validate_path_access(self, file_path: str, require_write: bool = False) -> str:
        """
        Validate path against authorized sandbox.
        
        Args:
            file_path: Relative or absolute file path.
            require_write: Whether write permission is required.
            
        Returns:
            Canonical real path if authorized.
            
        Raises:
            ValueError: If path contains traversal sequences ('..') or null bytes.
            PermissionError: If path is not within an authorized folder or lacks write permission.
        """
        if not file_path:
            raise ValueError("Target file path cannot be empty.")

        if "\0" in file_path:
            raise ValueError("Path traversal attack detected: null byte in path.")

        # Check explicit path traversal strings
        parts = file_path.replace("\\", "/").split("/")
        if ".." in parts:
            raise ValueError(f"Path traversal attack detected: '..' in path '{file_path}'.")

        canonical_file = os.path.realpath(os.path.abspath(file_path))

        # Check system root protection
        if self._is_disallowed_system_path(canonical_file):
            raise PermissionError(f"Access to protected system path denied: {canonical_file}")

        # Check against authorized folders
        matched_folder = None
        for entry in self._memory_cache.values():
            folder_root = entry["path"]
            # Common path check: canonical_file must be inside folder_root
            try:
                if os.path.commonpath([folder_root, canonical_file]) == folder_root:
                    matched_folder = entry
                    break
            except ValueError:
                # Different drives on Windows (e.g., C: vs D:)
                continue

        if not matched_folder:
            raise PermissionError(
                f"Access denied: Path '{canonical_file}' is outside all authorized folder sandboxes."
            )

        if require_write and matched_folder["access_level"] != "read_write":
            raise PermissionError(
                f"Write access denied: Folder '{matched_folder['path']}' is configured as 'read_only'."
            )

        return canonical_file

    def _insert_db(self, record: Dict[str, Any]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO authorized_folders VALUES (?, ?, ?, ?)",
                    (record["folder_id"], record["path"], record["access_level"], record["authorized_at"]),
                )
                conn.commit()
        except Exception as e:
            log.warning(f"Error persisting authorized folder to DB: {e}")

    def _update_db(self, record: Dict[str, Any]):
        self._insert_db(record)


authorized_folder_sandbox = AuthorizedFolderSandbox()
