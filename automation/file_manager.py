"""J.A.R.V.I.S. — Safe File Manager (Phase 12).

Provides safe filesystem navigation, reading, writing, and inspection.

Strict Security Constraints (Section 13):
- Ask before deleting files (delete_file requires user_approved=True).
- Ask before deleting directories.
- Protects critical system directories (Windows, Program Files, System32).
- Enforces safe byte read limits to prevent RAM exhaustion.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import log


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FileOperationResult:
    """Outcome of a file operation."""
    operation: str
    path: str
    success: bool
    message: str
    requires_approval: bool = False
    data: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# File Manager
# ---------------------------------------------------------------------------

class FileManager:
    """Sandboxed, permission-gated local file manager."""

    # Protected paths that should never be modified or deleted
    PROTECTED_ROOTS = [
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\windows\\system32",
    ]

    def __init__(self, default_root: Optional[str] = None):
        self.default_root = Path(default_root or os.getcwd()).resolve()

    def _is_protected(self, path: Path) -> bool:
        """Check if target path is in a protected system directory."""
        resolved = str(path.resolve()).lower()
        return any(resolved.startswith(prot) for prot in self.PROTECTED_ROOTS)

    # ------------------------------------------------------------------
    # Directory Listing & Inspection
    # ------------------------------------------------------------------

    def list_directory(self, dir_path: Optional[str] = None) -> FileOperationResult:
        """List files and subdirectories."""
        target = Path(dir_path).resolve() if dir_path else self.default_root
        if not target.exists():
            return FileOperationResult(
                operation="list_directory",
                path=str(target),
                success=False,
                message=f"Directory does not exist: {target}",
            )
        if not target.is_dir():
            return FileOperationResult(
                operation="list_directory",
                path=str(target),
                success=False,
                message=f"Path is not a directory: {target}",
            )

        try:
            items = []
            for entry in sorted(target.iterdir()):
                items.append({
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                    "modified": datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).isoformat(),
                })
            return FileOperationResult(
                operation="list_directory",
                path=str(target),
                success=True,
                message=f"Listed {len(items)} items in '{target.name}'.",
                data=items,
            )
        except Exception as e:
            return FileOperationResult(
                operation="list_directory",
                path=str(target),
                success=False,
                message=f"Error listing directory: {e}",
            )

    # ------------------------------------------------------------------
    # File Reading
    # ------------------------------------------------------------------

    def read_file(self, file_path: str, max_chars: int = 50000) -> FileOperationResult:
        """Read text from a local file safely."""
        p = Path(file_path).resolve()
        if not p.exists() or not p.is_file():
            return FileOperationResult(
                operation="read_file",
                path=str(p),
                success=False,
                message=f"File not found: {p}",
            )

        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)

            truncated = p.stat().st_size > max_chars
            msg = f"Read {len(content)} characters from '{p.name}'" + (" (truncated)" if truncated else ".")
            return FileOperationResult(
                operation="read_file",
                path=str(p),
                success=True,
                message=msg,
                data={"content": content, "truncated": truncated, "total_bytes": p.stat().st_size},
            )
        except Exception as e:
            return FileOperationResult(
                operation="read_file",
                path=str(p),
                success=False,
                message=f"Error reading file: {e}",
            )

    # ------------------------------------------------------------------
    # File Writing
    # ------------------------------------------------------------------

    def write_file(self, file_path: str, content: str, overwrite: bool = False) -> FileOperationResult:
        """Write content to a file."""
        p = Path(file_path).resolve()

        if self._is_protected(p):
            return FileOperationResult(
                operation="write_file",
                path=str(p),
                success=False,
                message=f"Security Alert: Target path '{p}' is a protected system directory.",
            )

        if p.exists() and not overwrite:
            return FileOperationResult(
                operation="write_file",
                path=str(p),
                success=False,
                message=f"File '{p.name}' already exists and overwrite=False.",
            )

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)

            return FileOperationResult(
                operation="write_file",
                path=str(p),
                success=True,
                message=f"Successfully wrote {len(content)} characters to '{p.name}'.",
                data={"bytes_written": p.stat().st_size},
            )
        except Exception as e:
            return FileOperationResult(
                operation="write_file",
                path=str(p),
                success=False,
                message=f"Error writing file: {e}",
            )

    # ------------------------------------------------------------------
    # Deletion — Strict Security Gate (Must ask before deleting)
    # ------------------------------------------------------------------

    def delete_file(self, file_path: str, user_approved: bool = False) -> FileOperationResult:
        """Delete a file. STRICT SECURITY RULE: Requires explicit user approval."""
        p = Path(file_path).resolve()

        if not p.exists() or not p.is_file():
            return FileOperationResult(
                operation="delete_file",
                path=str(p),
                success=False,
                message=f"File not found: {p}",
            )

        if self._is_protected(p):
            return FileOperationResult(
                operation="delete_file",
                path=str(p),
                success=False,
                message=f"Security Alert: Cannot delete files from protected system path: {p}",
            )

        # Mandatory Human Confirmation Gate
        if not user_approved:
            log.warning(f"File deletion blocked without user approval: '{p}'")
            return FileOperationResult(
                operation="delete_file",
                path=str(p),
                success=False,
                requires_approval=True,
                message=(
                    f"⚠️ **Authorization Required:** Deletion of file '{p.name}' is a destructive action. "
                    "Confirmation is required before deletion."
                ),
            )

        try:
            p.unlink()
            log.info(f"File deleted with user authorization: '{p}'")
            return FileOperationResult(
                operation="delete_file",
                path=str(p),
                success=True,
                message=f"File '{p.name}' deleted successfully with approval.",
            )
        except Exception as e:
            return FileOperationResult(
                operation="delete_file",
                path=str(p),
                success=False,
                message=f"Error deleting file: {e}",
            )

    def delete_directory(self, dir_path: str, user_approved: bool = False) -> FileOperationResult:
        """Delete a directory. STRICT SECURITY RULE: Requires explicit user approval."""
        p = Path(dir_path).resolve()

        if not p.exists() or not p.is_dir():
            return FileOperationResult(
                operation="delete_directory",
                path=str(p),
                success=False,
                message=f"Directory not found: {p}",
            )

        if self._is_protected(p):
            return FileOperationResult(
                operation="delete_directory",
                path=str(p),
                success=False,
                message=f"Security Alert: Cannot delete protected directory: {p}",
            )

        if not user_approved:
            return FileOperationResult(
                operation="delete_directory",
                path=str(p),
                success=False,
                requires_approval=True,
                message=f"⚠️ **Authorization Required:** Deleting directory '{p.name}' requires explicit confirmation.",
            )

        try:
            shutil.rmtree(p)
            return FileOperationResult(
                operation="delete_directory",
                path=str(p),
                success=True,
                message=f"Directory '{p.name}' and all contents removed with approval.",
            )
        except Exception as e:
            return FileOperationResult(
                operation="delete_directory",
                path=str(p),
                success=False,
                message=f"Error deleting directory: {e}",
            )


file_manager = FileManager()

__all__ = ["FileManager", "FileOperationResult", "file_manager"]
