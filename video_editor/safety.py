"""J.A.R.V.I.S. Video Editing Safety & Authorization Guard.

Enforces strict safety rules for media operations:
- Never overwrite or delete original source files without explicit confirmation
- Enforces non-destructive export directory isolation
- Validates file extensions and prevents directory traversal attacks
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set

from core.logger import log


class MediaSafetyError(PermissionError):
    """Raised when an operation attempts unauthorized modification or deletion of source media."""
    pass


class MediaSafetyGuard:
    """Guards media assets from accidental deletion or destructive overwriting."""

    ALLOWED_EXTENSIONS: Set[str] = {
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
        ".mp3", ".wav", ".aac", ".m4a", ".ogg", ".srt", ".vtt"
    }

    DEFAULT_OUTPUT_DIR = Path("D:/Documents/JARVIS/output/videos")

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_file_path(self, path: str | Path, must_exist: bool = True) -> Path:
        """Verify that path has a permitted video/audio extension and exists if required."""
        p = Path(path).resolve()
        if p.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: '{p.suffix}'. Permitted: {sorted(self.ALLOWED_EXTENSIONS)}"
            )
        if must_exist and not p.exists():
            raise FileNotFoundError(f"Media file not found: {p}")
        return p

    def get_safe_output_path(self, filename: str) -> Path:
        """Generate a guaranteed non-destructive output path within the output directory."""
        clean_name = Path(filename).name
        target = self.output_dir / clean_name

        # If file already exists, generate a timestamped or numbered alternative
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            counter = 1
            while (self.output_dir / f"{stem}_{counter}{suffix}").exists():
                counter += 1
            target = self.output_dir / f"{stem}_{counter}{suffix}"

        return target

    def verify_action_permission(
        self,
        source_file: str | Path,
        destination_file: Optional[str | Path] = None,
        is_deletion: bool = False,
        user_confirmed: bool = False,
    ) -> bool:
        """Verify that the media action does not destructively modify original content.

        Raises:
            MediaSafetyError: If action would delete or overwrite original media without consent.
        """
        src = Path(source_file).resolve()

        # Rule 1: Deletion of original source requires explicit user confirmation
        if is_deletion and not user_confirmed:
            raise MediaSafetyError(
                f"Safety Violation: Cannot delete original media '{src.name}' without explicit user confirmation."
            )

        # Rule 2: Destination cannot match source unless user explicitly approved in-place edit
        if destination_file is not None:
            dst = Path(destination_file).resolve()
            if src == dst and not user_confirmed:
                raise MediaSafetyError(
                    f"Safety Violation: In-place overwrite of original media '{src.name}' is blocked. "
                    "Provide a separate destination path or grant explicit approval."
                )

        return True


media_safety_guard = MediaSafetyGuard()

__all__ = ["MediaSafetyError", "MediaSafetyGuard", "media_safety_guard"]
