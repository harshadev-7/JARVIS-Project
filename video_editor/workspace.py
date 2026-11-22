"""
JARVIS Video Editing Engine — Workspace Manager
================================================
Manages organized workspace directories for video inputs, audio assets,
temporary caches, exports, and project timelines:

JARVIS/
    media/
        input/        <- Source video clips (MP4, MOV, MKV, AVI)
        audio/        <- Background music, voiceovers, sound FX
        images/       <- Overlays, watermarks, graphics
        temporary/    <- Keyframes, thumbnails, chunk caches
    output/
        reels/        <- 9:16 Instagram Reels
        shorts/       <- 9:16 YouTube Shorts
        cinematic/    <- 16:9 Widescreen exports
    editing/
        projects/     <- Saved project metadata (JSON)
        timelines/    <- Serialized cutlists and timeline tracks
        presets/      <- Style and effect presets
"""

from pathlib import Path
from typing import List, Dict, Optional
import shutil
import time

from core.logger import log

# Base directory for the JARVIS project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Workspace Paths
WORKSPACE_DIRS: Dict[str, Path] = {
    "media_input": PROJECT_ROOT / "media" / "input",
    "media_audio": PROJECT_ROOT / "media" / "audio",
    "media_images": PROJECT_ROOT / "media" / "images",
    "media_temporary": PROJECT_ROOT / "media" / "temporary",
    "output_reels": PROJECT_ROOT / "output" / "reels",
    "output_shorts": PROJECT_ROOT / "output" / "shorts",
    "output_cinematic": PROJECT_ROOT / "output" / "cinematic",
    "editing_projects": PROJECT_ROOT / "editing" / "projects",
    "editing_timelines": PROJECT_ROOT / "editing" / "timelines",
    "editing_presets": PROJECT_ROOT / "editing" / "presets",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class WorkspaceManager:
    """Initializes, queries, and maintains the video editing workspace."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or PROJECT_ROOT
        self.dirs = {k: self.root / v.relative_to(PROJECT_ROOT) for k, v in WORKSPACE_DIRS.items()}
        self.ensure_workspace()

    def ensure_workspace(self) -> None:
        """Create all required workspace directories if they do not exist."""
        for name, dir_path in self.dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
        log.debug(f"Workspace directories verified at {self.root}")

    def list_input_videos(self) -> List[Path]:
        """Return all supported video files found in media/input/."""
        input_dir = self.dirs["media_input"]
        if not input_dir.exists():
            return []
        files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

    def list_input_audio(self) -> List[Path]:
        """Return all supported music/audio files found in media/audio/."""
        audio_dir = self.dirs["media_audio"]
        if not audio_dir.exists():
            return []
        files = [
            f for f in audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

    def get_output_dir(self, format_type: str = "reels") -> Path:
        """Resolve output folder for the given target format."""
        fmt = format_type.lower()
        if "short" in fmt:
            return self.dirs["output_shorts"]
        elif "cinematic" in fmt or "16:9" in fmt or "widescreen" in fmt:
            return self.dirs["output_cinematic"]
        else:
            return self.dirs["output_reels"]

    def cleanup_temporary(self, max_age_hours: float = 24.0) -> int:
        """Remove temporary cached frames and intermediate audio files older than max_age_hours."""
        temp_dir = self.dirs["media_temporary"]
        if not temp_dir.exists():
            return 0
        now = time.time()
        removed_count = 0
        for item in temp_dir.iterdir():
            try:
                if (now - item.stat().st_mtime) > (max_age_hours * 3600):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                    removed_count += 1
            except Exception as e:
                log.warning(f"Failed to cleanup temp item '{item.name}': {e}")
        return removed_count


# Global singleton workspace instance
workspace_manager = WorkspaceManager()
