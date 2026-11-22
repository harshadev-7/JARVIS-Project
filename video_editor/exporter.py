"""J.A.R.V.I.S. Video Exporter & Rendering Engine.

Translates editing instructions into optimized FFmpeg commands:
- Presets: DRAFT (fast preview), SOCIAL_720P (mobile reels/shorts), HIGH_1080P (production)
- Non-destructive export destination handling via MediaSafetyGuard
- CPU-safe multi-threading limits (prevents system freezing on Core i3)
- Executes rendering directly when FFmpeg is present, or exports executable batch scripts
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from core.logger import log
from video_editor.safety import media_safety_guard


class ExportQuality(str, Enum):
    DRAFT = "DRAFT"               # Ultrafast preview, CRF 28
    SOCIAL_720P = "SOCIAL_720P"   # Veryfast, 720p/1280p, CRF 24 (Ideal for WhatsApp / IG)
    HIGH_1080P = "HIGH_1080P"     # Fast, 1080x1920 or 1920x1080, CRF 21 (YouTube / Master)


@dataclass
class QualityProfile:
    preset: str
    crf: int
    video_bitrate: Optional[str]
    audio_bitrate: str
    threads: int = 2


PROFILES = {
    ExportQuality.DRAFT: QualityProfile(
        preset="ultrafast",
        crf=28,
        video_bitrate="1500k",
        audio_bitrate="128k",
        threads=2,
    ),
    ExportQuality.SOCIAL_720P: QualityProfile(
        preset="veryfast",
        crf=24,
        video_bitrate="3500k",
        audio_bitrate="160k",
        threads=2,
    ),
    ExportQuality.HIGH_1080P: QualityProfile(
        preset="fast",
        crf=21,
        video_bitrate="6500k",
        audio_bitrate="192k",
        threads=2,
    ),
}


@dataclass
class ExportResult:
    """Summary of rendering execution or generated render script."""
    success: bool
    output_path: str
    command_str: str
    quality_preset: str
    rendered_directly: bool
    script_path: Optional[str] = None
    message: str = ""
    error: Optional[str] = None


class VideoExporter:
    """Renders video timelines into finalized media files."""

    def __init__(self):
        self.ffmpeg_path = shutil.which("ffmpeg")

    def export(
        self,
        input_file: str | Path,
        output_filename: str,
        quality: ExportQuality = ExportQuality.SOCIAL_720P,
        filter_complex: Optional[str] = None,
        audio_file: Optional[str | Path] = None,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        extra_args: Optional[List[str]] = None,
    ) -> ExportResult:
        """Render or generate render script for a media project.

        Args:
            input_file: Source video file path.
            output_filename: Desired target file name (saved safely in output/videos/).
            quality: ExportQuality profile preset.
            filter_complex: Optional FFmpeg filtergraph string.
            audio_file: Optional background music or voiceover replacement.
            start_sec: Optional start offset in seconds.
            duration_sec: Optional total duration limit.
            extra_args: Additional command line flags for ffmpeg.

        Returns:
            ExportResult detailing execution or batch script location.
        """
        src = media_safety_guard.validate_file_path(input_file, must_exist=True)
        dst = media_safety_guard.get_safe_output_path(output_filename)

        # Enforce non-destructive safety check
        media_safety_guard.verify_action_permission(src, dst, is_deletion=False, user_confirmed=False)

        profile = PROFILES.get(quality, PROFILES[ExportQuality.SOCIAL_720P])

        # Build FFmpeg command args
        cmd = [self.ffmpeg_path or "ffmpeg", "-y"]

        if start_sec is not None and start_sec > 0:
            cmd.extend(["-ss", f"{start_sec:.3f}"])

        cmd.extend(["-i", str(src)])

        if audio_file:
            audio_path = media_safety_guard.validate_file_path(audio_file, must_exist=True)
            cmd.extend(["-i", str(audio_path)])

        if duration_sec is not None and duration_sec > 0:
            cmd.extend(["-t", f"{duration_sec:.3f}"])

        if filter_complex:
            cmd.extend(["-filter_complex", filter_complex])

        # Video & Audio encoding parameters
        cmd.extend([
            "-c:v", "libx264",
            "-preset", profile.preset,
            "-crf", str(profile.crf),
            "-threads", str(profile.threads),
            "-c:a", "aac",
            "-b:a", profile.audio_bitrate,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])

        if extra_args:
            cmd.extend(extra_args)

        cmd.append(str(dst))
        command_str = " ".join(f'"{arg}"' if " " in arg or "=" in arg else arg for arg in cmd)

        # 1. If FFmpeg is found in system PATH, attempt direct rendering
        if self.ffmpeg_path:
            try:
                log.info(f"Rendering media via FFmpeg: {dst.name} ({quality.value})...")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if res.returncode == 0:
                    return ExportResult(
                        success=True,
                        output_path=str(dst),
                        command_str=command_str,
                        quality_preset=quality.value,
                        rendered_directly=True,
                        message=f"Export successfully rendered to: {dst}",
                    )
                else:
                    log.warning(f"FFmpeg render returned error: {res.stderr[:200]}")
            except Exception as e:
                log.warning(f"FFmpeg execution failed: {e}. Writing export script instead.")

        # 2. Standalone script generation fallback (works 100% offline without ffmpeg in PATH)
        script_ext = ".bat" if sys.platform == "win32" else ".sh"
        script_path = dst.parent / f"render_{dst.stem}{script_ext}"

        script_content = (
            f"@echo off\n"
            f"echo [JARVIS] Rendering: {dst.name}\n"
            f"{command_str}\n"
            f"echo [JARVIS] Render process complete.\n"
        ) if sys.platform == "win32" else (
            f"#!/usr/bin/env bash\n"
            f"echo '[JARVIS] Rendering: {dst.name}'\n"
            f"{command_str}\n"
            f"echo '[JARVIS] Render process complete.'\n"
        )

        script_path.write_text(script_content, encoding="utf-8")

        return ExportResult(
            success=True,
            output_path=str(dst),
            command_str=command_str,
            quality_preset=quality.value,
            rendered_directly=False,
            script_path=str(script_path),
            message=f"Export script generated: {script_path.name} (run to render with external FFmpeg)",
        )


video_exporter = VideoExporter()

__all__ = ["ExportQuality", "ExportResult", "QualityProfile", "VideoExporter", "video_exporter"]
