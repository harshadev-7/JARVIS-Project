"""JARVIS AutoClip Engine — Video Processor & Vertical Short Slicer
==================================================================
Performs frame-accurate media clipping, 9:16 vertical conversion for Shorts/Reels,
subtitle burn-in, and faststart metadata tagging using local FFmpeg.

Inspired by AutoClip's `video_processor.py` and `step6_video.py`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import imageio_ffmpeg
    DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    DEFAULT_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Manages FFmpeg video extraction, cropping, and subtitle burning."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or DEFAULT_FFMPEG

    def is_ffmpeg_available(self) -> bool:
        """Checks if the configured FFmpeg executable is operational."""
        try:
            res = subprocess.run([self.ffmpeg_bin, "-version"], capture_output=True, timeout=5)
            return res.returncode == 0
        except Exception:
            return False

    def build_extract_command(
        self,
        input_video: Union[str, Path],
        output_path: Union[str, Path],
        start_sec: float,
        duration_sec: float,
        vertical: bool = True,
        subtitles_file: Optional[Union[str, Path]] = None,
        watermark_text: Optional[str] = None,
    ) -> List[str]:
        """Constructs the optimized FFmpeg command line parameters."""
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss", f"{max(0.0, float(start_sec)):.3f}",
            "-i", str(input_video),
            "-t", f"{max(0.1, float(duration_sec)):.3f}",
        ]

        # Assemble video filter chain
        filters: List[str] = []

        if vertical:
            # 9:16 vertical crop: scale to fill 1080x1920 then center crop
            filters.append("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920")

        if subtitles_file and Path(subtitles_file).exists():
            # Escape path for FFmpeg subtitles filter on Windows
            sub_path_escaped = str(subtitles_file).replace("\\", "/").replace(":", "\\:")
            filters.append(f"subtitles='{sub_path_escaped}':force_style='FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3'")

        if watermark_text:
            escaped_watermark = watermark_text.replace(":", "\\:").replace("'", "")
            filters.append(f"drawtext=text='{escaped_watermark}':fontcolor=white@0.8:fontsize=24:x=w-tw-30:y=30")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ])

        return cmd

    def extract_clip(
        self,
        input_video: Union[str, Path],
        output_path: Union[str, Path],
        start_sec: float,
        duration_sec: float,
        vertical: bool = True,
        subtitles_file: Optional[Union[str, Path]] = None,
        watermark_text: Optional[str] = None,
    ) -> bool:
        """Executes FFmpeg to render a highlight clip."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = self.build_extract_command(
            input_video=input_video,
            output_path=output_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            vertical=vertical,
            subtitles_file=subtitles_file,
            watermark_text=watermark_text,
        )

        try:
            logger.info("Executing FFmpeg clip extraction: %s", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                logger.error("FFmpeg extraction failed (code %d): %s", proc.returncode, proc.stderr)
                return False
            return out_file.exists() and out_file.stat().st_size > 0
        except Exception as exc:
            logger.error("FFmpeg execution error: %s", exc)
            return False

    def generate_thumbnail(
        self,
        video_path: Union[str, Path],
        output_thumb_path: Union[str, Path],
        timestamp_sec: float = 1.0,
    ) -> bool:
        """Captures a representative still frame as thumbnail."""
        out_thumb = Path(output_thumb_path)
        out_thumb.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss", f"{max(0.0, float(timestamp_sec)):.2f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_thumb_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=15)
            return res.returncode == 0 and out_thumb.exists()
        except Exception as exc:
            logger.warning("Thumbnail generation failed: %s", exc)
            return False
