"""JARVIS AutoClip Engine — Compilation Builder
==============================================
Assembles multiple extracted highlight clips into a unified compilation reel
(e.g., "Top 5 Highlights", "Best Moments").

Inspired by AutoClip's `step6_video.py` collection compilation features.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

try:
    import imageio_ffmpeg
    DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    DEFAULT_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

logger = logging.getLogger(__name__)


class CompilationBuilder:
    """Concatenates highlight clips into a single compilation video."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or DEFAULT_FFMPEG

    def build_compilation(
        self,
        clip_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        reencode: bool = False,
    ) -> bool:
        """Merges multiple MP4 clips into one compilation MP4."""
        valid_clips = [Path(p) for p in clip_paths if Path(p).is_file() and Path(p).stat().st_size > 0]
        if not valid_clips:
            logger.warning("No valid clips provided for compilation.")
            return False

        if len(valid_clips) == 1:
            shutil.copy(valid_clips[0], output_path)
            return True

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary concat demuxer text file
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            concat_txt = tf.name
            for p in valid_clips:
                clean_path = str(p.resolve()).replace("\\", "/")
                tf.write(f"file '{clean_path}'\n")

        try:
            if not reencode:
                cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_txt,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(output_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
                    return True
                logger.info("Stream copy concat failed, retrying with re-encoding: %s", proc.stderr)

            # Fallback with re-encoding
            cmd_reencode = [
                self.ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_txt,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path),
            ]
            proc2 = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=300)
            return proc2.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0
        except Exception as exc:
            logger.error("Compilation assembly error: %s", exc)
            return False
        finally:
            if os.path.exists(concat_txt):
                try:
                    os.remove(concat_txt)
                except Exception:
                    pass
