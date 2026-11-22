"""
JARVIS Video Editing Engine — Video Input & Metadata Manager
=============================================================
Handles video ingestion, multi-clip scanning, stream probing, thumbnail
generation, and keyframe extraction for MP4, MOV, MKV, AVI, and WEBM.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import subprocess
import json
import re

import imageio_ffmpeg
from core.logger import log
from video_editor.workspace import workspace_manager


@dataclass
class VideoMediaMetadata:
    """Rich metadata profile for an ingested video clip."""
    file_path: str
    filename: str
    width: int
    height: int
    fps: float
    duration_sec: float
    total_frames: int
    video_codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_channels: int = 0
    audio_sample_rate: int = 0
    aspect_ratio: str = "16:9"
    bitrate_kbps: int = 0
    file_size_bytes: int = 0
    thumbnails: List[str] = field(default_factory=list)


class VideoInputManager:
    """Probes video assets, generates thumbnails, and extracts keyframes."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        try:
            self.ffmpeg_bin = ffmpeg_bin or imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            log.warning(f"Failed to resolve imageio-ffmpeg binary: {e}")
            self.ffmpeg_bin = "ffmpeg"

    def probe_video(self, file_path: str | Path) -> VideoMediaMetadata:
        """Inspect a video file and return a structured VideoMediaMetadata object."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        file_size = path.stat().st_size
        filename = path.name

        # Default fallbacks
        width, height = 1920, 1080
        fps = 30.0
        duration = 0.0
        video_codec = "h264"
        has_audio = False
        audio_codec = None
        audio_channels = 0
        audio_sample_rate = 0
        bitrate = 0

        # Execute FFmpeg probe command (-i file)
        cmd = [self.ffmpeg_bin, "-i", str(path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=15)
            output = res.stderr or res.stdout or ""

            # Extract Duration: 00:00:26.50, start: 0.000000, bitrate: 1200 kb/s
            dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", output)
            if dur_m:
                hours, mins, secs = float(dur_m.group(1)), float(dur_m.group(2)), float(dur_m.group(3))
                duration = hours * 3600 + mins * 60 + secs

            bitrate_m = re.search(r"bitrate:\s*(\d+)\s*kb/s", output)
            if bitrate_m:
                bitrate = int(bitrate_m.group(1))

            # Extract Video Stream: Video: h264 (...), yuv420p(...), 1080x1920 (...), 30 fps, 30 tbr
            vid_m = re.search(r"Video:\s*([a-zA-Z0-9_\-]+).*?,\s*(\d{2,5})x(\d{2,5})", output)
            if vid_m:
                video_codec = vid_m.group(1)
                width = int(vid_m.group(2))
                height = int(vid_m.group(3))

            fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", output)
            if fps_m:
                fps = float(fps_m.group(1))

            # Extract Audio Stream: Audio: aac (...), 44100 Hz, stereo, fltp, 128 kb/s
            aud_m = re.search(r"Audio:\s*([a-zA-Z0-9_\-]+).*?,\s*(\d+)\s*Hz,\s*([a-zA-Z0-9_\-]+)", output)
            if aud_m:
                has_audio = True
                audio_codec = aud_m.group(1)
                audio_sample_rate = int(aud_m.group(2))
                channel_str = aud_m.group(3).lower()
                audio_channels = 2 if "stereo" in channel_str else (1 if "mono" in channel_str else 6)

        except Exception as e:
            log.warning(f"FFmpeg probing encountered issue for '{filename}': {e}")

        # Compute total frames and aspect ratio
        total_frames = max(1, int(duration * fps))
        aspect_ratio = self._calculate_aspect_ratio(width, height)

        return VideoMediaMetadata(
            file_path=str(path),
            filename=filename,
            width=width,
            height=height,
            fps=fps,
            duration_sec=duration,
            total_frames=total_frames,
            video_codec=video_codec,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_channels=audio_channels,
            audio_sample_rate=audio_sample_rate,
            aspect_ratio=aspect_ratio,
            bitrate_kbps=bitrate,
            file_size_bytes=file_size,
        )

    def generate_thumbnail(
        self,
        file_path: str | Path,
        timestamp_sec: float = 1.0,
        output_path: Optional[str | Path] = None,
        width: int = 480,
    ) -> Path:
        """Extract a single frame as a JPG thumbnail."""
        src = Path(file_path).resolve()
        if not output_path:
            thumb_dir = workspace_manager.dirs["media_temporary"] / "thumbnails"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            out_file = thumb_dir / f"{src.stem}_thumb_{int(timestamp_sec*1000)}.jpg"
        else:
            out_file = Path(output_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss", str(timestamp_sec),
            "-i", str(src),
            "-vframes", "1",
            "-vf", f"scale={width}:-1",
            "-q:v", "3",
            str(out_file),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and out_file.exists():
                return out_file
        except Exception as e:
            log.warning(f"Thumbnail generation failed for '{src.name}': {e}")

        # Fallback: create empty placeholder if needed
        return out_file

    def extract_keyframes(
        self,
        file_path: str | Path,
        count: int = 5,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Extract evenly distributed keyframes across the video duration."""
        meta = self.probe_video(file_path)
        out_dir = output_dir or (workspace_manager.dirs["media_temporary"] / "keyframes" / Path(file_path).stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        duration = max(1.0, meta.duration_sec)
        interval = duration / (count + 1)
        extracted: List[Path] = []

        for i in range(1, count + 1):
            t = i * interval
            out_frame = out_dir / f"frame_{i:03d}_{int(t*1000)}ms.jpg"
            thumb = self.generate_thumbnail(file_path, timestamp_sec=t, output_path=out_frame)
            if thumb.exists():
                extracted.append(thumb)

        return extracted

    def scan_directory(self, folder_path: Optional[str | Path] = None) -> List[VideoMediaMetadata]:
        """Scan folder for all video clips and return their metadata."""
        target_dir = Path(folder_path) if folder_path else workspace_manager.dirs["media_input"]
        if not target_dir.exists():
            return []

        videos = [
            f for f in target_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
        ]
        results: List[VideoMediaMetadata] = []
        for v in sorted(videos):
            try:
                results.append(self.probe_video(v))
            except Exception as e:
                log.warning(f"Could not probe video '{v.name}': {e}")
        return results

    def _calculate_aspect_ratio(self, w: int, h: int) -> str:
        """Return standardized aspect ratio string e.g. '16:9', '9:16', '1:1', '4:3'."""
        if h == 0:
            return "16:9"
        ratio = w / h
        if abs(ratio - 9/16) < 0.05:
            return "9:16"
        elif abs(ratio - 16/9) < 0.05:
            return "16:9"
        elif abs(ratio - 1.0) < 0.05:
            return "1:1"
        elif abs(ratio - 4/3) < 0.05:
            return "4:3"
        elif abs(ratio - 21/9) < 0.08:
            return "21:9"
        return f"{w}:{h}"


# Global input manager singleton
input_manager = VideoInputManager()
