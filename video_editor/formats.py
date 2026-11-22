"""
JARVIS Video Editing Engine — Autonomous Video Formats & Presets
================================================================
Defines standardized output formats, geometry aspect ratios, and export
parameter presets for Instagram Reels, YouTube Shorts, Cinematic 16:9, and Square 1:1.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class ExportFormatConfig:
    """Standardized technical profile for an output video format."""
    format_name: str
    width: int
    height: int
    aspect_ratio: str
    fps: int = 30
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 22
    preset: str = "veryfast"
    video_bitrate_kbps: int = 4500
    audio_bitrate_kbps: int = 192
    pixel_format: str = "yuv420p"
    max_duration_sec: float = 60.0
    safe_zone_margin_v: int = 140

    def get_scale_crop_filter(self) -> str:
        """
        Generate FFmpeg filter expression to scale and crop any input
        media cleanly into this format's target resolution without black bars.
        """
        w, h = self.width, self.height
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


FORMAT_PRESETS: Dict[str, ExportFormatConfig] = {
    "reels": ExportFormatConfig(
        format_name="Instagram Reels",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        fps=30,
        crf=22,
        video_bitrate_kbps=5000,
        audio_bitrate_kbps=192,
        max_duration_sec=90.0,
        safe_zone_margin_v=160,
    ),
    "shorts": ExportFormatConfig(
        format_name="YouTube Shorts",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        fps=30,
        crf=21,
        video_bitrate_kbps=6000,
        audio_bitrate_kbps=192,
        max_duration_sec=60.0,
        safe_zone_margin_v=140,
    ),
    "cinematic": ExportFormatConfig(
        format_name="YouTube Cinematic Widescreen",
        width=1920,
        height=1080,
        aspect_ratio="16:9",
        fps=24,
        crf=20,
        video_bitrate_kbps=8000,
        audio_bitrate_kbps=256,
        max_duration_sec=300.0,
        safe_zone_margin_v=80,
    ),
    "square": ExportFormatConfig(
        format_name="Square Feed Video",
        width=1080,
        height=1080,
        aspect_ratio="1:1",
        fps=30,
        crf=22,
        video_bitrate_kbps=4000,
        audio_bitrate_kbps=160,
        max_duration_sec=120.0,
        safe_zone_margin_v=90,
    ),
}


def resolve_format_config(preset_query: str = "reels") -> ExportFormatConfig:
    """Resolve format configuration from user string query."""
    q = preset_query.lower().strip()
    if "short" in q:
        return FORMAT_PRESETS["shorts"]
    elif "cinematic" in q or "16:9" in q or "widescreen" in q or "youtube" in q:
        return FORMAT_PRESETS["cinematic"]
    elif "square" in q or "1:1" in q:
        return FORMAT_PRESETS["square"]
    else:  # default: reels (9:16, 1080x1920)
        return FORMAT_PRESETS["reels"]
