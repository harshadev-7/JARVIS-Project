"""
JARVIS AI — Vertical Video Production Engine
=============================================
Assembles 9:16 vertical short-form videos (YouTube Shorts, Instagram Reels, TikTok)
using FFmpeg / MoviePy specifications with voiceover audio sync and subtitles.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.logger import log


class VideoProject(BaseModel):
    """Specification for vertical 9:16 short video."""
    project_id: str
    title: str
    aspect_ratio: str = "9:16"
    width: int = 1080
    height: int = 1920
    fps: int = 30
    duration_sec: float = 45.0
    audio_track_path: Optional[str] = None
    subtitles_srt: Optional[str] = None
    output_path: str = ""
    status: str = "draft"


class VideoBuilder:
    """Generates rendering specifications and FFmpeg commands for vertical videos."""

    def __init__(self):
        self.ffmpeg_available = shutil.which("ffmpeg") is not None

    def create_short_project(
        self,
        project_id: str,
        title: str,
        duration_sec: float = 45.0,
        audio_path: Optional[str] = None,
    ) -> VideoProject:
        """Initialize vertical video configuration."""
        out_path = f"logs/rendered_{project_id}.mp4"
        return VideoProject(
            project_id=project_id,
            title=title,
            aspect_ratio="9:16",
            width=1080,
            height=1920,
            fps=30,
            duration_sec=duration_sec,
            audio_track_path=audio_path,
            output_path=out_path,
            status="ready_to_render",
        )

    def generate_ffmpeg_render_command(self, project: VideoProject, background_video: str) -> List[str]:
        """
        Generate FFmpeg rendering command parameters:
        Scales & crops to 9:16 (1080x1920), overlays voiceover audio, sets H.264 video codec.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i", background_video,
        ]
        if project.audio_track_path:
            cmd.extend(["-i", project.audio_track_path])

        cmd.extend([
            "-vf", f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-t", str(project.duration_sec),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            project.output_path,
        ])
        return cmd


video_builder = VideoBuilder()
