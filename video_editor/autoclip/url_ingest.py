"""JARVIS AutoClip Engine — URL Ingestion & Video Downloader
===========================================================
Downloads public long-form videos and accompanying subtitles from YouTube,
Bilibili, and web sources using `yt-dlp` for automated clipping.

Inspired by AutoClip's `download_recovery.py` and link import pipeline.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class UrlDownloadResult:
    """Result of a video and subtitle download operation."""
    success: bool
    video_path: Optional[Path]
    srt_path: Optional[Path]
    title: str
    duration_sec: float
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "video_path": str(self.video_path) if self.video_path else None,
            "srt_path": str(self.srt_path) if self.srt_path else None,
            "title": self.title,
            "duration_sec": self.duration_sec,
            "error_message": self.error_message,
        }


def is_supported_url(text: str) -> bool:
    """Checks whether the text represents a web video URL."""
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not (s.startswith("http://") or s.startswith("https://")):
        return False
    try:
        parsed = urlparse(s)
        hostname = (parsed.hostname or "").lower()
        return bool(hostname and "." in hostname)
    except Exception:
        return False


class UrlIngest:
    """Handles downloading of video footage and automatic captions."""

    @classmethod
    def download_media(
        cls,
        url: str,
        output_dir: Union[str, Path],
        max_height: int = 1080,
    ) -> UrlDownloadResult:
        """Downloads video and subtitles for the specified URL."""
        if not is_supported_url(url):
            return UrlDownloadResult(
                success=False,
                video_path=None,
                srt_path=None,
                title="",
                duration_sec=0.0,
                error_message=f"Invalid URL: {url}",
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            import yt_dlp
        except ImportError:
            return UrlDownloadResult(
                success=False,
                video_path=None,
                srt_path=None,
                title="",
                duration_sec=0.0,
                error_message="yt-dlp is not installed. Please install it with `pip install yt-dlp`.",
            )

        out_template = str(out_path / "%(id)s_%(title).40s.%(ext)s")

        ydl_opts: Dict[str, Any] = {
            "outtmpl": out_template,
            "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-orig", "hi", "zh-Hans", "zh"],
            "subtitlesformat": "srt/vtt/best",
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "ignoreerrors": False,
        }

        try:
            logger.info("Starting yt-dlp download for URL: %s", url)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return UrlDownloadResult(
                        success=False,
                        video_path=None,
                        srt_path=None,
                        title="",
                        duration_sec=0.0,
                        error_message="Could not extract video metadata from URL.",
                    )

                title = info.get("title", "Online Video")
                duration = float(info.get("duration", 0.0))
                video_id = info.get("id", "")

                # Locate downloaded video file
                video_file: Optional[Path] = None
                for ext in ["mp4", "mkv", "webm"]:
                    matches = list(out_path.glob(f"*{video_id}*.{ext}"))
                    if matches:
                        video_file = matches[0]
                        break

                if not video_file:
                    # Fallback to newest video in directory
                    vids = sorted(
                        [p for p in out_path.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm"}],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if vids:
                        video_file = vids[0]

                # Locate downloaded subtitles file
                srt_file: Optional[Path] = None
                sub_matches = sorted(
                    [p for p in out_path.iterdir() if p.suffix.lower() in {".srt", ".vtt"}],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if sub_matches:
                    srt_file = sub_matches[0]

                if not video_file or not video_file.exists():
                    return UrlDownloadResult(
                        success=False,
                        video_path=None,
                        srt_path=None,
                        title=title,
                        duration_sec=duration,
                        error_message="Download completed but output media file was not found.",
                    )

                return UrlDownloadResult(
                    success=True,
                    video_path=video_file,
                    srt_path=srt_file,
                    title=title,
                    duration_sec=duration,
                    error_message="",
                )

        except Exception as exc:
            logger.error("Failed to download video from URL %s: %s", url, exc)
            return UrlDownloadResult(
                success=False,
                video_path=None,
                srt_path=None,
                title="",
                duration_sec=0.0,
                error_message=f"Video download failed: {str(exc)}",
            )
