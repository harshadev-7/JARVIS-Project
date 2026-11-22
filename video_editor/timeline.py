"""J.A.R.V.I.S. Video Timeline & Cutlist Manager.

Handles multi-clip editing:
- Define in/out trim points
- Reorder, concatenate, and split video clips
- Manage transition points between segments
- Generates FFmpeg concat demuxer scripts
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from video_editor.safety import media_safety_guard


@dataclass
class ClipSegment:
    """Individual segment in a video timeline."""
    source_path: str
    start_sec: float = 0.0
    end_sec: Optional[float] = None
    transition_in: Optional[str] = None   # e.g. "fade", "dissolve"
    transition_out: Optional[str] = None
    volume: float = 1.0
    speed: float = 1.0

    @property
    def duration(self) -> float:
        if self.end_sec is not None:
            return max(0.0, (self.end_sec - self.start_sec) / self.speed)
        return 0.0


class VideoTimeline:
    """Manages ordered video clips, trimming, and composition instructions."""

    def __init__(self, title: str = "JARVIS Project"):
        self.title = title
        self.segments: List[ClipSegment] = []

    def add_clip(
        self,
        source_path: str | Path,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        speed: float = 1.0,
        volume: float = 1.0,
    ) -> ClipSegment:
        """Add a trimmed clip segment to the timeline."""
        p = media_safety_guard.validate_file_path(source_path, must_exist=True)
        seg = ClipSegment(
            source_path=str(p.resolve()),
            start_sec=max(0.0, start_sec),
            end_sec=end_sec,
            speed=max(0.25, min(4.0, speed)),
            volume=max(0.0, min(2.0, volume)),
        )
        self.segments.append(seg)
        return seg

    def clear(self) -> None:
        """Clear all clips from timeline."""
        self.segments.clear()

    @property
    def total_clips(self) -> int:
        return len(self.segments)

    def total_duration(self) -> float:
        """Compute total timeline length in seconds."""
        return sum(s.duration for s in self.segments if s.duration > 0)

    def generate_concat_file(self, output_file: Optional[Path] = None) -> Path:
        """Generate an FFmpeg concat demuxer text file for zero-reencode fast merging."""
        target = output_file or Path(tempfile.gettempdir()) / f"jarvis_concat_{id(self)}.txt"
        lines = []
        for s in self.segments:
            # Escape single quotes for ffmpeg concat file
            clean_p = s.source_path.replace("'", "'\\''")
            lines.append(f"file '{clean_p}'")
            if s.start_sec > 0:
                lines.append(f"inpoint {s.start_sec:.3f}")
            if s.end_sec is not None:
                lines.append(f"outpoint {s.end_sec:.3f}")

        target.write_text("\n".join(lines), encoding="utf-8")
        return target

    def build_trim_filter_args(self, segment: ClipSegment) -> List[str]:
        """Construct FFmpeg command options for a specific trim segment."""
        args = ["-ss", f"{segment.start_sec:.3f}"]
        if segment.end_sec is not None:
            dur = segment.end_sec - segment.start_sec
            args.extend(["-t", f"{dur:.3f}"])
        args.extend(["-i", segment.source_path])
        return args


__all__ = ["ClipSegment", "VideoTimeline"]
