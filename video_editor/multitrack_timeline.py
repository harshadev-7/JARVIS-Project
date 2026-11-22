"""
JARVIS Video Editing Engine V2.0 — Multi-Track NLE Timeline Model
=================================================================
Implements an industry-standard 6-track non-linear editing timeline:
- Track 1 (V1): Primary video footage sequence
- Track 2 (V2): Overlay / B-roll / picture-in-picture
- Track 3 (V3): Visual effects & color grade adjustment layer
- Track 4 (T1): Typography, captions, KANKI.7 watermark & outro card
- Track 5 (A1): Primary music soundtrack with beat markers
- Track 6 (A2): Secondary audio effects / voiceover with audio ducking
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid
import time

from core.logger import log
from video_editor.workspace import workspace_manager


@dataclass
class TimelineClip:
    """Individual item placed on a timeline track."""
    clip_id: str
    source_path: str
    track_id: str             # "v1", "v2", "v3", "t1", "a1", "a2"
    timeline_in_sec: float
    timeline_out_sec: float
    source_in_sec: float = 0.0
    source_out_sec: Optional[float] = None
    speed: float = 1.0
    volume: float = 1.0
    effects: List[str] = field(default_factory=list)
    opacity: float = 1.0


@dataclass
class MultiTrackTimeline:
    """Multi-track editing composition project."""
    project_id: str
    title: str
    aspect_ratio: str = "9:16"
    width: int = 1080
    height: int = 1920
    fps: int = 30
    duration_sec: float = 30.0
    tracks: Dict[str, List[TimelineClip]] = field(default_factory=lambda: {
        "v1": [],  # Main footage
        "v2": [],  # Overlays & B-Roll
        "v3": [],  # VFX & Adjustment layers
        "t1": [],  # Titles & Branding
        "a1": [],  # Music & Soundtrack
        "a2": [],  # Sound FX & Voice
    })

    def add_clip(
        self,
        track_id: str,
        source_path: str | Path,
        timeline_in: float,
        duration: float,
        source_in: float = 0.0,
        speed: float = 1.0,
        effects: Optional[List[str]] = None,
    ) -> TimelineClip:
        """Place a media clip onto a designated track."""
        if track_id not in self.tracks:
            self.tracks[track_id] = []

        cid = f"clip-{uuid.uuid4().hex[:6]}"
        clip = TimelineClip(
            clip_id=cid,
            source_path=str(source_path),
            track_id=track_id,
            timeline_in_sec=round(timeline_in, 3),
            timeline_out_sec=round(timeline_in + duration, 3),
            source_in_sec=round(source_in, 3),
            source_out_sec=round(source_in + (duration * speed), 3),
            speed=speed,
            effects=effects or [],
        )
        self.tracks[track_id].append(clip)
        return clip

    def save_project_json(self, output_file: Optional[Path] = None) -> Path:
        """Serialize complete multi-track timeline to JSON."""
        target = output_file or (workspace_manager.dirs["editing_timelines"] / f"{self.project_id}_multitrack.json")
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "project_id": self.project_id,
            "title": self.title,
            "aspect_ratio": self.aspect_ratio,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tracks": {
                tid: [
                    {
                        "clip_id": c.clip_id,
                        "source": Path(c.source_path).name,
                        "timeline_in": c.timeline_in_sec,
                        "timeline_out": c.timeline_out_sec,
                        "speed": c.speed,
                        "effects": c.effects,
                    }
                    for c in clips
                ]
                for tid, clips in self.tracks.items()
            },
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.debug(f"Saved MultiTrackTimeline -> {target}")
        return target


def create_multitrack_timeline(
    project_id: str,
    title: str,
    duration_sec: float = 30.0,
    aspect_ratio: str = "9:16",
) -> MultiTrackTimeline:
    """Helper factory for initializing a 6-track timeline."""
    w, h = (1080, 1920) if aspect_ratio == "9:16" else ((1920, 1080) if aspect_ratio == "16:9" else (1080, 1080))
    return MultiTrackTimeline(
        project_id=project_id,
        title=title,
        aspect_ratio=aspect_ratio,
        width=w,
        height=h,
        duration_sec=duration_sec,
    )
