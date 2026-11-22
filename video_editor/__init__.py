"""J.A.R.V.I.S. Video Editing Automation Module (Phase 19).

Provides AI-powered, non-destructive video manipulation:
- Media inspection and stream property extraction
- Timeline assembly, cutlists, and multi-clip concatenation
- Scene detection and visual moment ranking
- Audio beat extraction and cut synchronization
- SRT subtitle creation, parsing, and video burning
- 9:16 vertical formatting for YouTube Shorts & Instagram Reels
- Multi-tier export quality (DRAFT, SOCIAL_720P, HIGH_1080P)
- Strict non-destructive media safety guardrails
"""

from video_editor.safety import MediaSafetyError, MediaSafetyGuard, media_safety_guard
from video_editor.inspector import MediaMetadata, VideoInspector
from video_editor.timeline import ClipSegment, VideoTimeline
from video_editor.scene_detector import SceneChange, SceneDetectionResult, SceneDetector, scene_detector
from video_editor.beat_sync import BeatPoint, BeatAnalysisResult, BeatSyncEngine, beat_sync_engine
from video_editor.subtitles import SubtitleEntry, SubtitleManager, subtitle_manager
from video_editor.effects import SocialFormattingConfig, VideoEffectsEngine, effects_engine
from video_editor.exporter import ExportQuality, ExportResult, QualityProfile, VideoExporter, video_exporter
from video_editor.video_editing_agent import VideoEditingCoordinator, video_editing_coordinator

__all__ = [
    "MediaSafetyError",
    "MediaSafetyGuard",
    "media_safety_guard",
    "MediaMetadata",
    "VideoInspector",
    "ClipSegment",
    "VideoTimeline",
    "SceneChange",
    "SceneDetectionResult",
    "SceneDetector",
    "scene_detector",
    "BeatPoint",
    "BeatAnalysisResult",
    "BeatSyncEngine",
    "beat_sync_engine",
    "SubtitleEntry",
    "SubtitleManager",
    "subtitle_manager",
    "SocialFormattingConfig",
    "VideoEffectsEngine",
    "effects_engine",
    "ExportQuality",
    "ExportResult",
    "QualityProfile",
    "VideoExporter",
    "video_exporter",
    "VideoEditingCoordinator",
    "video_editing_coordinator",
]
