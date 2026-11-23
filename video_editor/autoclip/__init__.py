"""JARVIS AutoClip Module
========================
Automated long-to-short video clipping, viral highlight extraction,
timeline pause snapping, 9:16 vertical short conversion, and compilation generation.

Inspired by AutoClip (https://github.com/zhouxiaoka/autoclip).
"""

from video_editor.autoclip.transcript_slicer import (
    TranscriptCue,
    SlicingWindow,
    TranscriptSlicer,
    to_seconds,
    to_srt_time,
    to_clock_time,
)
from video_editor.autoclip.timeline_refiner import (
    DurationProfile,
    RefinedClip,
    TimelineRefiner,
    profile_from_duration,
)
from video_editor.autoclip.clip_evaluator import (
    ClipEvaluator,
    ScoredSegment,
)
from video_editor.autoclip.video_processor import VideoProcessor
from video_editor.autoclip.compilation_builder import CompilationBuilder
from video_editor.autoclip.url_ingest import (
    UrlIngest,
    UrlDownloadResult,
    is_supported_url,
)
from video_editor.autoclip.autoclip_engine import (
    AutoClipEngine,
    AutoClipResult,
    autoclip_engine,
)

__all__ = [
    "TranscriptCue",
    "SlicingWindow",
    "TranscriptSlicer",
    "to_seconds",
    "to_srt_time",
    "to_clock_time",
    "DurationProfile",
    "RefinedClip",
    "TimelineRefiner",
    "profile_from_duration",
    "ClipEvaluator",
    "ScoredSegment",
    "VideoProcessor",
    "CompilationBuilder",
    "UrlIngest",
    "UrlDownloadResult",
    "is_supported_url",
    "AutoClipEngine",
    "AutoClipResult",
    "autoclip_engine",
]
