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
from video_editor.autonomous_creator import ViralStoryBrief, TrendingTopicCurator, AutonomousVideoCreator, autonomous_video_creator
from video_editor.workspace import WorkspaceManager, workspace_manager
from video_editor.input_manager import VideoInputManager, VideoMediaMetadata, input_manager
from video_editor.analyzer import VideoAnalyzer, SceneSegment, FootageAnalysisReport, video_analyzer
from video_editor.editing_brain import EditingBrain, EditingDecisionPlan, editing_brain
from video_editor.branding import BrandingConfig, BrandingManager, branding_manager
from video_editor.quality_control import QualityControlEngine, QualityInspectionReport, quality_control_engine
from video_editor.rendering_engine import RenderingEngine, RenderOutputResult, rendering_engine
from video_editor.project_memory import ProjectMemory, project_memory
from video_editor.adaptive_editor import AdaptiveEditor, adaptive_editor
from video_editor.reel_controller import ReelController, reel_controller
from video_editor.media_discovery import (
    MediaDiscoveryEngine,
    DiscoveredMediaItem,
    MediaCollectionReport,
    SubjectAnalyzer,
    CameraAngleClassifier,
    FootageProvider,
    WikimediaCommonsProvider,
    InternetArchiveProvider,
    PexelsProvider,
    PixabayProvider,
    media_discovery_engine,
)
from video_editor.capcut_engine import CapCutEngine, CapCutStyleProfile, capcut_engine
from video_editor.multitrack_timeline import TimelineClip, MultiTrackTimeline, create_multitrack_timeline
from video_editor.autoclip import (
    AutoClipEngine,
    AutoClipResult,
    autoclip_engine,
    TranscriptCue,
    SlicingWindow,
    TranscriptSlicer,
    DurationProfile,
    RefinedClip,
    TimelineRefiner,
    ClipEvaluator,
    VideoProcessor as AutoClipVideoProcessor,
    CompilationBuilder as AutoClipCompilationBuilder,
)

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
    "ViralStoryBrief",
    "TrendingTopicCurator",
    "AutonomousVideoCreator",
    "autonomous_video_creator",
    "WorkspaceManager",
    "workspace_manager",
    "VideoInputManager",
    "VideoMediaMetadata",
    "input_manager",
    "VideoAnalyzer",
    "SceneSegment",
    "FootageAnalysisReport",
    "video_analyzer",
    "EditingBrain",
    "EditingDecisionPlan",
    "editing_brain",
    "BrandingConfig",
    "BrandingManager",
    "branding_manager",
    "QualityControlEngine",
    "QualityInspectionReport",
    "quality_control_engine",
    "RenderingEngine",
    "RenderOutputResult",
    "rendering_engine",
    "ProjectMemory",
    "project_memory",
    "AdaptiveEditor",
    "adaptive_editor",
    "ReelController",
    "reel_controller",
    "MediaDiscoveryEngine",
    "DiscoveredMediaItem",
    "MediaCollectionReport",
    "SubjectAnalyzer",
    "CameraAngleClassifier",
    "FootageProvider",
    "WikimediaCommonsProvider",
    "InternetArchiveProvider",
    "PexelsProvider",
    "PixabayProvider",
    "media_discovery_engine",
    "CapCutEngine",
    "CapCutStyleProfile",
    "capcut_engine",
    "TimelineClip",
    "MultiTrackTimeline",
    "create_multitrack_timeline",
    "AutoClipEngine",
    "AutoClipResult",
    "autoclip_engine",
    "TranscriptCue",
    "SlicingWindow",
    "TranscriptSlicer",
    "DurationProfile",
    "RefinedClip",
    "TimelineRefiner",
    "ClipEvaluator",
    "AutoClipVideoProcessor",
    "AutoClipCompilationBuilder",
]
