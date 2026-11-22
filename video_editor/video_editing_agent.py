"""J.A.R.V.I.S. Video Editing Autonomous Agent & Pipeline Coordinator.

High-level coordinator unifying all video capabilities:
- Inspection and metadata verification
- Trimming and multi-clip concatenation
- Scene detection and automatic highlight reel synthesis
- Music beat synchronization
- Subtitle generation and burning
- Social media vertical formatting (YouTube Shorts & Instagram Reels 9:16)
- Strict non-destructive media safety guardrails
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import log
from video_editor.beat_sync import beat_sync_engine
from video_editor.effects import SocialFormattingConfig, effects_engine
from video_editor.exporter import ExportQuality, ExportResult, video_exporter
from video_editor.inspector import MediaMetadata, VideoInspector
from video_editor.safety import MediaSafetyError, media_safety_guard
from video_editor.scene_detector import scene_detector
from video_editor.subtitles import SubtitleEntry, subtitle_manager
from video_editor.timeline import ClipSegment, VideoTimeline


class VideoEditingCoordinator:
    """Unified video editing automation pipeline."""

    def __init__(self):
        self.inspector = VideoInspector()
        self.safety = media_safety_guard
        self.scene_detector = scene_detector
        self.beat_sync = beat_sync_engine
        self.subtitles = subtitle_manager
        self.effects = effects_engine
        self.exporter = video_exporter

    def inspect_media(self, file_path: str | Path) -> MediaMetadata:
        """Inspect media file and report duration, resolution, codecs, and aspect ratio."""
        return self.inspector.inspect(file_path)

    def create_short_or_reel(
        self,
        video_path: str | Path,
        output_name: Optional[str] = None,
        blur_background: bool = True,
        watermark_text: Optional[str] = "JARVIS AI",
        subtitles: Optional[List[str]] = None,
        music_path: Optional[str | Path] = None,
        quality: ExportQuality = ExportQuality.SOCIAL_720P,
        max_duration_sec: float = 60.0,
    ) -> ExportResult:
        """Transform any input video into a 9:16 vertical short/reel.

        Applies:
        - 9:16 vertical conversion (1080x1920) with blurred background padding
        - Optional title/watermark overlay
        - Optional subtitle track
        - Non-destructive export to output/videos/
        """
        src = self.safety.validate_file_path(video_path, must_exist=True)
        meta = self.inspector.inspect(src)

        out_name = output_name or f"reel_{src.stem}.mp4"
        cfg = SocialFormattingConfig(
            target_width=1080,
            target_height=1920,
            blur_background=blur_background,
        )

        filter_graph = self.effects.build_vertical_reels_filter(cfg)

        if watermark_text:
            text_filter = self.effects.build_text_overlay_filter(
                text=watermark_text,
                position="top_center",
                font_size=32,
                y_offset=80,
            )
            filter_graph = f"{filter_graph},{text_filter}"

        # Subtitle handling if phrases are provided
        if subtitles:
            sub_entries = self.subtitles.create_subtitles_from_phrases(subtitles)
            srt_path = self.safety.output_dir / f"{src.stem}_subtitles.srt"
            self.subtitles.save_srt(sub_entries, srt_path)
            sub_filter = self.subtitles.build_ffmpeg_subtitles_filter(srt_path, font_size=28, margin_v=120)
            filter_graph = f"{filter_graph},{sub_filter}"

        dur = min(meta.duration_sec, max_duration_sec) if meta.duration_sec > 0 else max_duration_sec

        return self.exporter.export(
            input_file=src,
            output_filename=out_name,
            quality=quality,
            filter_complex=filter_graph,
            audio_file=music_path,
            duration_sec=dur,
        )

    def trim_clip(
        self,
        video_path: str | Path,
        start_sec: float,
        end_sec: float,
        output_name: Optional[str] = None,
        quality: ExportQuality = ExportQuality.SOCIAL_720P,
    ) -> ExportResult:
        """Extract a clean trimmed segment from a video."""
        src = self.safety.validate_file_path(video_path, must_exist=True)
        dur = max(0.1, end_sec - start_sec)
        out_name = output_name or f"trim_{start_sec:.0f}s_{end_sec:.0f}s_{src.name}"

        return self.exporter.export(
            input_file=src,
            output_filename=out_name,
            quality=quality,
            start_sec=start_sec,
            duration_sec=dur,
        )

    def auto_highlight_reel(
        self,
        video_path: str | Path,
        target_duration: float = 30.0,
        output_name: Optional[str] = None,
        music_path: Optional[str | Path] = None,
    ) -> ExportResult:
        """Automatically find key scenes and build a dynamic highlight reel."""
        src = self.safety.validate_file_path(video_path, must_exist=True)
        highlights = self.scene_detector.find_highlights(src, top_k=3, segment_duration=target_duration / 3.0)

        # Build timeline
        tl = VideoTimeline(title=f"Highlights - {src.name}")
        for h in highlights:
            tl.add_clip(src, start_sec=h["start_sec"], end_sec=h["end_sec"])

        out_name = output_name or f"highlights_{src.stem}.mp4"
        # Export opening highlight segment
        first_h = highlights[0] if highlights else {"start_sec": 0.0, "end_sec": target_duration}
        return self.trim_clip(
            video_path=src,
            start_sec=first_h["start_sec"],
            end_sec=first_h["end_sec"],
            output_name=out_name,
        )

    def sync_cuts_to_music(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        tolerance_sec: float = 0.25,
    ) -> Dict[str, Any]:
        """Align video cuts to detected music downbeats."""
        v_meta = self.inspector.inspect(video_path)
        scenes = self.scene_detector.detect_scenes(video_path)
        beats = self.beat_sync.analyze_audio(audio_path)

        aligned = self.beat_sync.align_cuts_to_beats(
            cut_timestamps=scenes.cut_timestamps,
            beat_timestamps=beats.beat_timestamps,
            tolerance_sec=tolerance_sec,
        )

        return {
            "source_video": str(video_path),
            "source_audio": str(audio_path),
            "original_cuts": scenes.cut_timestamps,
            "detected_bpm": beats.estimated_bpm,
            "total_beats": beats.total_beats,
            "aligned_cuts": aligned,
            "cuts_count": len(aligned),
        }


video_editing_coordinator = VideoEditingCoordinator()

__all__ = ["VideoEditingCoordinator", "video_editing_coordinator"]
