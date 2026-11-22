"""
JARVIS Video Editing Engine — Automatic Video & Scene Analyzer
==============================================================
Performs scene segmentation, blur/quality scoring, duplicate scene detection,
action/motion intensity profiling, and scene selection algorithms.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import math
import json
import re

import imageio_ffmpeg
from core.logger import log
from video_editor.input_manager import input_manager, VideoMediaMetadata
from video_editor.workspace import workspace_manager


@dataclass
class SceneSegment:
    """Detailed profile of an identified scene within a video clip."""
    scene_index: int
    source_file: str
    start_sec: float
    end_sec: float
    duration_sec: float
    sharpness_score: float = 50.0
    is_blurry: bool = False
    motion_energy: float = 0.5       # 0.0 (static) to 1.0 (explosive action)
    is_action_sequence: bool = False
    is_cinematic: bool = False
    lighting_condition: str = "balanced"
    dominant_color: Tuple[int, int, int] = (128, 128, 128)
    keyframe_path: Optional[str] = None
    selection_score: float = 0.0     # Composite utility score for editing brain
    tags: List[str] = field(default_factory=list)


@dataclass
class FootageAnalysisReport:
    """Full breakdown of inspected video clips and detected scenes."""
    total_clips: int
    total_duration_sec: float
    total_scenes_detected: int
    usable_scenes_count: int
    discarded_scenes_count: int
    scenes: List[SceneSegment] = field(default_factory=list)
    average_sharpness: float = 0.0
    action_scenes_count: int = 0
    cinematic_scenes_count: int = 0


class VideoAnalyzer:
    """Analyzes raw video footage, detects scene boundaries, and scores usability."""

    def __init__(self):
        try:
            self.ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            self.ffmpeg_bin = "ffmpeg"

    def detect_scenes_ffmpeg(
        self,
        file_path: str | Path,
        threshold: float = 0.35,
        min_scene_len_sec: float = 1.0,
    ) -> List[Tuple[float, float]]:
        """
        Use FFmpeg's 'select=gt(scene,threshold)' filter to detect cut timestamps.
        Returns a list of (start_sec, end_sec) intervals.
        """
        src = Path(file_path).resolve()
        meta = input_manager.probe_video(src)
        dur = meta.duration_sec
        if dur <= min_scene_len_sec:
            return [(0.0, dur)]

        cmd = [
            self.ffmpeg_bin,
            "-i", str(src),
            "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null",
            "-",
        ]
        cut_timestamps = [0.0]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
            output = res.stderr or res.stdout or ""
            # Parse showinfo pts_time:12.345
            for match in re.finditer(r"pts_time:\s*(\d+\.?\d*)", output):
                t = float(match.group(1))
                if t - cut_timestamps[-1] >= min_scene_len_sec and t < dur:
                    cut_timestamps.append(t)
        except Exception as e:
            log.warning(f"FFmpeg scene detection error on '{src.name}': {e}")

        if cut_timestamps[-1] < dur:
            cut_timestamps.append(dur)

        # Pair into segments
        segments = []
        for i in range(len(cut_timestamps) - 1):
            s, e = cut_timestamps[i], cut_timestamps[i+1]
            if e - s >= min_scene_len_sec * 0.8:
                segments.append((s, e))

        if not segments:
            segments.append((0.0, dur))
        return segments

    def analyze_clip(self, file_path: str | Path, min_scene_len: float = 1.0) -> List[SceneSegment]:
        """Perform full multi-dimensional visual and quality analysis on a video clip."""
        src = Path(file_path).resolve()
        meta = input_manager.probe_video(src)
        raw_scenes = self.detect_scenes_ffmpeg(src, min_scene_len_sec=min_scene_len)

        # Lazy import of Vision subsystem to leverage Phase 20 capabilities
        try:
            from vision.scene_understanding import scene_analyzer
        except Exception:
            scene_analyzer = None

        segments: List[SceneSegment] = []
        temp_dir = workspace_manager.dirs["media_temporary"] / "scene_keyframes" / src.stem
        temp_dir.mkdir(parents=True, exist_ok=True)

        for idx, (s, e) in enumerate(raw_scenes):
            seg_dur = e - s
            mid_point = s + (seg_dur / 2.0)
            kf_path = temp_dir / f"scene_{idx:03d}_{int(mid_point*1000)}ms.jpg"

            # Extract keyframe for visual profiling
            input_manager.generate_thumbnail(src, timestamp_sec=mid_point, output_path=kf_path)

            sharpness = 50.0
            is_blurry = False
            lighting = "balanced"
            dominant_color = (128, 128, 128)

            if kf_path.exists() and scene_analyzer:
                try:
                    rep = scene_analyzer.analyze_scene(kf_path)
                    sharpness = rep.sharpness
                    is_blurry = rep.is_blurry
                    lighting = rep.lighting_condition
                    dominant_color = rep.dominant_color_rgb
                except Exception as ex:
                    log.debug(f"Scene analysis fallback on keyframe: {ex}")

            # Heuristic motion energy based on duration and scene frequency
            # Shorter scenes in raw footage typically correspond to rapid camera pans/cuts
            motion_energy = min(1.0, max(0.1, 1.5 / max(1.0, seg_dur)))
            is_action = motion_energy > 0.65 or (seg_dur < 2.5 and sharpness > 40.0)
            is_cinematic = (lighting == "balanced" and sharpness > 35.0 and seg_dur >= 2.0)

            tags = []
            if is_action:
                tags.append("action")
            if is_cinematic:
                tags.append("cinematic")
            if is_blurry:
                tags.append("blurry")

            # Selection utility score
            score = (sharpness * 0.4) + (motion_energy * 30.0) + (10.0 if is_cinematic else 0.0)
            if is_blurry:
                score -= 40.0

            segment = SceneSegment(
                scene_index=idx,
                source_file=str(src),
                start_sec=round(s, 3),
                end_sec=round(e, 3),
                duration_sec=round(seg_dur, 3),
                sharpness_score=round(sharpness, 2),
                is_blurry=is_blurry,
                motion_energy=round(motion_energy, 2),
                is_action_sequence=is_action,
                is_cinematic=is_cinematic,
                lighting_condition=lighting,
                dominant_color=dominant_color,
                keyframe_path=str(kf_path) if kf_path.exists() else None,
                selection_score=round(score, 2),
                tags=tags,
            )
            segments.append(segment)

        return segments

    def analyze_all_footage(self, files: Optional[List[str | Path]] = None) -> FootageAnalysisReport:
        """Scan and analyze all source footage in media/input/ or specified files."""
        if not files:
            files = workspace_manager.list_input_videos()

        all_scenes: List[SceneSegment] = []
        total_duration = 0.0

        for f in files:
            try:
                scenes = self.analyze_clip(f)
                all_scenes.extend(scenes)
                meta = input_manager.probe_video(f)
                total_duration += meta.duration_sec
            except Exception as e:
                log.error(f"Failed to analyze clip '{f}': {e}")

        usable = [s for s in all_scenes if not s.is_blurry and s.duration_sec >= 0.8]
        discarded = len(all_scenes) - len(usable)
        avg_sharp = sum(s.sharpness_score for s in all_scenes) / max(1, len(all_scenes))
        action_count = sum(1 for s in all_scenes if s.is_action_sequence)
        cinematic_count = sum(1 for s in all_scenes if s.is_cinematic)

        return FootageAnalysisReport(
            total_clips=len(files),
            total_duration_sec=round(total_duration, 2),
            total_scenes_detected=len(all_scenes),
            usable_scenes_count=len(usable),
            discarded_scenes_count=discarded,
            scenes=all_scenes,
            average_sharpness=round(avg_sharp, 2),
            action_scenes_count=action_count,
            cinematic_scenes_count=cinematic_count,
        )

    def select_best_scenes(
        self,
        scenes: List[SceneSegment],
        target_duration: float = 30.0,
        style: str = "reels",
    ) -> List[SceneSegment]:
        """
        Autonomous Scene Selection Algorithm:
        Selects top scenes based on the requested editing aesthetic and target duration.
        """
        # Filter out corrupted or blurry scenes
        valid_scenes = [s for s in scenes if not s.is_blurry and s.duration_sec >= 0.5]
        if not valid_scenes:
            valid_scenes = scenes  # fallback if all were borderline

        style_lower = style.lower()

        # Weighting based on style
        def rank_fn(s: SceneSegment) -> float:
            base = s.selection_score
            if "anime" in style_lower or "velocity" in style_lower:
                base += s.motion_energy * 40.0
            elif "cinematic" in style_lower or "luxury" in style_lower:
                base += (25.0 if s.is_cinematic else 0.0) + (s.sharpness_score * 0.3)
            elif "car" in style_lower:
                base += (30.0 if s.is_action_sequence else 0.0) + (s.motion_energy * 30.0)
            return base

        sorted_scenes = sorted(valid_scenes, key=rank_fn, reverse=True)

        selected: List[SceneSegment] = []
        accumulated_dur = 0.0

        for sc in sorted_scenes:
            # If a single scene alone exceeds target duration, trim it cleanly
            if accumulated_dur == 0.0 and sc.duration_sec > target_duration:
                trimmed = SceneSegment(
                    scene_index=sc.scene_index,
                    source_file=sc.source_file,
                    start_sec=sc.start_sec,
                    end_sec=round(sc.start_sec + target_duration, 3),
                    duration_sec=round(target_duration, 3),
                    sharpness_score=sc.sharpness_score,
                    is_blurry=sc.is_blurry,
                    motion_energy=sc.motion_energy,
                    is_action_sequence=sc.is_action_sequence,
                    is_cinematic=sc.is_cinematic,
                    lighting_condition=sc.lighting_condition,
                    dominant_color=sc.dominant_color,
                    keyframe_path=sc.keyframe_path,
                    selection_score=sc.selection_score,
                    tags=sc.tags,
                )
                selected.append(trimmed)
                accumulated_dur += target_duration
                break

            if accumulated_dur + sc.duration_sec <= target_duration + 0.5:
                selected.append(sc)
                accumulated_dur += sc.duration_sec
            elif accumulated_dur < target_duration - 1.0:
                # Trim scene to fit remaining duration
                remaining = target_duration - accumulated_dur
                if remaining >= 0.5:
                    trimmed = SceneSegment(
                        scene_index=sc.scene_index,
                        source_file=sc.source_file,
                        start_sec=sc.start_sec,
                        end_sec=round(sc.start_sec + remaining, 3),
                        duration_sec=round(remaining, 3),
                        sharpness_score=sc.sharpness_score,
                        is_blurry=sc.is_blurry,
                        motion_energy=sc.motion_energy,
                        is_action_sequence=sc.is_action_sequence,
                        is_cinematic=sc.is_cinematic,
                        lighting_condition=sc.lighting_condition,
                        dominant_color=sc.dominant_color,
                        keyframe_path=sc.keyframe_path,
                        selection_score=sc.selection_score,
                        tags=sc.tags,
                    )
                    selected.append(trimmed)
                    accumulated_dur += remaining
                    break

        return selected


# Global video analyzer singleton
video_analyzer = VideoAnalyzer()
