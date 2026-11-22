"""J.A.R.V.I.S. Video Scene & Highlight Detector.

Identifies visual cut points, scene changes, and key moments:
- Frame luminance and color delta calculation
- Native OpenCV acceleration when available, with pure-Python / FFmpeg fallback
- Peak delta detection for scene change boundaries
- Highlight moment ranking for short-form content generation
- RAM-safe processing (<100MB preview downsampling)
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.logger import log
from video_editor.inspector import VideoInspector
from video_editor.safety import media_safety_guard


@dataclass
class SceneChange:
    """Represents a detected scene transition boundary."""
    timestamp_sec: float
    frame_idx: int
    score: float
    description: str = "Scene transition"


@dataclass
class SceneDetectionResult:
    """Summary of scene detection analysis."""
    source_file: str
    total_scenes: int
    scenes: List[SceneChange] = field(default_factory=list)
    cut_timestamps: List[float] = field(default_factory=list)
    method: str = "heuristic"


class SceneDetector:
    """Detects scene cuts and visually important moments."""

    def __init__(self):
        self.inspector = VideoInspector()
        self.ffmpeg_path = shutil.which("ffmpeg")

    def detect_scenes(
        self,
        video_path: str | Path,
        threshold: float = 30.0,
        min_scene_len_sec: float = 1.5,
    ) -> SceneDetectionResult:
        """Detect scene cut points in a video file.

        Args:
            video_path: Path to target media file.
            threshold: Sensitivity threshold (0.0 - 100.0). Higher means fewer scenes.
            min_scene_len_sec: Minimum duration in seconds between consecutive scenes.

        Returns:
            SceneDetectionResult with detected cut timestamps.
        """
        p = media_safety_guard.validate_file_path(video_path, must_exist=True)
        meta = self.inspector.inspect(p)
        dur = meta.duration_sec

        # 1. Try OpenCV if installed
        try:
            import cv2
            return self._detect_with_opencv(p, meta, threshold, min_scene_len_sec)
        except ImportError:
            pass

        # 2. Try FFmpeg scene filter if available
        if self.ffmpeg_path and dur > 0:
            res = self._detect_with_ffmpeg(p, meta, threshold, min_scene_len_sec)
            if res:
                return res

        # 3. Fallback: Heuristic keyframe segmenter
        return self._detect_fallback(p, meta, min_scene_len_sec)

    def _detect_with_opencv(
        self,
        path: Path,
        meta,
        threshold: float,
        min_scene_len_sec: float,
    ) -> SceneDetectionResult:
        """Frame difference analysis via OpenCV with low-resolution frame sampling."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return self._detect_fallback(path, meta, min_scene_len_sec)

        fps = cap.get(cv2.CAP_PROP_FPS) or meta.fps or 30.0
        min_frames = int(min_scene_len_sec * fps)

        scenes: List[SceneChange] = []
        prev_gray = None
        frame_idx = 0
        last_cut_frame = -min_frames

        # Sample every 2nd or 3rd frame to optimize CPU on Core i3
        stride = 2 if fps >= 24 else 1

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    # Downscale to 160x90 to keep memory footprint under 30MB
                    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                    if prev_gray is not None:
                        # Absolute difference in luminance
                        diff = cv2.absdiff(gray, prev_gray)
                        score = float(diff.mean())

                        if score > threshold and (frame_idx - last_cut_frame) >= min_frames:
                            t_sec = round(frame_idx / fps, 2)
                            scenes.append(
                                SceneChange(
                                    timestamp_sec=t_sec,
                                    frame_idx=frame_idx,
                                    score=round(score, 2),
                                    description=f"Cut detected at {t_sec}s (delta: {score:.1f})",
                                )
                            )
                            last_cut_frame = frame_idx

                    prev_gray = gray
                frame_idx += 1
        finally:
            cap.release()

        cut_times = [s.timestamp_sec for s in scenes]
        return SceneDetectionResult(
            source_file=str(path),
            total_scenes=len(scenes) + 1,
            scenes=scenes,
            cut_timestamps=cut_times,
            method="opencv_luminance_delta",
        )

    def _detect_with_ffmpeg(
        self,
        path: Path,
        meta,
        threshold: float,
        min_scene_len_sec: float,
    ) -> Optional[SceneDetectionResult]:
        """Use FFmpeg scene score filter if available."""
        # Normalize threshold from 0-100 to 0.0-1.0 (default 0.3-0.4)
        norm_th = max(0.1, min(0.9, threshold / 100.0))
        cmd = [
            self.ffmpeg_path,
            "-i", str(path),
            "-filter_complex", f"select='gt(scene,{norm_th})',metadata=print",
            "-f", "null", "-",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            scenes: List[SceneChange] = []
            import re
            pts_times = re.findall(r"pts_time:([0-9\.]+)", res.stderr)
            last_t = -min_scene_len_sec
            for pt in pts_times:
                t = float(pt)
                if t - last_t >= min_scene_len_sec:
                    scenes.append(
                        SceneChange(
                            timestamp_sec=round(t, 2),
                            frame_idx=int(t * meta.fps),
                            score=threshold,
                            description=f"FFmpeg scene boundary at {t:.2f}s",
                        )
                    )
                    last_t = t

            return SceneDetectionResult(
                source_file=str(path),
                total_scenes=len(scenes) + 1,
                scenes=scenes,
                cut_timestamps=[s.timestamp_sec for s in scenes],
                method="ffmpeg_scene_filter",
            )
        except Exception as e:
            log.debug(f"FFmpeg scene detection skipped: {e}")
            return None

    def _detect_fallback(
        self,
        path: Path,
        meta,
        min_scene_len_sec: float,
    ) -> SceneDetectionResult:
        """Heuristic scene breakdown based on duration intervals."""
        dur = meta.duration_sec
        scenes: List[SceneChange] = []

        if dur > min_scene_len_sec * 2:
            # Segment into dynamic 5s to 10s intervals
            interval = max(min_scene_len_sec, min(8.0, dur / 4.0))
            current = interval
            while current < dur - 1.0:
                scenes.append(
                    SceneChange(
                        timestamp_sec=round(current, 2),
                        frame_idx=int(current * meta.fps),
                        score=25.0,
                        description=f"Heuristic interval cut at {current:.2f}s",
                    )
                )
                current += interval

        return SceneDetectionResult(
            source_file=str(path),
            total_scenes=len(scenes) + 1,
            scenes=scenes,
            cut_timestamps=[s.timestamp_sec for s in scenes],
            method="heuristic_rhythmic_intervals",
        )

    def find_highlights(
        self,
        video_path: str | Path,
        top_k: int = 3,
        segment_duration: float = 5.0,
    ) -> List[dict]:
        """Rank and return key highlight segments suitable for Shorts/Reels."""
        res = self.detect_scenes(video_path)
        meta = self.inspector.inspect(video_path)
        dur = meta.duration_sec

        highlights = []
        if res.cut_timestamps:
            sorted_scenes = sorted(res.scenes, key=lambda s: s.score, reverse=True)
            for s in sorted_scenes[:top_k]:
                start = max(0.0, s.timestamp_sec - 1.0)
                end = min(dur, start + segment_duration)
                highlights.append({
                    "start_sec": round(start, 2),
                    "end_sec": round(end, 2),
                    "duration": round(end - start, 2),
                    "score": s.score,
                    "description": s.description,
                })
        else:
            # If no cut points detected, return beginning, middle, and peak
            highlights.append({"start_sec": 0.0, "end_sec": min(dur, segment_duration), "duration": min(dur, segment_duration), "score": 10.0, "description": "Opening hook"})
            if dur > segment_duration * 2:
                mid = dur / 2.0
                highlights.append({"start_sec": round(mid, 2), "end_sec": round(min(dur, mid + segment_duration), 2), "duration": segment_duration, "score": 8.0, "description": "Middle peak"})

        return highlights


scene_detector = SceneDetector()

__all__ = ["SceneChange", "SceneDetectionResult", "SceneDetector", "scene_detector"]
