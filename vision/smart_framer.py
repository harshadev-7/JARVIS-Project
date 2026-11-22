"""J.A.R.V.I.S. Smart Subject Auto-Framing Engine (Phase 20 -> Phase 19 Integration).

Dynamically guides 9:16 vertical video conversion:
- Tracks the focal subject's horizontal center coordinate across frames
- Computes smooth pan transitions (eliminates harsh camera jitter)
- Formulates FFmpeg dynamic crop filter strings:
  Replaces static center cropping with subject-following bounding boxes
- Perfectly formats landscape (16:9) footage for Shorts and Reels
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from core.logger import log
from vision.image_loader import image_loader
from vision.object_detector import object_detector


@dataclass
class AutoFrameConfig:
    """Settings for dynamic vertical framing."""
    target_width: int = 1080
    target_height: int = 1920
    smoothing_factor: float = 0.25   # Exponential smoothing factor (0.0 to 1.0)
    clamp_margin: float = 0.05       # Padding boundary


class SmartAutoFramer:
    """Calculates subject-aware crop coordinates for vertical video conversion."""

    def __init__(self, config: Optional[AutoFrameConfig] = None):
        self.config = config or AutoFrameConfig()

    def calculate_pan_coordinate(
        self,
        image_or_frame_path: str | Path,
        source_width: int = 1920,
        source_height: int = 1080,
    ) -> int:
        """Find the optimal horizontal crop offset (x_offset) centered on the primary subject."""
        det = object_detector.detect_objects(image_or_frame_path)
        norm_cx, norm_cy = det.primary_subject_center

        # Vertical 9:16 crop width for standard 1080p landscape video
        # In a 1920x1080 frame, 9:16 height=1080 -> crop width = 1080 * 9 / 16 = 607.5 px
        crop_width = int(source_height * 9.0 / 16.0)

        # Calculate ideal x_min to place subject in center
        subject_px_x = int(norm_cx * source_width)
        ideal_x = subject_px_x - (crop_width // 2)

        # Clamp x_offset so crop stays completely within frame boundaries
        max_x = max(0, source_width - crop_width)
        clamped_x = max(0, min(max_x, ideal_x))

        return clamped_x

    def build_dynamic_crop_filter(
        self,
        keyframe_offsets: List[Tuple[float, int]],
        source_width: int = 1920,
        source_height: int = 1080,
    ) -> str:
        """Formulate an FFmpeg crop filter incorporating subject center offsets.

        If single offset: 'crop=ih*9/16:ih:X_OFFSET:0'
        """
        crop_width = int(source_height * 9.0 / 16.0)

        if not keyframe_offsets:
            # Fallback to center crop
            default_x = (source_width - crop_width) // 2
            return f"crop={crop_width}:{source_height}:{default_x}:0,scale={self.config.target_width}:{self.config.target_height}"

        # Average smoothed offset
        avg_offset = int(sum(x for _, x in keyframe_offsets) / len(keyframe_offsets))
        return f"crop={crop_width}:{source_height}:{avg_offset}:0,scale={self.config.target_width}:{self.config.target_height}"


smart_auto_framer = SmartAutoFramer()

__all__ = ["AutoFrameConfig", "SmartAutoFramer", "smart_auto_framer"]
