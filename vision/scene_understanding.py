"""J.A.R.V.I.S. Scene Understanding & Image Quality Analyzer.

Evaluates high-level visual attributes:
- Brightness, contrast, and color saturation metrics
- Blur and sharpness estimation (Laplacian high-frequency variance)
- Lighting condition classification (Underexposed, Balanced, Overexposed)
- Color warmth and dominant palette estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from core.logger import log
from vision.image_loader import image_loader


@dataclass
class SceneReport:
    """Comprehensive visual quality and scene composition analysis."""
    image_path: str
    brightness: float          # 0.0 (pitch black) to 1.0 (pure white)
    contrast: float            # Standard deviation of luminance
    sharpness: float           # High-frequency edge energy
    is_blurry: bool
    lighting_condition: str    # "underexposed", "balanced", "overexposed"
    dominant_color_rgb: Tuple[int, int, int]
    composition_notes: List[str] = field(default_factory=list)


class SceneAnalyzer:
    """Analyzes lighting, sharpness, and scene composition attributes."""

    def __init__(self):
        # 3x3 Laplacian kernel for sharpness estimation
        self.laplacian = torch.tensor(
            [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]], dtype=torch.float32
        ).unsqueeze(0)

    def analyze_scene(self, image_path: str | Path) -> SceneReport:
        """Compute scene lighting and sharpness telemetry."""
        tensor = image_loader.load_tensor(image_path, target_size=(160, 160), normalize=True)

        # 1. Grayscale luminance
        gray = 0.299 * tensor[:, 0:1, :, :] + 0.587 * tensor[:, 1:2, :, :] + 0.114 * tensor[:, 2:3, :, :]

        # 2. Brightness & Contrast
        brightness = float(gray.mean().item())
        contrast = float(gray.std().item())

        # 3. Sharpness via Laplacian variance
        edge_map = F.conv2d(gray, self.laplacian, padding=1)
        sharpness = float(edge_map.var().item() * 1000.0)
        is_blurry = sharpness < 1.5

        # 4. Lighting condition
        if brightness < 0.25:
            lighting = "underexposed"
        elif brightness > 0.75:
            lighting = "overexposed"
        else:
            lighting = "balanced"

        # 5. Dominant color (mean RGB channel values)
        r = int(float(tensor[:, 0, :, :].mean().item()) * 255)
        g = int(float(tensor[:, 1, :, :].mean().item()) * 255)
        b = int(float(tensor[:, 2, :, :].mean().item()) * 255)

        notes = []
        if is_blurry:
            notes.append("Image exhibits motion blur or low optical focus.")
        else:
            notes.append("Image focus is sharp with clear edge boundaries.")

        if lighting == "underexposed":
            notes.append("Low-light scene. Consider increasing exposure or fill lighting.")
        elif lighting == "overexposed":
            notes.append("High-key scene with potential highlight clipping.")
        else:
            notes.append("Exposure and dynamic range are well-balanced.")

        return SceneReport(
            image_path=str(image_path),
            brightness=round(brightness, 3),
            contrast=round(contrast, 3),
            sharpness=round(sharpness, 3),
            is_blurry=is_blurry,
            lighting_condition=lighting,
            dominant_color_rgb=(r, g, b),
            composition_notes=notes,
        )


scene_analyzer = SceneAnalyzer()

__all__ = ["SceneReport", "SceneAnalyzer", "scene_analyzer"]
