"""
JARVIS Video Editing Engine — Text Overlays & Branding Manager
==============================================================
Generates dynamic text overlays, introduction hook titles, cinematic captions,
subtitles, outro cards, and configurable watermarks (Default Brand: KANKI.7).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path


@dataclass
class BrandingConfig:
    """Configurable brand styling and positioning."""
    brand_name: str = "KANKI.7"
    font_family: str = "Arial"
    font_size: int = 32
    font_color: str = "white"
    position: str = "bottom-center"  # "top-left", "top-right", "bottom-left", "bottom-center", "bottom-right"
    opacity: float = 0.85
    show_watermark: bool = True
    show_intro_title: bool = True
    show_outro_card: bool = True
    intro_title_text: Optional[str] = None
    outro_title_text: Optional[str] = "EDIT BY KANKI.7"


class BrandingManager:
    """Constructs FFmpeg drawtext filter chains for branding and text overlays."""

    def __init__(self, config: Optional[BrandingConfig] = None):
        self.config = config or BrandingConfig()

    def update_brand(
        self,
        brand_name: Optional[str] = None,
        font_size: Optional[int] = None,
        position: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> None:
        """Dynamically update active branding attributes."""
        if brand_name:
            self.config.brand_name = brand_name
        if font_size:
            self.config.font_size = font_size
        if position:
            self.config.position = position
        if opacity is not None:
            self.config.opacity = max(0.1, min(1.0, opacity))

    def _resolve_coordinates(self, position: str, margin_x: int = 30, margin_y: int = 80) -> Tuple[str, str]:
        """Convert friendly position names into FFmpeg x/y coordinate expressions."""
        pos = position.lower()
        if pos == "top-left":
            return str(margin_x), str(margin_y)
        elif pos == "top-right":
            return f"w-text_w-{margin_x}", str(margin_y)
        elif pos == "bottom-left":
            return str(margin_x), f"h-text_h-{margin_y}"
        elif pos == "bottom-right":
            return f"w-text_w-{margin_x}", f"h-text_h-{margin_y}"
        elif pos == "top-center":
            return "(w-text_w)/2", str(margin_y)
        else:  # default: bottom-center
            return "(w-text_w)/2", f"h-text_h-{margin_y}"

    def build_watermark_filter(self, config: Optional[BrandingConfig] = None) -> str:
        """Generate subtle semi-transparent watermark filter."""
        cfg = config or self.config
        if not cfg.show_watermark:
            return ""

        x_expr, y_expr = self._resolve_coordinates(cfg.position, margin_x=32, margin_y=110)
        alpha = f"@{cfg.opacity:.2f}"

        # Clean text for FFmpeg escaping
        safe_text = cfg.brand_name.replace(":", "\\:").replace("'", "")

        return (
            f"drawtext=text='{safe_text}':"
            f"font='{cfg.font_family}':fontsize={cfg.font_size}:"
            f"fontcolor={cfg.font_color}{alpha}:"
            f"x={x_expr}:y={y_expr}:"
            f"box=1:boxcolor=0x000000@0.45:boxborderw=8"
        )

    def build_intro_title_filter(
        self,
        title_text: str,
        duration_sec: float = 2.5,
        font_size: int = 56,
        accent_color: str = "0x00f0ff",
    ) -> str:
        """Generate high-impact intro title card during opening seconds."""
        safe_title = title_text.replace(":", "\\:").replace("'", "")
        # Enable between 0.2s and duration_sec with subtle fade
        enable_expr = f"between(t,0.2,{duration_sec})"
        return (
            f"drawtext=text='{safe_title}':"
            f"fontsize={font_size}:fontcolor={accent_color}:"
            f"x=(w-text_w)/2:y='(h-text_h)/2 - 120':"
            f"box=1:boxcolor=0x0b0f19@0.85:boxborderw=16:"
            f"enable='{enable_expr}'"
        )

    def build_outro_card_filter(
        self,
        total_duration_sec: float,
        outro_text: Optional[str] = None,
        duration_sec: float = 1.5,
        font_size: int = 42,
    ) -> str:
        """Generate outro signature branding card at the conclusion."""
        card_text = (outro_text or self.config.outro_title_text or f"EDIT BY {self.config.brand_name}").replace(":", "\\:").replace("'", "")
        start_time = max(0.0, total_duration_sec - duration_sec)
        enable_expr = f"gte(t,{start_time:.2f})"
        return (
            f"drawtext=text='{card_text}':"
            f"fontsize={font_size}:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"box=1:boxcolor=black@0.9:boxborderw=24:"
            f"enable='{enable_expr}'"
        )

    def build_subtitle_filter(self, srt_path: str | Path) -> str:
        """Generate FFmpeg subtitles filter string."""
        path_str = str(Path(srt_path).resolve()).replace("\\", "/")
        # In Windows FFmpeg, escape colon: D\:/path
        if ":" in path_str and not path_str.startswith("http"):
            drive, rest = path_str.split(":", 1)
            path_str = f"{drive}\\:{rest}"

        style = (
            "Fontname=Arial,Fontsize=26,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,Bold=1,Outline=2,Shadow=1,MarginV=140"
        )
        return f"subtitles='{path_str}':force_style='{style}'"


# Global branding manager singleton with default KANKI.7
branding_manager = BrandingManager(BrandingConfig(brand_name="KANKI.7"))
