"""J.A.R.V.I.S. Video Effects & 9:16 Social Media Framing.

Generates FFmpeg filter graphs for video transformations:
- YouTube Shorts & Instagram Reels 9:16 vertical formatting (1080x1920)
- Smart center crop vs aesthetic blurred background padding
- Watermark / logo overlay positioning with opacity
- Clean title and lower-third text overlays (drawtext)
- Visual transitions (fade, crossfade, dissolve)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SocialFormattingConfig:
    """Configures 9:16 vertical formatting for Shorts & Reels."""
    target_width: int = 1080
    target_height: int = 1920
    blur_background: bool = True
    blur_strength: int = 25
    center_scale_mode: str = "fit"  # "fit" (preserves entire frame) or "fill" (zooms in)


class VideoEffectsEngine:
    """Generates FFmpeg complex filter expressions for visual styling."""

    def build_vertical_reels_filter(
        self,
        config: Optional[SocialFormattingConfig] = None,
    ) -> str:
        """Construct FFmpeg filtergraph to convert any video into 9:16 (1080x1920).

        If blur_background=True:
            Creates a blurred background scaling to 1080x1920, and overlays the
            original video scaled to fit neatly in the center without black bars.
        If blur_background=False:
            Crops the video to 9:16 center aspect ratio and scales to 1080x1920.
        """
        cfg = config or SocialFormattingConfig()
        w = cfg.target_width
        h = cfg.target_height

        if cfg.blur_background:
            # Complex filter splitting input: background is blurred and scaled to target;
            # foreground is scaled to fit width and centered vertically
            return (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},"
                f"boxblur={cfg.blur_strength}:5[bg];"
                f"[0:v]scale={w}:-2:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
            )
        else:
            # Direct center crop to 9:16 ratio then scale
            return f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={w}:{h}"

    def build_text_overlay_filter(
        self,
        text: str,
        position: str = "bottom_center",
        font_size: int = 36,
        font_color: str = "white",
        box: bool = True,
        box_color: str = "black@0.6",
        box_border_w: int = 10,
        y_offset: int = 120,
    ) -> str:
        """Generate a drawtext filter for title or caption display.

        Positions: 'top_center', 'center', 'bottom_center', 'bottom_third'
        """
        # Escape special characters for FFmpeg drawtext
        clean_text = text.replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")

        if position == "bottom_center":
            x_expr = "(w-text_w)/2"
            y_expr = f"h-text_h-{y_offset}"
        elif position == "top_center":
            x_expr = "(w-text_w)/2"
            y_expr = f"{y_offset}"
        elif position == "center":
            x_expr = "(w-text_w)/2"
            y_expr = "(h-text_h)/2"
        elif position == "bottom_third":
            x_expr = "(w-text_w)/2"
            y_expr = "h*0.75-text_h/2"
        else:
            x_expr = "(w-text_w)/2"
            y_expr = f"h-text_h-{y_offset}"

        box_str = f":box=1:boxcolor={box_color}:boxborderw={box_border_w}" if box else ""
        return f"drawtext=text='{clean_text}':fontsize={font_size}:fontcolor={font_color}:x={x_expr}:y={y_expr}{box_str}"

    def build_watermark_filter(
        self,
        watermark_img_path: str | Path,
        position: str = "top_right",
        margin: int = 24,
        scale_w: int = 120,
    ) -> str:
        """Generate overlay filter string for brand logo or watermark.

        Positions: 'top_right', 'top_left', 'bottom_right', 'bottom_left'
        """
        clean_p = str(Path(watermark_img_path).resolve()).replace("\\", "/")

        if position == "top_right":
            coord = f"W-w-{margin}:{margin}"
        elif position == "top_left":
            coord = f"{margin}:{margin}"
        elif position == "bottom_right":
            coord = f"W-w-{margin}:H-h-{margin}"
        elif position == "bottom_left":
            coord = f"{margin}:H-h-{margin}"
        else:
            coord = f"W-w-{margin}:{margin}"

        return f"movie='{clean_p}',scale={scale_w}:-1[wm];[in][wm]overlay={coord}[out]"

    def build_fade_filter(
        self,
        fade_in_sec: float = 0.5,
        fade_out_start_sec: Optional[float] = None,
        fade_out_sec: float = 0.5,
    ) -> str:
        """Formulate fade in/out filter parameters."""
        filters = []
        if fade_in_sec > 0:
            filters.append(f"fade=t=in:st=0:d={fade_in_sec:.2f}")
        if fade_out_start_sec is not None and fade_out_sec > 0:
            filters.append(f"fade=t=out:st={fade_out_start_sec:.2f}:d={fade_out_sec:.2f}")
        return ",".join(filters)

    def build_crossfade_filter(
        self,
        duration_sec: float = 0.5,
        offset_sec: float = 5.0,
        transition: str = "fade",
    ) -> str:
        """Formulate xfade video filter between two input video streams."""
        return f"xfade=transition={transition}:duration={duration_sec:.2f}:offset={offset_sec:.2f}"


effects_engine = VideoEffectsEngine()

__all__ = ["SocialFormattingConfig", "VideoEffectsEngine", "effects_engine"]
