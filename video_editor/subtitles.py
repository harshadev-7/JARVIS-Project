"""J.A.R.V.I.S. Video Subtitle & Caption Engine.

Generates, parses, and formats synchronized subtitles:
- Standards-compliant SubRip (.srt) generation and parsing
- Accurate millisecond timestamp formatting (HH:MM:SS,mmm)
- Subtitle burning filter synthesis for FFmpeg with styling options
- Automatic subtitle chunking from transcript words or sentences
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from video_editor.safety import media_safety_guard


@dataclass
class SubtitleEntry:
    """Single subtitle entry in a caption track."""
    index: int
    start_sec: float
    end_sec: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


class SubtitleManager:
    """Manages subtitle timing, serialization, and burning configuration."""

    @staticmethod
    def seconds_to_srt_time(seconds: float) -> str:
        """Convert float seconds to SubRip timestamp format: HH:MM:SS,mmm."""
        total_ms = int(round(max(0.0, seconds) * 1000))
        hours = total_ms // (3600 * 1000)
        rem = total_ms % (3600 * 1000)
        minutes = rem // (60 * 1000)
        rem = rem % (60 * 1000)
        secs = rem // 1000
        ms = rem % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    @staticmethod
    def srt_time_to_seconds(srt_time: str) -> float:
        """Convert SubRip timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm) to float seconds."""
        clean = srt_time.replace(",", ".").strip()
        parts = clean.split(":")
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        return float(clean)

    def generate_srt(self, entries: List[SubtitleEntry]) -> str:
        """Serialize a list of SubtitleEntry items into standard SRT text format."""
        blocks = []
        for idx, entry in enumerate(entries, start=1):
            start_str = self.seconds_to_srt_time(entry.start_sec)
            end_str = self.seconds_to_srt_time(entry.end_sec)
            clean_text = entry.text.strip()
            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{clean_text}\n")
        return "\n".join(blocks).strip() + "\n"

    def parse_srt(self, srt_content: str) -> List[SubtitleEntry]:
        """Parse raw SRT content into structured SubtitleEntry objects."""
        entries: List[SubtitleEntry] = []
        pattern = re.compile(
            r"(\d+)\s*\n"
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"
            r"([\s\S]*?)(?=\n\s*\d+\s*\n|\Z)",
            re.MULTILINE,
        )

        for match in pattern.finditer(srt_content.strip()):
            idx = int(match.group(1))
            start_t = self.srt_time_to_seconds(match.group(2))
            end_t = self.srt_time_to_seconds(match.group(3))
            text = match.group(4).strip()
            entries.append(SubtitleEntry(index=idx, start_sec=start_t, end_sec=end_t, text=text))

        return entries

    def save_srt(self, entries: List[SubtitleEntry], output_path: str | Path) -> Path:
        """Write subtitle entries to an SRT file."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_srt(entries)
        target.write_text(content, encoding="utf-8")
        return target

    def create_subtitles_from_phrases(
        self,
        phrases: List[str],
        start_time: float = 0.5,
        phrase_duration: float = 2.5,
        gap: float = 0.3,
    ) -> List[SubtitleEntry]:
        """Synthesize subtitle entries from ordered phrases or script sections."""
        entries = []
        current = start_time
        for i, text in enumerate(phrases, start=1):
            end = current + phrase_duration
            entries.append(SubtitleEntry(
                index=i,
                start_sec=round(current, 2),
                end_sec=round(end, 2),
                text=text.strip(),
            ))
            current = end + gap
        return entries

    def build_ffmpeg_subtitles_filter(
        self,
        srt_path: str | Path,
        font_name: str = "Arial",
        font_size: int = 24,
        primary_color_hex: str = "&H00FFFFFF",  # ASS White
        outline_color_hex: str = "&H00000000",  # ASS Black outline
        bold: bool = True,
        margin_v: int = 40,
    ) -> str:
        """Construct FFmpeg video filter parameter for burning subtitles into video.

        Properly escapes Windows drive colons and backslashes for FFmpeg filtergraph:
        e.g. subtitles='D\\:/path/to/sub.srt':force_style='FontSize=24,...'
        """
        p = Path(srt_path).resolve()
        # FFmpeg filter parser requires escaping colons and backslashes
        escaped_path = str(p).replace("\\", "/").replace(":", "\\:")

        bold_val = "1" if bold else "0"
        style = (
            f"Fontname={font_name},"
            f"Fontsize={font_size},"
            f"PrimaryColour={primary_color_hex},"
            f"OutlineColour={outline_color_hex},"
            f"Bold={bold_val},"
            f"Outline=2,"
            f"Shadow=1,"
            f"MarginV={margin_v}"
        )
        return f"subtitles='{escaped_path}':force_style='{style}'"


subtitle_manager = SubtitleManager()

__all__ = ["SubtitleEntry", "SubtitleManager", "subtitle_manager"]
