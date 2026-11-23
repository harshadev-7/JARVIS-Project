"""JARVIS AutoClip Engine — Transcript Slicer & Window Manager
============================================================
Parses subtitles (SRT, VTT, JSON, raw text) into structured cues and
divides long-duration transcripts into token-efficient overlapping windows
for highlight evaluation.

Inspired by AutoClip's one-pass context windowing algorithm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union


def to_seconds(t: str | float | int) -> float:
    """Converts 'HH:MM:SS,mmm' / 'HH:MM:SS.mmm' / 'MM:SS' or numeric seconds to float seconds."""
    if isinstance(t, (int, float)):
        return max(0.0, float(t))
    s = str(t).strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return max(0.0, int(h) * 3600 + int(m) * 60 + float(sec))
    elif len(parts) == 2:
        m, sec = parts
        return max(0.0, int(m) * 60 + float(sec))
    try:
        return max(0.0, float(s))
    except ValueError:
        raise ValueError(f"Invalid timestamp format: '{t}'")


def to_srt_time(sec: float) -> str:
    """Converts seconds into standard SRT timestamp 'HH:MM:SS,mmm'."""
    total_ms = int(round(max(0.0, float(sec)) * 1000))
    h, rest = divmod(total_ms, 3600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_clock_time(sec: float) -> str:
    """Formats seconds into readable clock format 'M:SS' or 'H:MM:SS'."""
    total_sec = int(round(max(0.0, float(sec))))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


@dataclass(frozen=True)
class TranscriptCue:
    """Represents a single timed sentence or speech unit."""
    id: int
    start_sec: float
    end_sec: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_row(self) -> str:
        """Token-efficient representation: id|time|text"""
        return f"{self.id}|{to_clock_time(self.start_sec)}|{self.text}"


@dataclass(frozen=True)
class SlicingWindow:
    """An analysis window spanning a contiguous range of cues."""
    first_idx: int
    last_idx: int
    start_sec: float
    end_sec: float
    total_chars: int

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


class TranscriptSlicer:
    """Parses subtitle inputs and produces sliding analysis windows."""

    # Default window profiles
    LOCAL_WINDOW_SEC = 20.0 * 60.0       # 20 minutes
    LOCAL_OVERLAP_SEC = 4.0 * 60.0       # 4 minutes
    LOCAL_MAX_CHARS = 12_000             # ~8K tokens

    CLOUD_WINDOW_SEC = 60.0 * 60.0       # 60 minutes
    CLOUD_OVERLAP_SEC = 4.0 * 60.0       # 4 minutes
    CLOUD_MAX_CHARS = 70_000             # ~45K tokens

    @classmethod
    def parse_srt(cls, srt_content_or_path: Union[str, Path]) -> List[TranscriptCue]:
        """Parses SRT content string or file path into a list of TranscriptCues."""
        content = ""
        if isinstance(srt_content_or_path, Path) or (
            isinstance(srt_content_or_path, str) and "\n" not in srt_content_or_path and Path(srt_content_or_path).is_file()
        ):
            content = Path(srt_content_or_path).read_text(encoding="utf-8", errors="replace")
        else:
            content = str(srt_content_or_path)

        cues: List[TranscriptCue] = []
        # Match SRT blocks: index, timestamp --> timestamp, text lines
        pattern = re.compile(
            r"(\d+)\s*\r?\n"
            r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*\r?\n"
            r"(.*?)(?=\r?\n\r?\n\d+\s*\r?\n|\r?\n\r?\n?$|\Z)",
            re.DOTALL
        )

        for match in pattern.finditer(content):
            idx_str, start_str, end_str, text_str = match.groups()
            idx = int(idx_str)
            start = to_seconds(start_str)
            end = to_seconds(end_str)
            cleaned_text = re.sub(r"<[^>]+>", "", text_str).strip()
            cleaned_text = " ".join(cleaned_text.split())
            if cleaned_text:
                cues.append(TranscriptCue(id=idx, start_sec=start, end_sec=end, text=cleaned_text))

        # Fallback if standard SRT regex missed cues (e.g. non-standard numbering)
        if not cues and "-->" in content:
            alt_pattern = re.compile(
                r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*\r?\n(.*?)(?=\n\n|\Z)",
                re.DOTALL
            )
            for i, match in enumerate(alt_pattern.finditer(content), start=1):
                start_str, end_str, text_str = match.groups()
                start = to_seconds(start_str)
                end = to_seconds(end_str)
                cleaned_text = re.sub(r"<[^>]+>", "", text_str).strip()
                cleaned_text = " ".join(cleaned_text.split())
                if cleaned_text:
                    cues.append(TranscriptCue(id=i, start_sec=start, end_sec=end, text=cleaned_text))

        return cues

    @classmethod
    def parse_vtt(cls, vtt_content_or_path: Union[str, Path]) -> List[TranscriptCue]:
        """Parses WebVTT format into TranscriptCues."""
        content = ""
        if isinstance(vtt_content_or_path, Path) or (
            isinstance(vtt_content_or_path, str) and "\n" not in vtt_content_or_path and Path(vtt_content_or_path).is_file()
        ):
            content = Path(vtt_content_or_path).read_text(encoding="utf-8", errors="replace")
        else:
            content = str(vtt_content_or_path)

        cues: List[TranscriptCue] = []
        pattern = re.compile(
            r"((?:\d{1,2}:)?\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{2}:\d{2}[,\.]\d{1,3})(?:[^\r\n]*)\r?\n(.*?)(?=\r?\n\r?\n|\Z)",
            re.DOTALL
        )
        for i, match in enumerate(pattern.finditer(content), start=1):
            start_str, end_str, text_str = match.groups()
            start = to_seconds(start_str)
            end = to_seconds(end_str)
            cleaned_text = re.sub(r"<[^>]+>", "", text_str).strip()
            cleaned_text = " ".join(cleaned_text.split())
            if cleaned_text:
                cues.append(TranscriptCue(id=i, start_sec=start, end_sec=end, text=cleaned_text))

        return cues

    @classmethod
    def parse_raw_text(cls, text: str, total_duration_sec: float = 60.0) -> List[TranscriptCue]:
        """Parses plain text sentences and assigns proportional timestamps across total duration."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
        if not sentences:
            return []

        cues: List[TranscriptCue] = []
        step = max(1.0, total_duration_sec / len(sentences))
        for i, sent in enumerate(sentences, start=1):
            start = (i - 1) * step
            end = min(total_duration_sec, i * step)
            cues.append(TranscriptCue(id=i, start_sec=start, end_sec=end, text=sent))
        return cues

    @classmethod
    def format_rows(cls, cues: List[TranscriptCue]) -> str:
        """Formats cues as compact rows 'id|time|text' to conserve tokens."""
        return "\n".join(cue.to_row() for cue in cues)

    @classmethod
    def compute_windows(
        cls,
        cues: List[TranscriptCue],
        window_sec: float = LOCAL_WINDOW_SEC,
        overlap_sec: float = LOCAL_OVERLAP_SEC,
        max_chars: int = LOCAL_MAX_CHARS,
    ) -> List[SlicingWindow]:
        """Calculates overlapping sliding windows over the transcript cues."""
        if not cues:
            return []

        windows: List[SlicingWindow] = []
        first = 0
        total_cues = len(cues)

        while first < total_cues:
            start_sec = cues[first].start_sec
            chars = 0
            last = first

            while last + 1 < total_cues:
                next_cue = cues[last + 1]
                if (next_cue.end_sec - start_sec > window_sec) or (chars + len(next_cue.text) > max_chars):
                    break
                last += 1
                chars += len(cues[last].text)

            end_sec = cues[last].end_sec
            windows.append(
                SlicingWindow(
                    first_idx=first,
                    last_idx=last,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    total_chars=chars,
                )
            )

            if last + 1 >= total_cues:
                break

            # Backtrack by overlap
            back = last
            while back > first and (cues[last].end_sec - cues[back].start_sec) < overlap_sec:
                back -= 1
            first = max(back, first + 1)

        return windows
