"""JARVIS AutoClip Engine — Timeline Refiner & Boundary Snapper
=============================================================
Refines candidate highlight boundaries by snapping them to precise subtitle cue borders
and speech pauses, enforcing minimum and maximum duration profiles, and deduplicating
overlapping moments.

Inspired by AutoClip's quality engineering and `refine_timeline` algorithms.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from video_editor.autoclip.transcript_slicer import TranscriptCue, to_seconds, to_srt_time


@dataclass(frozen=True)
class DurationProfile:
    """Configures target length and retention parameters based on source video length."""
    tier: str                             # "short", "medium", "long"
    total_sec: float
    min_clip_sec: float = 30.0            # Min allowed length for a viral clip
    target_clip_sec: Tuple[float, float] = (60.0, 120.0)  # Ideal length (1–2 minutes)
    max_clip_sec: float = 180.0           # Max allowed clip length
    min_keep: int = 1                     # Minimum clips to keep
    max_clips: int = 10                   # Maximum clips to export
    snap_window_sec: float = 4.0          # Cue boundary search window
    duplicate_overlap_ratio: float = 0.5  # Shared overlap ratio threshold to mark duplicate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "total_sec": self.total_sec,
            "min_clip_sec": self.min_clip_sec,
            "target_clip_sec": list(self.target_clip_sec),
            "max_clip_sec": self.max_clip_sec,
            "min_keep": self.min_keep,
            "max_clips": self.max_clips,
            "snap_window_sec": self.snap_window_sec,
            "duplicate_overlap_ratio": self.duplicate_overlap_ratio,
        }


def profile_from_duration(total_sec: float) -> DurationProfile:
    """Builds an optimized DurationProfile based on input video length."""
    total_sec = max(1.0, float(total_sec))
    if total_sec < 15 * 60:   # < 15 minutes
        return DurationProfile(
            tier="short",
            total_sec=total_sec,
            min_clip_sec=20.0,
            target_clip_sec=(30.0, 75.0),
            max_clip_sec=120.0,
            min_keep=1,
            max_clips=5,
        )
    elif total_sec < 45 * 60: # 15 to 45 minutes
        return DurationProfile(
            tier="medium",
            total_sec=total_sec,
            min_clip_sec=30.0,
            target_clip_sec=(60.0, 120.0),
            max_clip_sec=180.0,
            min_keep=2,
            max_clips=8,
        )
    else:                     # > 45 minutes
        return DurationProfile(
            tier="long",
            total_sec=total_sec,
            min_clip_sec=45.0,
            target_clip_sec=(60.0, 150.0),
            max_clip_sec=240.0,
            min_keep=3,
            max_clips=12,
        )


@dataclass
class RefinedClip:
    """A verified, boundary-snapped clip candidate ready for extraction."""
    id: int
    title: str
    reason: str
    score: float
    start_sec: float
    end_sec: float
    start_row: int
    end_row: int
    text_snippet: str
    tags: List[str] = dataclasses.field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "reason": self.reason,
            "score": round(self.score, 3),
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.duration_sec, 2),
            "start_srt": to_srt_time(self.start_sec),
            "end_srt": to_srt_time(self.end_sec),
            "start_row": self.start_row,
            "end_row": self.end_row,
            "text_snippet": self.text_snippet,
            "tags": self.tags,
        }


class TimelineRefiner:
    """Snaps boundaries, eliminates duplicate moments, and enforces duration constraints."""

    @classmethod
    def snap_to_cue_boundary(
        cls,
        target_sec: float,
        cues: List[TranscriptCue],
        prefer_start: bool = True,
        max_dist: float = 4.0,
    ) -> float:
        """Finds the nearest cue timestamp within max_dist seconds."""
        if not cues:
            return target_sec

        best_time = target_sec
        best_delta = float("inf")

        for cue in cues:
            edge = cue.start_sec if prefer_start else cue.end_sec
            delta = abs(edge - target_sec)
            if delta < best_delta and delta <= max_dist:
                best_delta = delta
                best_time = edge

        return best_time

    @classmethod
    def deduplicate(
        cls,
        clips: List[RefinedClip],
        overlap_threshold: float = 0.5,
    ) -> List[RefinedClip]:
        """Suppresses lower-scoring clips that share > overlap_threshold of the shorter clip."""
        if not clips:
            return []

        # Sort highest score first
        sorted_clips = sorted(clips, key=lambda c: c.score, reverse=True)
        retained: List[RefinedClip] = []

        for candidate in sorted_clips:
            is_dup = False
            for kept in retained:
                # Calculate intersection
                s = max(candidate.start_sec, kept.start_sec)
                e = min(candidate.end_sec, kept.end_sec)
                shared = max(0.0, e - s)

                dur_candidate = candidate.duration_sec
                dur_kept = kept.duration_sec
                min_dur = min(dur_candidate, dur_kept)

                if min_dur > 0 and (shared / min_dur) >= overlap_threshold:
                    is_dup = True
                    break

            if not is_dup:
                retained.append(candidate)

        # Re-sort chronologically by start time
        return sorted(retained, key=lambda c: c.start_sec)

    @classmethod
    def refine_clips(
        cls,
        raw_candidates: List[Dict[str, Any]],
        cues: List[TranscriptCue],
        profile: Optional[DurationProfile] = None,
    ) -> List[RefinedClip]:
        """Refines raw candidate dictionaries into high-precision, snapped RefinedClips."""
        if not raw_candidates or not cues:
            return []

        total_dur = cues[-1].end_sec - cues[0].start_sec if cues else 60.0
        active_profile = profile or profile_from_duration(total_dur)
        refined: List[RefinedClip] = []

        for i, item in enumerate(raw_candidates, start=1):
            title = str(item.get("title", f"Highlight #{i}")).strip()
            reason = str(item.get("reason", "Top viral insight moment")).strip()
            score = float(item.get("score", 0.7))

            # Determine start and end rows
            start_row = item.get("start_row")
            end_row = item.get("end_row")

            if start_row is not None and end_row is not None and 0 <= start_row < len(cues) and 0 <= end_row < len(cues):
                s_idx = min(start_row, end_row)
                e_idx = max(start_row, end_row)
                start_sec = cues[s_idx].start_sec
                end_sec = cues[e_idx].end_sec
            else:
                raw_s = to_seconds(item.get("start_time", 0.0))
                raw_e = to_seconds(item.get("end_time", raw_s + 60.0))
                start_sec = cls.snap_to_cue_boundary(raw_s, cues, prefer_start=True, max_dist=active_profile.snap_window_sec)
                end_sec = cls.snap_to_cue_boundary(raw_e, cues, prefer_start=False, max_dist=active_profile.snap_window_sec)

                # Locate nearest row indices
                s_idx = min(range(len(cues)), key=lambda idx: abs(cues[idx].start_sec - start_sec))
                e_idx = min(range(len(cues)), key=lambda idx: abs(cues[idx].end_sec - end_sec))

            # Enforce minimum clip duration: expand forward if needed
            duration = end_sec - start_sec
            if duration < active_profile.min_clip_sec:
                needed = active_profile.min_clip_sec - duration
                # Try expanding e_idx
                while e_idx + 1 < len(cues) and (cues[e_idx].end_sec - start_sec) < active_profile.min_clip_sec:
                    e_idx += 1
                end_sec = cues[e_idx].end_sec

            # Enforce maximum clip duration: trim back if needed
            if (end_sec - start_sec) > active_profile.max_clip_sec:
                while e_idx > s_idx and (cues[e_idx].end_sec - start_sec) > active_profile.max_clip_sec:
                    e_idx -= 1
                end_sec = cues[e_idx].end_sec

            # Extract text snippet from included cues
            snippet = " ".join(cues[idx].text for idx in range(s_idx, min(e_idx + 1, len(cues))))

            refined.append(
                RefinedClip(
                    id=i,
                    title=title,
                    reason=reason,
                    score=max(0.0, min(1.0, score)),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_row=s_idx,
                    end_row=e_idx,
                    text_snippet=snippet,
                    tags=item.get("tags", ["viral", "highlight"]),
                )
            )

        # Deduplicate overlapping clips
        deduped = cls.deduplicate(refined, overlap_threshold=active_profile.duplicate_overlap_ratio)

        # Filter by profile capacity
        if len(deduped) > active_profile.max_clips:
            # Keep top scoring within max_clips, then restore chronological order
            top_scored = sorted(deduped, key=lambda c: c.score, reverse=True)[:active_profile.max_clips]
            deduped = sorted(top_scored, key=lambda c: c.start_sec)

        return deduped
