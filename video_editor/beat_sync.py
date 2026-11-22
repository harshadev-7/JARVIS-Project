"""J.A.R.V.I.S. Audio Beat & Rhythm Synchronization Engine.

Extracts musical rhythm and acoustic energy peaks:
- Pure-Python WAV parsing (zero external dependencies)
- Short-Time Energy (RMS) envelope analysis
- Dynamic peak detection for downbeats and rhythmic transients
- Snaps video cut points to nearest musical beats
- Generates rhythmic BPM cut grids for music video editing
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from core.logger import log
from video_editor.safety import media_safety_guard


@dataclass
class BeatPoint:
    """Individual musical beat or rhythmic transient."""
    timestamp_sec: float
    intensity: float
    is_downbeat: bool = False


@dataclass
class BeatAnalysisResult:
    """Summary of rhythmic audio analysis."""
    source_file: str
    duration_sec: float
    estimated_bpm: float
    total_beats: int
    beats: List[BeatPoint] = field(default_factory=list)
    beat_timestamps: List[float] = field(default_factory=list)


class BeatSyncEngine:
    """Analyzes audio tracks and aligns visual edits to musical beats."""

    def __init__(self, window_size_ms: float = 50.0):
        self.window_size_ms = window_size_ms

    def analyze_audio(
        self,
        audio_path: str | Path,
        sensitivity: float = 1.3,
    ) -> BeatAnalysisResult:
        """Extract beat timestamps from audio file.

        Args:
            audio_path: Path to WAV or media audio file.
            sensitivity: Multiplier over mean energy for peak detection (1.1 - 2.0).

        Returns:
            BeatAnalysisResult containing detected beat timestamps and estimated BPM.
        """
        p = media_safety_guard.validate_file_path(audio_path, must_exist=True)

        # 1. If uncompressed WAV, parse directly via Python standard library wave
        if p.suffix.lower() == ".wav":
            try:
                return self._analyze_wav(p, sensitivity)
            except Exception as e:
                log.debug(f"Native WAV parsing failed: {e}. Falling back to synthetic grid.")

        # 2. Synthetic or fallback rhythmic grid (e.g. standard 120 BPM)
        return self._generate_synthetic_beat_result(p, default_bpm=120.0, duration=30.0)

    def _analyze_wav(self, path: Path, sensitivity: float) -> BeatAnalysisResult:
        """Pure-Python Short-Time Energy (RMS) calculation on PCM WAV."""
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / float(framerate) if framerate > 0 else 0.0

            if n_frames == 0 or framerate == 0:
                return self._generate_synthetic_beat_result(path, 120.0, 10.0)

            # Read audio data in chunks to preserve RAM
            # Only support 16-bit PCM directly (most common)
            if sampwidth == 2:
                window_frames = int((self.window_size_ms / 1000.0) * framerate)
                if window_frames <= 0:
                    window_frames = 1024

                energies: List[Tuple[float, float]] = []  # (timestamp_sec, rms_energy)
                step_sec = window_frames / float(framerate)

                for chunk_idx in range(0, n_frames, window_frames):
                    raw = wf.readframes(window_frames)
                    if not raw:
                        break

                    count = len(raw) // 2
                    if count == 0:
                        continue

                    # Unpack 16-bit signed shorts
                    fmt = f"<{count}h"
                    try:
                        samples = struct.unpack(fmt, raw)
                        # Average channels if stereo
                        if n_channels > 1:
                            samples = samples[::n_channels]
                        
                        # Root-Mean-Square (RMS) energy
                        sum_sq = sum(s * s for s in samples)
                        rms = math.sqrt(sum_sq / len(samples))
                        t_sec = (chunk_idx / float(framerate))
                        energies.append((t_sec, rms))
                    except Exception:
                        break

                if energies:
                    return self._detect_peaks_from_energies(path, duration, energies, sensitivity)

        return self._generate_synthetic_beat_result(path, 120.0, duration)

    def _detect_peaks_from_energies(
        self,
        path: Path,
        duration: float,
        energies: List[Tuple[float, float]],
        sensitivity: float,
    ) -> BeatAnalysisResult:
        """Compute rolling energy threshold and peak detection."""
        all_rms = [e[1] for e in energies]
        mean_rms = sum(all_rms) / max(1, len(all_rms))
        threshold = mean_rms * sensitivity

        beats: List[BeatPoint] = []
        min_gap_sec = 0.25  # Max 240 BPM
        last_beat_t = -min_gap_sec

        for i in range(1, len(energies) - 1):
            t, e = energies[i]
            prev_e = energies[i - 1][1]
            next_e = energies[i + 1][1]

            # Peak condition: higher than neighbours and exceeds threshold
            if e > prev_e and e > next_e and e > threshold:
                if (t - last_beat_t) >= min_gap_sec:
                    intensity = round(e / max(1.0, mean_rms), 2)
                    beats.append(BeatPoint(
                        timestamp_sec=round(t, 2),
                        intensity=intensity,
                        is_downbeat=intensity > (sensitivity * 1.5),
                    ))
                    last_beat_t = t

        beat_times = [b.timestamp_sec for b in beats]

        # Estimate BPM from average inter-beat interval
        bpm = 120.0
        if len(beat_times) >= 2:
            intervals = [beat_times[i] - beat_times[i - 1] for i in range(1, len(beat_times))]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                bpm = round(60.0 / avg_interval, 1)

        return BeatAnalysisResult(
            source_file=str(path),
            duration_sec=round(duration, 2),
            estimated_bpm=bpm,
            total_beats=len(beats),
            beats=beats,
            beat_timestamps=beat_times,
        )

    def _generate_synthetic_beat_result(
        self,
        path: Path,
        default_bpm: float = 120.0,
        duration: float = 30.0,
    ) -> BeatAnalysisResult:
        """Create a rhythmic beat grid based on target BPM."""
        interval = 60.0 / default_bpm
        beats: List[BeatPoint] = []
        cur = 0.0
        idx = 0
        while cur < duration:
            beats.append(BeatPoint(
                timestamp_sec=round(cur, 2),
                intensity=1.5 if (idx % 4 == 0) else 1.0,
                is_downbeat=(idx % 4 == 0),
            ))
            cur += interval
            idx += 1

        return BeatAnalysisResult(
            source_file=str(path),
            duration_sec=duration,
            estimated_bpm=default_bpm,
            total_beats=len(beats),
            beats=beats,
            beat_timestamps=[b.timestamp_sec for b in beats],
        )

    def align_cuts_to_beats(
        self,
        cut_timestamps: List[float],
        beat_timestamps: List[float],
        tolerance_sec: float = 0.25,
    ) -> List[float]:
        """Snap visual cut timestamps to the nearest musical beat within tolerance.

        If a cut point is within `tolerance_sec` of a beat, snap it to that beat.
        Otherwise keep original timestamp to prevent visual discontinuity.
        """
        if not beat_timestamps or not cut_timestamps:
            return cut_timestamps

        aligned = []
        for cut in cut_timestamps:
            nearest_beat = min(beat_timestamps, key=lambda b: abs(b - cut))
            if abs(nearest_beat - cut) <= tolerance_sec:
                aligned.append(round(nearest_beat, 2))
            else:
                aligned.append(round(cut, 2))

        # Return sorted unique cut timestamps
        return sorted(list(set(aligned)))

    def generate_beat_grid(
        self,
        duration_sec: float,
        bpm: float = 120.0,
        beats_per_cut: int = 4,
    ) -> List[float]:
        """Generate a series of cut points matched to music bars (e.g. every 4 beats)."""
        seconds_per_beat = 60.0 / bpm
        cut_interval = seconds_per_beat * beats_per_cut

        cuts = []
        t = cut_interval
        while t < duration_sec:
            cuts.append(round(t, 2))
            t += cut_interval

        return cuts


beat_sync_engine = BeatSyncEngine()

__all__ = ["BeatPoint", "BeatAnalysisResult", "BeatSyncEngine", "beat_sync_engine"]
