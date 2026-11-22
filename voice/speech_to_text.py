"""J.A.R.V.I.S. — Local Speech-to-Text Engine (Phase 10).

Offline speech recognition powered by faster-whisper (CTranslate2).

Design Principles:
- Runs OpenAI Whisper model **locally** on CPU — zero cloud API calls
- Lazy model loading: model is instantiated only when first needed
- Model sizes safe for 8 GB RAM Intel i3:
    * tiny  ≈  75 MB RAM  —  fastest, good for wake-word detection
    * base  ≈ 150 MB RAM  —  better accuracy for full commands
- Transcribes WAV / MP3 / M4A / FLAC / OGG audio files
- Accepts raw numpy float32 arrays (from microphone capture)
- Graceful fallback to text input when model is unavailable

Educational Notes:
    faster-whisper re-implements OpenAI Whisper using CTranslate2,
    an inference engine that applies 8-bit integer quantisation and
    batch-level parallelism to run Transformer encoder–decoder models
    up to 4× faster on CPU compared to the original PyTorch weights.

    The *tiny* model contains ~39 M parameters (encoder + decoder)
    and processes a 5-second audio clip in ~1–2 s on a modern i3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from core.logger import log


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    """Structured result from a speech-to-text transcription."""
    text: str
    language: str
    duration_sec: float
    model_used: str
    confidence: float
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Local STT Engine
# ---------------------------------------------------------------------------

class LocalSTTEngine:
    """Offline speech-to-text engine using faster-whisper on CPU.

    Parameters
    ----------
    model_size : str
        Whisper model variant — ``"tiny"`` (default, ~75 MB) or ``"base"`` (~150 MB).
    device : str
        ``"cpu"`` (default).  GPU is not expected on the target hardware.
    compute_type : str
        ``"int8"`` (default) for memory-efficient quantised inference.
    """

    SUPPORTED_MODELS = ("tiny", "base", "small")
    SUPPORTED_FORMATS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_size}' not supported. Choose from: {self.SUPPORTED_MODELS}"
            )
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._available: bool = False
        self._model_loaded: bool = False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the Whisper model into memory (lazy, called on first use).

        On the very first invocation the model weights are downloaded from
        HuggingFace (~75 MB for *tiny*).  After that they are cached and
        all subsequent loads are fully offline.
        """
        if self._model_loaded:
            return True

        try:
            from faster_whisper import WhisperModel

            log.info(
                f"Loading faster-whisper '{self._model_size}' on {self._device} "
                f"(compute_type={self._compute_type})…"
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._model_loaded = True
            self._available = True
            log.info(f"Whisper '{self._model_size}' ready for local CPU inference.")
            return True

        except ImportError:
            log.warning("faster-whisper is not installed. STT uses text fallback.")
            self._available = False
            return False
        except Exception as e:
            log.warning(f"Whisper model load failed: {e}. STT uses text fallback.")
            self._available = False
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    # ------------------------------------------------------------------
    # Transcription — from file
    # ------------------------------------------------------------------

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Parameters
        ----------
        audio_path : str
            Path to a WAV, MP3, M4A, FLAC or OGG file.
        language : str, optional
            ISO-639-1 code (e.g. ``"en"``).  Auto-detected when *None*.
        """
        start = time.perf_counter()
        path = Path(audio_path)

        if not path.exists():
            return TranscriptionResult(
                text="", language="", duration_sec=0,
                model_used=self._model_size, confidence=0,
                success=False, error=f"Audio file not found: {audio_path}",
            )

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return TranscriptionResult(
                text="", language="", duration_sec=0,
                model_used=self._model_size, confidence=0,
                success=False,
                error=f"Unsupported format '{path.suffix}'. Use: {self.SUPPORTED_FORMATS}",
            )

        # Lazy-load model
        if not self._model_loaded and not self.load_model():
            return TranscriptionResult(
                text="", language="", duration_sec=0,
                model_used=self._model_size, confidence=0,
                success=False, error="Whisper model not available.",
            )

        try:
            segments, info = self._model.transcribe(
                str(path),
                language=language,
                beam_size=1,       # fastest on CPU
                vad_filter=True,   # skip silence
            )
            full_text = " ".join(seg.text.strip() for seg in segments).strip()
            dur = time.perf_counter() - start

            log.info(f"Transcribed '{path.name}' in {dur:.2f}s: '{full_text[:60]}…'")
            return TranscriptionResult(
                text=full_text,
                language=info.language or "unknown",
                duration_sec=dur,
                model_used=self._model_size,
                confidence=info.language_probability or 0.0,
                success=True,
            )
        except Exception as e:
            dur = time.perf_counter() - start
            log.error(f"Transcription error: {e}")
            return TranscriptionResult(
                text="", language="", duration_sec=dur,
                model_used=self._model_size, confidence=0,
                success=False, error=str(e),
            )

    # ------------------------------------------------------------------
    # Transcription — from numpy array (microphone)
    # ------------------------------------------------------------------

    def transcribe_audio_array(
        self,
        audio_array,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe a raw numpy audio array (e.g. from microphone).

        Parameters
        ----------
        audio_array : numpy.ndarray
            Mono float32 audio samples.
        sample_rate : int
            Sample rate in Hz (Whisper expects 16 000).
        language : str, optional
            ISO-639-1 code or *None* for auto-detection.
        """
        start = time.perf_counter()

        if not self._model_loaded and not self.load_model():
            return TranscriptionResult(
                text="", language="", duration_sec=0,
                model_used=self._model_size, confidence=0,
                success=False, error="Whisper model not available.",
            )

        try:
            import numpy as np

            arr = audio_array
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 32768.0
            if len(arr.shape) > 1:
                arr = arr.mean(axis=1)

            segments, info = self._model.transcribe(
                arr,
                language=language,
                beam_size=1,
                vad_filter=True,
            )
            full_text = " ".join(seg.text.strip() for seg in segments).strip()
            dur = time.perf_counter() - start

            return TranscriptionResult(
                text=full_text,
                language=info.language or "unknown",
                duration_sec=dur,
                model_used=self._model_size,
                confidence=info.language_probability or 0.0,
                success=True,
            )
        except Exception as e:
            dur = time.perf_counter() - start
            return TranscriptionResult(
                text="", language="", duration_sec=dur,
                model_used=self._model_size, confidence=0,
                success=False, error=str(e),
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return STT engine status for the system status panel."""
        return {
            "available": self._available,
            "model_loaded": self._model_loaded,
            "model_size": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "engine": "faster_whisper",
            "offline": True,
            "requires_internet": False,
        }


__all__ = ["LocalSTTEngine", "TranscriptionResult"]
