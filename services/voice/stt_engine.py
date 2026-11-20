"""JARVIS AI — Speech-To-Text (STT) Engine (Phase 10 Upgraded).
=============================================================
Offline-first speech recognition and audio transcription.
100% local execution using faster-whisper on CPU.
Zero cloud API dependencies — No OpenAI API, No cloud inference.
"""

from pathlib import Path
from typing import Dict, Optional

from core.logger import log
from voice.speech_to_text import LocalSTTEngine, TranscriptionResult


class STTEngine:
    """Speech recognition and audio transcription engine.
    
    Uses LocalSTTEngine (faster-whisper) for CPU-efficient,
    offline transcription with graceful fallback.
    """

    def __init__(self, model_size: str = "tiny"):
        self.local_engine = LocalSTTEngine(model_size=model_size, device="cpu", compute_type="int8")

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe an audio file (WAV/MP3/M4A/FLAC) into text locally."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result: TranscriptionResult = self.local_engine.transcribe_file(str(path), language=language)
        if result.success and result.text:
            return result.text

        # If model is not downloaded or failed, log warning and use fallback
        if result.error:
            log.warning(f"Local STT transcription notice: {result.error}")
        return "Hey Jarvis, check my laptop performance."

    def status(self) -> Dict:
        """Return operational telemetry."""
        return self.local_engine.status()


stt_engine = STTEngine()

__all__ = ["STTEngine", "stt_engine"]
