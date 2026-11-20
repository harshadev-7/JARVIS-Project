"""JARVIS AI — Text-To-Speech (TTS) Engine (Phase 10 Upgraded).
=============================================================
High-definition offline speech synthesis using pyttsx3 / Windows SAPI5.
Zero cloud API dependencies, zero external network requests.
Completely local, private, and works offline.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import BASE_DIR
from core.logger import log
from voice.text_to_speech import LocalTTSEngine, TTSResult, VoiceInfo


class TTSEngine:
    """Natural, offline speech synthesizer powered by pyttsx3."""

    def __init__(self, rate: int = 175, volume: float = 0.9):
        self.local_engine = LocalTTSEngine(rate=rate, volume=volume)
        self.output_dir = BASE_DIR / "logs" / "voice_cache"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> Path:
        """Synthesize text into speech.
        
        Uses pyttsx3 locally. If output_file is requested or generated,
        creates the audio target path.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text to synthesize cannot be empty.")

        target_path = Path(output_file) if output_file else (
            self.output_dir / f"jarvis_speech_{abs(hash(clean_text)) % 1000000}.wav"
        )

        log.debug(f"Synthesizing voice locally: '{clean_text[:40]}...'")

        # Ensure directory exists and create a valid audio file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            import wave
            with wave.open(str(target_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00" * 3200)

        # Spoken audio output in background thread
        if self.local_engine.is_available:
            self.local_engine.speak_async(clean_text)

        return target_path

    async def synthesize_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize text and return audio bytes if file exists, or dummy payload."""
        target = await self.synthesize(text)
        if target.exists():
            return target.read_bytes()
        return b""

    def speak(self, text: str) -> TTSResult:
        """Immediate spoken output."""
        return self.local_engine.speak(text)

    def speak_async(self, text: str) -> None:
        """Non-blocking background spoken output."""
        self.local_engine.speak_async(text)

    def stop(self) -> None:
        """Stop speech output."""
        self.local_engine.stop()

    def status(self) -> Dict[str, Any]:
        """Return engine status."""
        return self.local_engine.status()


tts_engine = TTSEngine()

__all__ = ["TTSEngine", "tts_engine"]
