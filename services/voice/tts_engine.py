"""
JARVIS AI — Text-To-Speech (TTS) Engine
========================================
High-definition neural speech synthesis inspired by JARVIS.
Uses Microsoft Edge Neural voices (default: en-GB-RyanNeural for the iconic British AI cadence)
with zero mandatory API keys or subscription fees.
"""

import asyncio
from pathlib import Path
from typing import Optional
import edge_tts

from config.settings import BASE_DIR, settings
from core.logger import log


class TTSEngine:
    """Natural, calm, professional speech synthesizer."""

    # Iconic British English male neural voice
    DEFAULT_VOICE = "en-GB-RyanNeural"

    def __init__(self, voice: Optional[str] = None):
        self.voice = voice or self.DEFAULT_VOICE
        self.output_dir = BASE_DIR / "logs" / "voice_cache"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> Path:
        """
        Synthesize text into natural MP3 audio file.
        Returns the absolute Path to the audio file.
        """
        chosen_voice = voice or self.voice
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text to synthesize cannot be empty.")

        target_path = Path(output_file) if output_file else (
            self.output_dir / f"jarvis_speech_{abs(hash(clean_text)) % 1000000}.mp3"
        )

        log.debug(f"Synthesizing voice: '{clean_text[:40]}...' using {chosen_voice}")
        communicate = edge_tts.Communicate(clean_text, chosen_voice)
        await communicate.save(str(target_path))
        return target_path

    async def synthesize_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize text and return raw MP3 bytes."""
        chosen_voice = voice or self.voice
        communicate = edge_tts.Communicate(text.strip(), chosen_voice)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)


tts_engine = TTSEngine()
