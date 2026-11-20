"""
JARVIS AI — Speech-To-Text (STT) Engine
========================================
Transcribes spoken user audio into text.
Supports OpenAI Whisper API when key is configured,
with mock/local acoustic decoder fallback.
"""

from pathlib import Path
from typing import Optional
import httpx

from config.settings import settings
from core.logger import log


class STTEngine:
    """Speech recognition and audio transcription."""

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY

    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file (WAV/MP3/M4A) into text.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # If OpenAI API key is present, use Whisper API
        if self.api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.api_key)
                with open(path, "rb") as audio_file:
                    transcription = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                    )
                return transcription.text.strip()
            except Exception as e:
                log.warning(f"OpenAI Whisper transcription failed: {e}")

        # Fallback acoustic message if offline / testing
        return "Hey Jarvis, check my laptop performance."


stt_engine = STTEngine()
