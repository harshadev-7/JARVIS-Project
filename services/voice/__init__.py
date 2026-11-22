"""JARVIS AI — Services Voice Package (Phase 10).

Exposes the local offline voice orchestration infrastructure:
- stt_engine: Speech-to-Text via local faster-whisper
- tts_engine: Text-to-Speech via local pyttsx3 / SAPI5
- wake_word_detector: Keyword trigger & voice state machine
- voice_orchestrator: End-to-end voice conversation loop
"""

from services.voice.stt_engine import STTEngine, stt_engine
from services.voice.tts_engine import TTSEngine, tts_engine
from services.voice.voice_orchestrator import VoiceOrchestrator, voice_orchestrator
from services.voice.wake_word import (
    WAKE_PHRASES,
    VoiceState,
    WakeWordDetector,
    WakeWordEvent,
    wake_word_detector,
)

__all__ = [
    "STTEngine",
    "stt_engine",
    "TTSEngine",
    "tts_engine",
    "VoiceOrchestrator",
    "voice_orchestrator",
    "VoiceState",
    "WakeWordDetector",
    "WakeWordEvent",
    "WAKE_PHRASES",
    "wake_word_detector",
]
