"""J.A.R.V.I.S. Voice Intelligence Package (Phase 10).

Local, offline-first voice system:
- Text-to-Speech via pyttsx3 (Windows SAPI5)
- Speech-to-Text via faster-whisper (local CPU Whisper)
- Wake Word Detection via continuous microphone monitoring

All components are completely optional. JARVIS operates
seamlessly without microphone or audio output hardware.
"""

from voice.text_to_speech import LocalTTSEngine, TTSResult, VoiceInfo
from voice.speech_to_text import LocalSTTEngine, TranscriptionResult
from voice.wake_word import (
    VoiceState,
    WakeWordDetector,
    WakeWordEvent,
    WAKE_PHRASES,
)

__all__ = [
    "LocalTTSEngine",
    "TTSResult",
    "VoiceInfo",
    "LocalSTTEngine",
    "TranscriptionResult",
    "VoiceState",
    "WakeWordDetector",
    "WakeWordEvent",
    "WAKE_PHRASES",
]
