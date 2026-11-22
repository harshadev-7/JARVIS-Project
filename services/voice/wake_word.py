"""JARVIS AI — Wake-Word Detection & Voice State Engine (Phase 10 Upgraded).
==========================================================================
Manages acoustic trigger "Hey Jarvis", conversation continuity,
voice state lifecycle, and interruption handling.
Supports both live microphone monitoring and text-mode trigger matching.
"""

from voice.wake_word import (
    WAKE_PHRASES,
    VoiceState,
    WakeWordDetector,
    WakeWordEvent,
)

# Shared singleton instance
wake_word_detector = WakeWordDetector()

__all__ = [
    "WAKE_PHRASES",
    "VoiceState",
    "WakeWordDetector",
    "WakeWordEvent",
    "wake_word_detector",
]
