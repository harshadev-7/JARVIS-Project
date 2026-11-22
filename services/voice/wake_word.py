"""JARVIS AI — Wake-Word Detection & Voice State Engine (Phase 10 Upgraded).
==========================================================================
Manages acoustic trigger "Hey Jarvis", conversation continuity,
voice state lifecycle, and interruption handling.
Supports both live microphone monitoring and text-mode trigger matching.
"""

from voice.wake_word import (
    WAKE_PHRASES,
    VoiceState,
    WakeWordEvent,
    WakeWordDetector as _BaseWakeWordDetector,
)


class WakeWordDetector(_BaseWakeWordDetector):
    """Voice trigger detector supporting boolean evaluation for Phase 5 tests."""

    def check_wake_phrase(self, text: str) -> bool:
        """Evaluate if text contains the wake-word trigger (returns bool)."""
        event = super().check_wake_phrase(text)
        return True if event is not None else False


# Shared singleton instance
wake_word_detector = WakeWordDetector()

__all__ = [
    "WAKE_PHRASES",
    "VoiceState",
    "WakeWordDetector",
    "WakeWordEvent",
    "wake_word_detector",
]
