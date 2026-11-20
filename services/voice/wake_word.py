"""
JARVIS AI — Wake-Word Detection & Voice State Engine
=====================================================
Manages the acoustic trigger "Hey Jarvis", conversation continuity,
voice state lifecycle, and interruption handling.
"""

from enum import Enum
from typing import Optional, Callable
from pydantic import BaseModel
from core.logger import log


class VoiceState(str, Enum):
    STANDBY = "standby"          # Passive listening for wake-word
    LISTENING = "listening"      # Actively recording user speech
    PROCESSING = "processing"    # AI Brain / reasoning engine thinking
    SPEAKING = "speaking"        # Synthesizing and speaking response
    INTERRUPTED = "interrupted"  # Voice playback halted by user voice


class WakeWordDetector:
    """Detects 'Hey Jarvis' wake word and controls voice state transitions."""

    WAKE_PHRASES = ["hey jarvis", "jarvis", "okay jarvis", "hi jarvis"]

    def __init__(self):
        self.state = VoiceState.STANDBY
        self._interruption_flag = False

    def check_wake_phrase(self, text: str) -> bool:
        """Evaluate if text contains the wake-word trigger."""
        cleaned = text.strip().lower()
        for phrase in self.WAKE_PHRASES:
            if phrase in cleaned:
                log.info(f"Wake word detected: '{phrase}'")
                self.state = VoiceState.LISTENING
                return True
        return False

    def trigger_interruption(self):
        """Immediately halt ongoing speech synthesis when user speaks."""
        if self.state == VoiceState.SPEAKING:
            log.info("Voice interruption triggered by user.")
            self.state = VoiceState.INTERRUPTED
            self._interruption_flag = True

    def reset_state(self):
        """Return to passive standby listening."""
        self.state = VoiceState.STANDBY
        self._interruption_flag = False


wake_word_detector = WakeWordDetector()
