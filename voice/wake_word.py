"""J.A.R.V.I.S. — Wake Word Detection & Voice Listener (Phase 10).

Continuous microphone monitoring for "Hey Jarvis" wake word detection.

Design Principles:
- Uses sounddevice for real-time audio capture
- Voice Activity Detection via RMS energy threshold
- Transcribes short audio chunks through local STT engine
- Detects wake phrases: "hey jarvis", "jarvis", "ok jarvis", "hi jarvis"
- State machine: STANDBY → LISTENING → PROCESSING → SPEAKING → STANDBY
- Completely optional — JARVIS operates seamlessly without a microphone

Educational Notes:
    Wake word detection in production systems typically uses a dedicated
    lightweight neural keyword spotter (e.g. Porcupine, Snowboy) that
    runs continuously at very low CPU cost.

    Our implementation takes a simpler educational approach:
    1. Record 2-second audio chunks from the microphone
    2. Measure RMS energy to detect voice activity
    3. If energy exceeds threshold → transcribe with Whisper tiny
    4. Check transcription for wake phrases

    This is heavier than a dedicated keyword spotter but requires no
    extra models and re-uses our existing STT engine.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from core.logger import log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAKE_PHRASES: List[str] = [
    "hey jarvis",
    "jarvis",
    "ok jarvis",
    "okay jarvis",
    "hi jarvis",
]


# ---------------------------------------------------------------------------
# Voice state machine
# ---------------------------------------------------------------------------

class VoiceState(str, Enum):
    """States of the voice interaction lifecycle."""
    STANDBY = "standby"          # Passively waiting for wake word
    LISTENING = "listening"      # Actively recording user speech
    PROCESSING = "processing"    # Brain / reasoning engine thinking
    SPEAKING = "speaking"        # TTS synthesising / playing response
    INTERRUPTED = "interrupted"  # User spoke during TTS playback
    DISABLED = "disabled"        # Voice subsystem not activated


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class WakeWordEvent:
    """Event emitted when the wake word is detected."""
    phrase_detected: str
    timestamp: float
    audio_energy: float = 0.0


# ---------------------------------------------------------------------------
# Wake Word Detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    WAKE_PHRASES = WAKE_PHRASES
    """Detect "Hey Jarvis" in text *or* from live microphone audio.

    Two operating modes
    -------------------
    **Text mode** (always available):
        Call ``check_wake_phrase(text)`` on any string — if it
        contains a wake phrase, a ``WakeWordEvent`` is returned.

    **Audio mode** (requires microphone + sounddevice):
        Call ``start_continuous_listening(stt_engine)`` to spawn a
        daemon thread that records 2-second chunks, checks voice
        activity, transcribes via the local STT engine, and fires
        the ``on_wake`` callback when a wake phrase is heard.

    Parameters
    ----------
    energy_threshold : float
        Minimum RMS energy to consider a chunk as containing speech.
    listen_duration_sec : float
        Length of each recording chunk in seconds.
    sample_rate : int
        Audio sample rate (Whisper expects 16 000 Hz).
    on_wake : callable, optional
        Callback fired when a wake phrase is detected.
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        listen_duration_sec: float = 3.0,
        sample_rate: int = 16000,
        on_wake: Optional[Callable[[WakeWordEvent], None]] = None,
    ):
        self.state = VoiceState.STANDBY
        self.energy_threshold = energy_threshold
        self.listen_duration_sec = listen_duration_sec
        self.sample_rate = sample_rate
        self._on_wake_callback = on_wake
        self._listening_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._mic_available: bool = False
        self._interruption_flag: bool = False
        self._check_microphone()

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------

    def _check_microphone(self) -> None:
        """Probe for input audio devices."""
        try:
            import sounddevice as sd  # type: ignore

            devs = sd.query_devices()
            if isinstance(devs, dict):
                devs = [devs]
            input_devs = [d for d in devs if d.get("max_input_channels", 0) > 0]
            self._mic_available = len(input_devs) > 0
            if self._mic_available:
                log.info(f"Microphone detected: {len(input_devs)} input device(s).")
            else:
                log.info("No microphone found. Voice input uses text fallback.")
        except ImportError:
            log.info("sounddevice not installed. Voice input uses text fallback.")
            self._mic_available = False
        except Exception as e:
            log.info(f"Microphone check error: {e}. Using text fallback.")
            self._mic_available = False

    @property
    def mic_available(self) -> bool:
        """Whether an input microphone was detected."""
        return self._mic_available

    # ------------------------------------------------------------------
    # Text-mode wake word check
    # ------------------------------------------------------------------

    def check_wake_phrase(self, text: str) -> Optional[WakeWordEvent]:
        """Return a ``WakeWordEvent`` if *text* contains a wake phrase."""
        cleaned = text.strip().lower()
        for phrase in WAKE_PHRASES:
            if phrase in cleaned:
                log.info(f"Wake word detected: '{phrase}' in '{cleaned[:50]}'")
                self.state = VoiceState.LISTENING
                event = WakeWordEvent(phrase_detected=phrase, timestamp=time.time())
                if self._on_wake_callback:
                    self._on_wake_callback(event)
                return event
        return None

    @staticmethod
    def extract_command(text: str) -> str:
        """Strip the leading wake phrase from *text* to get the user command."""
        cleaned = text.strip()
        lower = cleaned.lower()
        for phrase in sorted(WAKE_PHRASES, key=len, reverse=True):
            if lower.startswith(phrase):
                return cleaned[len(phrase):].lstrip(" ,.?!:").strip()
        return cleaned

    # ------------------------------------------------------------------
    # Audio recording
    # ------------------------------------------------------------------

    def record_audio_chunk(
        self, duration_sec: Optional[float] = None
    ) -> Optional["numpy.ndarray"]:
        """Record a mono audio chunk from the default microphone.

        Returns a 1-D float32 numpy array, or *None* if recording fails.
        """
        dur = duration_sec or self.listen_duration_sec
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np

            log.debug(f"Recording {dur}s at {self.sample_rate} Hz …")
            audio = sd.rec(
                int(dur * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            return audio.flatten()
        except Exception as e:
            log.warning(f"Audio recording failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Voice Activity Detection (simple RMS energy gate)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_voice_activity(audio_array, threshold: float = 0.02) -> bool:
        """Return *True* if the RMS energy of *audio_array* exceeds *threshold*."""
        try:
            import numpy as np
            energy = float(np.sqrt(np.mean(audio_array ** 2)))
            return energy > threshold
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Continuous background listening
    # ------------------------------------------------------------------

    def start_continuous_listening(self, stt_engine=None) -> None:
        """Spawn a daemon thread that listens for the wake word via mic.

        Parameters
        ----------
        stt_engine : LocalSTTEngine, optional
            The STT engine used to transcribe audio chunks.
        """
        if not self._mic_available:
            log.warning("Cannot start mic listener: no microphone available.")
            return
        if self._listening_thread and self._listening_thread.is_alive():
            log.info("Continuous listening is already active.")
            return

        self._stop_event.clear()
        self.state = VoiceState.STANDBY

        def _loop() -> None:
            log.info("Wake word listener started — say 'Hey Jarvis' …")
            while not self._stop_event.is_set():
                try:
                    audio = self.record_audio_chunk(duration_sec=2.0)
                    if audio is None:
                        time.sleep(0.5)
                        continue
                    if not self.detect_voice_activity(audio, self.energy_threshold):
                        continue
                    if stt_engine and stt_engine.is_available:
                        result = stt_engine.transcribe_audio_array(audio, self.sample_rate)
                        if result.success and result.text:
                            self.check_wake_phrase(result.text)
                except Exception as e:
                    log.debug(f"Listen loop error: {e}")
                    time.sleep(0.5)
            log.info("Wake word listener stopped.")

        self._listening_thread = threading.Thread(
            target=_loop, daemon=True, name="jarvis-wake-listener"
        )
        self._listening_thread.start()

    def stop_continuous_listening(self) -> None:
        """Stop the background wake-word listener."""
        self._stop_event.set()
        self.state = VoiceState.DISABLED
        if self._listening_thread:
            self._listening_thread.join(timeout=5.0)
            self._listening_thread = None
        log.info("Wake word listener stopped.")

    # ------------------------------------------------------------------
    # Interruption handling
    # ------------------------------------------------------------------

    def trigger_interruption(self) -> None:
        """Signal that the user spoke while JARVIS was speaking."""
        if self.state == VoiceState.SPEAKING:
            log.info("Voice interruption triggered.")
            self.state = VoiceState.INTERRUPTED
            self._interruption_flag = True

    def reset_state(self) -> None:
        """Return to passive standby."""
        self.state = VoiceState.STANDBY
        self._interruption_flag = False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        return {
            "state": self.state.value,
            "mic_available": self._mic_available,
            "energy_threshold": self.energy_threshold,
            "listen_duration_sec": self.listen_duration_sec,
            "continuous_listening": (
                self._listening_thread is not None
                and self._listening_thread.is_alive()
            ),
            "wake_phrases": list(WAKE_PHRASES),
        }


__all__ = ["VoiceState", "WakeWordDetector", "WakeWordEvent", "WAKE_PHRASES"]
