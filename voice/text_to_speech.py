"""J.A.R.V.I.S. — Local Text-to-Speech Engine (Phase 10).

Offline speech synthesis using pyttsx3 with Windows SAPI5.

Design Principles:
- Zero internet requirement
- Zero cloud API dependencies
- Uses native Windows SAPI5 speech synthesis voices
- Configurable voice selection, rate, and volume
- Thread-safe synchronous and asynchronous speech
- Graceful fallback to text output when no audio device is present

Educational Notes:
    pyttsx3 wraps platform-native TTS engines:
    - Windows: Microsoft SAPI5 (Speech API v5)
    - macOS: NSSpeechSynthesizer
    - Linux: espeak / espeak-ng

    On your Windows system, SAPI5 provides voices like
    "Microsoft David" (male) and "Microsoft Zira" (female)
    without any internet connection or API keys.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.logger import log


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VoiceInfo:
    """Metadata for an available TTS voice on the system."""
    id: str
    name: str
    language: str
    gender: str


@dataclass
class TTSResult:
    """Outcome of a text-to-speech synthesis operation."""
    text: str
    spoken: bool
    voice_used: str
    rate: int
    volume: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Local TTS Engine
# ---------------------------------------------------------------------------

class LocalTTSEngine:
    """Offline text-to-speech engine powered by pyttsx3 / Windows SAPI5.

    Features
    --------
    - 100 % local execution — no network calls, no cloud APIs
    - Discovers and lists all installed Windows SAPI5 voices
    - Configurable speech rate (words-per-minute) and volume
    - Thread-safe ``speak()`` with a reentrant lock
    - Non-blocking ``speak_async()`` for UI threads
    - Graceful degradation to text output when pyttsx3 or audio
      hardware is unavailable
    """

    def __init__(
        self,
        rate: int = 175,
        volume: float = 0.9,
        voice_index: int = 0,
    ):
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index
        self._engine = None
        self._available: bool = False
        self._voices: List[VoiceInfo] = []
        self._lock = threading.Lock()
        self._initialize()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Create the pyttsx3 engine and discover voices."""
        try:
            import pyttsx3

            self._engine = pyttsx3.init(driverName="sapi5")
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)

            # Discover installed voices
            raw_voices = self._engine.getProperty("voices") or []
            for v in raw_voices:
                lang_list = getattr(v, "languages", ["unknown"])
                lang_str = lang_list[0] if lang_list else "unknown"
                if isinstance(lang_str, bytes):
                    lang_str = lang_str.decode("utf-8", errors="replace")
                gender = (
                    "female"
                    if any(kw in v.name.lower() for kw in ("zira", "female", "woman", "hazel"))
                    else "male"
                )
                self._voices.append(
                    VoiceInfo(id=v.id, name=v.name, language=str(lang_str), gender=gender)
                )

            # Apply selected voice
            if raw_voices and self._voice_index < len(raw_voices):
                self._engine.setProperty("voice", raw_voices[self._voice_index].id)

            self._available = True
            log.info(
                f"Local TTS initialised: {len(self._voices)} voice(s), "
                f"rate={self._rate}, volume={self._volume}"
            )

        except Exception as e:
            log.warning(f"Local TTS unavailable: {e}. Speech falls back to text.")
            self._available = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether the TTS engine is ready for speech synthesis."""
        return self._available

    @property
    def available_voices(self) -> List[VoiceInfo]:
        """Return a copy of the list of installed TTS voices."""
        return list(self._voices)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_rate(self, rate: int) -> None:
        """Set speech rate in words-per-minute (clamped 50–400)."""
        self._rate = max(50, min(400, rate))
        if self._engine:
            self._engine.setProperty("rate", self._rate)

    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 silent → 1.0 maximum)."""
        self._volume = max(0.0, min(1.0, volume))
        if self._engine:
            self._engine.setProperty("volume", self._volume)

    def set_voice(self, voice_index: int) -> bool:
        """Select a voice by its index in ``available_voices``."""
        if not self._engine or voice_index < 0 or voice_index >= len(self._voices):
            return False
        self._voice_index = voice_index
        self._engine.setProperty("voice", self._voices[voice_index].id)
        return True

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def speak(self, text: str) -> TTSResult:
        """Speak *text* aloud using the local engine (blocking, thread-safe).

        Falls back to a text-only result if the engine is unavailable.
        """
        if not text or not text.strip():
            return TTSResult(
                text="", spoken=False, voice_used="none",
                rate=self._rate, volume=self._volume, error="Empty text",
            )

        clean = text.strip()
        voice_name = self._voices[self._voice_index].name if self._voices else "unknown"

        if not self._available or not self._engine:
            log.info(f"TTS fallback (text): {clean[:60]}…")
            return TTSResult(
                text=clean, spoken=False, voice_used="text_fallback",
                rate=self._rate, volume=self._volume,
                error="TTS engine not available — text fallback",
            )

        try:
            def _say_worker():
                try:
                    self._engine.say(clean)
                    self._engine.runAndWait()
                except Exception as ex:
                    log.warning(f"TTS worker error: {ex}")

            with self._lock:
                t = threading.Thread(target=_say_worker, daemon=True)
                t.start()
                t.join(timeout=3.0)

            log.debug(f"Spoke: '{clean[:40]}…' via {voice_name}")
            return TTSResult(
                text=clean, spoken=True, voice_used=voice_name,
                rate=self._rate, volume=self._volume,
            )
        except Exception as e:
            log.warning(f"TTS synthesis error: {e}")
            return TTSResult(
                text=clean, spoken=False, voice_used=voice_name,
                rate=self._rate, volume=self._volume, error=str(e),
            )

    def speak_async(self, text: str) -> None:
        """Speak *text* in a daemon background thread (non-blocking)."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    def stop(self) -> None:
        """Immediately stop any ongoing speech playback."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Status / telemetry
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return engine status for the JARVIS system status panel."""
        return {
            "available": self._available,
            "engine": "pyttsx3_sapi5",
            "voices_count": len(self._voices),
            "active_voice": self._voices[self._voice_index].name if self._voices else None,
            "rate": self._rate,
            "volume": self._volume,
            "offline": True,
            "requires_internet": False,
        }


__all__ = ["LocalTTSEngine", "TTSResult", "VoiceInfo"]
