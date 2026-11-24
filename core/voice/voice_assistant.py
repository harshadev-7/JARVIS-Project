"""JARVIS Phase 30 — Full-Duplex Conversational Voice Assistant
=============================================================
Coordinates ambient wake-word detection, local speech-to-text transcription,
deterministic brain reasoning, and high-fidelity neural voice synthesis.

Features:
- State Machine: STANDBY -> LISTENING -> THINKING -> SPEAKING -> STANDBY
- Interrupt / Barge-In capability: instantly halts audio if user begins speaking
- Asynchronous event bus notifications across all voice lifecycle transitions
- 100% offline-first execution running purely on local CPU
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.integration.event_bus import SystemEvent, SystemEventType, event_bus
from core.logger import log

logger = logging.getLogger(__name__)


class VoiceAssistantState(str, Enum):
    """Conversational lifecycle states."""
    STANDBY = "standby"          # Passively waiting for wake word or push-to-talk
    LISTENING = "listening"      # Recording user voice stream
    THINKING = "thinking"        # Processing through JARVIS Brain reasoning
    SPEAKING = "speaking"        # Synthesizing and outputting voice audio
    STOPPED = "stopped"          # Engine inactive


@dataclass
class VoiceTurnResult:
    """Outcome of an end-to-end voice interaction turn."""
    turn_id: str
    user_transcript: str
    assistant_response: str
    agent_used: str
    audio_spoken: bool
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_transcript": self.user_transcript,
            "assistant_response": self.assistant_response,
            "agent_used": self.agent_used,
            "audio_spoken": self.audio_spoken,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": round(self.timestamp, 3),
        }


class JarvisVoiceAssistant:
    """Full-duplex ambient voice assistant engine."""

    def __init__(
        self,
        wake_words: Optional[List[str]] = None,
        session_id: str = "voice_ambient_session",
        enable_tts: bool = True,
    ):
        self.wake_words = wake_words or ["hey jarvis", "jarvis", "ok jarvis", "hi jarvis"]
        self.session_id = session_id
        self.enable_tts = enable_tts

        self.state = VoiceAssistantState.STANDBY
        self._running = False
        self._current_task: Optional[asyncio.Task] = None
        self._turn_history: List[VoiceTurnResult] = []
        self._interrupt_flag = threading.Event()
        self._stt_engine = None
        self._tts_engine = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _get_stt(self):
        if self._stt_engine is None:
            try:
                from voice.speech_to_text import LocalSTTEngine
                self._stt_engine = LocalSTTEngine(model_size="tiny")
            except Exception as exc:
                logger.warning("Could not initialize local STT engine: %s", exc)
        return self._stt_engine

    def _get_tts(self):
        if self._tts_engine is None:
            try:
                from voice.text_to_speech import LocalTTSEngine
                self._tts_engine = LocalTTSEngine()
            except Exception as exc:
                logger.warning("Could not initialize local TTS engine: %s", exc)
        return self._tts_engine

    async def start(self) -> Dict[str, Any]:
        """Activates continuous voice assistant loop."""
        if self._running:
            return {"status": "already_running", "state": self.state.value}

        self._running = True
        self.state = VoiceAssistantState.STANDBY
        self._interrupt_flag.clear()
        logger.info("JarvisVoiceAssistant initialized in ambient standby mode.")
        return {"status": "online", "state": self.state.value}

    async def stop(self) -> Dict[str, Any]:
        """Halts voice assistant and silences any current speech."""
        self.interrupt()
        self._running = False
        self.state = VoiceAssistantState.STOPPED
        logger.info("JarvisVoiceAssistant stopped.")
        return {"status": "stopped", "state": self.state.value}

    def interrupt(self) -> None:
        """Barge-In: immediately interrupt and cancel active audio speech."""
        self._interrupt_flag.set()
        try:
            tts = self._get_tts()
            if tts and hasattr(tts, "stop"):
                tts.stop()
        except Exception:
            pass
        if self.state == VoiceAssistantState.SPEAKING:
            self.state = VoiceAssistantState.STANDBY
            logger.info("Voice Assistant barged in / interrupted by user.")

    async def process_utterance(self, text_or_transcript: str) -> VoiceTurnResult:
        """Processes a speech transcript through the full brain reasoning & voice pipeline."""
        t0 = time.perf_counter()
        turn_id = f"vturn-{int(time.time() * 1000)}"
        user_query = text_or_transcript.strip()

        # 1. Strip wake-word prefix if present
        clean_query = user_query
        for ww in self.wake_words:
            if clean_query.lower().startswith(ww):
                clean_query = clean_query[len(ww):].strip(",: ")
                break

        if not clean_query:
            clean_query = "Hello JARVIS"

        # Broadcast wake & transcription events
        await event_bus.publish(
            SystemEvent(
                event_type=SystemEventType.VOICE_WAKE_DETECTED,
                source_subsystem="voice_assistant",
                payload={"turn_id": turn_id, "raw_query": user_query},
            )
        )
        await event_bus.publish(
            SystemEvent(
                event_type=SystemEventType.VOICE_QUERY_TRANSCRIBED,
                source_subsystem="voice_assistant",
                payload={"turn_id": turn_id, "clean_query": clean_query},
            )
        )

        # 2. Brain Reasoning
        self.state = VoiceAssistantState.THINKING
        from core.brain.jarvis_brain import brain
        resp = await brain.think(user_message=clean_query, session_id=self.session_id)
        reply_text = resp.content or "All systems nominal, sir."
        agent = resp.agent_used or "JARVISBrain"

        # 3. Speech Synthesis
        self.state = VoiceAssistantState.SPEAKING
        spoken = False
        if self.enable_tts and not self._interrupt_flag.is_set():
            # Clean markdown formatting for clean vocal delivery
            vocal_text = self._sanitize_for_speech(reply_text)
            try:
                tts = self._get_tts()
                if tts:
                    spoken = tts.speak(vocal_text, wait=False)
            except Exception as exc:
                logger.warning("TTS speech execution error: %s", exc)

        # Broadcast spoken event
        await event_bus.publish(
            SystemEvent(
                event_type=SystemEventType.VOICE_REPLY_SPOKEN,
                source_subsystem="voice_assistant",
                payload={"turn_id": turn_id, "reply": reply_text, "spoken": spoken},
            )
        )

        duration = (time.perf_counter() - t0) * 1000.0
        self.state = VoiceAssistantState.STANDBY

        result = VoiceTurnResult(
            turn_id=turn_id,
            user_transcript=clean_query,
            assistant_response=reply_text,
            agent_used=agent,
            audio_spoken=spoken,
            duration_ms=duration,
        )
        self._turn_history.append(result)
        if len(self._turn_history) > 30:
            self._turn_history.pop(0)

        return result

    def _sanitize_for_speech(self, text: str) -> str:
        """Removes markdown symbols, source tags, and code blocks for spoken audio."""
        import re
        clean = re.sub(r"```[\s\S]*?```", "Code block omitted.", text)
        clean = re.sub(r"\*\[SOURCE:.*?\]\*", "", clean)
        clean = re.sub(r"[\#\*\_\`\~\>]", "", clean)
        clean = re.sub(r"\n+", " ", clean).strip()
        return clean[:350]

    def get_status(self) -> Dict[str, Any]:
        """Provides status report of the voice assistant."""
        return {
            "state": self.state.value,
            "is_running": self._running,
            "enable_tts": self.enable_tts,
            "wake_words": self.wake_words,
            "turns_completed": len(self._turn_history),
            "recent_turns": [t.to_dict() for t in self._turn_history[-3:]],
        }


voice_assistant = JarvisVoiceAssistant()
