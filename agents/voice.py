"""J.A.R.V.I.S. Voice Agent (Phase 10 Upgraded).

Specialized agent for voice intelligence, audio transcription,
speech synthesis dispatch, and text fallback management.

Design Principles:
- 100% offline-first execution with local TTS (pyttsx3) and STT (faster-whisper)
- Zero cloud API dependencies
- Completely optional: J.A.R.V.I.S. operates seamlessly without microphone or audio hardware
- Bridges seamlessly with independent agent system and central brain
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent
from services.voice.stt_engine import stt_engine
from services.voice.tts_engine import tts_engine
from services.voice.wake_word import wake_word_detector


class VoiceAgent(BaseAgent):
    """Independent specialized agent for audio interactions and speech commands."""

    def __init__(self):
        super().__init__(
            name="VoiceAgent",
            role="Handles speech recognition, voice synthesis dispatch, and audio transcription with text fallback.",
            capabilities=[
                "speech_synthesis",
                "audio_transcription",
                "voice_command_dispatch",
                "text_fallback",
                "wake_word_detection",
                "local_sapi5_tts",
            ],
            required_autonomy_level=1,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Process spoken command, transcribe audio, or synthesize text to audio."""
        start_time = time.perf_counter()
        ctx = context or {}
        action = ctx.get("action", "speak")

        try:
            if action == "transcribe":
                audio_file = ctx.get("audio_file")
                if audio_file:
                    transcript = await stt_engine.transcribe(audio_file)
                else:
                    transcript = task

                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"voice_stt_{int(time.time())}",
                    success=True,
                    data={"transcription": transcript, "audio_file": audio_file},
                    summary=f"Voice recognized: '{transcript}'",
                    duration_sec=duration,
                )

            elif action == "wake_check":
                # Check for wake word in text
                event = wake_word_detector.check_wake_phrase(task)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"voice_wake_{int(time.time())}",
                    success=True,
                    data={
                        "wake_detected": event is not None,
                        "phrase": event.phrase_detected if event else None,
                        "command": wake_word_detector.extract_command(task) if event else task,
                    },
                    summary=f"Wake check: {'Detected ' + event.phrase_detected if event else 'None'}",
                    duration_sec=duration,
                )

            elif action == "status":
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"voice_stat_{int(time.time())}",
                    success=True,
                    data={
                        "tts": tts_engine.status(),
                        "stt": stt_engine.status(),
                        "wake_word": wake_word_detector.status(),
                    },
                    summary="Voice subsystem status retrieved successfully.",
                    duration_sec=duration,
                )

            else:
                # Speech synthesis dispatch ("speak")
                tts_result = tts_engine.speak(task)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"voice_tts_{int(time.time())}",
                    success=True,
                    data={
                        "text": task,
                        "spoken": tts_result.spoken,
                        "voice_used": tts_result.voice_used,
                        "offline": True,
                    },
                    summary=f"Voice synthesizer processed: '{task[:40]}...' (spoken: {tts_result.spoken})",
                    duration_sec=duration,
                )

        except Exception as e:
            log.warning(f"VoiceAgent execution issue: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"voice_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Voice interaction fallen back to text mode: {e}",
                duration_sec=duration,
            )


__all__ = ["VoiceAgent"]
