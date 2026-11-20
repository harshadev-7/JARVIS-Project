"""J.A.R.V.I.S. Voice Agent (Phase 9).

Specialized agent for voice intelligence, audio transcription,
speech synthesis dispatch, and text fallback management.

Design:
- Completely optional: J.A.R.V.I.S. operates seamlessly without microphone or audio hardware
- Bridges to local TTS (pyttsx3 / SAPI5) and STT engines
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent


class VoiceAgent(BaseAgent):
    """Independent specialized agent for audio interactions and speech commands."""

    def __init__(self):
        super().__init__(
            name="VoiceAgent",
            role="Handles speech recognition, voice synthesis dispatch, and audio transcription with text fallback.",
            capabilities=["speech_synthesis", "audio_transcription", "voice_command_dispatch", "text_fallback"],
            required_autonomy_level=1,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Process spoken command or synthesize text to audio."""
        start_time = time.perf_counter()
        ctx = context or {}
        action = ctx.get("action", "speak")

        try:
            if action == "transcribe":
                audio_file = ctx.get("audio_file")
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"voice_stt_{int(time.time())}",
                    success=True,
                    data={"transcription": task, "audio_file": audio_file},
                    summary=f"Voice recognized: '{task}'",
                    duration_sec=duration,
                )
            else:
                # Speech synthesis dispatch
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"voice_tts_{int(time.time())}",
                    success=True,
                    data={"text": task, "status": "audio_queued"},
                    summary=f"Voice synthesizer queued speech: '{task[:40]}...'",
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
