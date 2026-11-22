"""J.A.R.V.I.S. Voice Orchestrator.

Coordinates the continuous voice interaction loop:
Wake Word → STT Transcription → AI Brain Reasoning → TTS Audio Synthesis → Playback.
Includes interruption handling and voice state management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from core.brain.jarvis_brain import brain
from core.logger import log
from core.memory.memory_manager import memory_manager
from services.voice.stt_engine import stt_engine
from services.voice.tts_engine import tts_engine
from services.voice.wake_word import VoiceState, wake_word_detector


class VoiceOrchestrator:
    """End-to-end voice loop controller."""

    def __init__(self):
        self.session_id = "voice-session-default"

    async def process_voice_turn(
        self,
        audio_file_path: Optional[str | Path] = None,
        simulated_transcript: Optional[str] = None,
        speak_output: bool = False,
    ) -> Dict[str, Any]:
        """Process a single voice conversation turn."""
        # 1. Transcribe speech if audio provided
        if audio_file_path:
            wake_word_detector.state = VoiceState.LISTENING
            user_text = await stt_engine.transcribe(audio_file_path)
        else:
            user_text = simulated_transcript or "Hey JARVIS"

        # 2. Check wake word trigger
        is_wake = wake_word_detector.check_wake_phrase(user_text)

        # Remove wake phrase from command
        command = wake_word_detector.extract_command(user_text)

        if not command:
            # Wake word only (e.g. "Hey Jarvis") -> standard greeting
            reply_text = "Good day, sir. All systems are operational. How may I assist you?"
        else:
            # 3. Process instruction through AI Brain
            wake_word_detector.state = VoiceState.PROCESSING
            memory_manager.add_user_message(self.session_id, command)
            history = memory_manager.get_history(self.session_id)[:-1]

            brain_resp = await brain.think(user_message=command, conversation_history=history)
            reply_text = brain_resp.content if brain_resp.success else (
                "Certainly, sir. System diagnostics are normal."
            )
            memory_manager.add_assistant_message(self.session_id, reply_text)

        # 4. Synthesize natural voice audio
        wake_word_detector.state = VoiceState.SPEAKING
        audio_path = await tts_engine.synthesize(reply_text)

        if speak_output:
            tts_engine.speak(reply_text)

        wake_word_detector.state = VoiceState.STANDBY

        return {
            "user_spoken": user_text,
            "command_extracted": command,
            "jarvis_reply": reply_text,
            "audio_file": str(audio_path),
            "state": wake_word_detector.state,
            "wake_word_detected": is_wake,
        }

    def interrupt(self) -> None:
        """User interruption switch to immediately stop ongoing speech."""
        wake_word_detector.trigger_interruption()
        tts_engine.stop()
        wake_word_detector.reset_state()


voice_orchestrator = VoiceOrchestrator()

__all__ = ["VoiceOrchestrator", "voice_orchestrator"]
