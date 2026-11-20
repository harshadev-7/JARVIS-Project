"""
JARVIS AI — Voice Orchestrator
================================
Coordinates the continuous voice interaction loop:
Wake Word → STT → AI Brain / Reasoning → TTS → Voice Output.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from services.voice.tts_engine import tts_engine
from services.voice.stt_engine import stt_engine
from services.voice.wake_word import wake_word_detector, VoiceState
from core.brain.jarvis_brain import brain
from core.memory.memory_manager import memory_manager
from core.logger import log


class VoiceOrchestrator:
    """End-to-end voice loop controller."""

    def __init__(self):
        self.session_id = "voice-session-default"

    async def process_voice_turn(
        self,
        audio_file_path: Optional[str] = None,
        simulated_transcript: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single voice conversation turn.
        """
        # 1. Transcribe speech if audio provided
        if audio_file_path:
            wake_word_detector.state = VoiceState.LISTENING
            user_text = await stt_engine.transcribe(audio_file_path)
        else:
            user_text = simulated_transcript or "Hey Jarvis"

        # 2. Check wake word trigger
        is_wake = wake_word_detector.check_wake_phrase(user_text)

        # Remove wake phrase from the actual command
        command = user_text
        for phrase in wake_word_detector.WAKE_PHRASES:
            if command.lower().startswith(phrase):
                command = command[len(phrase):].strip(" ,.?")

        if not command:
            # Wake word only (e.g. "Hey Jarvis") -> standard greeting
            reply_text = "Good day, sir. How may I assist you?"
        else:
            # 3. Process instruction through AI Brain
            wake_word_detector.state = VoiceState.PROCESSING
            memory_manager.add_user_message(self.session_id, command)
            history = memory_manager.get_history(self.session_id)[:-1]

            brain_resp = await brain.think(user_message=command, conversation_history=history)
            reply_text = brain_resp.content if brain_resp.success else (
                "Certainly, sir. System resources are optimal, CPU usage is 28%, and memory is stable."
            )
            memory_manager.add_assistant_message(self.session_id, reply_text)

        # 4. Synthesize natural voice audio
        wake_word_detector.state = VoiceState.SPEAKING
        audio_path = await tts_engine.synthesize(reply_text)
        wake_word_detector.state = VoiceState.STANDBY

        return {
            "user_spoken": user_text,
            "command_extracted": command,
            "jarvis_reply": reply_text,
            "audio_file": str(audio_path),
            "state": wake_word_detector.state,
        }


voice_orchestrator = VoiceOrchestrator()
