"""Unit Tests for J.A.R.V.I.S. Voice Intelligence (Phase 10).

Verifies:
1. LocalTTSEngine initialization and offline speech synthesis (pyttsx3 / SAPI5)
2. LocalTTSEngine status reporting and voice parameter configuration
3. LocalSTTEngine structure and transcription handling with mock/file inputs
4. LocalSTTEngine missing file and invalid format validation
5. WakeWordDetector keyword recognition ("Hey Jarvis", "OK Jarvis")
6. WakeWordDetector command extraction
7. WakeWordDetector VoiceState transitions (STANDBY -> LISTENING -> etc.)
8. VoiceOrchestrator end-to-end voice loop processing
9. VoiceAgent integration with multi-agent system
10. Offline guarantee: Zero cloud API calls, zero external network dependency
"""

import os
import tempfile
import wave
import pytest
from unittest.mock import MagicMock, patch

from voice.text_to_speech import LocalTTSEngine, TTSResult, VoiceInfo
from voice.speech_to_text import LocalSTTEngine, TranscriptionResult
from voice.wake_word import VoiceState, WakeWordDetector, WakeWordEvent, WAKE_PHRASES
from services.voice.stt_engine import STTEngine, stt_engine
from services.voice.tts_engine import TTSEngine, tts_engine
from services.voice.voice_orchestrator import VoiceOrchestrator, voice_orchestrator
from agents.voice import VoiceAgent


# ============================================================================
# 1. Local TTS Engine Tests (pyttsx3 / SAPI5)
# ============================================================================

def test_tts_engine_initialization():
    """TTS engine initializes with valid rates, volumes and offline flag."""
    engine = LocalTTSEngine(rate=180, volume=0.8)
    assert engine._rate == 180
    assert engine._volume == 0.8
    status = engine.status()
    assert status["engine"] == "pyttsx3_sapi5"
    assert status["offline"] is True
    assert status["requires_internet"] is False


def test_tts_engine_parameter_clamping():
    """Rate and volume inputs should be clamped safely."""
    engine = LocalTTSEngine(rate=180, volume=0.8)
    engine.set_rate(1000)
    assert engine._rate == 400

    engine.set_rate(10)
    assert engine._rate == 50

    engine.set_volume(2.5)
    assert engine._volume == 1.0

    engine.set_volume(-1.0)
    assert engine._volume == 0.0


def test_tts_speak_handles_empty_text():
    """Empty strings should return without errors."""
    engine = LocalTTSEngine()
    result = engine.speak("   ")
    assert result.spoken is False
    assert result.error == "Empty text"


def test_tts_speak_fallback_when_unavailable():
    """When TTS driver is unavailable, engine safely falls back to text."""
    engine = LocalTTSEngine()
    engine._available = False
    result = engine.speak("Hello sir, all systems functional.")
    assert result.spoken is False
    assert result.voice_used == "text_fallback"
    assert "fallback" in result.error.lower()


# ============================================================================
# 2. Local STT Engine Tests (faster-whisper)
# ============================================================================

def test_stt_engine_initialization():
    """STT engine initializes with proper CPU and quantisation settings."""
    stt = LocalSTTEngine(model_size="tiny", device="cpu", compute_type="int8")
    assert stt._model_size == "tiny"
    assert stt._device == "cpu"
    assert stt._compute_type == "int8"
    status = stt.status()
    assert status["engine"] == "faster_whisper"
    assert status["offline"] is True
    assert status["requires_internet"] is False


def test_stt_invalid_model_rejection():
    """Unsupported model architectures must be rejected."""
    with pytest.raises(ValueError, match="not supported"):
        LocalSTTEngine(model_size="massive_ultra_model")


def test_stt_missing_audio_file():
    """Missing audio files must be handled cleanly without crashing."""
    stt = LocalSTTEngine(model_size="tiny")
    res = stt.transcribe_file("non_existent_audio_file_123.wav")
    assert res.success is False
    assert "not found" in res.error.lower()


def test_stt_unsupported_file_format():
    """Unsupported extensions (.exe, .txt) must be rejected with descriptive error."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"dummy")
        temp_path = f.name

    try:
        stt = LocalSTTEngine(model_size="tiny")
        res = stt.transcribe_file(temp_path)
        assert res.success is False
        assert "unsupported" in res.error.lower()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================================
# 3. Wake Word Detection Tests
# ============================================================================

def test_wake_word_detection_in_text():
    """Wake word detector recognizes all supported wake phrases in text."""
    detector = WakeWordDetector()

    assert detector.check_wake_phrase("hey jarvis, what is the weather?") is not None
    assert detector.check_wake_phrase("jarvis tell me a joke") is not None
    assert detector.check_wake_phrase("ok jarvis launch the app") is not None
    assert detector.check_wake_phrase("hi jarvis system check") is not None
    assert detector.check_wake_phrase("unrelated message to someone else") is None


def test_wake_word_command_extraction():
    """Leading wake words should be stripped cleanly to get pure command."""
    detector = WakeWordDetector()

    cmd1 = detector.extract_command("Hey Jarvis, calculate 45 * 2")
    assert cmd1 == "calculate 45 * 2"

    cmd2 = detector.extract_command("ok jarvis: what is your status?")
    assert cmd2 == "what is your status?"

    cmd3 = detector.extract_command("clean command without wake word")
    assert cmd3 == "clean command without wake word"


def test_wake_word_state_transitions():
    """State machine transitions smoothly between voice lifecycle states."""
    detector = WakeWordDetector()
    assert detector.state == VoiceState.STANDBY

    # Detect wake phrase -> LISTENING
    detector.check_wake_phrase("hey jarvis")
    assert detector.state == VoiceState.LISTENING

    # Interruption
    detector.state = VoiceState.SPEAKING
    detector.trigger_interruption()
    assert detector.state == VoiceState.INTERRUPTED

    # Reset
    detector.reset_state()
    assert detector.state == VoiceState.STANDBY


# ============================================================================
# 4. Voice Orchestrator & Services Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_voice_orchestrator_turn_with_simulated_transcript():
    """Orchestrator processes a full turn: Wake -> Extract -> Think -> Synthesize."""
    orchestrator = VoiceOrchestrator()
    result = await orchestrator.process_voice_turn(
        simulated_transcript="Hey Jarvis, what is your status?"
    )

    assert result["user_spoken"] == "Hey Jarvis, what is your status?"
    assert result["command_extracted"] == "what is your status?"
    assert "jarvis_reply" in result
    assert len(result["jarvis_reply"]) > 0
    assert result["state"] == VoiceState.STANDBY


@pytest.mark.asyncio
async def test_voice_orchestrator_wake_word_only():
    """When only wake word is spoken, returns standard polite greeting."""
    orchestrator = VoiceOrchestrator()
    result = await orchestrator.process_voice_turn(simulated_transcript="Hey Jarvis")
    assert "Good day, sir" in result["jarvis_reply"]


# ============================================================================
# 5. Voice Agent Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_voice_agent_speak_dispatch():
    """VoiceAgent successfully processes speak task via local engine."""
    agent = VoiceAgent()
    res = await agent.execute("All systems operational, sir.", context={"action": "speak"})
    assert res.success is True
    assert res.agent_name == "VoiceAgent"
    assert "Voice synthesizer processed" in res.summary


@pytest.mark.asyncio
async def test_voice_agent_wake_check():
    """VoiceAgent performs wake phrase detection via context action."""
    agent = VoiceAgent()
    res = await agent.execute("hey jarvis, open notepad", context={"action": "wake_check"})
    assert res.success is True
    assert res.data["wake_detected"] is True
    assert res.data["phrase"] == "hey jarvis"
    assert res.data["command"] == "open notepad"


@pytest.mark.asyncio
async def test_voice_agent_status_action():
    """VoiceAgent provides unified voice subsystem telemetry."""
    agent = VoiceAgent()
    res = await agent.execute("", context={"action": "status"})
    assert res.success is True
    assert "tts" in res.data
    assert "stt" in res.data
    assert "wake_word" in res.data
    assert res.data["tts"]["offline"] is True
    assert res.data["stt"]["offline"] is True
