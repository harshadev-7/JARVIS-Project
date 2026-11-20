"""
JARVIS AI — Phase 5 Test Suite
================================
Tests for Advanced Voice Assistant:
- Wake-word detection ("Hey Jarvis") & State Transitions
- Voice interruption handling
- High-definition neural TTS synthesis
- Voice turn orchestration
- FastAPI voice endpoints
"""

import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from fastapi.testclient import TestClient

from services.voice.wake_word import wake_word_detector, VoiceState
from services.voice.tts_engine import tts_engine
from services.voice.voice_orchestrator import voice_orchestrator
from api.app import app


# ── 1. Wake-Word Tests ──────────────────────────────────────────────────────

def test_wake_word_recognition():
    """Detector should catch variations of 'Hey Jarvis'."""
    assert wake_word_detector.check_wake_phrase("Hey Jarvis, what time is it?") is True
    assert wake_word_detector.state == VoiceState.LISTENING

    wake_word_detector.reset_state()
    assert wake_word_detector.check_wake_phrase("Good morning Siri") is False
    assert wake_word_detector.state == VoiceState.STANDBY


def test_voice_interruption():
    """Interruption flag should activate if speaking when user interrupts."""
    wake_word_detector.state = VoiceState.SPEAKING
    wake_word_detector.trigger_interruption()
    assert wake_word_detector.state == VoiceState.INTERRUPTED
    assert wake_word_detector._interruption_flag is True
    wake_word_detector.reset_state()


# ── 2. TTS Neural Engine Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tts_synthesis(tmp_path):
    """TTS engine should synthesize text to an audio file."""
    output_mp3 = tmp_path / "test_jarvis.mp3"

    with patch("edge_tts.Communicate.save", new_callable=AsyncMock) as mock_save:
        mock_save.return_value = None
        path = await tts_engine.synthesize("At your service, sir.", output_file=str(output_mp3))
        assert str(path) == str(output_mp3)
        mock_save.assert_called_once()


# ── 3. Voice Orchestrator Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_orchestrator_turn():
    """Orchestrator should process spoken turn, generate reply, and return audio path."""
    with patch("services.voice.tts_engine.TTSEngine.synthesize", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = Path("dummy_speech.mp3")

        result = await voice_orchestrator.process_voice_turn(
            simulated_transcript="Hey Jarvis, what is my mission?"
        )

        assert "user_spoken" in result
        assert "jarvis_reply" in result
        assert result["audio_file"] == "dummy_speech.mp3"


# ── 4. FastAPI Voice Endpoints Tests ────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_voice_status(client):
    """GET /voice/status should return current voice state."""
    res = client.get("/voice/status")
    assert res.status_code == 200
    data = res.json()
    assert data["wake_word"] == "Hey Jarvis"
    assert "state" in data


def test_api_voice_process_turn(client):
    """POST /voice/process should accept speech transcript and respond."""
    with patch("services.voice.tts_engine.TTSEngine.synthesize", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = Path("dummy_speech.mp3")
        res = client.post("/voice/process", json={"transcript": "Hey Jarvis"})
        assert res.status_code == 200
        data = res.json()
        assert "Good day, sir" in data["jarvis_reply"] or "assist" in data["jarvis_reply"]


def test_api_voice_interrupt(client):
    """POST /voice/interrupt should signal speech pause."""
    res = client.post("/voice/interrupt")
    assert res.status_code == 200
    assert res.json()["message"] == "Voice playback interrupted"
