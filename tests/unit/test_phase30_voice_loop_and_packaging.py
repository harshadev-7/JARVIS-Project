"""JARVIS Phase 30 — Full-Duplex Voice Loop & Production Packaging Unit Tests
=============================================================================
Validates:
1. Subsystem Registry cataloging all 30 project subsystems.
2. Full-duplex voice assistant state machine, wake phrase stripping, and brain integration.
3. Barge-in interruption handling.
4. Voice lifecycle events published to SystemEventBus.
5. Brain router classification of /listen and /voice-mode.
6. Verification of 1-click production launchers (launch_jarvis.bat, launch_hud.ps1, Docker).
"""

from pathlib import Path
import pytest

from core.brain.router import IntentRouter, IntentType
from core.integration import (
    SubsystemRegistry,
    SystemEvent,
    SystemEventType,
    event_bus,
    subsystem_registry,
)
from core.voice import (
    JarvisVoiceAssistant,
    VoiceAssistantState,
    VoiceTurnResult,
    voice_assistant,
)


# 1. Subsystem Registry 30/30 Test
def test_subsystem_registry_all_30_subsystems():
    """Verify all 30 canonical subsystems are registered across all 30 project phases."""
    registry = SubsystemRegistry()
    subs = registry.list_subsystems()

    assert len(subs) >= 30, f"Expected 30 subsystems, found {len(subs)}"

    v_sub = registry.get_subsystem("voice_conversational_assistant")
    assert v_sub is not None
    assert v_sub.phase == 30
    assert v_sub.category == "operational"

    probe = registry.probe_all_health()
    assert probe["total_subsystems"] >= 30
    assert probe["healthy_count"] >= 30
    assert probe["overall_status"] == "healthy"
    assert probe["health_score_pct"] == 100.0


# 2. Voice Assistant Utterance & Brain Integration Test
@pytest.mark.asyncio
async def test_voice_assistant_utterance():
    """Verify end-to-end processing of a speech utterance through brain reasoning."""
    va = JarvisVoiceAssistant(enable_tts=False)
    assert va.state == VoiceAssistantState.STANDBY

    turn = await va.process_utterance("Hey Jarvis, what is 12 multiplied by 5?")
    assert isinstance(turn, VoiceTurnResult)
    assert "what is 12 multiplied by 5?" in turn.user_transcript
    assert "60" in turn.assistant_response
    assert turn.duration_ms > 0
    assert va.state == VoiceAssistantState.STANDBY

    st = va.get_status()
    assert st["turns_completed"] >= 1
    assert len(st["recent_turns"]) >= 1


# 3. Barge-In Interruption Test
def test_voice_assistant_barge_in():
    """Verify barge-in interruption halts audio playback and resets state."""
    va = JarvisVoiceAssistant(enable_tts=False)
    va.state = VoiceAssistantState.SPEAKING

    va.interrupt()
    assert va.state == VoiceAssistantState.STANDBY
    assert va._interrupt_flag.is_set()


# 4. Voice Event Bus Integration Test
@pytest.mark.asyncio
async def test_voice_events_dispatched_to_event_bus():
    """Verify SystemEventBus receives typed voice lifecycle notifications."""
    received = []

    async def voice_event_listener(event: SystemEvent):
        received.append(event)

    event_bus.subscribe(SystemEventType.VOICE_WAKE_DETECTED, voice_event_listener)
    event_bus.subscribe(SystemEventType.VOICE_QUERY_TRANSCRIBED, voice_event_listener)
    event_bus.subscribe(SystemEventType.VOICE_REPLY_SPOKEN, voice_event_listener)

    va = JarvisVoiceAssistant(enable_tts=False)
    await va.process_utterance("hey jarvis what is the capital of France")

    event_types = [e.event_type for e in received]
    assert SystemEventType.VOICE_WAKE_DETECTED in event_types
    assert SystemEventType.VOICE_QUERY_TRANSCRIBED in event_types
    assert SystemEventType.VOICE_REPLY_SPOKEN in event_types


# 5. Intent Router Voice Mode Test
def test_router_voice_mode_intent():
    """Verify IntentRouter routes /listen, /voice-mode, and spoken voice prompts."""
    router = IntentRouter()

    dec1 = router.route("/listen")
    assert dec1.intent == IntentType.VOICE_MODE
    assert dec1.agent_name == "VoiceAssistant"

    dec2 = router.route("/voice-mode")
    assert dec2.intent == IntentType.VOICE_MODE

    dec3 = router.route("/voice-loop")
    assert dec3.intent == IntentType.VOICE_MODE

    dec4 = router.route("start listening in ambient voice mode")
    assert dec4.intent == IntentType.VOICE_MODE


# 6. Production Packaging & Launcher Files Verification Test
def test_production_packaging_launchers():
    """Verify production batch, PowerShell, and Docker deployment artifacts."""
    root_dir = Path(__file__).resolve().parents[2]

    # 1. Windows batch launcher
    bat_file = root_dir / "launch_jarvis.bat"
    assert bat_file.exists(), "launch_jarvis.bat missing"
    bat_content = bat_file.read_text(encoding="utf-8")
    assert "J.A.R.V.I.S. Production Launcher" in bat_content
    assert "main.py" in bat_content

    # 2. PowerShell launcher
    ps_file = root_dir / "launch_hud.ps1"
    assert ps_file.exists(), "launch_hud.ps1 missing"
    ps_content = ps_file.read_text(encoding="utf-8")
    assert "Holographic Command Center Launcher" in ps_content

    # 3. Docker specification
    docker_file = root_dir / "docker" / "Dockerfile.api"
    assert docker_file.exists(), "Dockerfile.api missing"
    docker_content = docker_file.read_text(encoding="utf-8")
    assert "uvicorn" in docker_content
    assert "api.app:app" in docker_content
