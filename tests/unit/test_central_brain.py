"""Unit Tests for J.A.R.V.I.S. Central Brain & Intent Router (Phase 7).

Validates:
1. Local brain initialization without API keys or cloud dependencies
2. Intent classification and routing (Greeting, Identity, Diagnostics, Math, General)
3. Specialized agent delegation
4. Custom local Transformer inference on conversational queries
5. Multi-turn dialogue context management
6. Telemetry and system status inspection
"""

import pytest

from core.brain.context import ConversationContext
from core.brain.jarvis_brain import JARVISBrain, brain
from core.brain.router import IntentRouter, IntentType


@pytest.mark.asyncio
async def test_brain_initialization():
    test_brain = JARVISBrain()
    assert test_brain.is_initialized is False

    await test_brain.initialize()
    assert test_brain.is_initialized is True
    assert test_brain.local_inference is not None
    assert test_brain.local_inference.model is not None


@pytest.mark.asyncio
async def test_brain_greeting_intent():
    resp = await brain.think("hello jarvis", session_id="test_greeting")
    assert resp.success is True
    assert "operational" in resp.content.lower()
    assert resp.provider == "custom_local_ai"
    assert resp.duration_ms is not None and resp.duration_ms > 0


@pytest.mark.asyncio
async def test_brain_identity_intent():
    resp = await brain.think("who are you", session_id="test_identity")
    assert resp.success is True
    assert "JARVIS" in resp.content
    assert "custom" in resp.content.lower()
    assert "transformer" in resp.content.lower()


@pytest.mark.asyncio
async def test_brain_diagnostics_routing():
    resp = await brain.think("check system diagnostics", session_id="test_diag")
    assert resp.success is True
    assert resp.agent_used == "SystemMonitor"
    assert "CPU" in resp.content
    assert "Memory" in resp.content


@pytest.mark.asyncio
async def test_brain_math_routing():
    resp = await brain.think("calculate 15 * 10", session_id="test_math")
    assert resp.success is True
    assert resp.agent_used == "MathematicalAgent"
    assert "150" in resp.content


@pytest.mark.asyncio
async def test_brain_general_conversation_custom_transformer():
    resp = await brain.think("what is your operating mode?", session_id="test_general")
    assert resp.success is True
    assert resp.agent_used == "CustomTransformer"
    assert "Custom Transformer" in resp.content


@pytest.mark.asyncio
async def test_brain_context_tracking():
    session_id = "test_multi_turn_session"
    await brain.think("hello jarvis", session_id=session_id)
    await brain.think("check system diagnostics", session_id=session_id)

    ctx = brain.get_context(session_id)
    assert len(ctx.turns) >= 4  # 2 user turns + 2 assistant turns
    history = ctx.get_history()
    assert len(history) >= 4
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_brain_system_status():
    status = brain.get_system_status()
    assert status["provider"] == "custom_local_ai"
    assert status["offline_only"] is True
    assert status["requires_cloud_api"] is False
    assert status["requires_ollama"] is False
    assert status["active_model"] == "jarvis_micro_v1"
