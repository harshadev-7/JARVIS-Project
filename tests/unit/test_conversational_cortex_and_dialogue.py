"""
Test Suite: Conversational Cortex, Memory Routing & Natural Dialogue
====================================================================
Verifies that:
1. Memory queries like "tell about me" and "my profile" route to MemoryAgent.
2. Philosophical questions like "hi what is human in your perfective?" route through
   ConversationalCortex with profound insights rather than robotic server dumps.
3. Creator/origin and check-in dialogues return rich, polite, persona-consistent responses.
"""

import pytest
from core.brain.router import IntentRouter, IntentType
from core.brain.conversational_cortex import conversational_cortex
from core.brain.jarvis_brain import brain


@pytest.mark.asyncio
async def test_tell_about_me_routing():
    """Verify that 'tell about me' and profile queries route directly to MemoryAgent."""
    router = IntentRouter()

    dec1 = router.route("tell about me")
    assert dec1.intent == IntentType.MEMORY_QUERY

    dec2 = router.route("my profile")
    assert dec2.intent == IntentType.MEMORY_QUERY

    dec3 = router.route("what do you know about me")
    assert dec3.intent == IntentType.MEMORY_QUERY

    dec4 = router.route("about me")
    assert dec4.intent == IntentType.MEMORY_QUERY


@pytest.mark.asyncio
async def test_conversational_cortex_human_perspective():
    """Verify deep philosophical response on human nature, qualia, and empathy."""
    # Test with user's exact phrasing (including 'perfective' typo)
    reply1 = await conversational_cortex.respond("hi what is human in your perfective?", user_name="Harsha")
    assert reply1 is not None
    assert "Qualia" in reply1 or "qualia" in reply1
    assert "Emotional Valence" in reply1 or "empathy" in reply1.lower()
    assert "Harsha" in reply1

    # Test with standard 'perspective'
    reply2 = await conversational_cortex.respond("what is human in your perspective?", user_name="Harsha")
    assert reply2 is not None
    assert "Harsha" in reply2


@pytest.mark.asyncio
async def test_conversational_cortex_creator_and_feelings():
    """Verify transparent and articulate responses regarding AI identity and origins."""
    reply_creator = await conversational_cortex.respond("who created you", user_name="Harsha")
    assert reply_creator is not None
    assert "offline-first" in reply_creator
    assert "cloud" in reply_creator.lower()

    reply_feelings = await conversational_cortex.respond("do you have feelings", user_name="Harsha")
    assert reply_feelings is not None
    assert "subjective consciousness" in reply_feelings.lower() or "force-multiplier" in reply_feelings.lower()


@pytest.mark.asyncio
async def test_brain_think_human_perspective_flow():
    """Verify end-to-end brain think flow produces ConversationalCortex response."""
    await brain.initialize()
    resp = await brain.think("hi what is human in your perfective?", session_id="test-session-conv")

    assert resp.success is True
    assert resp.agent_used == "ConversationalCortex"
    assert "Qualia" in resp.content or "qualia" in resp.content
    assert "I hear you, Harsha. All JARVIS core systems are fully online" not in resp.content


@pytest.mark.asyncio
async def test_brain_think_tell_about_me_flow():
    """Verify end-to-end brain think flow produces MemoryAgent response for 'tell about me'."""
    await brain.initialize()
    from core.memory.persistent_memory import persistent_memory

    # Seed test profile preferences
    await persistent_memory.set_preference("user_name", "Harsha", category="identity")
    await persistent_memory.set_preference("user_role", "CSE student", category="background")
    await persistent_memory.set_preference("target_exam", "GATE CSE", category="academic_goals")

    resp = await brain.think("tell about me", session_id="test-session-mem")
    assert resp.success is True
    assert resp.agent_used == "MemoryAgent"
    assert "Harsha" in resp.content
    assert "CSE student" in resp.content or "GATE CSE" in resp.content
