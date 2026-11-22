"""JARVIS Critical Intelligence Architecture Repair Test Suite.

Automated regression suite verifying the resolution of all 4 observed failures:
1. Java addition of two numbers returns complete working code (not a fallback menu).
2. "Proceed." continues the pending programming task.
3. "/solve 25 multiplied by 48" returns 1200 with brief explanation.
4. "/solve If 5 machines..." returns 5 minutes with step-by-step derivation.
5. "Generate an image prompt for Goku Ultra Instinct" returns detailed prompt (0 searches).
6. "Explain Java inheritance" returns clear explanation with Java class code (0 searches).
7. "What is the latest NVIDIA GPU?" correctly triggers research and returns a direct answer.
8. "Create a Python program to reverse a string" returns working Python code.
9. "Make the previous prompt more cinematic" refines previous prompt via context continuity.
10. "Explain what RAM is in simple words" returns clear, simple desk-metaphor definition (0 searches).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.brain.router import IntentRouter, IntentType
from core.brain.decision_policy import decision_policy
from core.brain.jarvis_brain import JARVISBrain
from services.internet.research_pipeline import research_pipeline, ResearchReport, SourceCitation


@pytest.fixture
def brain():
    return JARVISBrain()


@pytest.fixture
def router():
    return IntentRouter()


# ==============================================================================
# 10-TEST CRITICAL INTELLIGENCE REPAIR SUITE
# ==============================================================================

@pytest.mark.asyncio
async def test_1_java_addition_code(brain, router):
    """Test 1: User asks for Java code to add two numbers -> returns complete working Java program."""
    initial_count = research_pipeline.research_call_count
    user_query = "Give me a code of addition of two numbers in Java programming language."

    decision = router.route(user_query)
    assert decision.intent in (IntentType.CODE_GENERATION, IntentType.PROGRAMMING)

    resp = await brain.think(user_query, session_id="test_repair_session")
    assert resp.success
    # Must NOT offer the 3-choice fallback menu
    assert "how would you prefer to proceed" not in resp.content.lower()
    assert "/research" not in resp.content and "/solve" not in resp.content
    # Must contain working Java code
    assert "public class" in resp.content or "class AddTwoNumbers" in resp.content
    assert "public static void main" in resp.content
    assert "num1 + num2" in resp.content or "double sum" in resp.content or "int sum" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search should not be called for Java addition!"


@pytest.mark.asyncio
async def test_2_proceed_continuity(brain):
    """Test 2: User says 'Proceed.' -> continues the pending Java task rather than resetting state."""
    initial_count = research_pipeline.research_call_count

    # Turn 1: Ask for Java code (sets pending task)
    await brain.think("Give me a Java program to add two numbers.", session_id="test_continuity_session")

    # Turn 2: User says "Proceed."
    resp = await brain.think("Proceed.", session_id="test_continuity_session")
    assert resp.success
    assert "public class" in resp.content or "class AddTwoNumbers" in resp.content or "java" in resp.content.lower()
    assert "main" in resp.content
    assert "[SOURCE: REASONED" in resp.content
    assert research_pipeline.research_call_count == initial_count


@pytest.mark.asyncio
async def test_3_solve_math_multiplication(brain, router):
    """Test 3: /solve 25 multiplied by 48 -> returns 1200 with explanation."""
    initial_count = research_pipeline.research_call_count
    user_query = "/solve 25 multiplied by 48"

    decision = router.route(user_query)
    assert decision.intent in (IntentType.MATHEMATICAL_PROBLEM, IntentType.MATH, IntentType.PROBLEM_SOLVING)

    resp = await brain.think(user_query)
    assert resp.success
    assert "1200" in resp.content
    assert "[SOURCE: CALCULATED" in resp.content or "[SOURCE: REASONED" in resp.content
    assert research_pipeline.research_call_count == initial_count


@pytest.mark.asyncio
async def test_4_solve_machines_rate_riddle(brain, router):
    """Test 4: /solve If 5 machines produce 5 products in 5 minutes, how long will 100 machines take to produce 100 products? -> 5 minutes."""
    initial_count = research_pipeline.research_call_count
    user_query = "/solve If 5 machines produce 5 products in 5 minutes, how long will 100 machines take to produce 100 products?"

    decision = router.route(user_query)
    assert decision.intent in (IntentType.LOGICAL_REASONING, IntentType.REASONING)

    resp = await brain.think(user_query)
    assert resp.success
    assert "5 minutes" in resp.content.lower()
    # Must NOT contain empty analytical templates
    assert "none specified" not in resp.content.lower()
    assert research_pipeline.research_call_count == initial_count


@pytest.mark.asyncio
async def test_5_image_prompt_goku(brain, router):
    """Test 5: Generate an image prompt for Goku Ultra Instinct -> original detailed prompt."""
    initial_count = research_pipeline.research_call_count
    user_query = "Generate an image prompt for Goku Ultra Instinct."

    decision = router.route(user_query)
    assert decision.intent == IntentType.IMAGE_PROMPT_GENERATION

    resp = await brain.think(user_query, session_id="test_goku_session")
    assert resp.success
    assert "goku" in resp.content.lower()
    assert any(term in resp.content.lower() for term in ["lighting", "aura", "composition", "style"])
    assert "[SOURCE: CREATIVE" in resp.content
    assert research_pipeline.research_call_count == initial_count


@pytest.mark.asyncio
async def test_6_explain_java_inheritance(brain, router):
    """Test 6: Explain Java inheritance -> clear explanation with Java example."""
    initial_count = research_pipeline.research_call_count
    user_query = "Explain Java inheritance."

    resp = await brain.think(user_query)
    assert resp.success
    assert "extends" in resp.content
    assert "class" in resp.content
    assert "animal" in resp.content.lower() or "super" in resp.content.lower() or "subclass" in resp.content.lower()
    assert "[SOURCE: REASONED" in resp.content or "[SOURCE: KNOWN" in resp.content
    assert research_pipeline.research_call_count == initial_count


@pytest.mark.asyncio
async def test_7_latest_nvidia_gpu_research(brain, router):
    """Test 7: What is the latest NVIDIA GPU? -> triggers research and provides direct answer."""
    user_query = "What is the latest NVIDIA GPU?"
    decision = router.route(user_query)
    assert decision.intent in (IntentType.INTERNET_RESEARCH, IntentType.RESEARCH)
    assert decision_policy.should_search_internet(user_query)

    mock_report = ResearchReport(
        topic="latest NVIDIA GPU",
        summary="NVIDIA has announced its next-generation Blackwell architecture (RTX 5090 and B200) for consumer and datacenter computing.",
        citations=[SourceCitation(title="NVIDIA News", url="https://nvidia.com/news", snippet="Blackwell GPUs")],
    )

    with patch("services.internet.research_pipeline.research_pipeline.conduct_research", new_callable=AsyncMock) as mock_res:
        mock_res.return_value = mock_report
        resp = await brain.think(user_query)
        assert resp.success
        assert mock_res.called
        assert "[SOURCE: RETRIEVED" in resp.content


@pytest.mark.asyncio
async def test_8_python_reverse_string(brain, router):
    """Test 8: Create a Python program to reverse a string -> working Python code."""
    initial_count = research_pipeline.research_call_count
    user_query = "Create a Python program to reverse a string."

    decision = router.route(user_query)
    assert decision.intent in (IntentType.CODE_GENERATION, IntentType.PROGRAMMING)

    resp = await brain.think(user_query)
    assert resp.success
    assert "def reverse_string" in resp.content
    assert "[::-1]" in resp.content or "reversed" in resp.content
    assert "[SOURCE: REASONED" in resp.content
    assert research_pipeline.research_call_count == initial_count


@pytest.mark.asyncio
async def test_9_make_previous_prompt_more_cinematic(brain):
    """Test 9: Make the previous prompt more cinematic -> modifies prompt via context continuity."""
    initial_count = research_pipeline.research_call_count

    # Turn 1: Generate initial Goku prompt
    await brain.think("Generate an image prompt for Goku Ultra Instinct.", session_id="test_goku_session")

    # Turn 2: User asks to make it more cinematic
    resp = await brain.think("Make the previous prompt more cinematic.", session_id="test_goku_session")
    assert resp.success
    assert "goku" in resp.content.lower()
    assert any(term in resp.content.lower() for term in ["cinematic", "anamorphic", "atmospheric", "lighting"])
    assert "[SOURCE: CREATIVE" in resp.content
    assert research_pipeline.research_call_count == initial_count


@pytest.mark.asyncio
async def test_10_explain_ram_in_simple_words(brain, router):
    """Test 10: Explain what RAM is in simple words -> direct answer without web search."""
    initial_count = research_pipeline.research_call_count
    user_query = "Explain what RAM is in simple words."

    resp = await brain.think(user_query)
    assert resp.success
    assert any(term in resp.content.lower() for term in ["memory", "desk", "short-term", "storage", "fast"])
    assert "[SOURCE: KNOWN" in resp.content
    assert research_pipeline.research_call_count == initial_count
