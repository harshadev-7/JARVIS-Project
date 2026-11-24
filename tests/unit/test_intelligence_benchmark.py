"""JARVIS Intelligence and Reasoning Benchmark Suite.

Executes the 10-test automated verification suite for the Intelligence Upgrade:
1. Image prompt generation (Local, 0 searches)
2. Machine production riddle (5 minutes, 0 searches)
3. Basic factual question (Paris, 0 searches)
4. Code generation (Python palindrome, 0 searches)
5. Creative concept (Sci-fi time travel film, 0 searches)
6. Bat and ball problem ($0.05 / 5 cents, 0 searches)
7. Deterministic arithmetic (25 * 18 = 450, 0 searches)
8. Live information query (Quantum computing, triggers research)
9. Local document summarization (0 searches)
10. Explicit web search (Electric vehicles, triggers research)
Bonus: Multi-turn conversational prompt generation and reference resolution (0 searches)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.brain.router import IntentRouter, IntentType
from core.brain.decision_policy import decision_policy
from core.brain.jarvis_brain import JARVISBrain
from services.internet.research_pipeline import research_pipeline, ResearchReport, SourceCitation


@pytest.fixture
def router():
    return IntentRouter()


@pytest.fixture
def brain():
    return JARVISBrain()


# ==============================================================================
# PHASE 1 & 2: INTENT CLASSIFICATION & DECISION POLICY TESTS
# ==============================================================================

def test_intent_classification_and_policy(router):
    """Verify that intents are properly classified and search policy correctly gates them."""
    # Test 1: Image Prompt
    d1 = router.route("Generate a prompt for AI to create an image of a cybernetic tiger in a neon Tokyo alley.")
    assert d1.intent == IntentType.IMAGE_PROMPT_GENERATION
    assert not decision_policy.should_search_internet("Generate a prompt for AI to create an image")

    # Test 2: Machine rate riddle
    d2 = router.route("If 5 machines produce 5 products in 5 minutes, how long will 100 machines take to produce 100 products?")
    assert d2.intent == IntentType.LOGICAL_REASONING
    assert not decision_policy.should_search_internet("If 5 machines produce 5 products in 5 minutes")

    # Test 3: Basic factual question
    d3 = router.route("What is the capital of France?")
    assert d3.intent in (IntentType.GENERAL_QUESTION, IntentType.KNOWLEDGE_QUESTION)
    assert not decision_policy.should_search_internet("What is the capital of France?")

    # Test 4: Code generation
    d4 = router.route("Write a Python function to check whether a string is a palindrome.")
    assert d4.intent == IntentType.CODE_GENERATION
    assert not decision_policy.should_search_internet("Write a Python function to check palindrome")

    # Test 5: Creative concept
    d5 = router.route("Create a concept for a sci-fi short film about time travel.")
    assert d5.intent == IntentType.CREATIVE_GENERATION
    assert not decision_policy.should_search_internet("Create a concept for a sci-fi short film")

    # Test 6: Bat and ball riddle
    d6 = router.route("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?")
    assert d6.intent == IntentType.LOGICAL_REASONING

    # Test 7: Arithmetic math
    d7 = router.route("What is 25 * 18?")
    assert d7.intent in (IntentType.MATHEMATICAL_PROBLEM, IntentType.MATH)
    assert not decision_policy.should_search_internet("What is 25 * 18?")

    # Test 8: Live / current developments
    d8 = router.route("What are the latest developments in quantum computing?")
    assert d8.intent in (IntentType.INTERNET_RESEARCH, IntentType.RESEARCH)
    assert decision_policy.should_search_internet("What are the latest developments in quantum computing?")

    # Test 9: Document summarization
    d9 = router.route("Summarize this document: Artificial Intelligence has evolved from rule-based systems to deep neural networks.")
    assert d9.intent == IntentType.PROBLEM_SOLVING
    assert not decision_policy.should_search_internet("Summarize this document: Artificial Intelligence has evolved")

    # Test 10: Explicit web search
    d10 = router.route("Search the web for recent electric vehicle releases.")
    assert d10.intent in (IntentType.INTERNET_RESEARCH, IntentType.RESEARCH)
    assert decision_policy.should_search_internet("Search the web for recent electric vehicle releases.")


# ==============================================================================
# 10-TEST END-TO-END INTELLIGENCE BENCHMARK SUITE
# ==============================================================================

@pytest.mark.asyncio
async def test_benchmark_1_image_prompt(brain):
    """Test 1: Image prompt generation without web search."""
    initial_count = research_pipeline.research_call_count
    resp = await brain.think("Generate a prompt for AI to create an image of a cybernetic tiger in a neon Tokyo alley.")
    assert resp.success
    assert "tiger" in resp.content.lower()
    assert any(term in resp.content.lower() for term in ["neon", "cyberpunk", "lighting", "shot on"])
    assert "[SOURCE: CREATIVE" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered!"


@pytest.mark.asyncio
async def test_benchmark_2_machine_riddle(brain):
    """Test 2: Deductive problem solving (5 machines puzzle) -> exact 5 minutes."""
    initial_count = research_pipeline.research_call_count
    resp = await brain.think("If 5 machines produce 5 products in 5 minutes, how long will 100 machines take to produce 100 products?")
    assert resp.success
    assert "5 minutes" in resp.content.lower()
    assert "[SOURCE: REASONED" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered!"


@pytest.mark.asyncio
async def test_benchmark_3_capital_of_france(brain):
    """Test 3: Basic factual question -> Paris without web search."""
    initial_count = research_pipeline.research_call_count
    resp = await brain.think("What is the capital of France?")
    assert resp.success
    assert "paris" in resp.content.lower()
    assert "[SOURCE: KNOWN" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered!"


@pytest.mark.asyncio
async def test_benchmark_4_code_generation(brain):
    """Test 4: Programming generation -> Python palindrome function."""
    initial_count = research_pipeline.research_call_count
    resp = await brain.think("Write a Python function to check whether a string is a palindrome.")
    assert resp.success
    assert "def is_palindrome" in resp.content
    assert "[::-1]" in resp.content or "reversed" in resp.content or "isalnum" in resp.content
    assert "[SOURCE: REASONED" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered!"


@pytest.mark.asyncio
async def test_benchmark_5_creative_concept(brain):
    """Test 5: Original creative concept -> Sci-fi time travel film."""
    initial_count = research_pipeline.research_call_count
    resp = await brain.think("Create a concept for a sci-fi short film about time travel.")
    assert resp.success
    assert "logline" in resp.content.lower()
    assert any(term in resp.content.lower() for term in ["temporal", "chrono", "time", "premise"])
    assert "[SOURCE: CREATIVE" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered!"


@pytest.mark.asyncio
async def test_benchmark_6_bat_and_ball_riddle(brain):
    """Test 6: Algebraic riddle -> Bat and ball costs -> Ball is $0.05 / 5 cents."""
    initial_count = research_pipeline.research_call_count
    resp = await brain.think("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?")
    assert resp.success
    assert any(ans in resp.content for ans in ["0.05", "5 cents", "$0.05"])
    assert "[SOURCE: REASONED" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered!"


@pytest.mark.asyncio
async def test_benchmark_7_arithmetic_calculation(brain):
    """Test 7: Deterministic math -> 25 * 18 = 450."""
    initial_count = research_pipeline.research_call_count
    resp = await brain.think("What is 25 * 18?")
    assert resp.success
    assert "450" in resp.content
    assert "[SOURCE: CALCULATED" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered!"


@pytest.mark.asyncio
async def test_benchmark_8_live_information_research(brain):
    """Test 8: Current developments query -> Correctly triggers research pipeline."""
    initial_count = research_pipeline.research_call_count

    # Mock research pipeline to verify routing without actual internet dependencies in offline CI
    mock_report = ResearchReport(
        topic="latest developments in quantum computing",
        summary="Recent breakthroughs in topological qubits and neutral-atom quantum processors have achieved higher coherence times.",
        citations=[SourceCitation(title="Quantum Progress", url="https://example.com/quantum", snippet="Topological qubits")],
    )

    with patch("services.internet.research_pipeline.research_pipeline.conduct_research", new_callable=AsyncMock) as mock_res:
        mock_res.return_value = mock_report
        resp = await brain.think("What are the latest developments in quantum computing?")
        assert resp.success
        assert mock_res.called, "Research pipeline should have been triggered for live developments query!"
        assert "[SOURCE: RETRIEVED" in resp.content


@pytest.mark.asyncio
async def test_benchmark_9_document_summarization(brain):
    """Test 9: Local document summarization -> 0 internet searches."""
    initial_count = research_pipeline.research_call_count
    doc = (
        "Summarize this document: Artificial Intelligence has evolved from rule-based expert systems to deep connectionist neural networks. "
        "Transformers revolutionized sequence modeling by replacing recurrence with self-attention mechanisms. "
        "Current focus is on autonomous agency, mathematical grounding, and edge-device efficiency."
    )
    resp = await brain.think(doc)
    assert resp.success
    assert "summary" in resp.content.lower()
    assert "[SOURCE: REASONED" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search was improperly triggered on document summarization!"


@pytest.mark.asyncio
async def test_benchmark_10_explicit_web_search(brain):
    """Test 10: Explicit web search request -> Triggers research pipeline."""
    mock_report = ResearchReport(
        topic="recent electric vehicle releases",
        summary="Automakers have announced next-generation solid-state battery platforms and affordable mid-size crossover EV models.",
        citations=[SourceCitation(title="EV Weekly", url="https://example.com/ev", snippet="New EV releases")],
    )

    with patch("services.internet.research_pipeline.research_pipeline.conduct_research", new_callable=AsyncMock) as mock_res:
        mock_res.return_value = mock_report
        resp = await brain.think("Search the web for recent electric vehicle releases.")
        assert resp.success
        assert mock_res.called, "Research pipeline should have been triggered for explicit web search!"
        assert "[SOURCE: RETRIEVED" in resp.content


# ==============================================================================
# BONUS: MULTI-TURN CONVERSATIONAL CONTEXT RESOLUTION
# ==============================================================================

@pytest.mark.asyncio
async def test_multi_turn_prompt_and_reference_resolution(brain):
    """Test conversational context: Multi-turn prompt disambiguation and ordinal reference."""
    initial_count = research_pipeline.research_call_count

    # Turn 1: Conversational context simulated
    history = [
        {"role": "user", "content": "Generate a prompt."},
        {"role": "assistant", "content": "What would you like the AI to create?"},
    ]

    # Turn 2: User provides subject
    resp = await brain.think("An image of Goku in Ultra Instinct.", conversation_history=history, session_id="test_conv")
    assert resp.success
    assert "goku" in resp.content.lower()
    assert "[SOURCE: CREATIVE" in resp.content
    assert research_pipeline.research_call_count == initial_count, "Internet search should not be called for multi-turn prompt creation!"

    # Turn 3: Follow-up reference "make the first one" or "the first one"
    resp2 = await brain.think("Proceed with the first one", session_id="test_conv")
    assert resp2.success
    assert "goku" in resp2.content.lower()
    assert "[SOURCE: KNOWN" in resp2.content
    assert research_pipeline.research_call_count == initial_count
