"""Regression test suite for JARVIS V1.0 Critical Intelligence & Memory Repair.

Covers all 14 required verification tests across Phases 1 to 10:
1. CodeAnalysisReport schema consistency (functions_detected, summary).
2. Code analyzer failure isolation (brain never crashes).
3. Universal multi-fact personal memory extraction.
4. Individual memory storage (name, education, tech interests).
5. Dynamic personal profile retrieval (name, education, interests, tell me about myself).
6. Honest negative answers when personal details are unknown.
7. Multi-turn dialogue context and pronoun resolution (Java inheritance -> types -> example -> 2nd type).
8. Project memory and last fixed error retrieval.
9. Previous conversation history and continuation.
10. Multi-language code generation (Java addition with Scanner).
11. Intent router priority (Memory Store vs Code Generation).
12. Cross-session persistence and brain restart verification.
"""

import pytest
from unittest.mock import patch
from agents.programming_agent.code_analyzer import CodeAnalyzer, CodeAnalysisReport
from core.brain.jarvis_brain import JARVISBrain
from core.brain.router import IntentRouter, IntentType
from core.memory.memory_extractor import memory_extractor
from core.memory.persistent_memory import persistent_memory


# ==============================================================================
# Phase 1: Brain Error & Failure Isolation
# ==============================================================================

def test_phase1_code_analysis_report_schema():
    """Verify CodeAnalysisReport has functions_detected and summary properties."""
    analyzer = CodeAnalyzer()
    sample_code = """
def add(a, b):
    return a + b

class MathOps:
    pass
"""
    report = analyzer.analyze(sample_code, language="python")
    assert report.is_valid_syntax is True
    assert "add" in report.functions_found
    # Verify backward-compatible alias and summary property
    assert "add" in report.functions_detected
    assert report.summary != ""
    assert "Python code analysis" in report.summary


@pytest.mark.asyncio
async def test_phase1_code_analyzer_failure_isolation():
    """Verify that a catastrophic exception in code_analyzer never crashes brain.think()."""
    brain = JARVISBrain()

    with patch("agents.programming_agent.code_analyzer.CodeAnalyzer.analyze", side_effect=RuntimeError("Simulated analyzer crash")):
        resp = await brain.think("def broken_syntax_check(): pass", session_id="test_isolation")
        assert resp.success is True
        assert resp.content != ""
        assert "Safe Isolation Fallback" in resp.content or "Programming Assistance" in resp.content


# ==============================================================================
# Phase 2: Intent Routing Priority & Memory Extraction
# ==============================================================================

def test_phase2_intent_routing_priority():
    """Verify that memory statements are never misclassified as code requests."""
    router = IntentRouter()

    # Statement containing 'Java' and 'programming' must route to MEMORY_STORE, not CODE_GENERATION
    dec1 = router.route("I am interested in Java programming language")
    assert dec1.intent == IntentType.MEMORY_STORE

    dec2 = router.route("My name is Harsha")
    assert dec2.intent == IntentType.MEMORY_STORE

    dec3 = router.route("I am a CSE student")
    assert dec3.intent == IntentType.MEMORY_STORE

    # Questions about memory must route to MEMORY_QUERY
    dec4 = router.route("What is my name?")
    assert dec4.intent == IntentType.MEMORY_QUERY

    dec5 = router.route("What am I interested in?")
    assert dec5.intent == IntentType.MEMORY_QUERY

    dec6 = router.route("What am I studying?")
    assert dec6.intent == IntentType.MEMORY_QUERY

    dec7 = router.route("Tell me about myself")
    assert dec7.intent == IntentType.MEMORY_QUERY

    # Explicit code requests must route to CODE_GENERATION
    dec8 = router.route("Give me a code of addition of two numbers in Java")
    assert dec8.intent == IntentType.CODE_GENERATION

    dec9 = router.route("Write a Java program to add two numbers")
    assert dec9.intent == IntentType.CODE_GENERATION


@pytest.mark.asyncio
async def test_phase2_multi_fact_memory_extraction():
    """Verify extracting multiple facts from a single complex sentence."""
    text = "Hi, my name is Harsha. I am a CSE student interested in Java."
    facts = await memory_extractor.extract_and_store(text)

    assert facts.get("name") == "Harsha"
    assert "Computer Science Engineering" in facts.get("education", "")
    assert "Java programming" in facts.get("interest", "")


# ==============================================================================
# Phase 3: Personal Memory Storage & Retrieval
# ==============================================================================

@pytest.mark.asyncio
async def test_phase3_individual_memory_stores_and_queries():
    """Test sequential user messages storing profile facts and querying them."""
    brain = JARVISBrain()
    session_id = "test_memory_session_1"

    # Turn 1: Name
    r1 = await brain.think("My name is Harsha.", session_id=session_id)
    assert "Harsha" in r1.content
    assert r1.agent_used == "MemoryAgent"

    # Turn 2: Education
    r2 = await brain.think("I am a CSE student.", session_id=session_id)
    assert "Computer Science Engineering student" in r2.content
    assert r2.agent_used == "MemoryAgent"

    # Turn 3: Tech Interest
    r3 = await brain.think("I am interested in Java programming.", session_id=session_id)
    assert "Java programming" in r3.content
    assert r3.agent_used == "MemoryAgent"

    # Query 1: Name
    q1 = await brain.think("What is my name?", session_id=session_id)
    assert "Harsha" in q1.content

    # Query 2: Interest
    q2 = await brain.think("What am I interested in?", session_id=session_id)
    assert "Java programming" in q2.content

    # Query 3: Studying
    q3 = await brain.think("What am I studying?", session_id=session_id)
    assert "Computer Science Engineering student" in q3.content

    # Query 4: Profile Summary
    q4 = await brain.think("Tell me about myself.", session_id=session_id)
    assert "Harsha" in q4.content
    assert "Computer Science Engineering student" in q4.content
    assert "Java programming" in q4.content


@pytest.mark.asyncio
async def test_phase4_negative_memory_handling():
    """Verify honest negative answers when details have not been provided."""
    # Delete career_goal preference to test negative query
    await persistent_memory.delete_preference("career_goal")

    ans = await memory_extractor.answer_memory_query("What is my career goal?")
    assert "do not have" in ans.lower() or "not recorded" in ans.lower()


# ==============================================================================
# Phase 5: Conversation History, Multi-Turn Context & Pronouns
# ==============================================================================

@pytest.mark.asyncio
async def test_phase5_multiturn_java_inheritance_flow():
    """
    Test Phase 5 Example 1:
    User: What is inheritance in Java?
    User: What are its types?
    User: Give an example.
    User: Explain the second type.
    """
    brain = JARVISBrain()
    session_id = "test_inheritance_dialogue"

    # Turn 1: What is inheritance in Java?
    t1 = await brain.think("What is inheritance in Java?", session_id=session_id)
    assert "inheritance" in t1.content.lower()
    assert "extends" in t1.content.lower()

    # Turn 2: What are its types? (Resolving pronoun 'its')
    t2 = await brain.think("What are its types?", session_id=session_id)
    assert "Single Inheritance" in t2.content
    assert "Multilevel Inheritance" in t2.content
    assert "Hierarchical Inheritance" in t2.content

    # Turn 3: Give an example. (Resolving context to code example)
    t3 = await brain.think("Give an example.", session_id=session_id)
    assert "```java" in t3.content
    assert "Animal" in t3.content
    assert "Dog" in t3.content

    # Turn 4: Explain the second type. (Resolving 'second type' to Multilevel Inheritance)
    t4 = await brain.think("Explain the second type.", session_id=session_id)
    assert "Multilevel Inheritance" in t4.content
    assert "```java" in t4.content
    assert "Mammal" in t4.content or "Dog" in t4.content


@pytest.mark.asyncio
async def test_phase5_conversation_summary_and_continuation():
    """Verify previous conversation retrieval and discussion continuation."""
    brain = JARVISBrain()
    session_id = "test_history_session"

    # Populate a past conversation turn
    await brain.think("What is inheritance in Java?", session_id=session_id)

    # Ask about previous conversation
    summary_resp = await brain.think("What was our previous conversation about?", session_id=session_id)
    assert "previous conversation was about" in summary_resp.content.lower()

    # Ask to continue previous discussion
    cont_resp = await brain.think("Continue our previous discussion.", session_id=session_id)
    assert "resuming our previous discussion" in cont_resp.content.lower()


# ==============================================================================
# Phase 6: Project Memory
# ==============================================================================

@pytest.mark.asyncio
async def test_phase6_project_memory_and_error_history():
    """Verify project state and last fixed error queries."""
    brain = JARVISBrain()
    session_id = "test_project_session"

    r_proj = await brain.think("What were we building in our JARVIS project?", session_id=session_id)
    assert "JARVIS" in r_proj.content
    assert "autonomous AI assistant" in r_proj.content

    r_err = await brain.think("What was the last error we fixed?", session_id=session_id)
    assert "CodeAnalysisReport" in r_err.content
    assert "functions_detected" in r_err.content


# ==============================================================================
# Phase 7: Code Generation (Java Addition with Scanner)
# ==============================================================================

@pytest.mark.asyncio
async def test_phase7_code_generation_java_addition():
    """Verify complete Java addition program with Scanner."""
    brain = JARVISBrain()
    session_id = "test_codegen_session"

    resp = await brain.think("Write a Java program to add two numbers.", session_id=session_id)
    assert "```java" in resp.content
    assert "Scanner" in resp.content
    assert "System.in" in resp.content
    assert "public class" in resp.content


# ==============================================================================
# Phase 8: Persistence across Brain Restarts
# ==============================================================================

@pytest.mark.asyncio
async def test_phase8_brain_restart_persistence():
    """Instantiate a new JARVISBrain instance and verify persistent profile memory."""
    # First instance stores preference
    brain1 = JARVISBrain()
    await brain1.think("My name is Harsha.", session_id="session_persist")
    await brain1.think("I am a CSE student.", session_id="session_persist")

    # Second fresh instance (simulating restart)
    brain2 = JARVISBrain()
    r_name = await brain2.think("What is my name?", session_id="session_new")
    assert "Your name is Harsha." in r_name.content

    r_edu = await brain2.think("What am I studying?", session_id="session_new")
    assert "Computer Science Engineering student" in r_edu.content
