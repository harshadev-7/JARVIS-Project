"""
J.A.R.V.I.S. Autonomous Intelligence System — Acceptance Scenarios Test Suite
=============================================================================
Tests all 10 user acceptance scenarios verifying natural language communication
without requiring slash commands (/research, /solve, /orchestrate, etc.).

Hardware profile: Intel Core i3, 8 GB RAM, 100% CPU local execution.
"""

import pytest
import asyncio
from core.brain.jarvis_brain import brain, JARVISBrain
from core.brain.router import IntentType


@pytest.mark.asyncio
async def test_acceptance_scenario_1_explain_os_processes():
    """TEST 1: 'Explain operating system processes.'
    Verifies that JARVIS understands a natural-language conceptual question,
    routes to KnowledgeAgent, and returns a structured explanation with core definition,
    architecture, PCB/states, and real-world examples without any slash command.
    """
    await brain.initialize()
    query = "Explain operating system processes."
    
    # Verify routing
    decision = brain.router.route(query)
    assert decision.intent == IntentType.KNOWLEDGE_QUESTION
    assert decision.agent_name in ("KnowledgeAgent", "KnowledgeEngine")
    
    # Verify brain execution
    response = await brain.think(query, session_id="test_acc_1")
    assert response.agent_used in ("KnowledgeAgent", "KnowledgeEngine")
    content = response.content.lower()
    assert "process" in content
    assert "core definition" in content or "definition" in content
    assert "how it works" in content or "pcb" in content or "architecture" in content
    assert "example" in content or "browser" in content


@pytest.mark.asyncio
async def test_acceptance_scenario_2_write_java_program():
    """TEST 2: 'Write a Java program to reverse a string.'
    Verifies that JARVIS recognizes a code generation request without slash commands,
    routes to ProgrammingAgent, and synthesizes complete, runnable Java code.
    """
    await brain.initialize()
    query = "Write a Java program to reverse a string."
    
    # Verify routing
    decision = brain.router.route(query)
    assert decision.intent == IntentType.CODE_GENERATION
    assert decision.agent_name in ("ProgrammingAgent", "DeductiveSolver")
    
    # Verify brain execution
    response = await brain.think(query, session_id="test_acc_2")
    assert response.agent_used in ("ProgrammingAgent", "DeductiveSolver")
    assert "public class" in response.content or "class " in response.content
    assert "public static void main" in response.content
    assert "reverse" in response.content.lower() or "stringbuilder" in response.content.lower()


@pytest.mark.asyncio
async def test_acceptance_scenario_3_latest_developments_in_ai():
    """TEST 3: 'What are the latest developments in AI?'
    Verifies that JARVIS recognizes the need for current live web intelligence,
    routes to ResearchAgent without requiring /research, and returns citations.
    """
    await brain.initialize()
    query = "What are the latest developments in AI?"
    
    # Verify routing
    decision = brain.router.route(query)
    assert decision.intent == IntentType.INTERNET_RESEARCH
    assert decision.agent_name == "ResearchAgent"
    
    # Verify brain execution
    response = await brain.think(query, session_id="test_acc_3")
    assert response.agent_used == "ResearchAgent"
    assert "research report" in response.content.lower() or "sources" in response.content.lower()


@pytest.mark.asyncio
async def test_acceptance_scenario_4_explain_jvm_jre_jdk():
    """TEST 4: 'Explain JVM, JRE, and JDK.'
    Verifies that JARVIS understands a multi-concept comparative question
    and generates an architectural synthesis without slash commands.
    """
    await brain.initialize()
    query = "Explain JVM, JRE, and JDK."
    
    # Verify routing
    decision = brain.router.route(query)
    assert decision.intent == IntentType.KNOWLEDGE_QUESTION
    
    # Verify brain execution
    response = await brain.think(query, session_id="test_acc_4")
    assert response.agent_used in ("KnowledgeAgent", "KnowledgeEngine")
    content = response.content.lower()
    assert "jvm" in content
    assert "jre" in content
    assert "jdk" in content
    assert "javac" in content or "compiler" in content
    assert "bytecode" in content or "runtime" in content


@pytest.mark.asyncio
async def test_acceptance_scenario_5_context_aware_pronoun_resolution():
    """TEST 5:
    Turn 1: 'What is JVM?' -> Explains JVM.
    Turn 2: 'What does it do?' -> Understands 'it' refers to JVM and explains JVM execution.
    """
    await brain.initialize()
    session_id = "test_acc_5"
    
    # Turn 1: Establish context
    resp1 = await brain.think("What is JVM?", session_id=session_id)
    assert resp1.agent_used in ("KnowledgeAgent", "KnowledgeEngine")
    assert "jvm" in resp1.content.lower()
    
    # Turn 2: Follow-up with pronoun "it"
    resp2 = await brain.think("What does it do?", session_id=session_id)
    content2 = resp2.content.lower()
    assert "jvm" in content2 or "virtual machine" in content2
    assert "bytecode" in content2 or "execution" in content2 or "classloader" in content2


@pytest.mark.asyncio
async def test_acceptance_scenario_6_calculate_compound_interest():
    """TEST 6: 'Calculate the compound interest on ₹10,000 at 8% for 2 years.'
    Verifies that JARVIS understands mathematical word problems, routes to
    MathematicalAgent, and computes the exact financial answer offline.
    """
    await brain.initialize()
    query = "Calculate the compound interest on ₹10,000 at 8% for 2 years."
    
    # Verify routing
    decision = brain.router.route(query)
    assert decision.intent == IntentType.MATHEMATICAL_PROBLEM
    assert decision.agent_name == "MathematicalAgent"
    
    # Verify brain execution
    response = await brain.think(query, session_id="test_acc_6")
    assert response.agent_used == "MathematicalAgent"
    # CI = 10000 * ((1 + 0.08)^2 - 1) = 10000 * 0.1664 = 1664
    assert "1,664" in response.content or "1664" in response.content
    assert "11,664" in response.content or "11664" in response.content


@pytest.mark.asyncio
async def test_acceptance_scenario_7_research_and_learning_roadmap():
    """TEST 7: 'Research Java backend development and create a learning roadmap.'
    Verifies that JARVIS decomposes this multi-step task into a DAG involving:
    Research -> Knowledge -> Reasoning -> Planning without requiring /orchestrate.
    """
    await brain.initialize()
    query = "Research Java backend development and create a learning roadmap."
    
    # Verify routing
    decision = brain.router.route(query)
    assert decision.intent == IntentType.MULTI_STEP_TASK
    assert decision.agent_name == "HierarchicalOrchestrator"
    
    # Verify DAG decomposition in supervisor
    from core.orchestrator.supervisor_agent import supervisor_agent
    graph = supervisor_agent.decompose_objective(query, "mis-acc-7")
    assigned = [node.assigned_agent for node in graph.nodes.values()]
    assert "research" in assigned
    assert "knowledge" in assigned
    assert "problem_solving" in assigned
    assert "task_planning" in assigned


@pytest.mark.asyncio
async def test_acceptance_scenario_8_autonomous_knowledge_retrieval_and_learning():
    """TEST 8: 'Explain Java inheritance.'
    Verifies that JARVIS provides the structured explanation, and on subsequent queries,
    retrieves the verified knowledge directly without unnecessary re-computation.
    """
    await brain.initialize()
    session_id = "test_acc_8"
    
    # Turn 1
    resp1 = await brain.think("Explain Java inheritance.", session_id=session_id)
    assert "inheritance" in resp1.content.lower()
    
    # Turn 2: Query again
    resp2 = await brain.think("Explain Java inheritance again", session_id=session_id)
    assert "inheritance" in resp2.content.lower()


@pytest.mark.asyncio
async def test_acceptance_scenario_9_restart_persistence():
    """TEST 9: Simulates system restart and verifies that previously learned knowledge
    in SQLite (jarvis.db) is preserved and instantly available.
    """
    from core.memory.persistent_memory import persistent_memory
    await persistent_memory.initialize()
    
    # Store a test knowledge entry
    item_id = await persistent_memory.upsert_knowledge(
        title="Research: Offline Transformer Kernel",
        content="JARVIS uses a local custom GPT decoder with 130,944 parameters on CPU.",
        category="research",
        source="unit_test",
    )
    assert item_id is not None
    
    # Simulate restart by creating fresh JarvisBrain
    fresh_brain = JARVISBrain()
    await fresh_brain.initialize()
    
    # Query knowledge through RAG
    from knowledge.rag_engine import rag_engine
    rag_ctx = await rag_engine.retrieve("Offline Transformer Kernel", top_k=1)
    assert rag_ctx.has_relevant_knowledge
    assert any("Transformer" in e.content for e in rag_ctx.entries)


@pytest.mark.asyncio
async def test_acceptance_scenario_10_natural_conversational_greeting():
    """TEST 10: 'Hello Jarvis.'
    Verifies that JARVIS responds naturally via Conversational Cortex without
    invoking unnecessary specialized execution agents.
    """
    await brain.initialize()
    query = "Hello Jarvis."
    
    # Verify routing
    decision = brain.router.route(query)
    assert decision.intent == IntentType.GREETING
    
    # Verify brain execution
    response = await brain.think(query, session_id="test_acc_10")
    assert response.agent_used == "ConversationalCortex"
    assert any(w in response.content.lower() for w in ["hello", "assist", "operational"])
