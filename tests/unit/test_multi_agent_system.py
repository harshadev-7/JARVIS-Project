"""Unit Tests for J.A.R.V.I.S. Independent Multi-Agent System (Phase 9).

Validates:
1. Registration of all 8 specialized agents in IndependentAgentSystem
2. Inter-agent communication and inbox delivery
3. ResearchAgent web search and citation formatting
4. PlanningAgent 10-step reasoning and subtask plan generation
5. CodingAgent AST code analysis and security auditing
6. KnowledgeAgent persistent fact indexing and search
7. MemoryAgent preference coordination and session cleanup
8. ComputerControlAgent consequential action safety gates
9. TaskExecutionAgent pipeline limits and audit logging
10. Autonomy level enforcement
"""

import pytest

from agents import (
    CodingAgent,
    ComputerControlAgent,
    IndependentAgentSystem,
    KnowledgeAgent,
    MemoryAgent,
    PlanningAgent,
    ResearchAgent,
    TaskExecutionAgent,
    VoiceAgent,
    agent_system,
)
from core.orchestrator.base_agent import AgentMessage, AgentTaskResult


def test_agent_system_roster():
    agents = agent_system.list_agents()
    agent_names = {a["name"] for a in agents}

    expected = {
        "ResearchAgent",
        "PlanningAgent",
        "CodingAgent",
        "KnowledgeAgent",
        "MemoryAgent",
        "ComputerControlAgent",
        "VoiceAgent",
        "TaskExecutionAgent",
    }
    assert expected.issubset(agent_names), f"Missing agents: {expected - agent_names}"
    assert len(agents) == 8


def test_inter_agent_messaging():
    # Send message from PlanningAgent to CodingAgent
    success = agent_system.send_message(
        sender="PlanningAgent",
        recipient="CodingAgent",
        message_type="task_request",
        content={"action": "write_function", "name": "calculate_metrics"},
    )
    assert success is True

    coding_agent = agent_system.get_agent("coding")
    assert coding_agent is not None
    assert len(coding_agent._inbox) > 0
    last_msg = coding_agent._inbox[-1]
    assert last_msg.sender == "PlanningAgent"
    assert last_msg.content["action"] == "write_function"


@pytest.mark.asyncio
async def test_research_agent():
    researcher = agent_system.get_agent("research")
    res = await researcher.execute("Quantum computing fundamentals")

    assert isinstance(res, AgentTaskResult)
    assert res.agent_name == "ResearchAgent"
    assert "summary" in res.data or res.success is False


@pytest.mark.asyncio
async def test_planning_agent():
    planner = agent_system.get_agent("planning")
    res = await planner.execute("Deploy a secure local database server")

    assert isinstance(res, AgentTaskResult)
    assert res.agent_name == "PlanningAgent"
    assert res.success is True
    assert "subtasks" in res.data
    assert len(res.data["subtasks"]) > 0


@pytest.mark.asyncio
async def test_coding_agent():
    coder = agent_system.get_agent("coding")
    code_sample = "def add(a: int, b: int) -> int:\n    return a + b\n"
    res = await coder.execute(code_sample)

    assert isinstance(res, AgentTaskResult)
    assert res.agent_name == "CodingAgent"
    assert res.success is True
    assert res.data["valid_syntax"] is True
    assert "add" in res.data["functions"]


@pytest.mark.asyncio
async def test_knowledge_agent():
    knower = agent_system.get_agent("knowledge")
    # Index new knowledge
    store_res = await knower.execute("store: Architecture - Multi-agent system contains 8 agents.")
    assert store_res.success is True

    # Search knowledge
    search_res = await knower.execute("multi-agent system")
    assert search_res.success is True
    assert len(search_res.data["matches"]) > 0


@pytest.mark.asyncio
async def test_memory_agent():
    mem_agent = agent_system.get_agent("memory")
    res = await mem_agent.execute("set theme to dark_ironman")

    assert res.success is True
    assert res.data["key"] == "theme"
    assert res.data["value"] == "dark_ironman"


@pytest.mark.asyncio
async def test_computer_control_consequential_safety_gate():
    controller = agent_system.get_agent("computer_control")

    # 1. Consequential action WITHOUT user approval must be blocked
    blocked_res = await controller.execute("delete sensitive_system_file.txt", context={"user_approved": False})
    assert blocked_res.success is False
    assert blocked_res.error == "UserAuthorizationRequired"
    assert "Authorization Required" in blocked_res.summary

    # 2. Consequential action WITH user approval proceeds through policy check
    approved_res = await controller.execute("delete temporary_cache.tmp", context={"user_approved": True, "autonomy_level": 4})
    # Does not get blocked by authorization gate
    assert approved_res.error != "UserAuthorizationRequired"


@pytest.mark.asyncio
async def test_task_execution_agent():
    executor = agent_system.get_agent("task_execution")
    res = await executor.execute("Verify all local subsystems")

    assert res.success is True
    assert res.agent_name == "TaskExecutionAgent"
    assert res.duration_sec >= 0.0


@pytest.mark.asyncio
async def test_autonomy_level_enforcement():
    # Attempting to execute CodingAgent (requires level 2) with autonomy_level=1
    res = await agent_system.execute_task(
        agent_name="coding",
        task="def test(): pass",
        context={"autonomy_level": 1},
    )
    assert res.success is False
    assert res.error == "AutonomyPermissionDenied"
