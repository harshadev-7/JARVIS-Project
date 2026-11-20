"""
JARVIS AI — Phase 9 Test Suite
================================
Tests for Multi-Agent Orchestration & All 15 Specialized Agents:
- Verification of all 15 agent registrations, roles & capabilities
- Mathematical calculation, data analysis, and scriptwriting agents
- Multi-agent mission coordination and collective intelligence synthesis
- FastAPI agent endpoints
"""

import pytest
from fastapi.testclient import TestClient

from core.orchestrator.agent_registry import agent_registry
from core.orchestrator.orchestrator import orchestrator
from api.app import app


# ── 1. 15 Agents Verification Tests ─────────────────────────────────────────

def test_all_15_agents_registered():
    """All 15 agents required by Section 3 must be present in registry."""
    agents = agent_registry.list_all()
    assert len(agents) == 15
    expected_ids = [
        "research", "problem_solving", "programming", "mathematical",
        "data_analysis", "computer_automation", "device_management",
        "social_media", "content_creation", "video_editing",
        "personal_assistant", "memory_management", "cybersecurity",
        "communication", "task_planning",
    ]
    registered_ids = [a["id"] for a in agents]
    for exp in expected_ids:
        assert exp in registered_ids


# ── 2. Specialized Agent Execution Tests ────────────────────────────────────

@pytest.mark.asyncio
async def test_mathematical_agent():
    """Math agent should evaluate arithmetic and trigonometric operations."""
    math_agent = agent_registry.get_agent("mathematical")
    res = await math_agent.execute("sqrt(144) + 10")
    assert res.success is True
    assert res.data["result"] == 22.0


@pytest.mark.asyncio
async def test_data_analysis_agent():
    """Data analysis agent should calculate mean and bounds."""
    da_agent = agent_registry.get_agent("data_analysis")
    res = await da_agent.execute("compute stats", context={"data": [10, 20, 30, 40, 50]})
    assert res.success is True
    assert res.data["mean"] == 30.0
    assert res.data["count"] == 5


@pytest.mark.asyncio
async def test_content_and_video_agents():
    """Content & Video agents should formulate script and 9:16 vertical framing."""
    cc_agent = agent_registry.get_agent("content_creation")
    vid_agent = agent_registry.get_agent("video_editing")

    res_cc = await cc_agent.execute("Quantum Computing in 60 seconds")
    assert res_cc.success is True
    assert "Brief" in res_cc.data["script"]

    res_vid = await vid_agent.execute("Render Short")
    assert res_vid.success is True
    assert res_vid.data["aspect_ratio"] == "9:16"


# ── 3. Multi-Agent Orchestrator Tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_multi_agent_mission():
    """Orchestrator should deploy multiple agents and synthesize results."""
    mission = await orchestrator.execute_mission(
        objective="Build an AI dashboard website",
        required_agents=["programming", "task_planning", "cybersecurity"],
    )
    assert mission.all_succeeded is True
    assert len(mission.results) == 3
    assert "Multi-Agent Mission Synthesis" in mission.synthesized_summary
    assert "ProgrammingAgent" in mission.synthesized_summary


# ── 4. FastAPI Agent Endpoints Tests ────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_agents_list(client):
    """GET /agents/list should return 15 agents."""
    res = client.get("/agents/list")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 15


def test_api_agents_orchestrate(client):
    """POST /agents/orchestrate should coordinate multiple agents."""
    res = client.post(
        "/agents/orchestrate",
        json={"objective": "Calculate system resource projections and formulate steps", "agents": ["mathematical", "task_planning"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["all_succeeded"] is True
    assert len(data["results"]) == 2
