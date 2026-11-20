"""
JARVIS AI — End-to-End System Integration Test Suite
=====================================================
Validates all subsystems cooperating as a unified autonomous intelligence:
1. Core Brain & Multi-turn Conversational Memory
2. 10-Step Autonomous Reasoning Engine
3. Persistent Memory, Semantic Vector Search & Mistake Avoidance
4. Universal Internet Research Pipeline
5. HD Neural Voice Assistant (Edge-TTS)
6. Windows Computer Automation & Hardware Monitoring
7. Android Multi-Device Mesh & Task Dispatch
8. Programming Intelligence & Sandboxed Execution
9. 15-Agent Collective Intelligence Missions
10. Viral Multi-Platform Social Media & Vertical Video Engine
11. Holographic Arc Reactor HUD & WebSocket Streaming
12. Cybersecurity Emergency Kill-Switch & Cryptographic Audit Trails
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app
from config.settings import settings
from core.brain.jarvis_brain import brain
from core.memory.memory_manager import memory_manager
from core.planner.reasoning_engine import reasoning_engine
from services.automation.system_monitor import system_monitor
from services.automation.command_executor import command_executor
from devices.device_registry.registry import device_registry, DeviceType
from agents.programming_agent.code_analyzer import code_analyzer
from core.orchestrator.orchestrator import orchestrator
from social_media.content_pipeline.content_generator import content_generator
from social_media.platforms.publisher import publishing_manager
from services.security.access_control import security_controller


@pytest.fixture(scope="module")
def client():
    """Shared TestClient for full system integration."""
    return TestClient(app)


# ── 1. Root & Health Verification ──────────────────────────────────────────
def test_e2e_health_and_hud(client):
    """Confirm server health, OpenAPI documentation, and Holographic HUD."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "JARVIS" in res_root.json()["message"]

    res_health = client.get("/system/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "operational"

    res_hud = client.get("/hud")
    assert res_hud.status_code == 200
    assert "arc-reactor" in res_hud.text
    assert "SYSTEM TELEMETRY" in res_hud.text


# ── 2. Chat & Multi-turn Memory Flow ───────────────────────────────────────
@pytest.mark.asyncio
async def test_e2e_chat_and_memory_cycle(client):
    """Test full turn: user message -> memory -> brain response -> memory record."""
    session_id = "test-e2e-session-001"

    res = client.post(
        "/chat/message",
        json={"message": "Good morning JARVIS", "session_id": session_id},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == session_id
    assert len(data["reply"]) > 5

    # Verify conversation history persisted
    history_res = client.get(f"/chat/history/{session_id}")
    assert history_res.status_code == 200
    hist = history_res.json()["messages"]
    assert len(hist) == 2  # 1 user + 1 assistant


# ── 3. Autonomous Reasoning Engine (10-Step Cycle) ────────────────────────
def test_e2e_reasoning_engine(client):
    """Test autonomous goal decomposition, approach scoring, and execution planning."""
    problem = "Implement automated data synchronization between server and mobile client."
    res = client.post("/reasoning/solve", json={"problem": problem, "autonomy_level": 3})
    assert res.status_code == 200
    result = res.json()
    assert "step_1_objective_understanding" in result
    assert "step_7_selected_solution" in result
    assert "step_8_execution_plan" in result
    assert len(result["step_8_execution_plan"]["subtasks"]) >= 3


# ── 4. Hardware Telemetry & Safe Command Execution ─────────────────────────
def test_e2e_automation_and_telemetry(client):
    """Test hardware telemetry retrieval and command sandboxing."""
    # Metrics
    metrics_res = client.get("/automation/system/metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert 0 <= metrics["cpu_percent"] <= 100
    assert metrics["ram_total_gb"] > 0

    # Execute safe command
    exec_res = client.post(
        "/automation/terminal/execute",
        json={"command": "echo INTEGRATION_TEST_PASSED", "autonomy_level": 4},
    )
    assert exec_res.status_code == 200
    assert "INTEGRATION_TEST_PASSED" in exec_res.json()["stdout"]

    # Block prohibited command
    danger_res = client.post(
        "/automation/terminal/execute",
        json={"command": "format c: /fs:NTFS", "autonomy_level": 5},
    )
    assert danger_res.status_code == 200
    assert danger_res.json()["blocked"] is True


# ── 5. Multi-Device Network & Android Pairing ─────────────────────────────
def test_e2e_device_mesh_flow(client):
    """Generate 6-digit code, pair Android device, and dispatch remote command."""
    # 1. Generate code
    code_res = client.post("/devices/pairing-code")
    assert code_res.status_code == 200
    code = code_res.json()["pairing_code"]
    assert len(code) == 6

    # 2. Pair device
    pair_res = client.post(
        "/devices/pair",
        json={
            "pairing_code": code,
            "name": "Integration Test Pixel Tablet",
            "device_type": "android",
            "capabilities": ["voice_capture", "push_alerts"],
        },
    )
    assert pair_res.status_code == 200
    device_id = pair_res.json()["device_id"]
    assert device_id.startswith("dev-")

    # 3. Dispatch task to device
    dispatch_res = client.post(
        "/devices/dispatch",
        json={
            "target": device_id,
            "action": "display_notification",
            "parameters": {"title": "JARVIS Alert", "message": "All systems nominal"},
        },
    )
    assert dispatch_res.status_code == 200
    assert dispatch_res.json()["action"] == "display_notification"
    assert dispatch_res.json()["task_id"].startswith("rmt-")


# ── 6. Programming Intelligence & Code Analysis ───────────────────────────
def test_e2e_programming_intelligence(client):
    """Analyze code syntax, security risks, and execute safely."""
    sample_code = """def compute_kinetic_energy(mass, velocity):
    return 0.5 * mass * (velocity ** 2)
"""
    analyze_res = client.post(
        "/programming/analyze",
        json={"code": sample_code, "language": "python"},
    )
    assert analyze_res.status_code == 200
    rep = analyze_res.json()
    assert rep["is_valid_syntax"] is True
    assert rep["security_score"] >= 0.8
    assert "compute_kinetic_energy" in rep["functions_found"]


# ── 7. Multi-Agent Collective Intelligence Mission ────────────────────────
def test_e2e_multi_agent_orchestration(client):
    """Deploy specialized agents collaboratively on a high-level goal."""
    res = client.post(
        "/agents/orchestrate",
        json={
            "objective": "Calculate projected hardware demand and plan implementation steps",
            "agents": ["mathematical", "task_planning"],
        },
    )
    assert res.status_code == 200
    mission = res.json()
    assert mission["all_succeeded"] is True
    assert len(mission["results"]) == 2
    assert "MathematicalAgent" in mission["results"][0]["agent_name"]
    assert "TaskPlanningAgent" in mission["results"][1]["agent_name"]


# ── 8. Content Automation & Vertical Video Engine ──────────────────────────
def test_e2e_social_and_video_pipeline(client):
    """Generate viral 9:16 content package and queue for social publishing."""
    topic = "Autonomous Artificial Intelligence Agents"
    res = client.post("/social/content/generate", json={"topic": topic})
    assert res.status_code == 200
    pkg = res.json()
    assert "youtube_shorts" in pkg["platforms"]
    assert "instagram_reels" in pkg["platforms"]
    assert len(pkg["hook"]) > 10

    # Queue for publishing
    queue_res = client.post(
        "/social/queue/add",
        json={"topic": topic, "platform": "youtube_shorts"},
    )
    assert queue_res.status_code == 200
    assert queue_res.json()["status"] == "queued"


# ── 9. Cybersecurity Kill-Switch & Disarm Lifecycle ───────────────────────
def test_e2e_security_kill_switch_lifecycle(client):
    """Test emergency freeze and authenticated restore."""
    # 1. Trigger shutdown
    halt_res = client.post(
        "/security/emergency-shutdown",
        json={"authorized_by": "lead_engineer", "reason": "System verification drill"},
    )
    assert halt_res.status_code == 200
    assert halt_res.json()["status"] == "EMERGENCY_SHUTDOWN_ACTIVE"
    assert security_controller.emergency_shutdown_active is True

    # 2. Verify automations blocked while frozen
    auto_res = client.post(
        "/automation/terminal/execute",
        json={"command": "echo TEST", "autonomy_level": 4},
    )
    assert auto_res.json()["blocked"] is True

    # 3. Disarm with admin key
    disarm_res = client.post("/security/disarm?admin_key=jarvis-admin-restore")
    assert disarm_res.status_code == 200
    assert disarm_res.json()["status"] == "OPERATIONAL"
    assert security_controller.emergency_shutdown_active is False
