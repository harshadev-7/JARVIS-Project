"""
JARVIS AI — Phase 12-14 Test Suite
====================================
Tests for Cloud Deployment, Holographic HUD UI, and Cybersecurity & Kill-Switch:
- Audit logger hash chaining
- Emergency kill-switch engagement & recovery
- Rate limiter token bucket
- Role-based access hierarchy
- FastAPI security and HUD routes
"""

import pytest
from fastapi.testclient import TestClient

from services.security.audit_logger import audit_logger
from services.security.access_control import security_controller, UserRole, RateLimiter
from api.app import app


# ── 1. Audit Logger Tests ───────────────────────────────────────────────────

def test_audit_logger_hash_chain():
    """Audit events should produce chained SHA256 hashes."""
    ev1 = audit_logger.log_event(
        actor="test_user",
        action="test_action_1",
        resource="system",
        granted=True,
    )
    ev2 = audit_logger.log_event(
        actor="test_user",
        action="test_action_2",
        resource="system",
        granted=True,
    )
    assert ev1.event_hash != ""
    assert ev2.event_hash != ""
    assert ev1.event_hash != ev2.event_hash


# ── 2. Security Controller & Kill-Switch Tests ──────────────────────────────

def test_emergency_kill_switch_lifecycle():
    """Emergency shutdown should activate alert and be disarmable with key."""
    # Engage
    security_controller.trigger_emergency_shutdown(authorized_by="security_officer")
    assert security_controller.emergency_shutdown_active is True

    # Disarm
    security_controller.reset_emergency_shutdown(authorized_by="admin")
    assert security_controller.emergency_shutdown_active is False


def test_rbac_hierarchy():
    """Admin should possess permissions over Operator, Auditor, and Readonly."""
    assert security_controller.check_access(UserRole.ADMIN, UserRole.OPERATOR) is True
    assert security_controller.check_access(UserRole.READONLY, UserRole.ADMIN) is False


def test_rate_limiter():
    """Rate limiter should reject requests exceeding maximum threshold."""
    limiter = RateLimiter(max_requests=3, time_window_sec=60.0)
    assert limiter.is_allowed("client-1") is True
    assert limiter.is_allowed("client-1") is True
    assert limiter.is_allowed("client-1") is True
    # 4th request within 60s should be blocked
    assert limiter.is_allowed("client-1") is False


# ── 3. FastAPI Endpoints Tests ──────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_hud_endpoint(client):
    """GET /hud should serve the futuristic holographic interface HTML."""
    res = client.get("/hud")
    assert res.status_code == 200
    assert "JARVIS AI // HUD" in res.text
    assert "arc-reactor" in res.text


def test_api_emergency_shutdown_flow(client):
    """POST /security/emergency-shutdown then disarm."""
    res = client.post(
        "/security/emergency-shutdown",
        json={"authorized_by": "master_user", "reason": "Integrity drill"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "EMERGENCY_SHUTDOWN_ACTIVE"

    # Disarm
    disarm_res = client.post("/security/disarm?admin_key=jarvis-admin-restore")
    assert disarm_res.status_code == 200
    assert disarm_res.json()["status"] == "OPERATIONAL"


def test_agents_status_endpoint(client):
    """GET /agents/status should return live startup metadata for the HUD roster."""
    res = client.get("/agents/status")
    assert res.status_code == 200
    payload = res.json()
    assert isinstance(payload, list)
    assert len(payload) >= 15
    first = payload[0]
    assert "id" in first
    assert "name" in first
    assert "status" in first
    assert first["status"] in {"ready", "standby", "active", "shielded"}


def test_system_autonomy_controls(client):
    """GET/POST /system/autonomy should expose and change the live autonomy level."""
    res = client.get("/system/autonomy")
    assert res.status_code == 200
    assert res.json()["autonomy_level"] in range(1, 6)

    set_res = client.post("/system/autonomy", json={"level": 4})
    assert set_res.status_code == 200
    assert set_res.json()["autonomy_level"] == 4

    reset_res = client.post("/system/autonomy", json={"level": 2})
    assert reset_res.status_code == 200
    assert reset_res.json()["autonomy_level"] == 2


def test_agent_missions_history_endpoint(client):
    """Mission orchestration should persist and expose a live mission ledger for the HUD."""
    mission_res = client.post(
        "/agents/orchestrate",
        json={"objective": "Assess mission readiness and system health", "agents": ["task_planning", "cybersecurity"]},
    )
    assert mission_res.status_code == 200
    mission_id = mission_res.json()["mission_id"]

    history_res = client.get("/agents/missions")
    assert history_res.status_code == 200
    payload = history_res.json()
    assert isinstance(payload, list)
    assert any(item.get("mission_id") == mission_id for item in payload)

    detail_res = client.get(f"/agents/missions/{mission_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["mission_id"] == mission_id
    assert detail_res.json()["objective"] == "Assess mission readiness and system health"


def test_agent_demo_mission_endpoint(client):
    """The HUD should be able to trigger a local demo mission with a single click."""
    res = client.post(
        "/agents/missions/demo",
        json={"objective": "Run a local readiness audit for the command interface", "agents": ["task_planning", "cybersecurity"]},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["objective"] == "Run a local readiness audit for the command interface"
    assert payload["status"] in {"running", "completed", "partial"}
    assert isinstance(payload.get("timeline", []), list)
    assert payload["timeline"]


def test_agent_orchestration_respects_autonomy_level(client):
    """The orchestrator should block missions that exceed the active autonomy policy."""
    level_res = client.post("/system/autonomy", json={"level": 1})
    assert level_res.status_code == 200

    blocked = client.post(
        "/agents/orchestrate",
        json={"objective": "Dispatch a public announcement campaign", "agents": ["communication"]},
    )
    assert blocked.status_code == 403
    assert "Autonomy" in blocked.json()["detail"]

    reset_res = client.post("/system/autonomy", json={"level": 2})
    assert reset_res.status_code == 200


def test_mission_status_tracking_endpoint(client):
    """Active mission telemetry should expose current status and execution snapshots to the HUD."""
    res = client.post(
        "/agents/orchestrate",
        json={"objective": "Track an active mission status snapshot", "agents": ["task_planning"]},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] in {"running", "completed"}
    assert isinstance(payload.get("agent_status", []), list)
    assert payload["agent_status"]
    assert isinstance(payload.get("timeline", []), list)
    assert payload["timeline"]

    active_res = client.get("/agents/missions/active")
    assert active_res.status_code == 200
    data = active_res.json()
    assert isinstance(data, list)
    assert any(item.get("mission_id") == payload["mission_id"] for item in data)
