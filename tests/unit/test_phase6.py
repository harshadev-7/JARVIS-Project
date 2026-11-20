"""
JARVIS AI — Phase 6 Test Suite
================================
Tests for Windows Computer Automation:
- System performance monitoring (CPU, RAM, Disk, Network)
- Application control and autonomy permissions
- Safe terminal command execution & danger blacklist enforcement
- FastAPI automation endpoints
"""

import pytest
from fastapi.testclient import TestClient

from services.automation.system_monitor import system_monitor, SystemMetrics
from services.automation.app_controller import app_controller
from services.automation.command_executor import command_executor
from api.app import app


# ── 1. System Monitor Tests ─────────────────────────────────────────────────

def test_system_monitor_metrics():
    """Monitor should return positive physical metrics."""
    metrics = system_monitor.get_metrics()
    assert isinstance(metrics, SystemMetrics)
    assert metrics.cpu_percent >= 0.0
    assert metrics.cpu_count >= 1
    assert metrics.ram_total_gb > 0.0
    assert metrics.disk_total_gb > 0.0


# ── 2. App Controller Tests ─────────────────────────────────────────────────

def test_app_controller_permission_gate():
    """Autonomy Level 1 should be blocked from launching desktop apps."""
    res = app_controller.launch_app("notepad", autonomy_level=1)
    assert res.success is False
    assert "Permission Denied" in res.message


# ── 3. Safe Command Executor Tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_command_executor_safe_run():
    """Safe commands like echo should execute and capture stdout."""
    res = await command_executor.execute("echo JARVIS_ACTIVE", autonomy_level=4)
    assert res.exit_code == 0
    assert "JARVIS_ACTIVE" in res.stdout
    assert res.blocked is False


@pytest.mark.asyncio
async def test_command_executor_blacklist_blocking():
    """Prohibited destructive commands must be blocked immediately."""
    res = await command_executor.execute("del /f /s /q c:\\", autonomy_level=5)
    assert res.blocked is True
    assert "prohibited pattern" in res.stderr.lower() or "blocked" in res.stderr.lower()


@pytest.mark.asyncio
async def test_command_executor_autonomy_gate():
    """Autonomy Level 1 should be prevented from executing shell commands."""
    res = await command_executor.execute("echo Hello", autonomy_level=1)
    assert res.blocked is True
    assert "Permission" in res.stderr or "not permitted" in res.stderr


# ── 4. FastAPI Automation Endpoints Tests ───────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_system_metrics(client):
    """GET /automation/system/metrics should return real-time metrics."""
    res = client.get("/automation/system/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "cpu_percent" in data
    assert "ram_total_gb" in data


def test_api_terminal_execute(client):
    """POST /automation/terminal/execute should execute safe command."""
    res = client.post(
        "/automation/terminal/execute",
        json={"command": "echo SYSTEM_ONLINE", "autonomy_level": 4, "timeout_sec": 10.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["exit_code"] == 0
    assert "SYSTEM_ONLINE" in data["stdout"]
