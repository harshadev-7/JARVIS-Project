"""JARVIS Phase 29 — Holographic Command Center & Sentinel Daemon Unit Tests
=============================================================================
Validates:
1. Proactive advisory heuristics and structured alert generation.
2. Autonomous self-healing maintenance and cache pruning.
3. JarvisSentinelDaemon lifecycle, single cycle execution, and event dispatch.
4. Subsystem registry inclusion of Phase 29 Sentinel subsystem.
5. Brain router classification of /sentinel, /maintenance, and /subsystems.
6. REST API surface: /system/subsystems, /system/sentinel, and /system/runtime.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

from api.app import app
from core.brain.router import IntentRouter, IntentType
from core.integration import (
    JARVISMasterRuntime,
    SubsystemRegistry,
    event_bus,
    subsystem_registry,
)
from core.sentinel import (
    AdvisorySeverity,
    AutonomousMaintenance,
    JarvisSentinelDaemon,
    MaintenanceResult,
    ProactiveAdvisory,
    ProactiveAdvisoryEngine,
    autonomous_maintenance,
    proactive_advisory_engine,
    sentinel_daemon,
)


# 1. Proactive Advisory Engine Tests
@pytest.mark.asyncio
async def test_proactive_advisory_engine():
    """Verify advisory engine generates structured alerts under controlled conditions."""
    engine = ProactiveAdvisoryEngine(ram_threshold_pct=0.01, cpu_threshold_pct=0.01)
    advisories = await engine.evaluate_heuristics()

    assert len(advisories) >= 1
    adv = advisories[0]
    assert isinstance(adv, ProactiveAdvisory)
    assert adv.id.startswith("adv-")
    assert adv.severity in {AdvisorySeverity.INFO, AdvisorySeverity.RECOMMENDATION, AdvisorySeverity.WARNING}
    
    d = adv.to_dict()
    assert "timestamp" in d
    assert "title" in d
    assert "category" in d


# 2. Autonomous Maintenance Engine Tests
@pytest.mark.asyncio
async def test_autonomous_maintenance(tmp_path):
    """Verify autonomous hygiene prunes old temp files and checks DB integrity."""
    test_temp = tmp_path / "temp_media"
    test_temp.mkdir(parents=True, exist_ok=True)
    
    # Create dummy temporary files
    dummy1 = test_temp / "clip_01.tmp"
    dummy1.write_text("temporary video frame buffer")
    dummy2 = test_temp / "sample.part"
    dummy2.write_text("download chunk")

    maint = AutonomousMaintenance(temp_dir=test_temp)
    result = await maint.run_maintenance(max_file_age_hours=0.0)

    assert isinstance(result, MaintenanceResult)
    assert result.files_cleaned >= 2
    assert result.bytes_freed > 0
    assert result.db_integrity_ok is True
    assert not dummy1.exists()
    assert not dummy2.exists()


# 3. Sentinel Daemon Lifecycle Tests
@pytest.mark.asyncio
async def test_sentinel_daemon_cycle():
    """Verify sentinel daemon start, cycle execution, and clean shutdown."""
    daemon = JarvisSentinelDaemon(check_interval_sec=0.1)
    assert daemon.is_running is False

    cycle_info = await daemon.run_single_cycle()
    assert cycle_info["cycle"] == 1
    assert "advisories_detected" in cycle_info
    assert "maintenance_ran" in cycle_info

    # Test status payload
    status = daemon.get_status()
    assert status["cycle_count"] == 1
    assert "recent_advisories" in status
    assert status["check_interval_sec"] == 0.1

    # Test manual maintenance invocation through daemon
    m_res = await daemon.run_maintenance_now()
    assert isinstance(m_res, MaintenanceResult)
    assert daemon.get_status()["last_maintenance"] is not None


# 4. Subsystem Registry Phase 29 Tests
def test_subsystem_registry_phase29():
    """Verify Sentinel daemon is cataloged in the unified SubsystemRegistry."""
    registry = SubsystemRegistry()
    sentinel_sub = registry.get_subsystem("sentinel_proactive_daemon")

    assert sentinel_sub is not None
    assert sentinel_sub.phase == 29
    assert sentinel_sub.category == "runtime"
    assert "Sentinel" in sentinel_sub.name

    # Total registered subsystems must be >= 28
    all_subs = registry.list_subsystems()
    assert len(all_subs) >= 28


# 5. Intent Router Sentinel Intent Tests
def test_router_sentinel_intent():
    """Verify IntentRouter routes /sentinel, /maintenance, and /subsystems."""
    router = IntentRouter()

    dec1 = router.route("/sentinel")
    assert dec1.intent == IntentType.SENTINEL_OPERATION
    assert dec1.agent_name == "SentinelDaemon"

    dec2 = router.route("/maintenance")
    assert dec2.intent == IntentType.SENTINEL_OPERATION

    dec3 = router.route("/subsystems")
    assert dec3.intent == IntentType.SENTINEL_OPERATION

    dec4 = router.route("show sentinel status and proactive advisories")
    assert dec4.intent == IntentType.SENTINEL_OPERATION


# 6. REST API Endpoints Tests
def test_rest_api_endpoints_phase29():
    """Verify FastAPI endpoints for subsystems, sentinel, and runtime."""
    with TestClient(app) as client:
        # Test /system/subsystems
        res_sub = client.get("/system/subsystems")
        assert res_sub.status_code == 200
        data_sub = res_sub.json()
        assert data_sub["total"] >= 28
        assert "health_summary" in data_sub
        assert any(s["subsystem_id"] == "sentinel_proactive_daemon" for s in data_sub["subsystems"])

        # Test /system/sentinel
        res_sen = client.get("/system/sentinel")
        assert res_sen.status_code == 200
        data_sen = res_sen.json()
        assert "is_running" in data_sen
        assert "cycle_count" in data_sen

        # Test /system/runtime
        res_rt = client.get("/system/runtime")
        assert res_rt.status_code == 200
        data_rt = res_rt.json()
        assert "runtime_state" in data_rt
        assert "health_summary" in data_rt
