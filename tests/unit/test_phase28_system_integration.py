"""JARVIS Phase 28 — Complete System Integration Unit Tests
============================================================
Validates:
1. Subsystem Registry cataloging all 28 project phases.
2. Health probing across all registered subsystems.
3. Asynchronous System Event Bus publish/subscribe and wildcard delivery.
4. Automated System Diagnostics smoke tests across intelligence, memory, and orchestration.
5. Master Runtime boot, status reporting, and graceful shutdown lifecycle.
6. Brain router diagnostics intent classification.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.brain.router import IntentRouter, IntentType
from core.integration import (
    DiagnosticCheckResult,
    DiagnosticsReport,
    JARVISMasterRuntime,
    RuntimeState,
    SubsystemInfo,
    SubsystemRegistry,
    SubsystemStatus,
    SystemDiagnostics,
    SystemEvent,
    SystemEventBus,
    SystemEventType,
    event_bus,
    master_runtime,
    subsystem_registry,
    system_diagnostics,
)


# 1. Subsystem Registry Tests
def test_subsystem_registry_catalog():
    """Verify all 28 functional subsystems are cataloged across project phases."""
    registry = SubsystemRegistry()
    all_subsystems = registry.list_subsystems()

    assert len(all_subsystems) == 28, f"Expected 28 subsystems, found {len(all_subsystems)}"

    phases = {s.phase for s in all_subsystems}
    assert 1 in phases
    assert 19 in phases
    assert 26 in phases
    assert 27 in phases
    assert 28 in phases

    categories = {s.category for s in all_subsystems}
    expected_categories = {"intelligence", "reasoning", "memory", "operational", "security", "media", "orchestration", "runtime"}
    assert expected_categories.issubset(categories)


def test_subsystem_registry_probing():
    """Verify health probing calculates accurate health percentages."""
    registry = SubsystemRegistry()
    probe = registry.probe_all_health()

    assert probe["total_subsystems"] == 28
    assert probe["healthy_count"] == 28
    assert probe["overall_status"] == "healthy"
    assert probe["health_score_pct"] == 100.0


# 2. System Event Bus Tests
@pytest.mark.asyncio
async def test_system_event_bus_pub_sub():
    """Verify typed publish/subscribe message delivery."""
    bus = SystemEventBus()
    received_events = []

    async def async_handler(event: SystemEvent):
        received_events.append(event)

    bus.subscribe(SystemEventType.TASK_COMPLETED, async_handler)

    test_event = SystemEvent(
        event_type=SystemEventType.TASK_COMPLETED,
        source_subsystem="test_runner",
        payload={"task_id": "test_1", "duration": 0.45},
    )

    dispatched = await bus.publish(test_event)
    assert dispatched == 1
    assert len(received_events) == 1
    assert received_events[0].event_type == SystemEventType.TASK_COMPLETED
    assert received_events[0].payload["task_id"] == "test_1"


@pytest.mark.asyncio
async def test_system_event_bus_wildcard_subscription():
    """Verify wildcard '*' listener receives all event types."""
    bus = SystemEventBus()
    wildcard_events = []

    def wildcard_handler(event: SystemEvent):
        wildcard_events.append(event)

    bus.subscribe("*", wildcard_handler)

    await bus.publish(SystemEvent(event_type=SystemEventType.SYSTEM_BOOT, source_subsystem="core"))
    await bus.publish(SystemEvent(event_type=SystemEventType.CIRCUIT_TRIPPED, source_subsystem="orchestrator"))
    await bus.publish(SystemEvent(event_type=SystemEventType.SECURITY_ALERT, source_subsystem="security"))

    assert len(wildcard_events) == 3
    event_types = [e.event_type for e in wildcard_events]
    assert SystemEventType.SYSTEM_BOOT in event_types
    assert SystemEventType.CIRCUIT_TRIPPED in event_types
    assert SystemEventType.SECURITY_ALERT in event_types


# 3. System Diagnostics & Self-Test Tests
@pytest.mark.asyncio
async def test_system_diagnostics_smoke_tests():
    """Verify automated diagnostic smoke tests across all 28 subsystems."""
    diagnostics = SystemDiagnostics()
    report: DiagnosticsReport = await diagnostics.run_full_diagnostics()

    assert isinstance(report, DiagnosticsReport)
    assert report.total_checks >= 10
    assert report.passed_count == report.total_checks
    assert report.failed_count == 0
    assert report.pass_rate_pct == 100.0
    assert report.overall_healthy is True

    # Check terminal formatting
    summary_text = report.format_terminal_summary()
    assert "JARVIS System Self-Diagnostics Report" in summary_text
    assert "100.0% (" in summary_text


# 4. Master Runtime Lifecycle Tests
@pytest.mark.asyncio
async def test_master_runtime_lifecycle():
    """Verify Master Runtime boot, status inspection, and graceful shutdown."""
    runtime = JARVISMasterRuntime()
    assert runtime.state == RuntimeState.STOPPED
    assert runtime.uptime_sec == 0.0

    # 1. Boot Sequence
    boot_info = await runtime.boot()
    assert runtime.state == RuntimeState.RUNNING
    assert boot_info["status"] == "online"
    assert runtime.uptime_sec >= 0.0

    # 2. Status check
    status = runtime.get_status()
    assert status["runtime_state"] == "running"
    assert status["health_summary"]["total_subsystems"] == 28

    # 3. Diagnostics invocation through runtime
    diag_rep = await runtime.run_diagnostics()
    assert diag_rep.overall_healthy is True

    # 4. Shutdown Sequence
    shutdown_info = await runtime.shutdown()
    assert runtime.state == RuntimeState.STOPPED
    assert shutdown_info["status"] == "stopped"
    assert runtime.uptime_sec == 0.0


# 5. Intent Router Diagnostics Intent Test
def test_router_diagnostics_intent():
    """Verify IntentRouter routes /diagnostics, /system-health, and /self-test."""
    router = IntentRouter()

    dec1 = router.route("/diagnostics")
    assert dec1.intent == IntentType.STATUS_DIAGNOSTICS
    assert dec1.agent_name == "SystemDiagnostics"

    dec2 = router.route("/system-health")
    assert dec2.intent == IntentType.STATUS_DIAGNOSTICS

    dec3 = router.route("/self-test")
    assert dec3.intent == IntentType.STATUS_DIAGNOSTICS

    dec4 = router.route("run system health check and diagnostics")
    assert dec4.intent == IntentType.STATUS_DIAGNOSTICS
