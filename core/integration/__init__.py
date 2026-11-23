"""JARVIS Phase 28 — Complete System Integration Core
===================================================
Unifies all 28 project phases into an end-to-end Master Runtime:
- Subsystem Registry & Health Probes
- Asynchronous System Event Bus
- Automated System Diagnostics & Self-Test Runner
- Master Runtime & Daemon Lifecycle Coordinator
"""

from core.integration.subsystem_registry import (
    SubsystemStatus,
    SubsystemInfo,
    SubsystemRegistry,
    subsystem_registry,
)
from core.integration.event_bus import (
    SystemEventType,
    SystemEvent,
    SystemEventBus,
    event_bus,
)
from core.integration.system_diagnostics import (
    DiagnosticCheckResult,
    DiagnosticsReport,
    SystemDiagnostics,
    system_diagnostics,
)
from core.integration.master_runtime import (
    RuntimeState,
    JARVISMasterRuntime,
    master_runtime,
)

__all__ = [
    "SubsystemStatus",
    "SubsystemInfo",
    "SubsystemRegistry",
    "subsystem_registry",
    "SystemEventType",
    "SystemEvent",
    "SystemEventBus",
    "event_bus",
    "DiagnosticCheckResult",
    "DiagnosticsReport",
    "SystemDiagnostics",
    "system_diagnostics",
    "RuntimeState",
    "JARVISMasterRuntime",
    "master_runtime",
]
