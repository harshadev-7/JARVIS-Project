"""JARVIS Phase 28 — Master System Runtime & Daemon Coordinator
=============================================================
Central coordinator managing boot sequence, health monitoring, background
daemons, event distribution, and graceful shutdown across all 28 project phases.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.integration.event_bus import SystemEvent, SystemEventType, event_bus
from core.integration.subsystem_registry import SubsystemRegistry, subsystem_registry
from core.integration.system_diagnostics import DiagnosticsReport, system_diagnostics

logger = logging.getLogger(__name__)


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    BOOTING = "booting"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


class JARVISMasterRuntime:
    """Master coordinator unifying all 28 project subsystems."""

    def __init__(
        self,
        registry: Optional[SubsystemRegistry] = None,
    ):
        self.registry = registry or subsystem_registry
        self.state: RuntimeState = RuntimeState.STOPPED
        self.boot_time: Optional[float] = None
        self._background_tasks: List[asyncio.Task] = []
        self._is_shutdown = False

    @property
    def uptime_sec(self) -> float:
        """Returns uptime in seconds since boot."""
        if self.boot_time and self.state == RuntimeState.RUNNING:
            return max(0.0, time.time() - self.boot_time)
        return 0.0

    async def boot(self) -> Dict[str, Any]:
        """Executes the standard system boot sequence across all subsystems."""
        if self.state == RuntimeState.RUNNING:
            logger.info("JARVIS Master Runtime already running.")
            return self.get_status()

        logger.info("Initiating JARVIS Master Runtime boot sequence (v%s)...", settings.JARVIS_VERSION)
        self.state = RuntimeState.BOOTING
        t0 = time.time()

        # 1. Broadcast boot event
        await event_bus.publish(
            SystemEvent(
                event_type=SystemEventType.SYSTEM_BOOT,
                source_subsystem="master_system_runtime",
                payload={"version": settings.JARVIS_VERSION, "autonomy_level": settings.DEFAULT_AUTONOMY_LEVEL},
            )
        )

        # 2. Probe initial subsystem health
        health = self.registry.probe_all_health()

        # 3. Launch Sentinel Daemon (Phase 29)
        try:
            from core.sentinel import sentinel_daemon
            await sentinel_daemon.start()
        except Exception as exc:
            logger.warning("Could not start Sentinel Daemon: %s", exc)

        # 4. Mark running
        self.boot_time = time.time()
        self.state = RuntimeState.RUNNING
        boot_duration_ms = (time.time() - t0) * 1000

        logger.info(
            "JARVIS Master Runtime boot complete in %.1fms. Overall health: %s (%d/%d subsystems healthy).",
            boot_duration_ms,
            health["overall_status"],
            health["healthy_count"],
            health["total_subsystems"],
        )

        return {
            "status": "online",
            "state": self.state.value,
            "version": settings.JARVIS_VERSION,
            "boot_duration_ms": round(boot_duration_ms, 2),
            "health": health,
        }

    async def shutdown(self) -> Dict[str, Any]:
        """Executes graceful shutdown of active daemons, background tasks, and event bus."""
        if self.state in {RuntimeState.STOPPED, RuntimeState.SHUTTING_DOWN}:
            return {"status": "already_stopped"}

        logger.info("Initiating JARVIS Master Runtime shutdown sequence...")
        self.state = RuntimeState.SHUTTING_DOWN
        t0 = time.time()

        # 1. Broadcast shutdown event
        await event_bus.publish(
            SystemEvent(
                event_type=SystemEventType.SYSTEM_SHUTDOWN,
                source_subsystem="master_system_runtime",
                payload={"uptime_sec": self.uptime_sec},
            )
        )

        # 2. Halt Sentinel Daemon
        try:
            from core.sentinel import sentinel_daemon
            await sentinel_daemon.stop()
        except Exception as exc:
            logger.warning("Error stopping Sentinel Daemon: %s", exc)

        # 3. Cancel active background tasks
        cancelled_count = 0
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        self.state = RuntimeState.STOPPED
        self.boot_time = None
        shutdown_duration_ms = (time.time() - t0) * 1000

        logger.info("JARVIS Master Runtime shutdown complete in %.1fms.", shutdown_duration_ms)
        return {
            "status": "stopped",
            "state": self.state.value,
            "cancelled_tasks": cancelled_count,
            "shutdown_duration_ms": round(shutdown_duration_ms, 2),
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns consolidated runtime state, uptime, subsystem health, and sentinel summary."""
        health = self.registry.probe_all_health()
        recent_events = [e.to_dict() for e in event_bus.get_history(limit=5)]
        sentinel_info = {}
        try:
            from core.sentinel import sentinel_daemon
            sentinel_info = sentinel_daemon.get_status()
        except Exception:
            pass

        return {
            "runtime_state": self.state.value,
            "version": settings.JARVIS_VERSION,
            "uptime_sec": round(self.uptime_sec, 2),
            "health_summary": health,
            "sentinel": sentinel_info,
            "recent_events": recent_events,
        }

    async def run_diagnostics(self) -> DiagnosticsReport:
        """Executes full diagnostic smoke tests across all subsystems."""
        return await system_diagnostics.run_full_diagnostics()


# Global singleton master runtime
master_runtime = JARVISMasterRuntime()
