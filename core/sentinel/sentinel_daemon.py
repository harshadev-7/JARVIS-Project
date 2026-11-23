"""JARVIS Phase 29 — Autonomous Sentinel Daemon
===============================================
Background supervisor continuously guarding system resources, generating
proactive advisories, and performing autonomous hygiene routines.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from core.integration.event_bus import SystemEvent, SystemEventType, event_bus
from core.sentinel.advisory_engine import (
    AdvisorySeverity,
    ProactiveAdvisory,
    ProactiveAdvisoryEngine,
    proactive_advisory_engine,
)
from core.sentinel.maintenance_engine import (
    AutonomousMaintenance,
    MaintenanceResult,
    autonomous_maintenance,
)

logger = logging.getLogger(__name__)


class JarvisSentinelDaemon:
    """Continuous background sentinel and proactive intelligence daemon."""

    def __init__(
        self,
        advisory_engine: Optional[ProactiveAdvisoryEngine] = None,
        maintenance_engine: Optional[AutonomousMaintenance] = None,
        check_interval_sec: float = 15.0,
        maintenance_interval_sec: float = 300.0,
    ):
        self.advisory_engine = advisory_engine or proactive_advisory_engine
        self.maintenance_engine = maintenance_engine or autonomous_maintenance
        self.check_interval_sec = check_interval_sec
        self.maintenance_interval_sec = maintenance_interval_sec

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_cycle_time: float = 0.0
        self._last_maintenance_time: float = 0.0
        self._cycle_count: int = 0
        self._recent_advisories: List[ProactiveAdvisory] = []
        self._last_maintenance_result: Optional[MaintenanceResult] = None
        self._active_advisory_ids: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> Dict[str, Any]:
        """Launches the sentinel daemon background task."""
        if self._running:
            return {"status": "already_running", "cycle_count": self._cycle_count}

        self._running = True
        self._task = asyncio.create_task(self._daemon_loop())
        logger.info("JarvisSentinelDaemon started with interval %.1fs", self.check_interval_sec)
        return {"status": "started", "interval_sec": self.check_interval_sec}

    async def stop(self) -> Dict[str, Any]:
        """Gracefully halts the sentinel daemon."""
        if not self._running:
            return {"status": "already_stopped"}

        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("JarvisSentinelDaemon halted cleanly.")
        return {"status": "stopped", "cycle_count": self._cycle_count}

    async def _daemon_loop(self) -> None:
        """Continuous execution loop."""
        while self._running:
            try:
                await self.run_single_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in sentinel daemon cycle: %s", exc)

            try:
                await asyncio.sleep(self.check_interval_sec)
            except asyncio.CancelledError:
                break

    async def run_single_cycle(self) -> Dict[str, Any]:
        """Runs a single inspection and proactive advisory evaluation cycle."""
        now = time.time()
        self._last_cycle_time = now
        self._cycle_count += 1

        # 1. Evaluate proactive heuristics
        new_advisories = await self.advisory_engine.evaluate_heuristics()
        dispatched_count = 0

        for adv in new_advisories:
            # Avoid re-spamming identical advisory id within a single session
            if adv.id not in self._active_advisory_ids:
                self._active_advisory_ids.add(adv.id)
                self._recent_advisories.append(adv)
                if len(self._recent_advisories) > 50:
                    self._recent_advisories.pop(0)

                # Broadcast advisory across SystemEventBus
                await event_bus.publish(
                    SystemEvent(
                        event_type=SystemEventType.PROACTIVE_ADVISORY,
                        source_subsystem="sentinel_daemon",
                        payload=adv.to_dict(),
                    )
                )
                dispatched_count += 1

        # 2. Check if maintenance is due
        ran_maintenance = False
        if (now - self._last_maintenance_time) >= self.maintenance_interval_sec:
            m_res = await self.run_maintenance_now()
            ran_maintenance = True

        return {
            "timestamp": now,
            "cycle": self._cycle_count,
            "advisories_detected": len(new_advisories),
            "new_advisories_broadcast": dispatched_count,
            "maintenance_ran": ran_maintenance,
        }

    async def run_maintenance_now(self) -> MaintenanceResult:
        """Executes immediate routine maintenance and broadcasts result."""
        res = await self.maintenance_engine.run_maintenance()
        self._last_maintenance_time = time.time()
        self._last_maintenance_result = res

        await event_bus.publish(
            SystemEvent(
                event_type=SystemEventType.MAINTENANCE_COMPLETED,
                source_subsystem="sentinel_daemon",
                payload=res.to_dict(),
            )
        )
        return res

    def get_status(self) -> Dict[str, Any]:
        """Provides status summary of the sentinel daemon."""
        return {
            "is_running": self._running,
            "cycle_count": self._cycle_count,
            "check_interval_sec": self.check_interval_sec,
            "last_cycle_timestamp": round(self._last_cycle_time, 2),
            "recent_advisories_count": len(self._recent_advisories),
            "recent_advisories": [a.to_dict() for a in self._recent_advisories[-5:]],
            "last_maintenance": (
                self._last_maintenance_result.to_dict() if self._last_maintenance_result else None
            ),
        }


sentinel_daemon = JarvisSentinelDaemon()
