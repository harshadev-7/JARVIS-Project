"""JARVIS Phase 28 — Asynchronous System Event Bus
==================================================
High-performance, decoupled event distribution engine allowing cross-subsystem
communication, telemetry collection, and reactive triggers without direct coupling.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class SystemEventType(str, Enum):
    # Lifecycle
    SYSTEM_BOOT = "system.boot"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    # Task & Orchestration
    TASK_DISPATCHED = "task.dispatched"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    CIRCUIT_TRIPPED = "circuit.tripped"
    CIRCUIT_RECOVERED = "circuit.recovered"
    CONSENSUS_REACHED = "consensus.reached"
    
    # Memory & Learning
    MEMORY_INDEXED = "memory.indexed"
    FEEDBACK_STAGED = "feedback.staged"
    FEEDBACK_APPROVED = "feedback.approved"
    MODEL_ACTIVATED = "model.activated"
    
    # Security & Devices
    SECURITY_ALERT = "security.alert"
    DEVICE_PAIRED = "device.paired"
    MEDIA_EXPORTED = "media.exported"
    
    # Diagnostics & Health
    HEALTH_PROBED = "health.probed"
    DIAGNOSTICS_COMPLETED = "diagnostics.completed"
    
    # Sentinel & Telemetry
    PROACTIVE_ADVISORY = "sentinel.advisory"
    MAINTENANCE_COMPLETED = "sentinel.maintenance"
    TELEMETRY_SAMPLE = "sentinel.telemetry"


@dataclass
class SystemEvent:
    """Standardized event packet broadcast across the JARVIS system."""
    event_type: Union[SystemEventType, str]
    source_subsystem: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        etype = self.event_type.value if isinstance(self.event_type, SystemEventType) else str(self.event_type)
        return {
            "event_id": self.event_id,
            "event_type": etype,
            "source_subsystem": self.source_subsystem,
            "payload": self.payload,
            "timestamp": round(self.timestamp, 3),
        }


class SystemEventBus:
    """Decoupled asynchronous event broker."""

    def __init__(self, max_history: int = 100):
        self._subscribers: Dict[str, List[Callable[[SystemEvent], Any]]] = {}
        self._history: List[SystemEvent] = []
        self._max_history = max_history

    def subscribe(self, event_type: Union[SystemEventType, str], handler: Callable[[SystemEvent], Any]) -> None:
        """Subscribes a callable handler to an event type (or '*' for all events)."""
        key = event_type.value if isinstance(event_type, SystemEventType) else str(event_type)
        self._subscribers.setdefault(key, []).append(handler)
        logger.debug("Handler registered for event type '%s'.", key)

    def unsubscribe(self, event_type: Union[SystemEventType, str], handler: Callable[[SystemEvent], Any]) -> bool:
        """Removes a registered handler."""
        key = event_type.value if isinstance(event_type, SystemEventType) else str(event_type)
        if key in self._subscribers and handler in self._subscribers[key]:
            self._subscribers[key].remove(handler)
            return True
        return False

    async def publish(self, event: SystemEvent) -> int:
        """Broadcasts an event asynchronously to all matching subscribers."""
        key = event.event_type.value if isinstance(event.event_type, SystemEventType) else str(event.event_type)
        
        # Append to circular ring buffer
        self._history.insert(0, event)
        if len(self._history) > self._max_history:
            self._history = self._history[:self._max_history]

        handlers = list(self._subscribers.get(key, []))
        # Add wildcard subscribers
        handlers.extend(self._subscribers.get("*", []))

        dispatched_count = 0
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                dispatched_count += 1
            except Exception as exc:
                logger.warning("Event subscriber error for event '%s': %s", key, exc)

        return dispatched_count

    def publish_sync(self, event: SystemEvent) -> int:
        """Synchronous wrapper for non-async calling contexts."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.publish(event))
            return 1
        except RuntimeError:
            return asyncio.run(self.publish(event))

    def get_history(self, limit: int = 20, event_type: Optional[str] = None) -> List[SystemEvent]:
        """Returns recent broadcast events."""
        if event_type:
            filtered = [e for e in self._history if (e.event_type.value if isinstance(e.event_type, SystemEventType) else e.event_type) == event_type]
            return filtered[:limit]
        return self._history[:limit]

    def clear(self) -> None:
        """Clears subscriptions and history (useful for test isolation)."""
        self._subscribers.clear()
        self._history.clear()


# Global singleton event bus
event_bus = SystemEventBus()
