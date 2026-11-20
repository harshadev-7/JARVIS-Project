"""
JARVIS AI — Cybersecurity Audit Logger
=======================================
Maintains an immutable, structured audit log of all security events:
authentication attempts, permission evaluations, remote executions,
and emergency shutdowns.
"""

import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from pathlib import Path

from config.settings import BASE_DIR
from core.logger import log


class SecurityAuditEvent(BaseModel):
    """An individual security audit trail entry."""
    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str  # e.g. "admin", "device_pixel", "system"
    action: str  # e.g. "terminal_command", "pair_device", "emergency_shutdown"
    resource: str
    granted: bool
    risk_level: str = "low"  # "low", "medium", "high", "critical"
    details: Dict[str, Any] = Field(default_factory=dict)
    event_hash: str = ""

    def compute_hash(self, prev_hash: str = "") -> str:
        payload = f"{self.event_id}:{self.timestamp.isoformat()}:{self.actor}:{self.action}:{self.granted}:{prev_hash}"
        return hashlib.sha256(payload.encode()).hexdigest()


class AuditLogger:
    """Records tamper-evident security audit events."""

    def __init__(self):
        self._audit_trail: List[SecurityAuditEvent] = []
        self._last_hash = "0000000000000000"
        self.log_file = BASE_DIR / "logs" / "audit.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        actor: str,
        action: str,
        resource: str,
        granted: bool,
        risk_level: str = "low",
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityAuditEvent:
        """Create and append an audit event."""
        eid = f"evt-{len(self._audit_trail) + 1:05d}"
        event = SecurityAuditEvent(
            event_id=eid,
            actor=actor,
            action=action,
            resource=resource,
            granted=granted,
            risk_level=risk_level,
            details=details or {},
        )
        event.event_hash = event.compute_hash(self._last_hash)
        self._last_hash = event.event_hash
        self._audit_trail.append(event)

        # Append to audit file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception as e:
            log.error(f"Audit log write failure: {e}")

        return event

    def get_recent_events(self, limit: int = 50) -> List[SecurityAuditEvent]:
        return self._audit_trail[-limit:]


audit_logger = AuditLogger()
