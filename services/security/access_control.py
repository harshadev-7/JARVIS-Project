"""
JARVIS AI — Access Control & Emergency Kill-Switch
===================================================
Role-Based Access Control (RBAC), API rate-limiting,
and instant Emergency Kill-Switch system shutdown.
"""

import time
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel

from services.security.audit_logger import audit_logger
from core.logger import log


class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    READONLY = "readonly"


class RateLimiter:
    """Token-bucket rate limiter per IP / client token."""

    def __init__(self, max_requests: int = 60, time_window_sec: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window_sec
        self._clients: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        timestamps = self._clients.get(client_id, [])

        # Prune expired
        timestamps = [t for t in timestamps if now - t < self.time_window]

        if len(timestamps) >= self.max_requests:
            return False

        timestamps.append(now)
        self._clients[client_id] = timestamps
        return True


class SecurityController:
    """Master security switchboard and emergency shutdown controller."""

    def __init__(self):
        self.emergency_shutdown_active = False
        self.rate_limiter = RateLimiter()

    def trigger_emergency_shutdown(self, authorized_by: str = "user") -> bool:
        """
        EMERGENCY KILL-SWITCH:
        Halts all running background automations, disconnects remote devices,
        and logs critical security event.
        """
        self.emergency_shutdown_active = True
        log.critical(f"EMERGENCY SHUTDOWN INITIATED BY: {authorized_by.upper()}!")

        audit_logger.log_event(
            actor=authorized_by,
            action="emergency_shutdown",
            resource="JARVIS_CORE_SYSTEM",
            granted=True,
            risk_level="critical",
            details={"status": "all_automations_frozen"},
        )
        return True

    def reset_emergency_shutdown(self, authorized_by: str = "admin") -> bool:
        """Disarm emergency halt and restore normal operations."""
        self.emergency_shutdown_active = False
        log.info(f"Emergency shutdown disarmed by: {authorized_by}")
        audit_logger.log_event(
            actor=authorized_by,
            action="disarm_emergency_shutdown",
            resource="JARVIS_CORE_SYSTEM",
            granted=True,
            risk_level="high",
        )
        return True

    def check_access(self, role: UserRole, required_role: UserRole) -> bool:
        """Evaluate hierarchical role permission."""
        hierarchy = {
            UserRole.ADMIN: 4,
            UserRole.OPERATOR: 3,
            UserRole.AUDITOR: 2,
            UserRole.READONLY: 1,
        }
        return hierarchy.get(role, 0) >= hierarchy.get(required_role, 0)


security_controller = SecurityController()
