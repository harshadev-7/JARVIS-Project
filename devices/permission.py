"""
JARVIS AI — Permission Scopes & Explicit Consent Manager
========================================================
Manages authorization scopes and user consent for connected devices.
Enforces the Zero-Trust mandate: consequential operations (e.g. file writing,
folder access, app execution) require explicit user authorization.
"""

import uuid
from enum import Enum
from typing import Set, Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

from core.logger import log


class PermissionScope(str, Enum):
    DEVICE_READ = "device:read"
    DEVICE_CONTROL = "device:control"
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    NOTIFICATION_SYNC = "notification:sync"
    APP_CONTROL = "app:control"


class ConsentStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ConsentRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req-{uuid.uuid4().hex[:8]}")
    device_id: str
    scope: PermissionScope
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    status: ConsentStatus = ConsentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=5))


class ConsentManager:
    """Zero-trust consent gatekeeper and permission auditor."""

    def __init__(self):
        # device_id -> set of granted PermissionScope
        self._device_permissions: Dict[str, Set[PermissionScope]] = {}
        # request_id -> ConsentRequest
        self._pending_consents: Dict[str, ConsentRequest] = {}
        # Chronological audit log
        self._audit_log: List[Dict[str, Any]] = []

    def grant_scope(self, device_id: str, scope: PermissionScope):
        """Permanently or explicitly grant a permission scope to a device."""
        if device_id not in self._device_permissions:
            self._device_permissions[device_id] = set()
        self._device_permissions[device_id].add(scope)
        self._log_audit("GRANT_SCOPE", device_id, {"scope": scope.value})
        log.info(f"Granted permission scope [{scope.value}] to device: {device_id}")

    def revoke_scope(self, device_id: str, scope: PermissionScope):
        """Revoke a previously granted scope."""
        if device_id in self._device_permissions:
            self._device_permissions[device_id].discard(scope)
            self._log_audit("REVOKE_SCOPE", device_id, {"scope": scope.value})
            log.info(f"Revoked permission scope [{scope.value}] from device: {device_id}")

    def revoke_all(self, device_id: str):
        """Revoke all permissions for a device upon disconnect."""
        if device_id in self._device_permissions:
            del self._device_permissions[device_id]
            self._log_audit("REVOKE_ALL", device_id, {})
            log.warning(f"All permissions revoked for device: {device_id}")

    def has_permission(self, device_id: str, scope: PermissionScope) -> bool:
        """Check whether device currently possesses an authorized scope."""
        scopes = self._device_permissions.get(device_id, set())
        return scope in scopes

    def request_consent(
        self,
        device_id: str,
        scope: PermissionScope,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        auto_approve_if_granted: bool = True,
    ) -> ConsentRequest:
        """
        Initiate a consent request.
        If auto_approve_if_granted is True and scope is already held, automatically approves.
        """
        req = ConsentRequest(
            device_id=device_id,
            scope=scope,
            action=action,
            details=details or {},
        )

        if auto_approve_if_granted and self.has_permission(device_id, scope):
            req.status = ConsentStatus.APPROVED
            self._log_audit("CONSENT_AUTO_APPROVED", device_id, {"scope": scope.value, "action": action})
            return req

        self._pending_consents[req.request_id] = req
        self._log_audit("CONSENT_REQUESTED", device_id, {"request_id": req.request_id, "action": action})
        log.info(f"Consent required for device [{device_id}] action: '{action}' [Req ID: {req.request_id}]")
        return req

    def approve_consent(self, request_id: str, persist_scope: bool = False) -> bool:
        """User approves pending consent request."""
        req = self._pending_consents.get(request_id)
        if not req:
            return False

        if datetime.now(timezone.utc) > req.expires_at:
            req.status = ConsentStatus.EXPIRED
            del self._pending_consents[request_id]
            return False

        req.status = ConsentStatus.APPROVED
        if persist_scope:
            self.grant_scope(req.device_id, req.scope)

        del self._pending_consents[request_id]
        self._log_audit("CONSENT_APPROVED", req.device_id, {"request_id": request_id, "action": req.action})
        log.info(f"Consent approved for request [{request_id}] (device: {req.device_id})")
        return True

    def reject_consent(self, request_id: str) -> bool:
        """User rejects pending consent request."""
        req = self._pending_consents.get(request_id)
        if not req:
            return False

        req.status = ConsentStatus.REJECTED
        del self._pending_consents[request_id]
        self._log_audit("CONSENT_REJECTED", req.device_id, {"request_id": request_id, "action": req.action})
        log.warning(f"Consent rejected for request [{request_id}] (device: {req.device_id})")
        return True

    def get_pending_requests(self) -> List[ConsentRequest]:
        """List active unexpired consent requests."""
        now = datetime.now(timezone.utc)
        valid = []
        for rid, req in list(self._pending_consents.items()):
            if now > req.expires_at:
                req.status = ConsentStatus.EXPIRED
                del self._pending_consents[rid]
            else:
                valid.append(req)
        return valid

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent permission audit logs."""
        return self._audit_log[-limit:]

    def _log_audit(self, event: str, device_id: str, data: Dict[str, Any]):
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "device_id": device_id,
            "data": data,
        })


consent_manager = ConsentManager()
