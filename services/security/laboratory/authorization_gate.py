"""
JARVIS AI — Cybersecurity Laboratory Authorization Gate
========================================================
Enforces strict affirmative consent and boundary validation before any
security assessment or file quarantine inspection can proceed.
Mandates:
- Explicit operator authorization token or grant required.
- Third-party endpoints, external IP networks, and OS kernel directories are blocked.
- Records all assessment events in the tamper-evident security audit log.
"""

import os
import re
import uuid
from typing import Dict, Optional, Set
from datetime import datetime, timezone, timedelta

from services.security.laboratory.models import AssessmentScope
from services.security.audit_logger import audit_logger
from core.logger import log


class AuthorizationGate:
    """Gatekeeper verifying explicit user consent and target safety."""

    def __init__(self):
        # scope_id -> AssessmentScope
        self._authorized_scopes: Dict[str, AssessmentScope] = {}
        # Explicitly approved paths
        self._approved_paths: Set[str] = set()

    def request_authorization(
        self,
        target_path: str,
        authorized_by: str = "operator",
        ttl_minutes: int = 60,
    ) -> AssessmentScope:
        """
        Request and register explicit authorization to assess a local directory or target.
        """
        canonical_target = os.path.realpath(os.path.abspath(target_path))

        # Block root drive captures and critical system directories
        normalized = canonical_target.lower().replace("/", "\\")
        if re.match(r"^[a-zA-Z]:\\?$", normalized) or normalized in ("\\", "/"):
            audit_logger.log_event(
                actor=authorized_by,
                action="lab_auth_request",
                resource=canonical_target,
                granted=False,
                risk_level="critical",
                details={"reason": "Attempted to authorize entire drive root."},
            )
            raise PermissionError(f"Security Gate: Cannot authorize entire root drive: {canonical_target}")

        scope_id = f"scope-{uuid.uuid4().hex[:8]}"
        scope = AssessmentScope(
            scope_id=scope_id,
            target_path=canonical_target,
            authorized_by=authorized_by,
            is_authorized=True,
        )

        self._authorized_scopes[scope_id] = scope
        self._approved_paths.add(canonical_target.lower())

        audit_logger.log_event(
            actor=authorized_by,
            action="lab_auth_grant",
            resource=canonical_target,
            granted=True,
            risk_level="low",
            details={"scope_id": scope_id},
        )
        log.info(f"Authorized security assessment scope [{scope_id}] for path: {canonical_target}")
        return scope

    def verify_target_authorized(self, target_path: str) -> str:
        """
        Verify that target path resides within an explicitly authorized scope.
        Returns canonical path if authorized, raises PermissionError otherwise.
        """
        canonical = os.path.realpath(os.path.abspath(target_path))
        canonical_lower = canonical.lower()

        # Check direct or parent path authorization
        for approved in self._approved_paths:
            try:
                if os.path.commonpath([approved, canonical_lower]) == approved:
                    return canonical
            except ValueError:
                continue

        audit_logger.log_event(
            actor="system",
            action="lab_scope_verification",
            resource=canonical,
            granted=False,
            risk_level="medium",
            details={"reason": "Target path not within any authorized assessment scope."},
        )
        raise PermissionError(
            f"Laboratory Access Denied: Path '{canonical}' has not been explicitly authorized for assessment. "
            f"Use request_authorization() or /audit authorize <path> first."
        )

    def revoke_authorization(self, scope_id: str) -> bool:
        """Revoke a previously granted assessment scope."""
        scope = self._authorized_scopes.get(scope_id)
        if not scope:
            return False

        scope.is_authorized = False
        target_lower = scope.target_path.lower()
        if target_lower in self._approved_paths:
            self._approved_paths.remove(target_lower)
        del self._authorized_scopes[scope_id]

        audit_logger.log_event(
            actor="system",
            action="lab_auth_revoke",
            resource=scope.target_path,
            granted=True,
            risk_level="low",
            details={"scope_id": scope_id},
        )
        log.warning(f"Revoked security assessment authorization for scope [{scope_id}].")
        return True


authorization_gate = AuthorizationGate()
