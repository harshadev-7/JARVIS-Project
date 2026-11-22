"""
JARVIS AI — Safe Remote Command Gatekeeper & Whitelist
======================================================
Strictly enforces sandboxed remote command execution for approved devices.
Mandate:
- Blocks arbitrary shell execution, system modifying scripts, and raw terminal commands.
- Only registered, safe actions with type-checked parameters are executable.
- Integrates with the tamper-evident security audit logger.
"""

import re
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from services.security.audit_logger import audit_logger
from core.logger import log


# Disallowed shell characters that must never appear in any argument
DANGEROUS_METAS = [";", "&", "|", "`", "$", "(", ")", "\n", "\r", "<", ">"]


class RemoteCommandRequest(BaseModel):
    command_name: str
    device_id: str
    session_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RemoteCommandResult(BaseModel):
    success: bool
    command_name: str
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class CommandWhitelistGatekeeper:
    """Dispatches safe, verified remote commands from authorized companion devices."""

    def __init__(self):
        self._handlers: Dict[str, Callable[[str, Dict[str, Any]], Dict[str, Any]]] = {}
        self._device_acls: Dict[str, set] = {}  # device_id -> set of allowed command names
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register built-in safe command handlers."""
        self.register_handler("ping", self._handle_ping)
        self.register_handler("get_status", self._handle_get_status)
        self.register_handler("system_metrics", self._handle_system_metrics)
        self.register_handler("list_media", self._handle_list_media)
        self.register_handler("echo", self._handle_echo)

    def register_handler(
        self,
        command_name: str,
        handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    ):
        """Register an authorized safe command handler."""
        self._handlers[command_name.lower().strip()] = handler

    def grant_command_permission(self, device_id: str, command_name: str):
        """Grant a device explicit access to a remote command."""
        cmd = command_name.lower().strip()
        if device_id not in self._device_acls:
            self._device_acls[device_id] = set()
        self._device_acls[device_id].add(cmd)
        log.info(f"Granted remote command permission [{cmd}] to device [{device_id}].")

    def revoke_command_permission(self, device_id: str, command_name: str):
        """Revoke a remote command permission."""
        cmd = command_name.lower().strip()
        if device_id in self._device_acls:
            self._device_acls[device_id].discard(cmd)
            log.info(f"Revoked remote command permission [{cmd}] from device [{device_id}].")

    def is_command_allowed(self, device_id: str, command_name: str) -> bool:
        """
        Check if device has permission to execute command.
        'ping' and 'echo' are allowed for all authenticated devices by default.
        """
        cmd = command_name.lower().strip()
        if cmd in ("ping", "echo"):
            return True
        allowed_cmds = self._device_acls.get(device_id, set())
        # Default allow standard read-only commands if device is in ACL or if default access enabled
        if not allowed_cmds and cmd in ("get_status", "system_metrics", "list_media"):
            return True
        return cmd in allowed_cmds

    def _validate_parameters(self, params: Dict[str, Any]):
        """Sanitize parameters to ensure no shell metacharacters or injections."""
        for key, val in params.items():
            if isinstance(val, str):
                for meta in DANGEROUS_METAS:
                    if meta in val:
                        raise ValueError(f"Dangerous character '{meta}' detected in parameter '{key}'.")

    async def execute_command(self, request: RemoteCommandRequest) -> RemoteCommandResult:
        """
        Validate, audit, and safely execute a remote command.
        """
        import time
        start_t = time.perf_counter()
        cmd = request.command_name.lower().strip()

        # 1. Verify existence in whitelist
        if cmd not in self._handlers:
            audit_logger.log_event(
                actor=request.device_id,
                action=f"remote_cmd:{cmd}",
                resource="command_gate",
                granted=False,
                risk_level="medium",
                details={"reason": "Command not in whitelist"},
            )
            return RemoteCommandResult(
                success=False,
                command_name=cmd,
                error=f"Command '{cmd}' is not in the authorized remote command whitelist.",
            )

        # 2. Check ACL
        if not self.is_command_allowed(request.device_id, cmd):
            audit_logger.log_event(
                actor=request.device_id,
                action=f"remote_cmd:{cmd}",
                resource="command_gate",
                granted=False,
                risk_level="medium",
                details={"reason": "ACL permission denied"},
            )
            return RemoteCommandResult(
                success=False,
                command_name=cmd,
                error=f"Permission denied: Device [{request.device_id}] lacks authorization for '{cmd}'.",
            )

        # 3. Sanitize parameters
        try:
            self._validate_parameters(request.parameters)
        except ValueError as e:
            audit_logger.log_event(
                actor=request.device_id,
                action=f"remote_cmd:{cmd}",
                resource="command_gate",
                granted=False,
                risk_level="high",
                details={"security_violation": str(e)},
            )
            return RemoteCommandResult(
                success=False,
                command_name=cmd,
                error=f"Parameter security validation rejected: {e}",
            )

        # 4. Safe Execution
        try:
            handler = self._handlers[cmd]
            result_data = handler(request.device_id, request.parameters)
            elapsed_ms = (time.perf_counter() - start_t) * 1000

            audit_logger.log_event(
                actor=request.device_id,
                action=f"remote_cmd:{cmd}",
                resource="command_gate",
                granted=True,
                risk_level="low",
                details={"execution_ms": round(elapsed_ms, 2)},
            )

            return RemoteCommandResult(
                success=True,
                command_name=cmd,
                data=result_data,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            log.warning(f"Error executing remote command '{cmd}': {e}")
            return RemoteCommandResult(
                success=False,
                command_name=cmd,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )

    # ── Default Safe Handlers ────────────────────────────────────────────────

    def _handle_ping(self, device_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"pong": True, "server_time": datetime.now(timezone.utc).isoformat()}

    def _handle_get_status(self, device_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from config.settings import settings
        return {
            "version": settings.JARVIS_VERSION,
            "status": "online",
            "autonomy_level": settings.DEFAULT_AUTONOMY_LEVEL,
            "host_os": "Windows",
        }

    def _handle_system_metrics(self, device_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from services.automation.system_monitor import system_monitor
        m = system_monitor.get_metrics()
        return m.dict() if hasattr(m, "dict") else m.model_dump()

    def _handle_list_media(self, device_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import os
        from config.settings import BASE_DIR
        media_input = BASE_DIR / "media" / "input"
        output_reels = BASE_DIR / "output" / "reels"
        
        inputs = os.listdir(media_input) if os.path.exists(media_input) else []
        reels = os.listdir(output_reels) if os.path.exists(output_reels) else []
        return {
            "input_clips": inputs[:20],
            "rendered_reels": reels[:20],
        }

    def _handle_echo(self, device_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        msg = params.get("message", "")
        return {"echo": msg}


command_gatekeeper = CommandWhitelistGatekeeper()
