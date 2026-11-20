"""
JARVIS AI — Safe Command & Terminal Executor
==============================================
Executes authorized terminal commands with strict safety filters,
timeout limits, and autonomy permission verification.
"""

import subprocess
import asyncio
from typing import Optional
from pydantic import BaseModel

from config.autonomy_levels import check_permission
from core.logger import log


class CommandResult(BaseModel):
    """Result of command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    blocked: bool = False
    block_reason: Optional[str] = None


class SafeCommandExecutor:
    """Executes system commands with safeguards."""

    # Prohibited dangerous patterns
    DANGEROUS_PATTERNS = [
        "format ",
        "mkfs",
        "diskpart",
        "reg delete",
        "del /f /s /q c:\\",
        "rm -rf /",
        "rmdir /s /q c:\\",
        ":(){ :|:& };:",
        "drop database",
    ]

    async def execute(
        self,
        command: str,
        autonomy_level: int = 4,
        timeout_sec: float = 30.0,
        working_directory: Optional[str] = None,
    ) -> CommandResult:
        """Run terminal command with safety validation."""
        # Check emergency shutdown
        from services.security.access_control import security_controller
        if security_controller.emergency_shutdown_active:
            log.warning(f"Blocked command during emergency shutdown: '{command}'")
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Security Alert: System is in EMERGENCY SHUTDOWN mode. All automations frozen.",
                duration_sec=0.0,
                blocked=True,
                block_reason="Emergency shutdown engaged.",
            )

        # Check autonomy permission
        allowed, reason = check_permission(autonomy_level, "execute")
        if not allowed:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=reason,
                duration_sec=0.0,
                blocked=True,
                block_reason=reason,
            )

        # Check blacklist
        lower_cmd = command.lower()
        for bad in self.DANGEROUS_PATTERNS:
            if bad in lower_cmd:
                log.warning(f"Blocked dangerous command attempt: '{command}'")
                return CommandResult(
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Security Alert: Command blocked due to prohibited pattern: '{bad}'.",
                    duration_sec=0.0,
                    blocked=True,
                    block_reason=f"Matched prohibited pattern: {bad}",
                )

        start = asyncio.get_event_loop().time()
        try:
            log.info(f"Executing system command: '{command}'")
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_directory,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            duration = asyncio.get_event_loop().time() - start

            stdout_txt = stdout_b.decode(errors="replace").strip()
            stderr_txt = stderr_b.decode(errors="replace").strip()

            return CommandResult(
                command=command,
                exit_code=proc.returncode if proc.returncode is not None else 0,
                stdout=stdout_txt,
                stderr=stderr_txt,
                duration_sec=round(duration, 2),
            )
        except asyncio.TimeoutError:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout_sec}s threshold.",
                duration_sec=timeout_sec,
                blocked=False,
            )
        except Exception as e:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_sec=0.0,
                blocked=False,
            )


command_executor = SafeCommandExecutor()
