"""
JARVIS AI — Application Controller
===================================
Manages Windows desktop applications: launching, listing,
and terminating processes with safety permissions.
"""

import subprocess
import psutil
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from config.settings import settings
from config.autonomy_levels import check_permission
from core.logger import log


class AppActionResult(BaseModel):
    """Result of an application management operation."""
    app_name: str
    success: bool
    pid: Optional[int] = None
    message: str


class AppController:
    """Controls local OS desktop applications."""

    # Common Windows application mappings
    APP_ALIASES = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "vscode": "code",
        "code": "code",
        "explorer": "explorer.exe",
        "terminal": "wt.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "browser": "start msedge",
        "edge": "msedge.exe",
        "chrome": "chrome.exe",
        "eclipse": "eclipse.exe",
    }

    def launch_app(self, app_identifier: str, autonomy_level: int = 4) -> AppActionResult:
        """Launch a desktop application."""
        from services.security.access_control import security_controller
        if security_controller.emergency_shutdown_active:
            return AppActionResult(
                app_name=app_identifier,
                success=False,
                message="Security Alert: System is in EMERGENCY SHUTDOWN mode. App launch blocked.",
            )

        allowed, reason = check_permission(autonomy_level, "execute")
        if not allowed:
            return AppActionResult(
                app_name=app_identifier,
                success=False,
                message=f"Permission Denied: {reason}",
            )

        cmd = self.APP_ALIASES.get(app_identifier.lower().strip(), app_identifier)
        try:
            log.info(f"Launching application command: {cmd}")
            proc = subprocess.Popen(cmd, shell=True)
            return AppActionResult(
                app_name=app_identifier,
                success=True,
                pid=proc.pid,
                message=f"Application '{app_identifier}' launched successfully.",
            )
        except Exception as e:
            log.error(f"Failed to launch application '{app_identifier}': {e}")
            return AppActionResult(
                app_name=app_identifier,
                success=False,
                message=f"Failed to launch application: {e}",
            )

    def close_app(self, app_name: str, autonomy_level: int = 4) -> AppActionResult:
        """Close running application by executable or process name."""
        allowed, reason = check_permission(autonomy_level, "execute")
        if not allowed:
            return AppActionResult(
                app_name=app_name,
                success=False,
                message=f"Permission Denied: {reason}",
            )

        target = app_name.lower().strip()
        closed_count = 0
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                p_name = proc.info["name"].lower()
                if target in p_name:
                    proc.terminate()
                    closed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if closed_count > 0:
            return AppActionResult(
                app_name=app_name,
                success=True,
                message=f"Terminated {closed_count} instance(s) of '{app_name}'.",
            )
        return AppActionResult(
            app_name=app_name,
            success=False,
            message=f"No running process matching '{app_name}' found.",
        )


app_controller = AppController()
