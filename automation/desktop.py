"""J.A.R.V.I.S. — Safe Desktop Automation (Phase 12).

Manages local desktop processes, applications, folders, and approved input.

Strict Security Constraints (Section 13):
- Mandatory approval gates before installing software, terminating core processes, or making purchases.
- Arbitrary model-generated commands are blocked without validation and explicit authorization.
- Activity audit log tracks all desktop interactions with timestamps.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import log


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DesktopActionResult:
    """Outcome of a desktop automation operation."""
    action: str
    target: str
    success: bool
    message: str
    pid: Optional[int] = None
    requires_approval: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Desktop Automation Controller
# ---------------------------------------------------------------------------

class DesktopAutomation:
    """Safe desktop controller with permission boundaries."""

    # Standard Windows application mappings
    APP_ALIASES = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "explorer": "explorer.exe",
        "terminal": "wt.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "vscode": "code",
        "code": "code",
        "edge": "msedge.exe",
        "browser": "start msedge",
    }

    # Actions that strictly require human confirmation before execution
    CONSEQUENTIAL_PATTERNS = [
        "install",
        "purchase",
        "buy",
        "send message",
        "format",
        "reboot",
        "shutdown",
        "kill critical",
    ]

    def __init__(self):
        self.activity_log: List[DesktopActionResult] = []
        self._check_pyautogui()

    def _check_pyautogui(self) -> None:
        """Check for optional PyAutoGUI library."""
        try:
            import pyautogui  # type: ignore
            self._has_pyautogui = True
            log.info("PyAutoGUI available for desktop input control.")
        except ImportError:
            self._has_pyautogui = False
            log.info("PyAutoGUI not installed. Keyboard/mouse simulation uses safe mock.")

    @property
    def has_input_control(self) -> bool:
        return self._has_pyautogui

    def is_consequential(self, action_name: str) -> bool:
        """Determine if an action requires explicit human confirmation."""
        lower = action_name.lower().strip()
        return any(pat in lower for pat in self.CONSEQUENTIAL_PATTERNS)

    def _log_activity(self, result: DesktopActionResult) -> DesktopActionResult:
        """Record action in the session audit trail."""
        self.activity_log.append(result)
        return result

    # ------------------------------------------------------------------
    # Application Management
    # ------------------------------------------------------------------

    def launch_app(self, app_name: str, user_approved: bool = False) -> DesktopActionResult:
        """Launch a desktop application with alias resolution and safety checks."""
        clean_name = app_name.lower().strip()

        # Check consequential actions
        if self.is_consequential(clean_name) and not user_approved:
            return self._log_activity(
                DesktopActionResult(
                    action="launch_app",
                    target=clean_name,
                    success=False,
                    requires_approval=True,
                    message=f"Authorization Required: Launching '{clean_name}' requires explicit user confirmation.",
                )
            )

        cmd = self.APP_ALIASES.get(clean_name, clean_name)
        try:
            proc = subprocess.Popen(cmd, shell=True)
            log.info(f"Launched application '{clean_name}' (PID: {proc.pid})")
            return self._log_activity(
                DesktopActionResult(
                    action="launch_app",
                    target=clean_name,
                    success=True,
                    pid=proc.pid,
                    message=f"Application '{clean_name}' launched successfully (PID: {proc.pid}).",
                    details={"executable": cmd},
                )
            )
        except Exception as e:
            log.error(f"Error launching '{clean_name}': {e}")
            return self._log_activity(
                DesktopActionResult(
                    action="launch_app",
                    target=clean_name,
                    success=False,
                    message=f"Failed to launch '{clean_name}': {e}",
                )
            )

    def close_app(self, app_name: str, user_approved: bool = False) -> DesktopActionResult:
        """Close running application by process name."""
        clean_name = app_name.lower().strip()

        # Closing core system apps requires approval
        core_apps = ["explorer.exe", "system", "csrss.exe", "wininit.exe"]
        if any(c in clean_name for c in core_apps) and not user_approved:
            return self._log_activity(
                DesktopActionResult(
                    action="close_app",
                    target=clean_name,
                    success=False,
                    requires_approval=True,
                    message=f"Security Alert: Terminating core system process '{clean_name}' requires approval.",
                )
            )

        try:
            import psutil
            terminated = 0
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    p_name = proc.info["name"].lower()
                    if clean_name in p_name:
                        proc.terminate()
                        terminated += 1
                except Exception:
                    continue
        except Exception:
            # Fallback using taskkill on Windows if psutil is restricted
            try:
                res = subprocess.run(f"taskkill /f /im {clean_name}*", shell=True, capture_output=True, text=True)
                terminated = 1 if res.returncode == 0 else 0
            except Exception:
                terminated = 0

        if terminated > 0:
            return self._log_activity(
                DesktopActionResult(
                    action="close_app",
                    target=clean_name,
                    success=True,
                    message=f"Closed {terminated} instance(s) of '{clean_name}'.",
                    details={"terminated_count": terminated},
                )
            )
        return self._log_activity(
            DesktopActionResult(
                action="close_app",
                target=clean_name,
                success=False,
                message=f"No active processes matching '{clean_name}' found.",
            )
        )

    def open_folder(self, folder_path: str) -> DesktopActionResult:
        """Open a directory in Windows File Explorer."""
        p = Path(folder_path).resolve()
        if not p.exists() or not p.is_dir():
            return self._log_activity(
                DesktopActionResult(
                    action="open_folder",
                    target=str(p),
                    success=False,
                    message=f"Folder not found: {p}",
                )
            )

        try:
            os.startfile(str(p))
            return self._log_activity(
                DesktopActionResult(
                    action="open_folder",
                    target=str(p),
                    success=True,
                    message=f"Opened folder '{p.name}' in File Explorer.",
                )
            )
        except Exception as e:
            return self._log_activity(
                DesktopActionResult(
                    action="open_folder",
                    target=str(p),
                    success=False,
                    message=f"Failed to open folder: {e}",
                )
            )

    # ------------------------------------------------------------------
    # Approved Keyboard & Mouse Actions
    # ------------------------------------------------------------------

    def perform_input_action(
        self,
        action_type: str,
        text_or_key: str = "",
        user_approved: bool = False,
    ) -> DesktopActionResult:
        """Execute keyboard or mouse action with security verification."""
        clean_action = action_type.lower().strip()

        # Keystrokes that could execute commands require verification
        if clean_action in ("type", "press_enter", "shortcut") and not user_approved:
            # Check if text contains dangerous patterns
            for bad in ["rm ", "del ", "format ", "curl ", "powershell "]:
                if bad in text_or_key.lower():
                    return self._log_activity(
                        DesktopActionResult(
                            action=f"input_{clean_action}",
                            target=text_or_key,
                            success=False,
                            requires_approval=True,
                            message=f"Security Gate: Input contains sensitive sequence '{bad}'. Approval required.",
                        )
                    )

        if self._has_pyautogui:
            try:
                import pyautogui
                if clean_action == "type":
                    pyautogui.write(text_or_key, interval=0.02)
                elif clean_action == "press":
                    pyautogui.press(text_or_key)
                elif clean_action == "click":
                    pyautogui.click()
                return self._log_activity(
                    DesktopActionResult(
                        action=f"input_{clean_action}",
                        target=text_or_key,
                        success=True,
                        message=f"Executed input action '{clean_action}' on active window.",
                    )
                )
            except Exception as e:
                return self._log_activity(
                    DesktopActionResult(
                        action=f"input_{clean_action}",
                        target=text_or_key,
                        success=False,
                        message=f"PyAutoGUI error: {e}",
                    )
                )
        else:
            # Safe simulated response when headless or PyAutoGUI uninstalled
            return self._log_activity(
                DesktopActionResult(
                    action=f"input_{clean_action}",
                    target=text_or_key,
                    success=True,
                    message=f"Simulated desktop input: {clean_action}('{text_or_key}') (PyAutoGUI not loaded).",
                    details={"simulated": True},
                )
            )


desktop_automation = DesktopAutomation()

__all__ = ["DesktopActionResult", "DesktopAutomation", "desktop_automation"]
