"""J.A.R.V.I.S. Computer Automation Package (Phase 12).

Safe desktop process management, file operations, hardware telemetry,
and human-in-the-loop security authorization gates.
"""

from automation.desktop import DesktopAutomation, DesktopActionResult, desktop_automation
from automation.file_manager import FileManager, FileOperationResult, file_manager
from automation.system_monitor import SystemMonitor, HardwareTelemetry, system_monitor

__all__ = [
    "DesktopAutomation",
    "DesktopActionResult",
    "desktop_automation",
    "FileManager",
    "FileOperationResult",
    "file_manager",
    "SystemMonitor",
    "HardwareTelemetry",
    "system_monitor",
]
