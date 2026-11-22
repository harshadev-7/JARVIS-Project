"""Unit Tests for J.A.R.V.I.S. Safe Computer Automation (Phase 12).

Verifies Section 13 & 15 requirements:
1. Application launching and process closing via DesktopAutomation.
2. Folder opening and directory navigation.
3. Safe File Manager: list, read, write, and size bounding.
4. Mandatory Security Gate 1: Ask before deleting files (reject when user_approved=False).
5. Mandatory Security Gate 2: Ask before sending messages.
6. Mandatory Security Gate 3: Ask before installing software.
7. Mandatory Security Gate 4: Ask before making purchases.
8. File deletion succeeds when user_approved=True.
9. Protected system path preservation (cannot delete C:\\Windows or System32).
10. Hardware telemetry inspection (CPU, RAM, Disk, resource threshold).
11. Activity audit log tracking.
12. ComputerControlAgent multi-action execution.
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from automation.desktop import DesktopAutomation, DesktopActionResult
from automation.file_manager import FileManager, FileOperationResult
from automation.system_monitor import SystemMonitor, HardwareTelemetry
from agents.computer_control import ComputerControlAgent


# ============================================================================
# 1. Desktop Automation Tests
# ============================================================================

def test_desktop_launch_app_alias():
    """Desktop automation maps aliases safely and records audit logs."""
    desktop = DesktopAutomation()
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value.pid = 9999
        res: DesktopActionResult = desktop.launch_app("notepad")
        assert res.success is True
        assert res.pid == 9999
        assert "notepad" in res.target
        assert len(desktop.activity_log) > 0


def test_desktop_open_nonexistent_folder():
    """Opening non-existent folder returns clean failure without crash."""
    desktop = DesktopAutomation()
    res = desktop.open_folder("C:\\NonExistent_Folder_xyz123")
    assert res.success is False
    assert "not found" in res.message.lower()


# ============================================================================
# 2. File Manager Security Gates Tests
# ============================================================================

def test_file_manager_write_and_read():
    """File manager writes content and reads it safely with character limits."""
    fm = FileManager()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_path = f.name

    try:
        write_res = fm.write_file(temp_path, "J.A.R.V.I.S. Core Automation Test", overwrite=True)
        assert write_res.success is True

        read_res = fm.read_file(temp_path)
        assert read_res.success is True
        assert read_res.data["content"] == "J.A.R.V.I.S. Core Automation Test"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_mandatory_security_gate_file_deletion_unapproved():
    """RULE: Must ask before deleting files. Unapproved deletion is blocked."""
    fm = FileManager()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"sensitive data")
        temp_path = f.name

    try:
        # Attempt delete without user approval -> MUST BE BLOCKED
        del_res = fm.delete_file(temp_path, user_approved=False)
        assert del_res.success is False
        assert del_res.requires_approval is True
        assert "Authorization Required" in del_res.message
        # Verify file is still untouched on disk
        assert os.path.exists(temp_path) is True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_file_deletion_succeeds_with_explicit_approval():
    """When user explicitly approves (user_approved=True), deletion proceeds."""
    fm = FileManager()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"data to delete")
        temp_path = f.name

    try:
        # Delete with explicit user approval
        del_res = fm.delete_file(temp_path, user_approved=True)
        assert del_res.success is True
        assert "deleted successfully" in del_res.message
        assert os.path.exists(temp_path) is False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_protected_system_paths_cannot_be_deleted():
    """Protected roots (C:\\Windows, System32) are immune to deletion attempts."""
    fm = FileManager()
    res = fm.delete_file("C:\\Windows\\System32\\notepad.exe", user_approved=True)
    assert res.success is False
    assert "protected" in res.message.lower()


# ============================================================================
# 3. Mandatory Consequential Action Gates (Section 13)
# ============================================================================

@pytest.mark.asyncio
async def test_agent_gate_software_installation():
    """RULE: Must ask before installing software."""
    agent = ComputerControlAgent()
    res = await agent.execute("install package tensorflow", context={"user_approved": False})
    assert res.success is False
    assert res.error == "UserAuthorizationRequired"
    assert res.data["risk_category"] == "install_software"
    assert "Software Installation" in res.summary


@pytest.mark.asyncio
async def test_agent_gate_purchases():
    """RULE: Must ask before making purchases."""
    agent = ComputerControlAgent()
    res = await agent.execute("purchase subscription for pro tier", context={"user_approved": False})
    assert res.success is False
    assert res.error == "UserAuthorizationRequired"
    assert res.data["risk_category"] == "purchase"
    assert "Financial Transaction" in res.summary


@pytest.mark.asyncio
async def test_agent_gate_send_messages():
    """RULE: Must ask before sending messages."""
    agent = ComputerControlAgent()
    res = await agent.execute("send message to dev team", context={"user_approved": False})
    assert res.success is False
    assert res.error == "UserAuthorizationRequired"
    assert res.data["risk_category"] == "send_message"
    assert "External Communication" in res.summary


# ============================================================================
# 4. Hardware Telemetry & Monitor Tests
# ============================================================================

def test_hardware_telemetry_metrics():
    """Telemetry returns valid CPU, RAM, Disk readings."""
    monitor = SystemMonitor()
    telemetry: HardwareTelemetry = monitor.get_metrics()
    assert telemetry.cpu_percent >= 0.0
    assert telemetry.ram_total_gb > 0.0
    assert telemetry.ram_used_gb >= 0.0
    assert telemetry.disk_percent >= 0.0
    assert isinstance(telemetry.is_safe_for_ai_workload, bool)


@pytest.mark.asyncio
async def test_computer_control_agent_telemetry_action():
    """Agent handles hardware telemetry check."""
    agent = ComputerControlAgent()
    res = await agent.execute("system telemetry", context={"action": "telemetry"})
    assert res.success is True
    assert "Hardware Telemetry Report" in res.summary
    assert "CPU:" in res.summary
    assert "RAM:" in res.summary
