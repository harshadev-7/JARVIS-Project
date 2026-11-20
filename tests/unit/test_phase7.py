"""
JARVIS AI — Phase 7 Test Suite
================================
Tests for Multi-Device Network & Android Control:
- Secure 6-digit pairing code generation & handshake
- Authentication token issuing
- Heartbeat and battery telemetry tracking
- Remote task dispatching across registered devices
- FastAPI device management endpoints
"""

import pytest
from fastapi.testclient import TestClient

from devices.device_registry.registry import (
    device_registry,
    DeviceType,
    DeviceStatus,
)
from api.app import app


# ── 1. Device Registry Engine Tests ─────────────────────────────────────────

def test_pairing_code_generation():
    """Pairing code should be a 6-digit numeric string."""
    code = device_registry.generate_pairing_code()
    assert len(code) == 6
    assert code.isdigit()


def test_device_pairing_flow():
    """Device should pair with valid code and receive auth token."""
    code = device_registry.generate_pairing_code()
    device = device_registry.pair_device(
        pairing_code=code,
        name="Harsh Workstation Laptop",
        device_type=DeviceType.WINDOWS,
        capabilities=["terminal", "app_control"],
    )
    assert device.device_id.startswith("dev-")
    assert len(device.auth_token) >= 32
    assert device.status == DeviceStatus.ONLINE

    # Pairing code cannot be reused
    with pytest.raises(ValueError):
        device_registry.pair_device(code, "Duplicate Phone", DeviceType.ANDROID)


def test_device_heartbeat_and_telemetry():
    """Heartbeat should refresh presence and battery level."""
    code = device_registry.generate_pairing_code()
    device = device_registry.pair_device(code, "Pixel Phone", DeviceType.ANDROID)

    success = device_registry.heartbeat(device.device_id, battery_level=85.0)
    assert success is True

    devices = device_registry.list_devices()
    matched = next((d for d in devices if d.device_id == device.device_id), None)
    assert matched is not None
    assert matched.battery_level == 85.0


def test_remote_task_dispatch():
    """Tasks should be dispatchable by device name or ID."""
    code = device_registry.generate_pairing_code()
    device = device_registry.pair_device(code, "Primary Laptop", DeviceType.WINDOWS)

    payload = device_registry.dispatch_remote_task(
        target_name_or_id="Primary Laptop",
        action="launch_app",
        parameters={"app": "eclipse"},
    )
    assert payload.target_device == device.device_id
    assert payload.action == "launch_app"
    assert payload.parameters["app"] == "eclipse"


# ── 2. FastAPI Device Endpoints Tests ───────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_pairing_and_dispatch_flow(client):
    """Full API device lifecycle: code -> pair -> list -> dispatch -> disconnect."""
    # 1. Generate code
    code_res = client.post("/devices/pairing-code")
    assert code_res.status_code == 200
    code = code_res.json()["pairing_code"]

    # 2. Pair
    pair_res = client.post(
        "/devices/pair",
        json={
            "pairing_code": code,
            "name": "Tablet Display",
            "device_type": "android",
            "capabilities": ["display", "voice"],
        },
    )
    assert pair_res.status_code == 200
    dev_id = pair_res.json()["device_id"]

    # 3. List
    list_res = client.get("/devices/list")
    assert list_res.status_code == 200
    all_devs = list_res.json()
    assert any(d["device_id"] == dev_id for d in all_devs)

    # 4. Dispatch
    disp_res = client.post(
        "/devices/dispatch",
        json={"target": dev_id, "action": "display_notification", "parameters": {"text": "Mission Active"}},
    )
    assert disp_res.status_code == 200
    assert disp_res.json()["action"] == "display_notification"

    # 5. Disconnect
    del_res = client.delete(f"/devices/{dev_id}")
    assert del_res.status_code == 200
