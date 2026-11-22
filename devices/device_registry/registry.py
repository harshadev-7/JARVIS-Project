"""
JARVIS AI — Multi-Device Registry & Network Hub
================================================
Manages cross-device registration, secure pairing, authentication,
heartbeat telemetry, remote task dispatching, and notifications across:
- Windows PCs & Laptops
- Android Smartphones & Tablets
- Raspberry Pi & IoT Modules
- Apple / macOS / Linux Devices
"""

import uuid
import secrets
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict
from pydantic import BaseModel, Field

from core.logger import log


class DeviceType(str, Enum):
    WINDOWS = "windows"
    ANDROID = "android"
    MACOS = "macos"
    LINUX = "linux"
    RASPBERRY_PI = "raspberry_pi"
    IOT = "iot"


class DeviceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"


class DeviceRecord(BaseModel):
    """Registered device profile."""
    device_id: str
    name: str
    device_type: DeviceType
    auth_token: str
    status: DeviceStatus = DeviceStatus.ONLINE
    battery_level: Optional[float] = None
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    capabilities: List[str] = Field(default_factory=list)


class RemoteTaskPayload(BaseModel):
    """Task dispatched to a remote registered device."""
    task_id: str = Field(default_factory=lambda: f"rmt-{uuid.uuid4().hex[:8]}")
    target_device: str
    action: str  # e.g., "launch_app", "notify", "fetch_status", "execute"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dispatched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeviceRegistry:
    """Central registry and dispatch engine for connected devices."""

    def __init__(self):
        self._devices: Dict[str, DeviceRecord] = {}
        self._pairing_codes: Dict[str, Dict[str, Any]] = {}  # code -> {created_at, ttl}
        self._pending_tasks: Dict[str, List[RemoteTaskPayload]] = defaultdict(list)
        self._completed_tasks: Dict[str, Dict[str, Any]] = {}

    def generate_pairing_code(self) -> str:
        """Create a 6-digit one-time cryptographic pairing PIN."""
        code = f"{secrets.randbelow(900000) + 100000}"
        self._pairing_codes[code] = {
            "created_at": datetime.now(timezone.utc),
        }
        log.info(f"Generated device pairing code: {code}")
        return code

    def pair_device(
        self,
        pairing_code: str,
        name: str,
        device_type: DeviceType,
        capabilities: Optional[List[str]] = None,
    ) -> DeviceRecord:
        """Validate pairing code and register device with auth token."""
        if pairing_code not in self._pairing_codes:
            raise ValueError("Invalid or expired pairing code.")

        del self._pairing_codes[pairing_code]

        dev_id = f"dev-{uuid.uuid4().hex[:8]}"
        auth_token = secrets.token_urlsafe(32)

        record = DeviceRecord(
            device_id=dev_id,
            name=name,
            device_type=device_type,
            auth_token=auth_token,
            status=DeviceStatus.ONLINE,
            capabilities=capabilities or ["notification", "remote_task"],
        )
        self._devices[dev_id] = record
        log.info(f"Device successfully paired: '{name}' ({device_type.value}) -> ID: {dev_id}")
        return record

    def heartbeat(self, device_id: str, battery_level: Optional[float] = None) -> bool:
        """Update device online presence and telemetry."""
        if device_id in self._devices:
            dev = self._devices[device_id]
            dev.last_seen = datetime.now(timezone.utc)
            dev.status = DeviceStatus.ONLINE
            if battery_level is not None:
                dev.battery_level = battery_level
            return True
        return False

    def list_devices(self) -> List[DeviceRecord]:
        """List all registered devices and check timeouts."""
        now = datetime.now(timezone.utc)
        for dev in self._devices.values():
            # Mark offline if no ping in 60 seconds
            if (now - dev.last_seen).total_seconds() > 60:
                dev.status = DeviceStatus.OFFLINE
        return list(self._devices.values())

    def find_device_by_name(self, name_query: str) -> Optional[DeviceRecord]:
        """Lookup registered device by case-insensitive name fragment."""
        query = name_query.lower().strip()
        for dev in self._devices.values():
            if query in dev.name.lower():
                return dev
        return None

    def dispatch_remote_task(
        self,
        target_name_or_id: str,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> RemoteTaskPayload:
        """
        Dispatch a task to a registered target device.
        Example: "Open Eclipse on my laptop"
        """
        # Resolve device
        device = self._devices.get(target_name_or_id) or self.find_device_by_name(target_name_or_id)
        if not device:
            raise LookupError(f"Target device '{target_name_or_id}' is not registered with JARVIS.")

        payload = RemoteTaskPayload(
            target_device=device.device_id,
            action=action,
            parameters=parameters or {},
        )
        self._pending_tasks[device.device_id].append(payload)
        log.info(f"Dispatched remote task [{payload.task_id}] to '{device.name}' ({device.device_id}): {action}")
        return payload

    def get_pending_tasks(self, device_id: str, clear: bool = True) -> List[RemoteTaskPayload]:
        """Fetch pending tasks queued for a specific device."""
        tasks = self._pending_tasks.get(device_id, [])
        if clear:
            self._pending_tasks[device_id] = []
        return tasks

    def complete_task(self, task_id: str, result_data: Optional[Dict[str, Any]] = None) -> bool:
        """Mark a remote task as successfully completed."""
        self._completed_tasks[task_id] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": result_data or {},
        }
        log.info(f"Remote task [{task_id}] reported complete.")
        return True

    def disconnect_device(self, device_id: str) -> bool:
        """Revoke device registration."""
        if device_id in self._devices:
            del self._devices[device_id]
            log.warning(f"Device revoked and disconnected: {device_id}")
            return True
        return False


device_registry = DeviceRegistry()
