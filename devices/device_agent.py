"""
JARVIS AI — Authorized Device NLP Agent
=======================================
Autonomous agent for natural language device mesh management, pairing,
local discovery, directory sandboxing, and telemetry queries.
"""

import re
from typing import Dict, Any, Optional
from core.orchestrator.base_agent import BaseAgent, AgentTaskResult
from devices.device_manager import device_manager
from core.logger import log


class DeviceAgent(BaseAgent):
    """Specialized agent for authorized cross-device operations and natural language control."""

    def __init__(self):
        super().__init__(
            name="DeviceManagementAgent",
            role="Multi-device registration, secure pairing, telemetry sync, file transfer, and remote control.",
            capabilities=[
                "device_pairing",
                "remote_dispatch",
                "device_telemetry",
                "lan_discovery",
                "folder_sandboxing",
                "file_transfer",
                "notification_sync",
            ],
            required_autonomy_level=3,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        query = task.lower().strip()
        log.info(f"DeviceAgent processing directive: '{task}'")

        # 1. Pairing Code Generation
        if any(w in query for w in ["pair", "pairing code", "pairing pin", "connect phone", "connect tablet"]):
            pin = device_manager.generate_pairing_pin()
            return AgentTaskResult(
                agent_name=self.name,
                task_id="dev-pair-01",
                success=True,
                data={"pairing_pin": pin, "expires_in_sec": 300},
                summary=(
                    f"Generated one-time 6-digit pairing PIN: **{pin}**.\n"
                    f"This code will expire in 5 minutes. Enter it on your companion device to pair."
                ),
            )

        # 2. Local Network Discovery
        if any(w in query for w in ["discover", "scan network", "nearby device", "find devices"]):
            peers = device_manager.scan_network()
            summary = (
                f"Discovered {len(peers)} device(s) on local network."
                if peers
                else "No active beaconing devices discovered on the local subnet."
            )
            return AgentTaskResult(
                agent_name=self.name,
                task_id="dev-disc-01",
                success=True,
                data={"discovered_peers": peers},
                summary=summary,
            )

        # 3. List Authorized Devices
        if any(w in query for w in ["list devices", "connected devices", "show devices", "device status"]):
            devices = device_manager.list_devices()
            dev_list = [
                f"- **{d.name}** ({d.device_type.value}): Status={d.status.value}, Battery={d.battery_level or 'N/A'}%"
                for d in devices
            ]
            summary = (
                f"**Authorized Devices ({len(devices)}):**\n" + "\n".join(dev_list)
                if devices
                else "No authorized devices are currently connected."
            )
            return AgentTaskResult(
                agent_name=self.name,
                task_id="dev-list-01",
                success=True,
                data={"devices": [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in devices]},
                summary=summary,
            )

        # 4. Folder Whitelist / Sandbox
        if "authorize folder" in query or "whitelist folder" in query:
            match = re.search(r"(?:authorize|whitelist)\s+folder\s+(.+)", task, re.IGNORECASE)
            folder_path = match.group(1).strip().strip("\"'") if match else ""
            if folder_path:
                try:
                    record = device_manager.authorize_folder(folder_path, access_level="read_write")
                    return AgentTaskResult(
                        agent_name=self.name,
                        task_id="dev-fld-01",
                        success=True,
                        data=record,
                        summary=f"Folder authorized for device transfers: `{record['path']}` (read_write).",
                    )
                except Exception as e:
                    return AgentTaskResult(
                        agent_name=self.name,
                        task_id="dev-fld-01",
                        success=False,
                        error=str(e),
                        summary=f"Failed to authorize folder: {e}",
                    )

        # 5. Revocation
        if "revoke" in query or "disconnect" in query:
            match = re.search(r"(?:revoke|disconnect)\s+(?:device\s+)?(.+)", task, re.IGNORECASE)
            target = match.group(1).strip() if match else ""
            dev = device_manager.find_device(target)
            if dev:
                device_manager.disconnect_device(dev.device_id)
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id="dev-rev-01",
                    success=True,
                    summary=f"Device '{dev.name}' ({dev.device_id}) has been revoked and disconnected.",
                )
            return AgentTaskResult(
                agent_name=self.name,
                task_id="dev-rev-01",
                success=False,
                summary=f"Device '{target}' was not found in registered devices.",
            )

        # Default fallback: summary of connected devices
        devices = device_manager.list_devices()
        return AgentTaskResult(
            agent_name=self.name,
            task_id="dev-gen-01",
            success=True,
            data={"total_devices": len(devices)},
            summary=f"Device Mesh: {len(devices)} registered hardware endpoint(s). Use 'pair' or 'list devices' to manage.",
        )
