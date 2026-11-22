"""
JARVIS AI — Multi-Device Network API Routes
============================================
Endpoints for device pairing, authentication, heartbeat telemetry,
cross-device task dispatching, LAN discovery, folder sandboxing,
verified file transfers, and filtered notifications.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from devices.device_registry.registry import (
    device_registry,
    DeviceRecord,
    DeviceType,
    RemoteTaskPayload,
)
from devices.device_manager import device_manager
from devices.permission import PermissionScope

router = APIRouter(prefix="/devices", tags=["Multi-Device Network"])


# ── Request / Response Models ────────────────────────────────────────────────

class PairingCodeResponse(BaseModel):
    pairing_code: str
    expires_in_sec: int = 300


class PairDeviceRequest(BaseModel):
    pairing_code: str = Field(..., min_length=6, max_length=6)
    name: str = Field(..., min_length=1, description="Friendly device name e.g. 'Harsh Phone'")
    device_type: DeviceType
    capabilities: Optional[List[str]] = Field(default_factory=list)


class HeartbeatRequest(BaseModel):
    device_id: str
    battery_level: Optional[float] = None


class DispatchTaskRequest(BaseModel):
    target: str = Field(..., description="Device name or ID e.g. 'laptop' or 'phone'")
    action: str = Field(..., description="Action to perform e.g. 'launch_app'")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AuthorizeFolderRequest(BaseModel):
    folder_path: str = Field(..., min_length=1, description="Absolute or relative folder path")
    access_level: str = Field(default="read_only", description="'read_only' or 'read_write'")


class SendNotificationRequest(BaseModel):
    source_device: str
    target_device: str
    title: str
    message: str
    category: str = Field(default="system_event", description="e.g. 'urgent', 'reminder', 'alert', 'system_event'")
    priority: str = Field(default="normal")


class InitiateTransferRequest(BaseModel):
    source_device: str
    target_device: str
    file_name: str
    file_size: int
    sha256_hash: str
    target_folder: str


# ── Phase 7 Compatible Endpoints ─────────────────────────────────────────────

@router.post("/pairing-code", response_model=PairingCodeResponse, summary="Generate a 6-digit one-time device pairing code")
async def generate_pairing_code():
    code = device_manager.generate_pairing_pin()
    return PairingCodeResponse(pairing_code=code)


@router.post("/pair", response_model=DeviceRecord, summary="Pair a new device using pairing code")
async def pair_device(request: PairDeviceRequest):
    try:
        record, _raw_token = device_manager.pair_device(
            pairing_code=request.pairing_code,
            name=request.name,
            device_type=request.device_type,
            capabilities=request.capabilities,
        )
        return record
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list", response_model=List[DeviceRecord], summary="List all registered devices and online status")
async def list_devices():
    return device_manager.list_devices()


@router.post("/heartbeat", summary="Transmit device presence and battery telemetry")
async def heartbeat(request: HeartbeatRequest):
    success = device_manager.heartbeat(
        device_id=request.device_id,
        battery_level=request.battery_level,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Device not found.")
    return {"status": "acknowledged"}


@router.post("/dispatch", response_model=RemoteTaskPayload, summary="Dispatch remote task to registered device")
async def dispatch_task(request: DispatchTaskRequest):
    try:
        payload = device_manager.dispatch_remote_task(
            target_name_or_id=request.target,
            action=request.action,
            parameters=request.parameters,
        )
        return payload
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{device_id}/tasks", response_model=List[RemoteTaskPayload], summary="Fetch pending remote tasks for a device")
async def get_device_tasks(device_id: str, clear: bool = True):
    return device_manager.registry.get_pending_tasks(device_id=device_id, clear=clear)


@router.post("/{device_id}/tasks/{task_id}/complete", summary="Report remote task completed")
async def complete_device_task(device_id: str, task_id: str, result: Optional[Dict[str, Any]] = None):
    device_manager.registry.complete_task(task_id=task_id, result_data=result)
    return {"status": "acknowledged", "task_id": task_id}


@router.delete("/{device_id}", summary="Revoke and disconnect device")
async def disconnect_device(device_id: str):
    success = device_manager.disconnect_device(device_id)
    if not success:
        raise HTTPException(status_code=404, detail="Device not found.")
    return {"message": f"Device '{device_id}' disconnected."}


# ── Phase 21 Authorized Device Extensions ────────────────────────────────────

@router.post("/discovery/scan", summary="Scan local network for beaconing companion devices")
async def discovery_scan():
    peers = device_manager.scan_network()
    return {"discovered_devices": peers, "total": len(peers)}


@router.post("/discovery/broadcast", summary="Broadcast presence beacon on local subnet")
async def discovery_broadcast():
    success = device_manager.broadcast_presence()
    return {"broadcast_sent": success}


@router.get("/folders", summary="List all explicitly authorized folders")
async def list_authorized_folders():
    return device_manager.list_authorized_folders()


@router.post("/folders/authorize", summary="Explicitly authorize a folder for device file operations")
async def authorize_folder(request: AuthorizeFolderRequest):
    try:
        record = device_manager.authorize_folder(
            folder_path=request.folder_path,
            access_level=request.access_level,
        )
        return record
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/folders/{folder_id}", summary="Revoke folder authorization")
async def revoke_folder(folder_id: str):
    success = device_manager.revoke_folder(folder_id)
    if not success:
        raise HTTPException(status_code=404, detail="Folder not found or already revoked.")
    return {"status": "revoked", "folder_id": folder_id}


@router.post("/notifications/send", summary="Send category-filtered notification to a device")
async def send_device_notification(request: SendNotificationRequest):
    try:
        notif = device_manager.send_notification(
            source_device=request.source_device,
            target_device=request.target_device,
            title=request.title,
            message=request.message,
            category=request.category,
            priority=request.priority,
        )
        return notif
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{device_id}/notifications", summary="Retrieve queued offline notifications for a device")
async def get_device_notifications(device_id: str):
    items = device_manager.get_queued_notifications(device_id)
    return {"notifications": items, "count": len(items)}


@router.post("/transfers/initiate", summary="Initiate a permission-gated file transfer")
async def initiate_file_transfer(request: InitiateTransferRequest):
    try:
        transfer = device_manager.transfer.initiate_transfer(
            source_device=request.source_device,
            target_device=request.target_device,
            file_name=request.file_name,
            file_size=request.file_size,
            sha256_hash=request.sha256_hash,
            target_folder=request.target_folder,
            require_consent=False,
        )
        return transfer
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=400, detail=str(e))
