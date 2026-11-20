"""
JARVIS AI — Multi-Device Network API Routes
============================================
Endpoints for device pairing, authentication, heartbeat telemetry,
and cross-device task dispatching.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from devices.device_registry.registry import (
    device_registry,
    DeviceRecord,
    DeviceType,
    RemoteTaskPayload,
)

router = APIRouter(prefix="/devices", tags=["Multi-Device Network"])


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


@router.post("/pairing-code", response_model=PairingCodeResponse, summary="Generate a 6-digit one-time device pairing code")
async def generate_pairing_code():
    code = device_registry.generate_pairing_code()
    return PairingCodeResponse(pairing_code=code)


@router.post("/pair", response_model=DeviceRecord, summary="Pair a new device using pairing code")
async def pair_device(request: PairDeviceRequest):
    try:
        record = device_registry.pair_device(
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
    return device_registry.list_devices()


@router.post("/heartbeat", summary="Transmit device presence and battery telemetry")
async def heartbeat(request: HeartbeatRequest):
    success = device_registry.heartbeat(
        device_id=request.device_id,
        battery_level=request.battery_level,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Device not found.")
    return {"status": "acknowledged"}


@router.post("/dispatch", response_model=RemoteTaskPayload, summary="Dispatch remote task to registered device")
async def dispatch_task(request: DispatchTaskRequest):
    try:
        payload = device_registry.dispatch_remote_task(
            target_name_or_id=request.target,
            action=request.action,
            parameters=request.parameters,
        )
        return payload
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{device_id}", summary="Revoke and disconnect device")
async def disconnect_device(device_id: str):
    success = device_registry.disconnect_device(device_id)
    if not success:
        raise HTTPException(status_code=404, detail="Device not found.")
    return {"message": f"Device '{device_id}' disconnected."}
