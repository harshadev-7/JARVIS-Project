"""
JARVIS AI — Cybersecurity & Privacy API Routes
===============================================
Endpoints for security audit logs, emergency shutdown kill switch,
and rate limiting protection.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from services.security.audit_logger import audit_logger, SecurityAuditEvent
from services.security.access_control import security_controller

router = APIRouter(prefix="/security", tags=["Cybersecurity & Privacy"])


class EmergencyShutdownRequest(BaseModel):
    authorized_by: str = Field("primary_user", description="Identity triggering shutdown")
    reason: str = Field(..., min_length=3, description="Justification for emergency halt")


@router.get("/audit", response_model=List[SecurityAuditEvent], summary="Inspect tamper-evident audit logs")
async def get_audit_trail(limit: int = 50):
    return audit_logger.get_recent_events(limit=limit)


@router.post("/emergency-shutdown", summary="EMERGENCY KILL-SWITCH: Halt all automation immediately")
async def trigger_emergency_shutdown(request: EmergencyShutdownRequest):
    security_controller.trigger_emergency_shutdown(authorized_by=request.authorized_by)
    return {
        "status": "EMERGENCY_SHUTDOWN_ACTIVE",
        "message": f"Kill-switch engaged by {request.authorized_by}. All automations halted.",
        "reason": request.reason,
    }


@router.post("/disarm", summary="Disarm emergency halt and restore operations")
async def disarm_emergency_shutdown(admin_key: str):
    if admin_key != "jarvis-admin-restore":
        raise HTTPException(status_code=403, detail="Invalid admin authorization key.")
    security_controller.reset_emergency_shutdown(authorized_by="admin")
    return {"status": "OPERATIONAL", "message": "Emergency shutdown disarmed."}


@router.get("/status", summary="Check system security lockdown status")
async def get_security_status():
    return {
        "emergency_shutdown_active": security_controller.emergency_shutdown_active,
        "rate_limiting_active": True,
        "audit_logging_active": True,
    }
