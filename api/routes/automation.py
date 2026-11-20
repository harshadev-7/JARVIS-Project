"""
JARVIS AI — Windows Computer Automation API Routes
===================================================
Endpoints for system performance monitoring, application control,
and safe terminal execution.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.automation.system_monitor import system_monitor, SystemMetrics
from services.automation.app_controller import app_controller, AppActionResult
from services.automation.command_executor import command_executor, CommandResult

router = APIRouter(prefix="/automation", tags=["Windows Computer Automation"])


class AppLaunchRequest(BaseModel):
    app: str = Field(..., min_length=1, description="Application name or executable (e.g. 'notepad', 'code')")
    autonomy_level: int = Field(4, ge=1, le=5)


class CommandExecuteRequest(BaseModel):
    command: str = Field(..., min_length=1, description="Shell command to run")
    autonomy_level: int = Field(4, ge=1, le=5)
    timeout_sec: float = Field(30.0, ge=1.0, le=120.0)


@router.get("/system/metrics", response_model=SystemMetrics, summary="Get real-time OS & Hardware performance")
async def get_metrics():
    return system_monitor.get_metrics()


@router.post("/app/launch", response_model=AppActionResult, summary="Launch authorized desktop application")
async def launch_app(request: AppLaunchRequest):
    return app_controller.launch_app(request.app, autonomy_level=request.autonomy_level)


@router.post("/app/close", response_model=AppActionResult, summary="Close desktop application")
async def close_app(request: AppLaunchRequest):
    return app_controller.close_app(request.app, autonomy_level=request.autonomy_level)


@router.post("/terminal/execute", response_model=CommandResult, summary="Execute safe terminal command")
async def execute_command(request: CommandExecuteRequest):
    return await command_executor.execute(
        command=request.command,
        autonomy_level=request.autonomy_level,
        timeout_sec=request.timeout_sec,
    )
