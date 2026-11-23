"""
JARVIS AI — Health & Status Routes
====================================
System health checks and status endpoints.
Used by monitoring tools, Docker health checks, and the UI dashboard.
"""

import time
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from api.schemas.models import HealthResponse
from core.memory.memory_manager import memory_manager
from config.settings import settings
from core.integration.subsystem_registry import subsystem_registry
from core.integration.system_diagnostics import system_diagnostics
from core.integration.master_runtime import master_runtime
from core.sentinel import sentinel_daemon

router = APIRouter(prefix="/system", tags=["System"])


class AutonomyLevelRequest(BaseModel):
    level: int = Field(..., ge=1, le=5, description="Autonomy level to apply, from 1 to 5")

# Track server start time for uptime calculation
_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check JARVIS system health",
)
async def health_check():
    """Returns current system health and configuration status."""
    uptime = time.time() - _start_time
    return HealthResponse(
        status="operational",
        version=settings.JARVIS_VERSION,
        provider=settings.AI_PROVIDER,
        model=_get_active_model(),
        environment=settings.ENVIRONMENT,
        uptime_seconds=round(uptime, 2),
    )


@router.get("/autonomy", summary="Read the current autonomy level and label")
async def get_autonomy_status():
    labels = {
        1: "advisor",
        2: "researcher",
        3: "planner",
        4: "executor",
        5: "autonomous",
    }
    level = settings.DEFAULT_AUTONOMY_LEVEL
    return {
        "autonomy_level": level,
        "label": labels.get(level, "researcher"),
        "display": f"LEVEL {level} [{labels.get(level, 'RESEARCHER').upper()}]",
    }


@router.post("/autonomy", summary="Set the current autonomy level")
async def set_autonomy_status(request: AutonomyLevelRequest):
    settings.DEFAULT_AUTONOMY_LEVEL = request.level
    return await get_autonomy_status()


@router.get(
    "/sessions",
    summary="List all active conversation sessions",
)
async def list_sessions():
    """Returns all active JARVIS conversation sessions."""
    sessions = memory_manager.list_sessions()
    return {
        "sessions": sessions,
        "total": len(sessions),
    }


@router.get(
    "/info",
    summary="JARVIS system information",
)
async def system_info():
    """Returns detailed system configuration (non-sensitive)."""
    return {
        "name": settings.JARVIS_NAME,
        "version": settings.JARVIS_VERSION,
        "environment": settings.ENVIRONMENT,
        "ai_provider": settings.AI_PROVIDER,
        "ai_model": _get_active_model(),
        "autonomy_level": settings.DEFAULT_AUTONOMY_LEVEL,
        "debug_mode": settings.DEBUG,
        "phase": "Phase 1 — Core AI Brain",
        "capabilities": [
            "natural_language_conversation",
            "multi_turn_memory",
            "context_awareness",
            "multi_provider_ai",
        ],
        "coming_soon": [
            "voice_assistant",
            "computer_automation",
            "multi_agent_system",
        ],
    }


@router.get("/subsystems", summary="List all registered subsystems with live health metrics")
async def get_subsystems(category: Optional[str] = None):
    """Retrieves all registered subsystems with their health and phase metadata."""
    subs = subsystem_registry.list_subsystems(category=category)
    probe = subsystem_registry.probe_all_health()
    return {
        "total": len(subs),
        "health_summary": probe,
        "subsystems": [s.to_dict() for s in subs],
    }


@router.post("/diagnostics/run", summary="Run automated self-tests across all subsystems")
async def run_diagnostics():
    """Runs on-demand non-destructive diagnostic smoke tests across all subsystems."""
    rep = await system_diagnostics.run_full_diagnostics()
    return {
        "status": "success",
        "report": rep.to_dict(),
        "summary_text": rep.format_terminal_summary(),
    }


@router.get("/runtime", summary="Get master runtime status")
async def get_runtime_status():
    """Returns the operational status, uptime, and state of the master runtime."""
    return master_runtime.get_status()


@router.get("/sentinel", summary="Get autonomous sentinel daemon status")
async def get_sentinel_status():
    """Returns the active status, recent advisories, and hygiene history of the sentinel daemon."""
    return sentinel_daemon.get_status()


@router.post("/sentinel/maintenance", summary="Run autonomous self-healing and routine maintenance now")
async def trigger_maintenance(max_file_age_hours: float = 24.0):
    """Forces an immediate execution of routine maintenance and cache cleanup."""
    res = await sentinel_daemon.run_maintenance_now()
    return {
        "status": "success",
        "maintenance": res.to_dict(),
    }


def _get_active_model() -> str:
    """Return the currently configured model name."""
    model_map = {
        "openai": settings.OPENAI_MODEL,
        "gemini": settings.GEMINI_MODEL,
        "anthropic": settings.ANTHROPIC_MODEL,
        "ollama": settings.OLLAMA_MODEL,
    }
    return model_map.get(settings.AI_PROVIDER, "unknown")
