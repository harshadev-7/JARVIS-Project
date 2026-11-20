"""
JARVIS AI — Health & Status Routes
====================================
System health checks and status endpoints.
Used by monitoring tools, Docker health checks, and the UI dashboard.
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel, Field
from api.schemas.models import HealthResponse
from core.memory.memory_manager import memory_manager
from config.settings import settings

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
            "reasoning_engine",
            "long_term_memory",
            "internet_research",
            "voice_assistant",
            "computer_automation",
            "multi_agent_system",
        ],
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
