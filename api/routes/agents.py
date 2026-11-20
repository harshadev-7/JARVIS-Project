"""
JARVIS AI — Multi-Agent Orchestration API Routes
=================================================
Endpoints for viewing the 15 specialized agents and executing
coordinated multi-agent missions.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.orchestrator.agent_registry import agent_registry
from core.orchestrator.orchestrator import orchestrator, OrchestrationMission
from config.settings import settings

router = APIRouter(prefix="/agents", tags=["Multi-Agent System"])


class MissionRequest(BaseModel):
    objective: str = Field(..., min_length=2, description="High-level objective requiring agent collaboration")
    agents: Optional[List[str]] = Field(None, description="Optional explicit agent keys to deploy")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


@router.get("/list", summary="List all 15 specialized AI agents and their capabilities")
async def list_agents():
    return agent_registry.list_all()


@router.get("/status", summary="Return live, HUD-friendly status metadata for each agent")
async def list_agent_status():
    status_map = {
        "research": "standby",
        "problem_solving": "ready",
        "programming": "ready",
        "mathematical": "ready",
        "data_analysis": "ready",
        "computer_automation": "ready",
        "device_management": "ready",
        "social_media": "ready",
        "content_creation": "ready",
        "video_editing": "ready",
        "personal_assistant": "ready",
        "memory_management": "active",
        "cybersecurity": "shielded",
        "communication": "ready",
        "task_planning": "ready",
    }

    roster = []
    for agent in agent_registry.list_all():
        agent_id = agent["id"]
        roster.append({
            "id": agent_id,
            "name": agent["name"],
            "role": agent["role"],
            "status": status_map.get(agent_id, "ready"),
            "required_autonomy_level": agent["required_autonomy_level"],
        })
    return roster


@router.get("/missions", summary="Return the recent multi-agent mission ledger for the HUD")
async def list_missions():
    return [mission.model_dump(mode="json") for mission in orchestrator.list_recent_missions()]


@router.get("/missions/active", summary="Return the currently active mission snapshots visible to the HUD")
async def list_active_missions():
    return [mission.model_dump(mode="json") for mission in orchestrator.list_active_missions()]


@router.post("/missions/demo", response_model=OrchestrationMission, summary="Create a single-click demo mission for the HUD")
async def create_demo_mission(request: MissionRequest):
    objective = request.objective or "Run a local readiness audit for the command interface"
    agents = request.agents or ["task_planning", "cybersecurity", "problem_solving"]
    try:
        return await orchestrator.execute_mission(
            objective=objective,
            required_agents=agents,
            context={
                "source": "hud_demo",
                "autonomy_level": settings.DEFAULT_AUTONOMY_LEVEL,
                "initiative": "quick-start local mission",
            },
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/missions/{mission_id}", response_model=OrchestrationMission, summary="Return a specific multi-agent mission")
async def get_mission_details(mission_id: str):
    mission = orchestrator.get_mission(mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
    return mission


@router.post("/orchestrate", response_model=OrchestrationMission, summary="Coordinate multi-agent mission")
async def orchestrate_mission(request: MissionRequest):
    try:
        return await orchestrator.execute_mission(
            objective=request.objective,
            required_agents=request.agents,
            context=request.context,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
