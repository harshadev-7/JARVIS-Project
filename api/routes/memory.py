"""
JARVIS AI — Memory & Learning API Routes
=========================================
Endpoints for managing Long-Term Memory, User Preferences,
Task History & Lessons, and Vector Semantic Search.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.memory.persistent_memory import persistent_memory

router = APIRouter(prefix="/memory", tags=["Memory & Learning"])


class SetPreferenceRequest(BaseModel):
    key: str = Field(..., min_length=1, description="Preference key e.g. 'theme', 'default_browser'")
    value: str = Field(..., min_length=1, description="Preference value")
    category: str = Field("general", description="Category grouping")
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class AddKnowledgeRequest(BaseModel):
    title: str = Field(..., min_length=2, description="Title of knowledge entry")
    content: str = Field(..., min_length=5, description="Factual content or documentation")
    category: str = Field("general", description="Domain classification")
    source: str = Field("user", description="Author or origin")


class TaskLogRequest(BaseModel):
    task_id: str
    goal: str
    status: str
    steps_count: int = 1
    duration_sec: float = 0.0
    error_summary: Optional[str] = None
    lessons_learned: Optional[str] = None


@router.post("/preferences", summary="Set or update a user preference")
async def set_preference(request: SetPreferenceRequest):
    pref = await persistent_memory.set_preference(
        key=request.key,
        value=request.value,
        category=request.category,
        confidence=request.confidence,
    )
    return {"message": "Preference saved successfully", "key": request.key, "value": request.value}


@router.get("/preferences", summary="List user preferences")
async def list_preferences(category: Optional[str] = None):
    return await persistent_memory.list_preferences(category=category)


@router.post("/knowledge", summary="Store verified knowledge")
async def store_knowledge(request: AddKnowledgeRequest):
    item_id = await persistent_memory.store_knowledge(
        title=request.title,
        content=request.content,
        category=request.category,
        source=request.source,
    )
    return {"message": "Knowledge stored and indexed in vector memory", "id": item_id}


@router.get("/knowledge/search", summary="Semantic search across knowledge base")
async def search_knowledge(
    q: str = Query(..., min_length=2, description="Semantic query text"),
    top_k: int = Query(5, ge=1, le=20),
):
    results = await persistent_memory.search_knowledge(query=q, top_k=top_k)
    return {"query": q, "results": results, "total": len(results)}


@router.post("/tasks/log", summary="Log task execution and lessons learned")
async def log_task(request: TaskLogRequest):
    await persistent_memory.log_task_execution(
        task_id=request.task_id,
        goal=request.goal,
        status=request.status,
        steps_count=request.steps_count,
        duration_sec=request.duration_sec,
        error_summary=request.error_summary,
        lessons_learned=request.lessons_learned,
    )
    return {"message": "Task history logged and indexed"}


@router.get("/tasks/mistakes", summary="Retrieve past mistakes matching a goal")
async def get_mistakes(goal: str = Query(..., min_length=2)):
    mistakes = await persistent_memory.get_mistakes_to_avoid(goal)
    return {"goal": goal, "mistakes_to_avoid": mistakes}


@router.delete("/{category}/{item_id}", summary="Delete memory item")
async def delete_memory_item(category: str, item_id: int):
    try:
        deleted = await persistent_memory.delete_item(category, item_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"message": f"Deleted item {item_id} from {category}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
