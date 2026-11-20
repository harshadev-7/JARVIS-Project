"""
JARVIS AI — Reasoning & Planning API Routes
============================================
Exposes endpoints for the 10-step reasoning engine, goal decomposition,
multi-approach evaluation, and outcome verification.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.planner.reasoning_engine import reasoning_engine, ReasoningCycleResult
from core.planner.task_decomposer import task_decomposer, TaskPlan
from core.planner.goal_analyzer import goal_analyzer, GoalAnalysis
from core.logger import log

router = APIRouter(prefix="/reasoning", tags=["Reasoning & Problem-Solving"])


class ProblemSolveRequest(BaseModel):
    problem: str = Field(..., min_length=3, description="Problem or high-level objective to solve")
    autonomy_level: Optional[int] = Field(None, ge=1, le=5, description="Autonomy level override (1-5)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contextual facts or variables")


class PlanGoalRequest(BaseModel):
    goal: str = Field(..., min_length=3, description="Goal to decompose into subtasks")
    steps: Optional[List[Dict[str, Any]]] = Field(None, description="Optional custom step blueprints")


@router.post(
    "/solve",
    response_model=ReasoningCycleResult,
    summary="Execute 10-Step Reasoning Cycle",
    description="Analyzes goal, facts vs assumptions, generates multiple approaches, evaluates them, and creates a verified plan.",
)
async def solve_problem(request: ProblemSolveRequest):
    """Run full 10-step problem-solving sequence."""
    try:
        result = reasoning_engine.solve(
            problem_description=request.problem,
            autonomy_level=request.autonomy_level,
            context=request.context,
        )
        return result
    except Exception as e:
        log.error(f"Error in reasoning route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze",
    response_model=GoalAnalysis,
    summary="Analyze Objective & Classify Facts/Assumptions",
)
async def analyze_goal(request: ProblemSolveRequest):
    """Breakdown objective into facts, assumptions, uncertainties, and constraints."""
    return goal_analyzer.analyze_heuristic(request.problem)


@router.post(
    "/plan",
    response_model=TaskPlan,
    summary="Generate Execution Plan",
)
async def generate_plan(request: PlanGoalRequest):
    """Decompose high-level goal into sequential subtasks."""
    steps = request.steps or [
        {"title": "Initial Assessment", "description": f"Gather parameters for {request.goal}"},
        {"title": "Execute Primary Action", "description": f"Implement core logic for {request.goal}"},
        {"title": "Verify & Validate", "description": "Ensure all requirements were met"},
    ]
    return task_decomposer.create_linear_plan(goal=request.goal, steps=steps)
