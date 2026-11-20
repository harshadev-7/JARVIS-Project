"""
JARVIS AI — Task Decomposer
============================
Breaks down high-level objectives into sequential and parallel subtasks,
assigning required capabilities, dependencies, and verification criteria.
"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SubTask(BaseModel):
    """An individual actionable step in a multi-step plan."""
    id: str = Field(..., description="Unique step identifier e.g. step-1")
    title: str = Field(..., description="Short name of the action")
    description: str = Field(..., description="Specific details of what to perform")
    assigned_agent: str = Field(default="CentralAI", description="Specialized agent responsible")
    dependencies: List[str] = Field(default_factory=list, description="IDs of steps that must finish first")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    estimated_duration_sec: int = Field(default=30)
    verification_check: str = Field(default="", description="How to confirm this step succeeded")
    fallback_plan: Optional[str] = Field(default=None, description="Alternative action if step fails")


class TaskPlan(BaseModel):
    """Complete multi-step execution plan."""
    plan_id: str
    goal: str
    subtasks: List[SubTask] = Field(default_factory=list)
    total_steps: int = 0
    estimated_total_duration_sec: int = 0

    def calculate_metrics(self):
        self.total_steps = len(self.subtasks)
        self.estimated_total_duration_sec = sum(t.estimated_duration_sec for t in self.subtasks)


class TaskDecomposer:
    """Decomposes goals into structured task plans."""

    def create_linear_plan(self, goal: str, steps: List[dict], plan_id: str = "plan-001") -> TaskPlan:
        """Construct a sequential TaskPlan from step definitions."""
        subtasks: List[SubTask] = []
        prev_id: Optional[str] = None

        for idx, s in enumerate(steps, 1):
            step_id = s.get("id", f"step-{idx}")
            deps = s.get("dependencies", [prev_id] if prev_id else [])
            
            task = SubTask(
                id=step_id,
                title=s.get("title", f"Step {idx}"),
                description=s.get("description", ""),
                assigned_agent=s.get("assigned_agent", "CentralAI"),
                dependencies=deps,
                priority=s.get("priority", TaskPriority.MEDIUM),
                verification_check=s.get("verification_check", "Check step exit code or output"),
                fallback_plan=s.get("fallback_plan"),
            )
            subtasks.append(task)
            prev_id = step_id

        plan = TaskPlan(
            plan_id=plan_id,
            goal=goal,
            subtasks=subtasks,
        )
        plan.calculate_metrics()
        return plan


task_decomposer = TaskDecomposer()
