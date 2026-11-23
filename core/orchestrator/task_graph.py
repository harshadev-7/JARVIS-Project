"""JARVIS Multi-Agent Orchestrator — Directed Acyclic Graph (DAG) Task Engine
=============================================================================
Manages structured task graphs with dependency resolution, topological staging,
circular dependency detection, and predecessor output context injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import asyncio


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CyclicDependencyError(ValueError):
    """Raised when a circular dependency is detected in the task graph."""
    pass


@dataclass
class RetryPolicy:
    """Configures automatic task retry behavior."""
    max_retries: int = 2
    base_delay_sec: float = 0.5
    backoff_factor: float = 2.0
    timeout_sec: float = 30.0


@dataclass
class TaskNode:
    """An atomic unit of work within a multi-agent orchestration mission."""
    task_id: str
    assigned_agent: str
    prompt: str
    dependencies: Set[str] = field(default_factory=set)
    context_keys: Dict[str, str] = field(default_factory=dict)  # maps task_id -> context_key
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    status: TaskStatus = TaskStatus.PENDING
    result_data: Any = None
    result_summary: str = ""
    error: Optional[str] = None
    retries_attempted: int = 0
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "assigned_agent": self.assigned_agent,
            "prompt": self.prompt,
            "dependencies": list(self.dependencies),
            "status": self.status.value,
            "result_summary": self.result_summary,
            "error": self.error,
            "retries_attempted": self.retries_attempted,
            "duration_sec": round(self.duration_sec, 2),
        }


class TaskGraph:
    """A Directed Acyclic Graph (DAG) coordinating multi-agent task execution."""

    def __init__(self, mission_id: str, objective: str):
        self.mission_id = mission_id
        self.objective = objective
        self.nodes: Dict[str, TaskNode] = {}

    def add_node(
        self,
        task_id: str,
        assigned_agent: str,
        prompt: str,
        dependencies: Optional[List[str]] = None,
        context_keys: Optional[Dict[str, str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> TaskNode:
        """Adds a new task node to the graph."""
        if task_id in self.nodes:
            raise ValueError(f"Task with ID '{task_id}' already exists in graph.")

        dep_set = set(dependencies) if dependencies else set()
        node = TaskNode(
            task_id=task_id,
            assigned_agent=assigned_agent,
            prompt=prompt,
            dependencies=dep_set,
            context_keys=context_keys or {},
            retry_policy=retry_policy or RetryPolicy(),
        )
        self.nodes[task_id] = node
        return node

    def validate_dag(self) -> None:
        """Validates that all dependencies exist and that there are no circular cycles."""
        # 1. Verify all dependencies exist
        for task_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Task '{task_id}' depends on non-existent task '{dep}'.")
                if dep == task_id:
                    raise CyclicDependencyError(f"Task '{task_id}' cannot depend on itself.")

        # 2. Cycle detection using Kahn's algorithm (indegree counting)
        in_degrees: Dict[str, int] = {k: 0 for k in self.nodes}
        adj_list: Dict[str, List[str]] = {k: [] for k in self.nodes}

        for task_id, node in self.nodes.items():
            in_degrees[task_id] = len(node.dependencies)
            for dep in node.dependencies:
                adj_list[dep].append(task_id)

        queue = [tid for tid, deg in in_degrees.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for nxt in adj_list[curr]:
                in_degrees[nxt] -= 1
                if in_degrees[nxt] == 0:
                    queue.append(nxt)

        if visited_count != len(self.nodes):
            raise CyclicDependencyError("Circular dependency detected in orchestration task graph.")

    def get_execution_stages(self) -> List[List[TaskNode]]:
        """Groups tasks into dependency-ordered stages for parallel execution."""
        self.validate_dag()

        # Compute longest path from any root for each node
        stages: Dict[int, List[TaskNode]] = {}
        depths: Dict[str, int] = {}

        def get_depth(tid: str) -> int:
            if tid in depths:
                return depths[tid]
            node = self.nodes[tid]
            if not node.dependencies:
                depths[tid] = 0
            else:
                depths[tid] = 1 + max(get_depth(d) for d in node.dependencies)
            return depths[tid]

        for tid in self.nodes:
            d = get_depth(tid)
            stages.setdefault(d, []).append(self.nodes[tid])

        return [stages[k] for k in sorted(stages.keys())]

    def build_task_context(self, task_id: str, global_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assembles the input context dictionary for a task by collecting predecessor outputs."""
        node = self.nodes.get(task_id)
        if not node:
            return global_context.copy() if global_context else {}

        ctx: Dict[str, Any] = global_context.copy() if global_context else {}
        ctx["predecessor_results"] = {}

        for dep_id in node.dependencies:
            dep_node = self.nodes.get(dep_id)
            if dep_node and dep_node.status == TaskStatus.COMPLETED:
                key = node.context_keys.get(dep_id, dep_id)
                ctx[key] = dep_node.result_data or dep_node.result_summary
                ctx["predecessor_results"][dep_id] = {
                    "agent": dep_node.assigned_agent,
                    "summary": dep_node.result_summary,
                    "data": dep_node.result_data,
                }

        return ctx

    def is_complete(self) -> bool:
        """Returns True if every node has reached a terminal status."""
        return all(
            node.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
            for node in self.nodes.values()
        )

    def is_all_successful(self) -> bool:
        """Returns True if all nodes finished successfully with no failures or skips."""
        return all(node.status == TaskStatus.COMPLETED for node in self.nodes.values())
