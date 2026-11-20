"""J.A.R.V.I.S. Planning Agent (Phase 9).

Specialized agent for autonomous goal analysis, 10-step problem-solving,
strategy evaluation, and subtask decomposition.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent


class PlanningAgent(BaseAgent):
    """Independent specialized agent for planning, reasoning, and task decomposition."""

    def __init__(self):
        super().__init__(
            name="PlanningAgent",
            role="Analyzes complex goals, evaluates alternative solutions, and drafts ordered execution plans.",
            capabilities=["goal_analysis", "10_step_reasoning", "task_decomposition", "risk_assessment"],
            required_autonomy_level=2,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute goal planning and multi-step decomposition."""
        start_time = time.perf_counter()
        clean_goal = task.replace("/solve", "").strip()

        try:
            from core.planner.reasoning_engine import reasoning_engine
            autonomy = (context or {}).get("autonomy_level", 2)
            solution = reasoning_engine.solve(
                problem_description=clean_goal,
                autonomy_level=autonomy,
                context=context,
            )
            duration = time.perf_counter() - start_time

            plan_tasks = [
                {"id": t.id, "title": t.title, "description": t.description}
                for t in solution.step_8_execution_plan.subtasks
            ]

            summary = (
                f"**Strategic Execution Plan formulated for: '{clean_goal}'**\n\n"
                f"- **Primary Goal:** {solution.step_1_objective_understanding.primary_goal}\n"
                f"- **Selected Strategy:** {solution.step_7_selected_solution.name} (Score: {solution.step_7_selected_solution.composite_score})\n"
                f"- **Safety Authorization:** {solution.step_6_evaluation.authorization_note}\n\n"
                f"**Ordered Subtasks ({len(plan_tasks)}):**\n"
                + "\n".join([f"  {idx+1}. [{t['id']}] {t['title']}: {t['description']}" for idx, t in enumerate(plan_tasks)])
            )

            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"plan_{int(time.time())}",
                success=True,
                data={
                    "primary_goal": solution.step_1_objective_understanding.primary_goal,
                    "selected_strategy": solution.step_7_selected_solution.name,
                    "subtasks": plan_tasks,
                    "verification": solution.step_9_verification.actual_outcome,
                },
                summary=summary,
                duration_sec=duration,
            )

        except Exception as e:
            log.error(f"PlanningAgent execution error: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"plan_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Failed to formulate strategic plan for '{clean_goal}': {e}",
                duration_sec=duration,
            )


__all__ = ["PlanningAgent"]
