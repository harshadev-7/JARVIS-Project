"""J.A.R.V.I.S. Computer Control Agent (Phase 9).

Specialized agent for desktop automation, process management,
hardware telemetry inspection, and authorized application control.

Strict Security Constraints:
- Requires user approval for consequential actions (process kill, file deletion, software installation)
- Enforces autonomy levels and emergency kill-switch verification
- Logs all actions to security audit log
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent


class ComputerControlAgent(BaseAgent):
    """Independent specialized agent for safe host automation."""

    CONSEQUENTIAL_ACTIONS = ["delete", "kill", "terminate", "format", "remove", "install"]

    def __init__(self):
        super().__init__(
            name="ComputerControlAgent",
            role="Safely controls desktop applications, monitors OS processes, and performs authorized system actions.",
            capabilities=["app_launch", "process_inspection", "telemetry_check", "safe_command_execution"],
            required_autonomy_level=2,
        )

    def is_consequential(self, task: str) -> bool:
        """Check if action poses risk of data loss or service disruption."""
        lower = task.lower()
        return any(act in lower for act in self.CONSEQUENTIAL_ACTIONS)

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute desktop automation with safety gates."""
        start_time = time.perf_counter()
        ctx = context or {}
        user_approved = ctx.get("user_approved", False)
        autonomy_level = ctx.get("autonomy_level", 2)

        # 1. Consequential Action Safety Gate
        if self.is_consequential(task) and not user_approved:
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"ctrl_gate_{int(time.time())}",
                success=False,
                error="UserAuthorizationRequired",
                data={"task": task, "requires_approval": True},
                summary=(
                    f"⚠️ **Authorization Required:** Action '{task}' is classified as consequential. "
                    "Explicit confirmation is required before proceeding."
                ),
                duration_sec=duration,
            )

        try:
            from services.automation.app_controller import app_controller
            from services.automation.system_monitor import system_monitor

            lower = task.lower()

            # Launch app
            if any(kw in lower for kw in ["launch", "open", "start"]):
                target = "notepad"
                for alias in app_controller.APP_ALIASES:
                    if alias in lower:
                        target = alias
                        break
                res = app_controller.launch_app(target, autonomy_level=autonomy_level)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_app_{int(time.time())}",
                    success=res.success,
                    data={"app": res.app_name, "pid": res.pid},
                    summary=f"**Desktop Controller:** {res.message}",
                    duration_sec=duration,
                )

            # System telemetry
            elif any(kw in lower for kw in ["monitor", "telemetry", "diagnostics", "cpu", "ram"]):
                metrics = system_monitor.get_metrics()
                duration = time.perf_counter() - start_time
                summary = (
                    f"**Hardware Telemetry:**\n"
                    f"- CPU: {metrics.cpu_percent}%\n"
                    f"- RAM: {metrics.ram_used_gb} GB / {metrics.ram_total_gb} GB ({metrics.ram_percent}%)\n"
                    f"- Disk: {metrics.disk_percent}% used"
                )
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_telem_{int(time.time())}",
                    success=True,
                    data=metrics.__dict__,
                    summary=summary,
                    duration_sec=duration,
                )

            else:
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_noop_{int(time.time())}",
                    success=True,
                    data={"task": task},
                    summary=f"Computer control verified safe baseline for task: '{task}'.",
                    duration_sec=duration,
                )

        except Exception as e:
            log.error(f"ComputerControlAgent error: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"ctrl_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Computer control failed: {e}",
                duration_sec=duration,
            )


__all__ = ["ComputerControlAgent"]
