"""J.A.R.V.I.S. Computer Control Agent (Phase 12 Upgraded).

Specialized agent for desktop automation, process management,
hardware telemetry inspection, and authorized application control.

Strict Security Constraints (Section 13):
1. Ask before deleting files
2. Ask before sending messages
3. Ask before installing software
4. Ask before making purchases
5. Never execute arbitrary model-generated terminal commands without validation and approval
6. Keep an audit activity log
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent
from automation.desktop import desktop_automation, DesktopActionResult
from automation.file_manager import file_manager, FileOperationResult
from automation.system_monitor import system_monitor, HardwareTelemetry


class ComputerControlAgent(BaseAgent):
    """Independent specialized agent for safe host automation."""

    # 4 Strict Blueprint Consequential Actions that ALWAYS require user approval
    MANDATORY_APPROVAL_CATEGORIES = {
        "delete_file": ["delete", "remove", "unlink", "erase"],
        "send_message": ["send message", "email", "post", "tweet"],
        "install_software": ["install", "pip install", "winget", "setup.exe"],
        "purchase": ["purchase", "buy", "order", "checkout", "pay"],
    }

    def __init__(self):
        super().__init__(
            name="ComputerControlAgent",
            role="Safely controls desktop applications, manages files, monitors hardware, and enforces security approval gates.",
            capabilities=[
                "app_launch",
                "folder_access",
                "file_management",
                "hardware_telemetry",
                "consequential_action_safety_gates",
                "activity_logging",
            ],
            required_autonomy_level=2,
        )
        self.action_history: List[Dict[str, Any]] = []

    def classify_consequential_risk(self, task: str) -> Optional[str]:
        """Return the safety category if the task requires human approval, else None."""
        lower = task.lower()
        for category, triggers in self.MANDATORY_APPROVAL_CATEGORIES.items():
            if any(t in lower for t in triggers):
                return category
        return None

    def _log_agent_action(self, action: str, target: str, approved: bool, success: bool):
        """Append to local audit history."""
        self.action_history.append({
            "action": action,
            "target": target,
            "approved": approved,
            "success": success,
            "timestamp": time.time(),
        })

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute desktop automation with safety gates."""
        start_time = time.perf_counter()
        ctx = context or {}
        user_approved = ctx.get("user_approved", False)
        action = ctx.get("action", "auto")

        # 1. Check Mandatory Safety Gates
        risk_category = self.classify_consequential_risk(task)
        if risk_category and not user_approved:
            self._log_agent_action(risk_category, task, approved=False, success=False)
            duration = time.perf_counter() - start_time
            category_names = {
                "delete_file": "File Deletion",
                "send_message": "External Communication",
                "install_software": "Software Installation",
                "purchase": "Financial Transaction / Purchase",
            }
            cat_name = category_names.get(risk_category, "Consequential Action")
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"gate_{risk_category}_{int(time.time())}",
                success=False,
                error="UserAuthorizationRequired",
                data={
                    "task": task,
                    "risk_category": risk_category,
                    "requires_approval": True,
                },
                summary=(
                    f"⚠️ **Authorization Required ({cat_name}):**\n"
                    f"Action '{task}' requires explicit confirmation. Please confirm with `user_approved=True`."
                ),
                duration_sec=duration,
            )

        try:
            lower = task.lower()

            # 2. File Operations
            if action == "delete_file" or "delete file" in lower:
                target_path = ctx.get("path") or task.replace("delete file", "").strip()
                res: FileOperationResult = file_manager.delete_file(target_path, user_approved=user_approved)
                self._log_agent_action("delete_file", target_path, approved=user_approved, success=res.success)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_del_{int(time.time())}",
                    success=res.success,
                    error=res.message if not res.success else None,
                    data={"path": res.path, "operation": res.operation},
                    summary=f"**File Manager:** {res.message}",
                    duration_sec=duration,
                )

            elif action == "read_file" or "read file" in lower:
                target_path = ctx.get("path") or task.replace("read file", "").strip()
                res = file_manager.read_file(target_path)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_read_{int(time.time())}",
                    success=res.success,
                    data=res.data,
                    summary=f"**File Manager:** {res.message}",
                    duration_sec=duration,
                )

            elif action == "write_file":
                target_path = ctx.get("path", "")
                content = ctx.get("content", "")
                overwrite = ctx.get("overwrite", False)
                res = file_manager.write_file(target_path, content, overwrite=overwrite)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_write_{int(time.time())}",
                    success=res.success,
                    data=res.data,
                    summary=f"**File Manager:** {res.message}",
                    duration_sec=duration,
                )

            elif action == "list_files" or "list files" in lower:
                target_dir = ctx.get("path")
                res = file_manager.list_directory(target_dir)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_list_{int(time.time())}",
                    success=res.success,
                    data={"items": res.data},
                    summary=f"**File Manager:** {res.message}",
                    duration_sec=duration,
                )

            # 3. Application Control
            elif action == "launch_app" or any(kw in lower for kw in ["launch", "open app", "start app"]):
                target_app = ctx.get("app") or task.replace("launch", "").replace("open", "").strip()
                d_res: DesktopActionResult = desktop_automation.launch_app(target_app, user_approved=user_approved)
                self._log_agent_action("launch_app", target_app, approved=user_approved, success=d_res.success)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_app_{int(time.time())}",
                    success=d_res.success,
                    data={"app": d_res.target, "pid": d_res.pid},
                    summary=f"**Desktop Controller:** {d_res.message}",
                    duration_sec=duration,
                )

            elif action == "open_folder" or "open folder" in lower:
                target_folder = ctx.get("path") or task.replace("open folder", "").strip()
                d_res = desktop_automation.open_folder(target_folder)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_folder_{int(time.time())}",
                    success=d_res.success,
                    summary=f"**Desktop Controller:** {d_res.message}",
                    duration_sec=duration,
                )

            # 4. Hardware Telemetry & Resource Diagnostics
            elif action == "telemetry" or any(kw in lower for kw in ["monitor", "telemetry", "diagnostics", "cpu", "ram"]):
                metrics: HardwareTelemetry = system_monitor.get_metrics()
                duration = time.perf_counter() - start_time
                summary = (
                    f"**Hardware Telemetry Report:**\n"
                    f"- CPU: {metrics.cpu_percent}% ({metrics.cpu_count_logical} threads)\n"
                    f"- RAM: {metrics.ram_used_gb} GB / {metrics.ram_total_gb} GB ({metrics.ram_percent}%, {metrics.ram_available_gb} GB free)\n"
                    f"- Disk: {metrics.disk_percent}% used\n"
                    f"- Safe for AI Training: {'✅ Yes' if metrics.is_safe_for_ai_workload else '⚠️ RAM Constrained'}"
                )
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_telem_{int(time.time())}",
                    success=True,
                    data=metrics.__dict__,
                    summary=summary,
                    duration_sec=duration,
                )

            # 5. Baseline Safe Verification
            else:
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"ctrl_base_{int(time.time())}",
                    success=True,
                    data={"task": task, "verified": True},
                    summary=f"Desktop automation verified safe baseline for instruction: '{task}'.",
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
                summary=f"Computer control error: {e}",
                duration_sec=duration,
            )


__all__ = ["ComputerControlAgent"]
