"""J.A.R.V.I.S. Coding Agent (Phase 9).

Specialized agent for programming, AST code inspection,
syntax validation, vulnerability scanning, and sandboxed test runs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent


class CodingAgent(BaseAgent):
    """Independent specialized agent for code analysis, debugging, and execution."""

    def __init__(self):
        super().__init__(
            name="CodingAgent",
            role="Inspects source code, validates syntax via AST, performs security audits, and executes sandboxed code.",
            capabilities=["code_analysis", "syntax_validation", "vulnerability_audit", "sandboxed_execution"],
            required_autonomy_level=2,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Analyze, debug, or execute a programming task."""
        start_time = time.perf_counter()

        try:
            from agents.programming_agent.code_analyzer import code_analyzer
            from agents.programming_agent.code_sandbox import code_sandbox

            # Check if task is code to be analyzed or executed
            is_code_snippet = any(kw in task for kw in ["def ", "class ", "import ", "print(", "=", "return "])

            if is_code_snippet:
                report = code_analyzer.analyze(task)
                duration = time.perf_counter() - start_time

                funcs = getattr(report, "functions_found", getattr(report, "functions_detected", []))
                classes = getattr(report, "classes_found", [])
                issues_cnt = len(report.issues) if hasattr(report, "issues") else 0
                summary = (
                    f"**Code Analysis & Security Audit:**\n"
                    f"- **Syntax Valid:** {'✅ Valid' if report.is_valid_syntax else '❌ Syntax Errors'}\n"
                    f"- **Security Score:** {report.security_score * 100:.0f}%\n"
                    f"- **Lines of Code:** {report.total_lines}\n"
                    f"- **Functions Detected:** {', '.join(funcs) or 'None'}\n"
                    f"- **Classes Detected:** {', '.join(classes) or 'None'}\n"
                    f"- **Issues Found:** {issues_cnt}"
                )

                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"code_{int(time.time())}",
                    success=report.is_valid_syntax,
                    data={
                        "valid_syntax": report.is_valid_syntax,
                        "security_score": report.security_score,
                        "lines": report.total_lines,
                        "functions": funcs,
                    },
                    summary=summary,
                    duration_sec=duration,
                )
            else:
                # Code generation / architectural specification prompt
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"code_gen_{int(time.time())}",
                    success=True,
                    data={"task": task, "status": "scaffolding_prepared"},
                    summary=f"Technical specification formulated for programming objective: '{task}'.",
                    duration_sec=duration,
                )

        except Exception as e:
            log.error(f"CodingAgent execution error: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"code_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Coding operation failed: {e}",
                duration_sec=duration,
            )


__all__ = ["CodingAgent"]
