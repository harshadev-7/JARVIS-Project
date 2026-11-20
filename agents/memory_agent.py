"""J.A.R.V.I.S. Memory Agent (Phase 9).

Specialized agent for managing user preferences, dialogue history recall,
session management, and privacy compliance screening.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent


class MemoryAgent(BaseAgent):
    """Independent specialized agent for memory inspection and preference coordination."""

    def __init__(self):
        super().__init__(
            name="MemoryAgent",
            role="Coordinates short-term and persistent memory tiers, manages preferences, and enforces privacy guards.",
            capabilities=["preference_management", "history_recall", "privacy_screening", "session_cleanup"],
            required_autonomy_level=1,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute memory operations (preferences, history, wipe)."""
        start_time = time.perf_counter()
        lower = task.lower().strip()

        try:
            from memory.long_term import long_term_memory
            from memory.retrieval import memory_retrieval
            from memory.short_term import short_term_memory

            # 1. Clear session history
            if "clear history" in lower or "delete session" in lower:
                session_id = (context or {}).get("session_id", "default_session")
                deleted = memory_retrieval.delete_session_history(session_id)
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"mem_del_{int(time.time())}",
                    success=True,
                    data={"session_id": session_id, "deleted_turns": deleted},
                    summary=f"Conversation memory cleared for session '{session_id}' ({deleted} turns removed).",
                    duration_sec=duration,
                )

            # 2. Preference query or set
            elif "preference" in lower or "set " in lower:
                if "=" in task or "to " in task:
                    # Save preference
                    parts = task.replace("set ", "").split("=" if "=" in task else "to ")
                    key = parts[0].strip()
                    val = parts[1].strip()
                    long_term_memory.set_preference(key, val, user_approved=True)
                    duration = time.perf_counter() - start_time
                    return AgentTaskResult(
                        agent_name=self.name,
                        task_id=f"mem_pref_set_{int(time.time())}",
                        success=True,
                        data={"key": key, "value": val},
                        summary=f"User preference saved: **{key}** = '{val}'",
                        duration_sec=duration,
                    )
                else:
                    # List preferences
                    prefs = long_term_memory.list_preferences()
                    duration = time.perf_counter() - start_time
                    pref_lines = [f"- **{p['key']}:** {p['value']} ({p['category']})" for p in prefs]
                    return AgentTaskResult(
                        agent_name=self.name,
                        task_id=f"mem_pref_list_{int(time.time())}",
                        success=True,
                        data={"preferences": prefs},
                        summary="**Stored User Preferences:**\n" + ("\n".join(pref_lines) if pref_lines else "None recorded yet."),
                        duration_sec=duration,
                    )

            # 3. Recall history summary
            else:
                session_id = (context or {}).get("session_id", "default_session")
                history = long_term_memory.get_history(session_id, limit=5)
                duration = time.perf_counter() - start_time
                summary_lines = [f"[{h['role'].upper()}]: {h['content'][:60]}..." for h in history]
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"mem_hist_{int(time.time())}",
                    success=True,
                    data={"count": len(history)},
                    summary=f"Recent History ({len(history)} turns):\n" + "\n".join(summary_lines),
                    duration_sec=duration,
                )

        except Exception as e:
            log.error(f"MemoryAgent execution error: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"mem_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Memory operation failed: {e}",
                duration_sec=duration,
            )


__all__ = ["MemoryAgent"]
