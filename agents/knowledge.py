"""J.A.R.V.I.S. Knowledge Agent (Phase 9).

Specialized agent for querying, indexing, and organizing curated persistent knowledge:
- Semantic keyword retrieval across knowledge categories
- Dynamic fact extraction and storage
- Context synthesis for conversational augmentation
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent


class KnowledgeAgent(BaseAgent):
    """Independent specialized agent for persistent knowledge repository management."""

    def __init__(self):
        super().__init__(
            name="KnowledgeAgent",
            role="Retrieves, catalogs, and indexes verified factual information from persistent memory.",
            capabilities=["knowledge_retrieval", "fact_indexing", "semantic_lookup", "topic_categorization"],
            required_autonomy_level=1,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Search knowledge base or index new knowledge."""
        start_time = time.perf_counter()

        try:
            from memory.long_term import long_term_memory
            from memory.retrieval import memory_retrieval

            # Determine if this is an indexing command or search query
            if task.lower().startswith("store:") or task.lower().startswith("remember:"):
                # Format: remember: [category] title - content
                raw = task.split(":", 1)[1].strip()
                title = raw.split("-")[0].strip() if "-" in raw else raw[:30]
                content = raw.split("-", 1)[1].strip() if "-" in raw else raw

                item_id = long_term_memory.store_knowledge(
                    title=title,
                    content=content,
                    category=(context or {}).get("category", "general"),
                )
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"know_add_{int(time.time())}",
                    success=True,
                    data={"item_id": item_id, "title": title},
                    summary=f"Knowledge indexed successfully [ID #{item_id}]: '{title}'",
                    duration_sec=duration,
                )
            else:
                # Query knowledge
                matches = memory_retrieval.search_memory(
                    query=task,
                    search_knowledge=True,
                    search_preferences=False,
                    search_conversations=False,
                    limit=3,
                )
                duration = time.perf_counter() - start_time

                if not matches:
                    return AgentTaskResult(
                        agent_name=self.name,
                        task_id=f"know_search_{int(time.time())}",
                        success=True,
                        data={"matches": []},
                        summary=f"No matching records found in persistent knowledge base for: '{task}'.",
                        duration_sec=duration,
                    )

                summary_lines = [f"- **[{m['category']}] {m['title']}:** {m['content']}" for m in matches]
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"know_search_{int(time.time())}",
                    success=True,
                    data={"matches": matches},
                    summary="**Knowledge Base Search Results:**\n" + "\n".join(summary_lines),
                    duration_sec=duration,
                )

        except Exception as e:
            log.error(f"KnowledgeAgent execution error: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"know_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Knowledge lookup error: {e}",
                duration_sec=duration,
            )


__all__ = ["KnowledgeAgent"]
