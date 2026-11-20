"""J.A.R.V.I.S. Research Agent (Phase 9).

Specialized agent for web search, document parsing, fact-checking, and source citation:
- Connects to DuckDuckGo zero-key search pipeline
- Extracts clean markdown snippets
- Distinguishes offline state from online web search results
- Indexes verified facts into persistent knowledge memory
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent


class ResearchAgent(BaseAgent):
    """Independent specialized agent for research and information retrieval."""

    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="Conducts internet research, extracts source citations, and indexes verified facts.",
            capabilities=["web_search", "content_extraction", "source_citation", "fact_indexing"],
            required_autonomy_level=2,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute research objective."""
        start_time = time.perf_counter()
        clean_query = task.replace("/research", "").strip()

        try:
            from services.internet.research_pipeline import research_pipeline
            report = await research_pipeline.conduct_research(clean_query, max_sources=3, auto_index_memory=True)
            duration = time.perf_counter() - start_time

            citations_list = [
                {"title": c.title, "url": c.url, "snippet": c.snippet} for c in report.citations
            ]

            summary = (
                f"**Research Synthesis on '{report.topic}':**\n\n"
                f"{report.summary}\n\n"
                f"**Sources ({len(citations_list)}):**\n"
                + "\n".join([f"- {c['title']} ({c['url']})" for c in citations_list])
            )

            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"res_{int(time.time())}",
                success=True,
                data={
                    "topic": report.topic,
                    "summary": report.summary,
                    "citations": citations_list,
                    "knowledge_indexed": report.knowledge_indexed,
                },
                summary=summary,
                duration_sec=duration,
            )

        except Exception as e:
            log.warning(f"ResearchAgent offline or query error: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"res_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Research on '{clean_query}' could not be completed: {e}. (System in offline fallback).",
                duration_sec=duration,
            )


__all__ = ["ResearchAgent"]
