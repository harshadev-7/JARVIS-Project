"""J.A.R.V.I.S. Research Agent (Phase 11 Upgraded).

Specialized agent for web search, document parsing, fact-checking, and source citation:
- Zero paid cloud AI dependencies (no OpenAI, no Google Gemini APIs).
- Strict offline detection: never fabricates live search results when disconnected.
- Distinguishes live web data from custom neural model weights.
- Extracts source citations with titles, URLs, and snippets.
- Indexes verified facts into persistent SQLite memory.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent
from research.browser import web_browser
from research.web_search import web_search_engine
from services.internet.research_pipeline import research_pipeline


class ResearchAgent(BaseAgent):
    """Independent specialized agent for research and information retrieval."""

    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="Conducts internet research, extracts source citations, and indexes verified facts.",
            capabilities=[
                "web_search",
                "content_extraction",
                "source_citation",
                "fact_indexing",
                "connectivity_check",
                "zero_api_key_research",
            ],
            required_autonomy_level=2,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute research objective."""
        start_time = time.perf_counter()
        ctx = context or {}
        action = ctx.get("action", "research")
        clean_query = task.replace("/research", "").strip()

        try:
            # 1. Connectivity Check Action
            if action == "connectivity_check":
                online = web_search_engine.is_connected()
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"res_conn_{int(time.time())}",
                    success=True,
                    data={"is_connected": online, "mode": "online" if online else "offline"},
                    summary=f"Connectivity status: {'Online' if online else 'Offline (No internet connection)'}",
                    duration_sec=duration,
                )

            # 2. Direct Web Page Extraction Action
            elif action == "browse":
                url = ctx.get("url") or clean_query
                page = await web_browser.fetch_page(url, max_words=ctx.get("max_words", 1000))
                duration = time.perf_counter() - start_time
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"res_browse_{int(time.time())}",
                    success=page.is_valid,
                    error=page.error,
                    data={
                        "url": page.url,
                        "title": page.title,
                        "word_count": page.word_count,
                        "content": page.text_content,
                    },
                    summary=f"Browsed '{page.title}' ({page.word_count} words extracted).",
                    duration_sec=duration,
                )

            # 3. Direct Search Action
            elif action == "search_only":
                resp = await web_search_engine.search(clean_query, max_results=ctx.get("max_results", 5))
                duration = time.perf_counter() - start_time
                citations = [r.to_dict() for r in resp.results]
                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"res_search_{int(time.time())}",
                    success=not resp.is_offline,
                    error=resp.error,
                    data={"query": clean_query, "citations": citations, "is_offline": resp.is_offline},
                    summary=f"Found {len(citations)} citations for '{clean_query}'. (Offline: {resp.is_offline})",
                    duration_sec=duration,
                )

            # 4. Standard End-to-End Research Pipeline
            else:
                max_sources = ctx.get("max_sources", 3)
                auto_index = ctx.get("auto_index", True)
                report = await research_pipeline.conduct_research(
                    clean_query,
                    max_sources=max_sources,
                    auto_index_memory=auto_index,
                )
                duration = time.perf_counter() - start_time

                citations_list = [
                    {"title": c.title, "url": c.url, "snippet": c.snippet, "type": c.source_type}
                    for c in report.citations
                ]

                if report.is_offline:
                    summary = (
                        f"**J.A.R.V.I.S. Research Notice (Offline):**\n\n"
                        f"{report.summary}\n\n"
                        f"*(No live internet queries were made)*"
                    )
                else:
                    sources_formatted = "\n".join([f"- **{c['title']}**: {c['url']}" for c in citations_list])
                    summary = (
                        f"**Universal Research Synthesis on '{report.topic}':**\n\n"
                        f"{report.summary}\n\n"
                        f"**Verified Sources ({len(citations_list)}):**\n"
                        f"{sources_formatted}\n\n"
                        f"**Indexed into Knowledge Base:** {'✅ Yes (ID: ' + str(report.knowledge_id) + ')' if report.knowledge_indexed else '❌ No'}"
                    )

                return AgentTaskResult(
                    agent_name=self.name,
                    task_id=f"res_{int(time.time())}",
                    success=True,
                    data={
                        "topic": report.topic,
                        "summary": report.summary,
                        "citations": citations_list,
                        "is_offline": report.is_offline,
                        "knowledge_indexed": report.knowledge_indexed,
                        "knowledge_id": report.knowledge_id,
                        "data_source": report.data_source,
                    },
                    summary=summary,
                    duration_sec=duration,
                )

        except Exception as e:
            log.warning(f"ResearchAgent execution issue: {e}")
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"res_err_{int(time.time())}",
                success=False,
                error=str(e),
                summary=f"Research on '{clean_query}' could not be completed: {e}.",
                duration_sec=duration,
            )


__all__ = ["ResearchAgent"]
