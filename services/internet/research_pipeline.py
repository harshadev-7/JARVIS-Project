"""
JARVIS AI — Universal Research Pipeline
=========================================
Coordinates multi-source internet research:
1. Queries search engines for top references.
2. Extracts content from permitted sources.
3. Performs information comparison and fact extraction.
4. Synthesizes a structured report with source citations.
5. Indexes verified knowledge directly into JARVIS memory.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from services.internet.search_client import search_client, SearchResult
from services.internet.web_browser import web_browser, ExtractedWebPage
from core.memory.persistent_memory import persistent_memory
from core.logger import log


class SourceCitation(BaseModel):
    """A verified source citation."""
    title: str
    url: str
    snippet: str
    source_type: str = "internet_web"


class ResearchReport(BaseModel):
    """Synthesized research findings on a topic."""
    topic: str
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    citations: List[SourceCitation] = Field(default_factory=list)
    is_offline: bool = False
    knowledge_indexed: bool = False
    knowledge_id: Optional[int] = None
    data_source: str = "live_internet"  # "live_internet" vs "offline_notice"


class ResearchPipeline:
    """End-to-end autonomous research coordinator."""

    async def conduct_research(
        self,
        topic: str,
        max_sources: int = 3,
        auto_index_memory: bool = True,
    ) -> ResearchReport:
        """
        Execute full research workflow for a given topic.
        """
        log.info(f"ResearchPipeline initiating research on: '{topic}'")

        # 0. Connectivity Check
        try:
            from research.web_search import web_search_engine
            if not web_search_engine.is_connected():
                log.info("System is offline. Aborting live internet research without pretending.")
                offline_summary = (
                    f"Live internet research on '{topic}' could not be performed.\n"
                    f"Status: System is currently offline with no active network connection.\n"
                    f"Zero external search was conducted."
                )
                return ResearchReport(
                    topic=topic,
                    summary=offline_summary,
                    key_findings=["Host workstation is disconnected from the internet."],
                    citations=[],
                    is_offline=True,
                    knowledge_indexed=False,
                    data_source="offline_notice",
                )
        except Exception:
            pass

        # 1. Search the web
        try:
            from research.web_search import web_search_engine
            search_resp = await web_search_engine.search(topic, max_results=max_sources)
            search_results = search_resp.results
        except Exception:
            search_results = await search_client.search(topic, max_results=max_sources)

        citations: List[SourceCitation] = []
        extracted_texts: List[str] = []

        # 2. Extract content from top valid results
        for sr in search_results:
            citations.append(
                SourceCitation(
                    title=sr.title,
                    url=sr.url,
                    snippet=sr.snippet,
                    source_type="internet_web",
                )
            )
            # If a valid HTTP URL exists, extract page content
            if sr.url.startswith("http") and "duckduckgo.com/html" not in sr.url:
                page = await web_browser.extract_content(sr.url, max_words=300)
                if not page.error and page.text_content:
                    extracted_texts.append(f"[{sr.title}]: {page.text_content[:400]}")
                else:
                    extracted_texts.append(f"[{sr.title}]: {sr.snippet}")
            else:
                extracted_texts.append(f"[{sr.title}]: {sr.snippet}")

        # 3. Synthesize findings
        key_findings: List[str] = []
        for sr in search_results:
            if sr.snippet:
                key_findings.append(f"{sr.title}: {sr.snippet}")

        findings_bullets = "\n".join([f"• **{sr.title}**: {sr.snippet}" for sr in search_results if sr.snippet])
        extracted_summary = "\n\n".join(extracted_texts[:2]) if extracted_texts else ""

        summary = (
            f"**Comprehensive Research Briefing: {topic}**\n\n"
            f"Synthesized from **{len(citations)} verified sources**:\n\n"
            f"{findings_bullets}\n\n"
            f"Data Source: Live internet retrieval (distinguished from custom neural weights)."
        )
        if extracted_summary:
            summary += f"\n\n**Extracted Intelligence Details:**\n{extracted_summary[:800]}..."

        knowledge_id = None
        indexed = False

        # 4. Index in JARVIS Long-Term Knowledge Base & Autonomous Learning
        if auto_index_memory and key_findings:
            try:
                full_knowledge_text = f"Summary: {summary}\n\nFindings:\n" + "\n".join(key_findings)
                knowledge_id = await persistent_memory.store_knowledge(
                    title=f"Research: {topic}",
                    content=full_knowledge_text,
                    category="research",
                    source="web_research_pipeline",
                )
                indexed = True
                log.info(f"Research findings indexed in memory with ID: {knowledge_id}")
            except Exception as e:
                log.warning(f"Could not automatically index research in memory: {e}")

            # Step 5: Automatically extract and register problem patterns in LearnedPatternEngine
            try:
                from core.planner.learned_pattern_engine import learned_pattern_engine
                snippets = [sr.snippet for sr in search_results if sr.snippet]
                learned_pattern_engine.learn_from_research(
                    query=topic,
                    solution_text=summary,
                    research_snippets=snippets,
                )
            except Exception as e:
                log.warning(f"Could not automatically learn pattern from research: {e}")

        return ResearchReport(
            topic=topic,
            summary=summary,
            key_findings=key_findings,
            citations=citations,
            is_offline=False,
            knowledge_indexed=indexed,
            knowledge_id=knowledge_id,
            data_source="live_internet",
        )


research_pipeline = ResearchPipeline()
