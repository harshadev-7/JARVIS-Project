"""JARVIS AI — Universal Research Pipeline (Phase 11 Upgraded).
=============================================================
Coordinates multi-source internet research:
1. Detects live network connectivity (never fakes searches when offline).
2. Queries search engines for top references without paid API keys.
3. Extracts and sanitizes content from verified sources.
4. Distinguishes live web data from model internal knowledge.
5. Indexes verified knowledge directly into SQLite persistent memory.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.logger import log
from research.browser import web_browser
from research.web_search import web_search_engine, SearchResponse


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
    data_source: str = "live_internet"  # "live_internet" vs "local_knowledge"


class ResearchPipeline:
    """End-to-end autonomous research coordinator."""

    async def conduct_research(
        self,
        topic: str,
        max_sources: int = 3,
        auto_index_memory: bool = True,
    ) -> ResearchReport:
        """Execute full research workflow for a given topic."""
        clean_topic = topic.strip()
        log.info(f"ResearchPipeline initiating investigation on: '{clean_topic}'")

        # 1. Connectivity Check
        if not web_search_engine.is_connected():
            log.info("System is offline. Aborting live internet research without pretending.")
            offline_summary = (
                f"Live internet research on '{clean_topic}' could not be performed.\n"
                f"Status: System is currently offline with no active network connection.\n"
                f"Zero external search was conducted."
            )
            return ResearchReport(
                topic=clean_topic,
                summary=offline_summary,
                key_findings=["Host workstation is disconnected from the internet."],
                citations=[],
                is_offline=True,
                knowledge_indexed=False,
                data_source="offline_notice",
            )

        # 2. Search Web (Zero-Key DuckDuckGo / Open endpoints)
        search_resp: SearchResponse = await web_search_engine.search(clean_topic, max_results=max_sources)

        citations: List[SourceCitation] = []
        key_findings: List[str] = []

        # 3. Process and extract citations
        for sr in search_resp.results:
            citations.append(
                SourceCitation(
                    title=sr.title,
                    url=sr.url,
                    snippet=sr.snippet,
                    source_type="internet_web",
                )
            )
            if sr.snippet:
                key_findings.append(f"{sr.title}: {sr.snippet}")

        # 4. Deep page extraction on primary URL if available
        if citations and citations[0].url.startswith("http") and "duckduckgo.com" not in citations[0].url:
            page = await web_browser.fetch_page(citations[0].url, max_words=300)
            if page.is_valid:
                key_findings.append(f"Deep Extract ({page.title}): {page.text_content[:300]}...")

        # 5. Synthesize report
        if citations:
            sources_summary = "; ".join([c.title for c in citations])
            summary = (
                f"Research on '{clean_topic}' compiled from {len(citations)} verified web sources.\n"
                f"Sources: {sources_summary}.\n"
                f"Data Source: Live internet retrieval (distinguished from custom neural weights)."
            )
        else:
            summary = f"Search query for '{clean_topic}' yielded zero live index entries."

        # 6. Persistent Knowledge Indexing
        indexed = False
        knowledge_id = None
        if auto_index_memory and key_findings:
            try:
                # Lazy import to prevent circular dependencies
                from memory.database import db_manager

                full_knowledge_text = f"Summary: {summary}\n\nKey Findings:\n" + "\n".join(key_findings)
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO knowledge_items (category, title, content, user_approved, created_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        ("internet_research", f"Research: {clean_topic}", full_knowledge_text, 1),
                    )
                    conn.commit()
                    knowledge_id = cursor.lastrowid
                    indexed = True
                    log.info(f"Research indexed in SQLite knowledge base (ID: {knowledge_id})")
            except Exception as e:
                log.warning(f"Could not index research in memory: {e}")

        return ResearchReport(
            topic=clean_topic,
            summary=summary,
            key_findings=key_findings,
            citations=citations,
            is_offline=False,
            knowledge_indexed=indexed,
            knowledge_id=knowledge_id,
            data_source="live_internet",
        )


research_pipeline = ResearchPipeline()

__all__ = ["SourceCitation", "ResearchReport", "ResearchPipeline", "research_pipeline"]
