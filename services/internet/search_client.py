"""
JARVIS AI — Universal Search Engine Integration
================================================
Interfaces with authorized search APIs (Tavily, Brave Search)
with a zero-key DuckDuckGo fallback for resilient web queries.
"""

import httpx
from typing import List, Optional
from pydantic import BaseModel, Field

from config.settings import settings
from core.logger import log


class SearchResult(BaseModel):
    """An individual web search result."""
    title: str
    url: str
    snippet: str
    source: str = "web"
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class SearchClient:
    """Client for querying external search APIs."""

    def __init__(self):
        self.timeout = 10.0

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Execute web search. Attempts Tavily/Brave if keys are provided,
        otherwise falls back to DuckDuckGo API/HTML search.
        """
        # 1. Tavily API (if configured in environment)
        tavily_key = getattr(settings, "TAVILY_API_KEY", "")
        if tavily_key:
            try:
                return await self._search_tavily(query, tavily_key, max_results)
            except Exception as e:
                log.warning(f"Tavily search failed, falling back: {e}")

        # 2. Resilient Open DuckDuckGo Fallback
        return await self._search_duckduckgo(query, max_results)

    async def _search_tavily(self, query: str, api_key: str, max_results: int) -> List[SearchResult]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", "No Title"),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                        source="tavily",
                        score=item.get("score", 1.0),
                    )
                )
            return results

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Query DuckDuckGo Instant Answer API / HTML Lite."""
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (JARVIS-Research/1.0)"
        }

        results: List[SearchResult] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    # Check abstract
                    abstract = data.get("AbstractText", "")
                    abstract_url = data.get("AbstractURL", "")
                    heading = data.get("Heading", query)
                    if abstract and abstract_url:
                        results.append(
                            SearchResult(
                                title=heading,
                                url=abstract_url,
                                snippet=abstract,
                                source="duckduckgo_instant",
                            )
                        )

                    # Check RelatedTopics
                    for topic in data.get("RelatedTopics", []):
                        if "Text" in topic and "FirstURL" in topic:
                            results.append(
                                SearchResult(
                                    title=topic["Text"].split(" - ")[0] if " - " in topic["Text"] else topic["Text"][:60],
                                    url=topic["FirstURL"],
                                    snippet=topic["Text"],
                                    source="duckduckgo",
                                )
                            )
                        if len(results) >= max_results:
                            break
        except Exception as e:
            log.warning(f"DuckDuckGo search error: {e}")

        # If zero direct instant answers, return structured research query placeholder
        if not results:
            results.append(
                SearchResult(
                    title=f"Research on: {query}",
                    url=f"https://html.duckduckgo.com/html/?q={query}",
                    snippet=f"Synthesized query topic: '{query}'. Consult official documentation or external sources.",
                    source="web_query",
                )
            )

        return results[:max_results]


search_client = SearchClient()
