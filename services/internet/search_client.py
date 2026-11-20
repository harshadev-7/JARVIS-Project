"""JARVIS AI — Universal Search Client (Phase 11 Upgraded).
==========================================================
Bridges to the modular research package (`research.web_search`).
100% zero-key open search with offline connectivity awareness.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from research.web_search import (
    SearchResult as NewSearchResult,
    SearchResponse,
    WebSearchEngine,
    web_search_engine,
)


class SearchResult(BaseModel):
    """An individual web search citation for services layer backward compatibility."""
    title: str
    url: str
    snippet: str
    source: str = "web"
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class SearchClient:
    """Client for performing web queries."""

    def __init__(self):
        self.engine = web_search_engine

    def is_online(self) -> bool:
        """Check active network connectivity."""
        return self.engine.is_connected()

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Execute search using zero-key open endpoints with offline awareness."""
        resp: SearchResponse = await self.engine.search(query=query, max_results=max_results)

        if resp.is_offline or not resp.results:
            return []

        return [
            SearchResult(
                title=r.title,
                url=r.url,
                snippet=r.snippet,
                source=r.source,
                score=r.score,
            )
            for r in resp.results
        ]


search_client = SearchClient()

__all__ = ["SearchResult", "SearchClient", "search_client"]
