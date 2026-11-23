"""J.A.R.V.I.S. — Local Web Search Engine (Phase 11 Upgraded).

Performs web queries using zero-key open search endpoints (DuckDuckGo Lite / Instant Answer & Wikipedia OpenSearch)
with strict offline detection and genuine source citation tracking.

Design Principles:
- Zero paid search APIs required.
- Zero fake placeholder citations ("Live query for...").
- Active network connectivity checking.
- Real search results with destination article URLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import socket
import time
from typing import Any, Dict, List, Optional
import urllib.parse

from core.logger import log


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """An individual verified web search citation."""
    title: str
    url: str
    snippet: str
    source: str = "web"
    score: float = 1.0
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
            "retrieved_at": self.retrieved_at,
        }


@dataclass
class SearchResponse:
    """Aggregated search result payload."""
    query: str
    results: List[SearchResult]
    is_offline: bool
    duration_ms: float
    error: Optional[str] = None

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0 and not self.is_offline


# ---------------------------------------------------------------------------
# Web Search Engine
# ---------------------------------------------------------------------------

class WebSearchEngine:
    """Zero-key, privacy-first web search engine for J.A.R.V.I.S."""

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 JARVIS-Research/2.0"
        )

    def is_connected(self, host: str = "1.1.1.1", port: int = 53, timeout: float = 2.0) -> bool:
        """Rapid DNS/socket connectivity probe to verify actual internet access."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except OSError:
            return False

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Search the web for a query with real results from multiple open providers."""
        start = time.perf_counter()
        clean_query = query.strip()
        if not clean_query:
            return SearchResponse(
                query="",
                results=[],
                is_offline=False,
                duration_ms=0.0,
                error="Query cannot be empty",
            )

        # 1. Connectivity Check
        if not self.is_connected():
            dur = (time.perf_counter() - start) * 1000.0
            log.info(f"Search request for '{clean_query}' aborted: host is offline.")
            return SearchResponse(
                query=clean_query,
                results=[],
                is_offline=True,
                duration_ms=dur,
                error="Network unreachable: No active internet connection detected.",
            )

        import httpx
        results: List[SearchResult] = []
        seen_urls = set()

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        # Provider 1: DuckDuckGo Instant Answer API
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": clean_query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    abstract = data.get("AbstractText", "").strip()
                    abstract_url = data.get("AbstractURL", "").strip()
                    heading = data.get("Heading", clean_query).strip() or clean_query
                    if abstract and abstract_url and abstract_url not in seen_urls:
                        seen_urls.add(abstract_url)
                        results.append(
                            SearchResult(
                                title=heading,
                                url=abstract_url,
                                snippet=abstract,
                                source="duckduckgo_instant",
                                score=1.0,
                            )
                        )

                    for topic in data.get("RelatedTopics", []):
                        if len(results) >= max_results:
                            break
                        if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                            text = topic["Text"].strip()
                            url_val = topic["FirstURL"].strip()
                            if text and url_val and url_val not in seen_urls:
                                seen_urls.add(url_val)
                                title = text.split(" - ")[0] if " - " in text else text[:60]
                                results.append(
                                    SearchResult(
                                        title=title,
                                        url=url_val,
                                        snippet=text,
                                        source="duckduckgo_topics",
                                        score=0.9,
                                    )
                                )
        except Exception as e:
            log.debug(f"DuckDuckGo API search notice: {e}")

        # Provider 2: DuckDuckGo HTML Lite Organic Results
        if len(results) < max_results:
            try:
                from bs4 import BeautifulSoup

                html_url = "https://html.duckduckgo.com/html/"
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                    resp = await client.post(html_url, data={"q": clean_query})
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, "html.parser")
                        for item in soup.select(".result"):
                            title_a = item.select_one(".result__a") or item.select_one(".result__title a")
                            snippet_td = item.select_one(".result__snippet")
                            if title_a:
                                title_text = title_a.get_text(separator=" ", strip=True)
                                title_text = re.sub(r"\s+", " ", title_text).strip()
                                raw_link = title_a.get("href", "")
                                real_link = raw_link
                                if "uddg=" in raw_link:
                                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_link).query)
                                    if "uddg" in parsed:
                                        real_link = parsed["uddg"][0]

                                snip_text = snippet_td.get_text(separator=" ", strip=True) if snippet_td else ""
                                snip_text = re.sub(r"\s+", " ", snip_text).strip()
                                if title_text and real_link.startswith("http") and real_link not in seen_urls:
                                    seen_urls.add(real_link)
                                    results.append(
                                        SearchResult(
                                            title=title_text,
                                            url=real_link,
                                            snippet=snip_text,
                                            source="duckduckgo_html",
                                            score=0.95,
                                        )
                                    )
                            if len(results) >= max_results:
                                break
            except Exception as e:
                log.debug(f"DuckDuckGo HTML search notice: {e}")

        # Provider 3: Open Wikipedia Search API
        if len(results) < max_results:
            try:
                wiki_url = "https://en.wikipedia.org/w/api.php"
                wiki_params = {
                    "action": "opensearch",
                    "search": clean_query,
                    "limit": max_results,
                    "format": "json",
                }
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                    resp = await client.get(wiki_url, params=wiki_params)
                    if resp.status_code == 200:
                        data = resp.json()
                        # data format: [query, [titles], [descriptions], [urls]]
                        if len(data) >= 4:
                            titles = data[1]
                            descs = data[2]
                            urls = data[3]
                            for t, d, u in zip(titles, descs, urls):
                                if u and u not in seen_urls:
                                    seen_urls.add(u)
                                    results.append(
                                        SearchResult(
                                            title=f"{t} (Wikipedia)",
                                            url=u,
                                            snippet=d or f"Encyclopedic entry for {t}.",
                                            source="wikipedia_opensearch",
                                            score=0.92,
                                        )
                                    )
                                if len(results) >= max_results:
                                    break
            except Exception as e:
                log.debug(f"Wikipedia search notice: {e}")

        dur = (time.perf_counter() - start) * 1000.0
        return SearchResponse(
            query=clean_query,
            results=results[:max_results],
            is_offline=False,
            duration_ms=dur,
            error=None if results else "No substantive search results retrieved from web providers.",
        )


web_search_engine = WebSearchEngine()
__all__ = ["SearchResult", "SearchResponse", "WebSearchEngine", "web_search_engine"]
