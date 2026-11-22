"""J.A.R.V.I.S. Research Subsystem (Phase 11).

Autonomous internet research, web search, content scraping,
and citation management without paid cloud AI APIs.
"""

from research.web_search import (
    SearchResult,
    SearchResponse,
    WebSearchEngine,
    web_search_engine,
)
from research.browser import (
    ExtractedPage,
    WebBrowser,
    web_browser,
)

__all__ = [
    "SearchResult",
    "SearchResponse",
    "WebSearchEngine",
    "web_search_engine",
    "ExtractedPage",
    "WebBrowser",
    "web_browser",
]
