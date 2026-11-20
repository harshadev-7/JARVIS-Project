"""JARVIS AI — Web Content Extractor (Phase 11 Upgraded).
========================================================
Bridges to the modular research package (`research.browser`).
Extracts clean, structured text and metadata from permitted web URLs.
"""

from typing import Optional
from pydantic import BaseModel

from research.browser import (
    ExtractedPage,
    WebBrowser as CoreWebBrowser,
    web_browser as core_web_browser,
)


class ExtractedWebPage(BaseModel):
    """Clean structured content extracted from a webpage."""
    url: str
    title: str
    text_content: str
    word_count: int = 0
    status_code: int = 200
    error: Optional[str] = None


class WebBrowser:
    """Safe web content extractor."""

    def __init__(self):
        self.browser = core_web_browser

    async def extract_content(self, url: str, max_words: int = 1500) -> ExtractedWebPage:
        """Fetch HTML and return clean text with word bounding."""
        res: ExtractedPage = await self.browser.fetch_page(url, max_words=max_words)
        return ExtractedWebPage(
            url=res.url,
            title=res.title,
            text_content=res.text_content,
            word_count=res.word_count,
            status_code=res.status_code,
            error=res.error,
        )

    def clean_html(self, html: str, url: str = "", max_words: int = 1500) -> ExtractedWebPage:
        """Strip tags from raw HTML."""
        res: ExtractedPage = self.browser.clean_html(html, url=url, max_words=max_words)
        return ExtractedWebPage(
            url=res.url,
            title=res.title,
            text_content=res.text_content,
            word_count=res.word_count,
            status_code=res.status_code,
            error=res.error,
        )


web_browser = WebBrowser()

__all__ = ["ExtractedWebPage", "WebBrowser", "web_browser"]
