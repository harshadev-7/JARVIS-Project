"""
JARVIS AI — Web Content Extractor
==================================
Extracts clean, structured text and metadata from permitted web URLs
using BeautifulSoup and httpx. Strictly respects web standards and security.
"""

import httpx
from bs4 import BeautifulSoup
from typing import Optional
from pydantic import BaseModel, Field

from core.logger import log


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
        self.timeout = 15.0
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (JARVIS-AI-Assistant/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    async def extract_content(self, url: str, max_words: int = 2000) -> ExtractedWebPage:
        """
        Fetch HTML from URL and extract readable text.
        Strips script, style, nav, and ads.
        """
        if not url.startswith(("http://", "https://")):
            return ExtractedWebPage(
                url=url,
                title="Invalid URL",
                text_content="",
                status_code=400,
                error="URL must begin with http:// or https://",
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=self.headers) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return ExtractedWebPage(
                        url=url,
                        title=f"HTTP {response.status_code}",
                        text_content="",
                        status_code=response.status_code,
                        error=f"Server returned status {response.status_code}",
                    )

                soup = BeautifulSoup(response.text, "html.parser")

                # Remove noise elements
                for element in soup(["script", "style", "nav", "footer", "aside", "noscript", "svg"]):
                    element.decompose()

                title = soup.title.string.strip() if soup.title and soup.title.string else url

                # Extract and clean text
                raw_text = soup.get_text(separator="\n")
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                clean_text = "\n".join(lines)

                # Clamp word count
                words = clean_text.split()
                if len(words) > max_words:
                    clean_text = " ".join(words[:max_words]) + "\n... [Content truncated for length]"

                return ExtractedWebPage(
                    url=url,
                    title=title,
                    text_content=clean_text,
                    word_count=len(words),
                    status_code=200,
                )

        except Exception as e:
            log.warning(f"Failed to fetch content from {url}: {e}")
            return ExtractedWebPage(
                url=url,
                title="Fetch Error",
                text_content="",
                status_code=500,
                error=str(e),
            )


web_browser = WebBrowser()
