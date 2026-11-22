"""J.A.R.V.I.S. — Safe Web Content Browser (Phase 11).

Extracts clean readable text, page titles, and metadata from web URLs
using httpx and BeautifulSoup with safe hardware memory boundaries.

Design Principles:
- Strips advertisements, navigation menus, scripts, CSS, and tracker tags.
- Clamps word count to prevent RAM exhaustion on Intel i3 / 8 GB systems.
- Offline-safe: rejects invalid schemas and handles connection drops gracefully.
- Provides fallback HTML parser using standard library if 3rd-party parser fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.logger import log


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class ExtractedPage:
    """Sanitized structured webpage content."""
    url: str
    title: str
    text_content: str
    word_count: int = 0
    status_code: int = 200
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and len(self.text_content.strip()) > 0


# ---------------------------------------------------------------------------
# Safe Web Browser
# ---------------------------------------------------------------------------

class WebBrowser:
    """Safe, sandboxed web page retriever and text cleaner."""

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36 JARVIS-Browser/1.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def fetch_page(self, url: str, max_words: int = 1500) -> ExtractedPage:
        """Fetch URL content and extract cleaned text.

        Parameters
        ----------
        url : str
            Target HTTP/HTTPS address.
        max_words : int
            Word limit clamp to preserve RAM.
        """
        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            return ExtractedPage(
                url=clean_url,
                title="Invalid URL",
                text_content="",
                status_code=400,
                error="URL must start with http:// or https://",
            )

        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=self.headers,
            ) as client:
                response = await client.get(clean_url)

                if response.status_code != 200:
                    return ExtractedPage(
                        url=clean_url,
                        title=f"HTTP Error {response.status_code}",
                        text_content="",
                        status_code=response.status_code,
                        error=f"Server returned HTTP status {response.status_code}",
                    )

                html = response.text
                return self.clean_html(html, url=clean_url, max_words=max_words)

        except Exception as e:
            log.warning(f"WebBrowser error accessing {clean_url}: {e}")
            return ExtractedPage(
                url=clean_url,
                title="Network Error",
                text_content="",
                status_code=500,
                error=str(e),
            )

    def clean_html(self, html: str, url: str = "", max_words: int = 1500) -> ExtractedPage:
        """Strip tags, navigation, scripts, and CSS from HTML text."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove noisy and non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"]):
                tag.decompose()

            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else (url or "Web Document")

            # Extract text
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

        except Exception:
            # Standard library regex fallback if BeautifulSoup unavailable
            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else (url or "Web Document")

            no_scripts = re.sub(r"<(script|style).*?>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            no_tags = re.sub(r"<[^>]+>", " ", no_scripts)
            lines = [line.strip() for line in no_tags.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

        # Word count bounding
        words = clean_text.split()
        word_count = len(words)
        if word_count > max_words:
            clean_text = " ".join(words[:max_words]) + "\n\n... [Content truncated for memory conservation]"

        return ExtractedPage(
            url=url,
            title=title,
            text_content=clean_text,
            word_count=min(word_count, max_words),
            status_code=200,
        )


web_browser = WebBrowser()

__all__ = ["ExtractedPage", "WebBrowser", "web_browser"]
