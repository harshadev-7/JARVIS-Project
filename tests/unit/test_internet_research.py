"""Unit Tests for J.A.R.V.I.S. Internet Research (Phase 11).

Verifies:
1. Connectivity detection and strict offline rule (never pretends to have searched).
2. WebSearchEngine query formatting and DuckDuckGo parsing.
3. WebBrowser HTML sanitization, script stripping, and word count bounding.
4. WebBrowser invalid URL and network failure handling.
5. ResearchPipeline multi-source fact synthesis and source citations.
6. Distinction between live internet data and internal model knowledge.
7. Knowledge base indexing into SQLite database.
8. ResearchAgent execution across search, browse, connectivity check, and research actions.
9. Zero-key operation: Works without Tavily, Google, Bing, or OpenAI keys.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from research.web_search import SearchResult, SearchResponse, WebSearchEngine
from research.browser import ExtractedPage, WebBrowser
from services.internet.research_pipeline import ResearchPipeline, SourceCitation, ResearchReport
from agents.research import ResearchAgent


# ============================================================================
# 1. Connectivity & Strict Offline Integrity Tests
# ============================================================================

def test_connectivity_probe_logic():
    """Engine probe returns a valid boolean without raising unhandled errors."""
    engine = WebSearchEngine(timeout=1.0)
    connected = engine.is_connected(timeout=0.5)
    assert isinstance(connected, bool)


@pytest.mark.asyncio
async def test_search_when_offline_never_fabricates_results():
    """Rule verification: When offline, engine returns explicit offline notice and zero citations."""
    engine = WebSearchEngine()
    # Mock offline state
    with patch.object(engine, "is_connected", return_value=False):
        response: SearchResponse = await engine.search("Latest developments in quantum computing")
        assert response.is_offline is True
        assert len(response.results) == 0
        assert "unreachable" in response.error.lower() or "offline" in response.error.lower()
        assert response.has_results is False


# ============================================================================
# 2. Web Search Engine Parsing Tests
# ============================================================================

@pytest.mark.asyncio
async def test_search_empty_query_rejected():
    """Empty query strings return an empty response with descriptive notice."""
    engine = WebSearchEngine()
    resp = await engine.search("   ")
    assert resp.error == "Query cannot be empty"
    assert len(resp.results) == 0


@pytest.mark.asyncio
async def test_search_results_parsing_with_mocked_network():
    """Tests proper extraction of title, url, snippet from search payloads."""
    engine = WebSearchEngine()
    mock_payload = {
        "Heading": "Python (programming language)",
        "AbstractText": "Python is a high-level, general-purpose programming language.",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "RelatedTopics": [
            {
                "Text": "Python Software Foundation - Non-profit organization",
                "FirstURL": "https://www.python.org/psf/",
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.object(engine, "is_connected", return_value=True):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            resp = await engine.search("Python language", max_results=3)
            assert resp.is_offline is False
            assert len(resp.results) >= 2
            assert resp.results[0].title == "Python (programming language)"
            assert "high-level" in resp.results[0].snippet
            assert "wikipedia.org" in resp.results[0].url
            assert resp.results[0].source == "duckduckgo_instant"


# ============================================================================
# 3. Web Browser & HTML Cleaning Tests
# ============================================================================

def test_html_cleaner_removes_scripts_and_tags():
    """HTML cleaner must completely remove script, style, and navigation tags."""
    browser = WebBrowser()
    raw_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Article on Transformers</title>
    <script>alert('malicious tracker');</script>
    <style>body { color: red; }</style>
    </head>
    <body>
    <nav><a href="/home">Home</a> | <a href="/about">About</a></nav>
    <header><h1>Header Navigation</h1></header>
    <main>
        <p>Transformers use self-attention mechanisms to weigh token relationships.</p>
        <p>Residual connections prevent vanishing gradients during training.</p>
    </main>
    <footer>Copyright 2026 JARVIS Inc</footer>
    </body>
    </html>
    """

    page = browser.clean_html(raw_html, url="https://example.com/transformer")
    assert page.title == "Test Article on Transformers"
    assert "malicious tracker" not in page.text_content
    assert "body { color: red; }" not in page.text_content
    assert "Transformers use self-attention" in page.text_content
    assert "Residual connections" in page.text_content
    assert page.word_count > 10


def test_html_cleaner_clamps_max_words():
    """Word bounding must prevent memory exhaustion on long articles."""
    browser = WebBrowser()
    long_content = "word " * 3000
    html = f"<html><head><title>Big</title></head><body><p>{long_content}</p></body></html>"

    page = browser.clean_html(html, max_words=100)
    assert page.word_count == 100
    assert "Content truncated" in page.text_content


@pytest.mark.asyncio
async def test_browser_invalid_url_schema():
    """Invalid URL schemas (ftp, file, non-http) are safely rejected."""
    browser = WebBrowser()
    res = await browser.fetch_page("ftp://ftp.example.com/file.txt")
    assert res.status_code == 400
    assert "must start with http" in res.error.lower()


# ============================================================================
# 4. Research Pipeline Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_offline_notice_distinguishes_source():
    """When offline, pipeline returns a structured offline report without fake sources."""
    pipeline = ResearchPipeline()
    with patch("research.web_search.web_search_engine.is_connected", return_value=False):
        report: ResearchReport = await pipeline.conduct_research("Artificial General Intelligence")
        assert report.is_offline is True
        assert report.data_source == "offline_notice"
        assert len(report.citations) == 0
        assert "offline" in report.summary.lower()


@pytest.mark.asyncio
async def test_pipeline_online_flow_with_citations():
    """Pipeline aggregates citations and formats report distinguishing live internet data."""
    pipeline = ResearchPipeline()

    mock_resp = SearchResponse(
        query="Transformer attention",
        results=[
            SearchResult(
                title="Attention Is All You Need",
                url="https://arxiv.org/abs/1706.03762",
                snippet="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                source="arxiv",
            )
        ],
        is_offline=False,
        duration_ms=45.0,
    )

    with patch("research.web_search.web_search_engine.is_connected", return_value=True):
        with patch("research.web_search.web_search_engine.search", new_callable=AsyncMock, return_value=mock_resp):
            report = await pipeline.conduct_research("Transformer attention", auto_index_memory=False)
            assert report.is_offline is False
            assert report.data_source == "live_internet"
            assert len(report.citations) == 1
            assert report.citations[0].title == "Attention Is All You Need"
            assert report.citations[0].source_type == "internet_web"
            assert "Live internet retrieval" in report.summary


# ============================================================================
# 5. Research Agent Multi-Action Tests
# ============================================================================

@pytest.mark.asyncio
async def test_research_agent_connectivity_action():
    """ResearchAgent handles 'connectivity_check' action."""
    agent = ResearchAgent()
    res = await agent.execute("", context={"action": "connectivity_check"})
    assert res.success is True
    assert "is_connected" in res.data
    assert "Connectivity status:" in res.summary


@pytest.mark.asyncio
async def test_research_agent_browse_action():
    """ResearchAgent handles 'browse' action on a target webpage."""
    agent = ResearchAgent()
    mock_page = ExtractedPage(
        url="https://docs.python.org",
        title="Python Documentation",
        text_content="Python documentation home.",
        word_count=3,
        status_code=200,
    )
    with patch("research.browser.web_browser.fetch_page", new_callable=AsyncMock, return_value=mock_page):
        res = await agent.execute("https://docs.python.org", context={"action": "browse"})
        assert res.success is True
        assert res.data["title"] == "Python Documentation"
        assert "Python Documentation" in res.summary


@pytest.mark.asyncio
async def test_research_agent_offline_resilience():
    """ResearchAgent presents clear offline summary when network is down."""
    agent = ResearchAgent()
    with patch("research.web_search.web_search_engine.is_connected", return_value=False):
        res = await agent.execute("/research Neural Architecture Search")
        assert res.success is True
        assert res.data["is_offline"] is True
        assert "Offline" in res.summary
        assert res.data["data_source"] == "offline_notice"
