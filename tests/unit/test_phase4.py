"""
JARVIS AI — Phase 4 Test Suite
================================
Tests for Universal Internet Access & Research:
- Search client querying and fallbacks
- Safe web page extraction and HTML noise reduction
- Autonomous multi-source research pipeline with memory indexing
- FastAPI research endpoints
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from services.internet.search_client import search_client, SearchResult
from services.internet.web_browser import web_browser, ExtractedWebPage
from services.internet.research_pipeline import research_pipeline
from api.app import app


# ── 1. Search Client Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_client_fallback():
    """Search client should return results even with DuckDuckGo open fallback."""
    results = await search_client.search("FastAPI documentation", max_results=3)
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    assert results[0].title is not None
    assert results[0].url is not None


# ── 2. Web Browser Extractor Tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_browser_invalid_url():
    """Browser extractor should reject invalid protocols."""
    result = await web_browser.extract_content("ftp://invalid-server.local")
    assert result.status_code == 400
    assert "http://" in result.error


@pytest.mark.asyncio
async def test_web_browser_html_cleaning():
    """Browser extractor should strip scripts and styles."""
    html_mock = """
    <html>
        <head><title>Test Page Title</title><style>.hidden { display:none; }</style></head>
        <body>
            <script>alert('malicious')</script>
            <nav>Menu Items</nav>
            <main>
                <h1>Welcome to JARVIS</h1>
                <p>JARVIS is an autonomous intelligence system.</p>
            </main>
        </body>
    </html>
    """
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = html_mock
        mock_get.return_value = mock_response

        extracted = await web_browser.extract_content("https://example.com/test")
        assert extracted.status_code == 200
        assert extracted.title == "Test Page Title"
        assert "alert" not in extracted.text_content
        assert "Welcome to JARVIS" in extracted.text_content


# ── 3. Research Pipeline Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_research_pipeline_execution():
    """Pipeline should synthesize findings and generate citations."""
    report = await research_pipeline.conduct_research(
        topic="Artificial Intelligence Agents",
        max_sources=2,
        auto_index_memory=True,
    )

    assert report.topic == "Artificial Intelligence Agents"
    assert len(report.citations) > 0
    assert report.summary != ""
    assert report.knowledge_indexed is True


# ── 4. FastAPI Research Endpoints Tests ──────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_research_search(client):
    """POST /research/search should return a list of search items."""
    res = client.post("/research/search", json={"query": "Quantum Computing", "max_results": 2})
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "title" in data[0]


def test_api_research_topic(client):
    """POST /research/topic should return complete research report."""
    res = client.post(
        "/research/topic",
        json={"topic": "Neural Network Transformers", "max_sources": 2, "auto_index_memory": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["topic"] == "Neural Network Transformers"
    assert "citations" in data
    assert "summary" in data
