"""
JARVIS AI — Internet & Research API Routes
===========================================
Endpoints for web search, webpage extraction, and autonomous topic research.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.internet.search_client import search_client, SearchResult
from services.internet.web_browser import web_browser, ExtractedWebPage
from services.internet.research_pipeline import research_pipeline, ResearchReport

router = APIRouter(prefix="/research", tags=["Internet & Research"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Search terms")
    max_results: int = Field(5, ge=1, le=20)


class ExtractRequest(BaseModel):
    url: str = Field(..., description="Target web page URL")
    max_words: int = Field(1500, ge=100, le=5000)


class ResearchTopicRequest(BaseModel):
    topic: str = Field(..., min_length=2, description="Topic to investigate")
    max_sources: int = Field(3, ge=1, le=10)
    auto_index_memory: bool = Field(True, description="Index results into JARVIS vector knowledge base")


@router.post("/search", response_model=List[SearchResult], summary="Execute web search query")
async def search_web(request: SearchRequest):
    return await search_client.search(request.query, max_results=request.max_results)


@router.post("/extract", response_model=ExtractedWebPage, summary="Extract clean readable content from URL")
async def extract_url(request: ExtractRequest):
    result = await web_browser.extract_content(request.url, max_words=request.max_words)
    if result.status_code >= 400 and result.error:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result


@router.post("/topic", response_model=ResearchReport, summary="Deep autonomous research on a topic")
async def research_topic(request: ResearchTopicRequest):
    return await research_pipeline.conduct_research(
        topic=request.topic,
        max_sources=request.max_sources,
        auto_index_memory=request.auto_index_memory,
    )
