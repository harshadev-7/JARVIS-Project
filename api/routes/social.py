"""
JARVIS AI — Autonomous Social Media & Video API Routes
=======================================================
Endpoints for viral script generation, vertical 9:16 video configuration,
and multi-platform post queue scheduling.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from social_media.content_pipeline.content_generator import content_generator, SocialContentPackage
from social_media.platforms.publisher import publishing_manager, ScheduledPost, PublishingMode
from video.ffmpeg_tools.video_builder import video_builder, VideoProject

router = APIRouter(prefix="/social", tags=["Social Media & Content Automation"])


class GenerateContentRequest(BaseModel):
    topic: str = Field(..., min_length=2, description="Topic for content package")


class SchedulePostRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    platform: str = Field("youtube_shorts", description="Target platform: youtube_shorts, instagram_reels, tiktok, x, linkedin")


class VideoRenderRequest(BaseModel):
    project_id: str
    title: str
    duration_sec: float = Field(45.0, ge=5.0, le=180.0)


@router.post("/content/generate", response_model=SocialContentPackage, summary="Generate multi-platform viral content package")
async def generate_content(request: GenerateContentRequest):
    return content_generator.generate_package(request.topic)


@router.post("/queue/add", response_model=ScheduledPost, summary="Enqueue post for publishing")
async def schedule_post(request: SchedulePostRequest):
    pkg = content_generator.generate_package(request.topic)
    try:
        return publishing_manager.enqueue_package(pkg, request.platform)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/queue/approve/{post_id}", summary="Approve a queued post")
async def approve_post(post_id: str):
    success = publishing_manager.approve_post(post_id)
    if not success:
        raise HTTPException(status_code=404, detail="Post not found.")
    return {"message": f"Post '{post_id}' approved."}


@router.post("/publish", response_model=List[ScheduledPost], summary="Dispatch all approved posts")
async def publish_pending():
    return publishing_manager.publish_pending()


@router.get("/queue", response_model=List[ScheduledPost], summary="Inspect active publishing queue")
async def get_queue():
    return publishing_manager.get_queue()


@router.post("/video/project", response_model=VideoProject, summary="Configure 9:16 vertical video project")
async def create_video_project(request: VideoRenderRequest):
    return video_builder.create_short_project(
        project_id=request.project_id,
        title=request.title,
        duration_sec=request.duration_sec,
    )
