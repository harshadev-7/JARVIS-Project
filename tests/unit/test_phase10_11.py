"""
JARVIS AI — Phase 10 & 11 Test Suite
======================================
Tests for Autonomous Content Creation & Social Media Publishing:
- Multi-Platform viral content package generation
- Vertical 9:16 video project specification
- Queue management with Approval Mode & Autonomous Mode
- FastAPI social media endpoints
"""

import pytest
from fastapi.testclient import TestClient

from social_media.content_pipeline.content_generator import content_generator
from social_media.platforms.publisher import publishing_manager, PublishingMode
from video.ffmpeg_tools.video_builder import video_builder
from api.app import app


# ── 1. Content Generator Tests ──────────────────────────────────────────────

def test_content_generator_multi_platform():
    """Generator should produce customized formats for 5 platforms."""
    pkg = content_generator.generate_package("Quantum Neural Networks")
    assert pkg.topic == "Quantum Neural Networks"
    assert len(pkg.platforms) == 5
    assert "youtube_shorts" in pkg.platforms
    assert "instagram_reels" in pkg.platforms
    assert "tiktok" in pkg.platforms
    assert "x" in pkg.platforms
    assert "linkedin" in pkg.platforms
    assert pkg.copyright_cleared is True
    assert len(pkg.voiceover_script) > 50


# ── 2. Video Builder Tests ──────────────────────────────────────────────────

def test_video_builder_9_16_specs():
    """Video builder should configure 1080x1920 9:16 vertical parameters."""
    proj = video_builder.create_short_project("proj-001", "AI Revolution", duration_sec=30.0)
    assert proj.aspect_ratio == "9:16"
    assert proj.width == 1080
    assert proj.height == 1920
    assert proj.duration_sec == 30.0

    cmd = video_builder.generate_ffmpeg_render_command(proj, "bg.mp4")
    assert "1080:1920" in " ".join(cmd)
    assert "libx264" in cmd


# ── 3. Publishing Manager Tests ─────────────────────────────────────────────

def test_publishing_queue_and_approval():
    """Posts in approval mode require manual approval before publish."""
    publishing_manager.set_mode(PublishingMode.APPROVAL_REQUIRED)
    pkg = content_generator.generate_package("Autonomous AI Employees")

    post = publishing_manager.enqueue_package(pkg, "youtube_shorts")
    assert post.status == "queued"

    # Publishing without approval returns empty
    published_early = publishing_manager.publish_pending()
    assert not any(p.post_id == post.post_id for p in published_early)

    # Approve
    approved = publishing_manager.approve_post(post.post_id)
    assert approved is True

    # Now publish succeeds
    published = publishing_manager.publish_pending()
    assert any(p.post_id == post.post_id for p in published)


# ── 4. FastAPI Social Media Endpoints Tests ─────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_social_generate(client):
    """POST /social/content/generate should return full package."""
    res = client.post("/social/content/generate", json={"topic": "Agentic Coding"})
    assert res.status_code == 200
    data = res.json()
    assert "hook" in data
    assert "youtube_shorts" in data["platforms"]


def test_api_social_queue_flow(client):
    """POST /social/queue/add and GET /social/queue."""
    add_res = client.post(
        "/social/queue/add",
        json={"topic": "Future of Robotics", "platform": "linkedin"},
    )
    assert add_res.status_code == 200
    post_id = add_res.json()["post_id"]

    queue_res = client.get("/social/queue")
    assert queue_res.status_code == 200
    queue_data = queue_res.json()
    assert any(p["post_id"] == post_id for p in queue_data)


def test_api_video_project(client):
    """POST /social/video/project should return 9:16 config."""
    res = client.post(
        "/social/video/project",
        json={"project_id": "vid-101", "title": "AI in 2026", "duration_sec": 40.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["aspect_ratio"] == "9:16"
    assert data["width"] == 1080
    assert data["height"] == 1920
