"""
JARVIS AI — Autonomous Publishing Manager
==========================================
Manages multi-platform publishing schedule (target: 5-6 posts/day),
queue inspection, and supports both 'Approval Required' and 'Autonomous' modes.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from social_media.content_pipeline.content_generator import SocialContentPackage, content_generator
from core.logger import log


class PublishingMode(str, Enum):
    APPROVAL_REQUIRED = "approval_required"  # Review required by user before release
    AUTONOMOUS = "autonomous"                # Auto-published based on schedule


class ScheduledPost(BaseModel):
    """Queued social post item."""
    post_id: str
    platform: str
    title: str
    caption: str
    status: str = "queued"  # "queued", "approved", "published", "rejected"
    scheduled_time: datetime
    mode: PublishingMode = PublishingMode.APPROVAL_REQUIRED
    published_at: Optional[datetime] = None


class PublishingManager:
    """Controls daily social queue and dispatches posts."""

    def __init__(self, target_daily_posts: int = 5):
        self.target_daily_posts = target_daily_posts
        self.mode = PublishingMode.APPROVAL_REQUIRED
        self._queue: List[ScheduledPost] = []

    def set_mode(self, mode: PublishingMode):
        """Toggle between approval mode and autonomous mode."""
        self.mode = mode
        log.info(f"Publishing mode set to: {mode.value}")

    def enqueue_package(self, package: SocialContentPackage, platform: str) -> ScheduledPost:
        """Schedule a post from a content package."""
        if platform not in package.platforms:
            raise ValueError(f"Platform '{platform}' not found in content package.")

        post_data = package.platforms[platform]
        post_id = f"post-{len(self._queue) + 1:03d}"

        scheduled = ScheduledPost(
            post_id=post_id,
            platform=platform,
            title=post_data.title,
            caption=post_data.caption,
            status="approved" if self.mode == PublishingMode.AUTONOMOUS else "queued",
            scheduled_time=datetime.now(timezone.utc),
            mode=self.mode,
        )
        self._queue.append(scheduled)
        log.info(f"Scheduled post [{post_id}] for {platform} (Mode: {self.mode.value})")
        return scheduled

    def approve_post(self, post_id: str) -> bool:
        """Approve a queued post in approval mode."""
        for p in self._queue:
            if p.post_id == post_id:
                p.status = "approved"
                log.info(f"Post [{post_id}] manually approved.")
                return True
        return False

    def publish_pending(self) -> List[ScheduledPost]:
        """Publish all approved posts."""
        published = []
        for p in self._queue:
            if p.status == "approved":
                p.status = "published"
                p.published_at = datetime.now(timezone.utc)
                published.append(p)
                log.info(f"Published post [{p.post_id}] to {p.platform}")
        return published

    def get_queue(self) -> List[ScheduledPost]:
        """Inspect all queued and published items."""
        return self._queue


publishing_manager = PublishingManager()
