"""
JARVIS AI — 24/7 Background Scheduler Service
==============================================
Manages recurring automations and background maintenance tasks
(social media queues, hardware audits, memory sync).
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.logger import log


class ScheduledJob(BaseModel):
    """A recurring background automation job."""
    job_id: str
    name: str
    interval_seconds: int
    is_active: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0


class BackgroundScheduler:
    """Lightweight in-process async scheduler for 24/7 autonomous routines."""

    def __init__(self):
        self._jobs: Dict[str, ScheduledJob] = {}
        self._is_running = False

    def register_job(self, job_id: str, name: str, interval_seconds: int = 3600) -> ScheduledJob:
        """Add a recurring background task."""
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            interval_seconds=interval_seconds,
            is_active=True,
        )
        self._jobs[job_id] = job
        log.info(f"Registered background routine: '{name}' (every {interval_seconds}s)")
        return job

    def list_jobs(self) -> List[ScheduledJob]:
        return list(self._jobs.values())

    def pause_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].is_active = False
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].is_active = True
            return True
        return False

    def mark_run(self, job_id: str):
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.last_run = datetime.now(timezone.utc)
            job.run_count += 1


scheduler_service = BackgroundScheduler()

# Pre-register default JARVIS autonomous maintenance routines
scheduler_service.register_job("job-sys-audit", "Hardware & Security Audit", interval_seconds=1800)
scheduler_service.register_job("job-social-queue", "Autonomous Social Media Publisher", interval_seconds=7200)
scheduler_service.register_job("job-mem-prune", "Vector Memory & Session Maintenance", interval_seconds=86400)
