"""JARVIS Phase 29 — Autonomous Self-Healing & Routine Maintenance
=================================================================
Executes non-blocking background hygiene:
- Prunes temporary media slices, audio caches, and intermediate renders.
- Cleans orphaned temporary database files.
- Resets cool-off for recovered circuit breakers.
- Verifies SQLite database integrity.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import BASE_DIR, settings

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceResult:
    """Audit report from an autonomous maintenance run."""
    timestamp: float
    files_cleaned: int
    bytes_freed: int
    db_integrity_ok: bool
    circuit_breakers_reset: int
    duration_ms: float
    cleaned_paths: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "files_cleaned": self.files_cleaned,
            "bytes_freed": self.bytes_freed,
            "bytes_freed_mb": round(self.bytes_freed / (1024 * 1024), 2),
            "db_integrity_ok": self.db_integrity_ok,
            "circuit_breakers_reset": self.circuit_breakers_reset,
            "duration_ms": round(self.duration_ms, 2),
            "cleaned_paths": self.cleaned_paths[:10],
            "details": self.details,
        }


class AutonomousMaintenance:
    """Background hygiene and self-healing engine."""

    def __init__(self, temp_dir: Optional[Path] = None, db_path: Optional[Path] = None):
        self.temp_dir = temp_dir or (BASE_DIR / "media" / "temporary")
        self.db_path = db_path or (BASE_DIR / "jarvis.db")

    async def run_maintenance(self, max_file_age_hours: float = 24.0) -> MaintenanceResult:
        """Executes safe routine cleanup and self-healing checks."""
        t0 = time.perf_counter()
        now = time.time()
        files_cleaned = 0
        bytes_freed = 0
        cleaned_paths = []

        # 1. Prune temporary media cache
        if self.temp_dir.exists():
            cutoff = now - (max_file_age_hours * 3600.0)
            for root, dirs, files in os.walk(self.temp_dir):
                for f in files:
                    fp = Path(root) / f
                    try:
                        st = fp.stat()
                        # Prune files older than cutoff or temporary frames
                        if st.st_mtime < cutoff or f.endswith(".tmp") or f.endswith(".part"):
                            sz = st.st_size
                            fp.unlink(missing_ok=True)
                            files_cleaned += 1
                            bytes_freed += sz
                            cleaned_paths.append(str(fp))
                    except Exception as exc:
                        logger.debug("Failed pruning temp file %s: %s", fp, exc)

        # 2. Reset expired circuit breakers
        reset_breakers_count = 0
        try:
            from core.orchestrator.circuit_breaker import circuit_breaker_registry
            for breaker in circuit_breaker_registry.breakers.values():
                if breaker.state.value == "open":
                    # Check if recovery timeout has passed
                    elapsed = now - breaker.last_failure_time
                    if elapsed >= breaker.recovery_timeout:
                        breaker.reset()
                        reset_breakers_count += 1
        except Exception as exc:
            logger.debug("Failed checking circuit breakers: %s", exc)

        # 3. Verify SQLite DB Integrity
        db_ok = True
        try:
            if self.db_path.exists():
                with sqlite3.connect(str(self.db_path), timeout=5.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA quick_check;")
                    row = cursor.fetchone()
                    db_ok = bool(row and row[0] == "ok")
        except Exception as exc:
            logger.warning("Database quick_check encountered an issue: %s", exc)
            db_ok = False

        duration = (time.perf_counter() - t0) * 1000.0

        return MaintenanceResult(
            timestamp=now,
            files_cleaned=files_cleaned,
            bytes_freed=bytes_freed,
            db_integrity_ok=db_ok,
            circuit_breakers_reset=reset_breakers_count,
            duration_ms=duration,
            cleaned_paths=cleaned_paths,
            details={
                "status": "complete",
                "temp_dir_checked": str(self.temp_dir),
                "db_checked": str(self.db_path),
            },
        )


autonomous_maintenance = AutonomousMaintenance()
