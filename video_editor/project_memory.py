"""
JARVIS Video Editing Engine — Project Memory & SQLite Persistence
==================================================================
Stores editing projects, timelines, rendering histories, and learned
style preferences in SQLite (jarvis.db).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import sqlite3
import json
import time

from core.logger import log

DB_PATH = Path(__file__).resolve().parents[1] / "jarvis.db"


class ProjectMemory:
    """Manages SQLite persistent storage for editing projects and history."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self) -> None:
        """Create project memory tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_editing_projects (
                    project_id TEXT PRIMARY KEY,
                    style TEXT NOT NULL,
                    target_duration_sec REAL NOT NULL,
                    color_grade TEXT,
                    branding_name TEXT,
                    timeline_json TEXT,
                    status TEXT DEFAULT 'created',
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_render_history (
                    render_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    style TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    duration_sec REAL,
                    file_size_bytes INTEGER,
                    render_time_sec REAL,
                    qc_passed INTEGER,
                    rendered_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES video_editing_projects(project_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_editing_preferences (
                    pref_key TEXT PRIMARY KEY,
                    pref_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_project(
        self,
        project_id: str,
        style: str,
        target_duration: float,
        color_grade: str,
        branding_name: str,
        timeline_data: Dict[str, Any],
        status: str = "completed",
    ) -> None:
        """Persist project timeline and parameters."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO video_editing_projects
                (project_id, style, target_duration_sec, color_grade, branding_name, timeline_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_id, style, target_duration, color_grade, branding_name,
                json.dumps(timeline_data), status, now
            ))
            conn.commit()
        log.debug(f"Saved editing project '{project_id}' to SQLite.")

    def record_render(
        self,
        project_id: str,
        style: str,
        output_path: str,
        duration_sec: float,
        file_size_bytes: int,
        render_time_sec: float,
        qc_passed: bool,
    ) -> None:
        """Log a completed render to history."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO video_render_history
                (project_id, style, output_path, duration_sec, file_size_bytes, render_time_sec, qc_passed, rendered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_id, style, output_path, duration_sec, file_size_bytes,
                render_time_sec, 1 if qc_passed else 0, now
            ))
            conn.commit()

    def get_recent_projects(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve latest projects."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT project_id, style, target_duration_sec, color_grade, branding_name, status, created_at
                FROM video_editing_projects
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_style_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch custom style or parameter preference."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pref_value FROM video_editing_preferences WHERE pref_key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default

    def set_style_preference(self, key: str, value: str) -> None:
        """Save calibrated style preference."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO video_editing_preferences (pref_key, pref_value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, now))
            conn.commit()


# Global project memory singleton
project_memory = ProjectMemory()
