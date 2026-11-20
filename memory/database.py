"""J.A.R.V.I.S. SQLite Persistent Memory Database Engine (Phase 8).

Provides:
- SQLite connection management
- Schema synchronization matching jarvis.db
- Safe atomic transactions
- Online timestamped database backups
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


class DatabaseManager:
    """Manages SQLite persistent storage for J.A.R.V.I.S. memory."""

    def __init__(self, db_path: str | Path = "jarvis.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a thread-safe SQLite connection with Row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_database(self) -> None:
        """Create or synchronize tables matching existing jarvis.db schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Conversation History Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id);")

            # 2. User Preferences Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key VARCHAR(64) UNIQUE NOT NULL,
                value TEXT NOT NULL,
                category VARCHAR(32) DEFAULT 'general',
                confidence FLOAT DEFAULT 1.0,
                user_approved INTEGER DEFAULT 1,
                updated_at DATETIME NOT NULL
            );
            """)
            # Check for user_approved column migration
            pref_cols = [c["name"] for c in cursor.execute("PRAGMA table_info(user_preferences);").fetchall()]
            if "user_approved" not in pref_cols:
                try:
                    cursor.execute("ALTER TABLE user_preferences ADD COLUMN user_approved INTEGER DEFAULT 1;")
                except Exception:
                    pass

            # 3. Knowledge Items Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title VARCHAR(128) NOT NULL,
                content TEXT NOT NULL,
                category VARCHAR(64) DEFAULT 'general',
                source VARCHAR(128) DEFAULT 'user',
                is_verified BOOLEAN DEFAULT 1,
                created_at DATETIME NOT NULL
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_cat ON knowledge_items(category);")

            # 4. Task Execution History Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(64) NOT NULL,
                goal TEXT NOT NULL,
                status VARCHAR(32) NOT NULL,
                steps_count INTEGER DEFAULT 1,
                duration_sec FLOAT DEFAULT 0.0,
                error_summary TEXT,
                lessons_learned TEXT,
                created_at DATETIME NOT NULL
            );
            """)
            conn.commit()

    def backup_database(self, backup_dir: str | Path = "backups") -> Path:
        """Create a timestamped backup copy of the SQLite database."""
        target_dir = Path(backup_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = target_dir / f"jarvis_memory_backup_{timestamp}.db"

        with self.get_connection() as src_conn:
            dst_conn = sqlite3.connect(str(backup_file))
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()

        return backup_file


# Default manager instance
db_manager = DatabaseManager()

__all__ = ["DatabaseManager", "db_manager"]
