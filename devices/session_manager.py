"""
JARVIS AI — Secure Network Session Lifecycle Manager
====================================================
Manages cryptographic session tokens, TTL expirations, inactivity timeouts,
and immediate revocation for connected mesh devices.
Persisted in SQLite (network_sessions table) with in-memory caching.
"""

import sqlite3
import hashlib
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

from core.logger import log


class SessionRecord(BaseModel):
    session_id: str
    device_id: str
    session_key_bytes: bytes
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"  # "active", "expired", "revoked"


class NetworkSessionManager:
    """Manages active network sessions, timeouts, and revocations."""

    def __init__(self, db_path: Optional[str] = None, default_ttl_sec: int = 86400, inactivity_timeout_sec: int = 1800):
        if db_path is None:
            from config.settings import BASE_DIR
            self.db_path = str(BASE_DIR / "jarvis.db")
        else:
            self.db_path = db_path

        self.default_ttl = default_ttl_sec
        self.inactivity_timeout = inactivity_timeout_sec
        self._sessions: Dict[str, SessionRecord] = {}
        self._init_db()

    def _init_db(self):
        """Create network_sessions table in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS network_sessions (
                        session_id TEXT PRIMARY KEY,
                        device_id TEXT NOT NULL,
                        session_key_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active'
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning(f"NetworkSessionManager: SQLite init notice ({e}).")

    def register_session(
        self,
        session_id: str,
        device_id: str,
        session_key: bytes,
        ttl_seconds: Optional[int] = None,
    ) -> SessionRecord:
        """Register a new active encrypted session."""
        ttl = ttl_seconds or self.default_ttl
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=session_id,
            device_id=device_id,
            session_key_bytes=session_key,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            last_activity=now,
            status="active",
        )

        self._sessions[session_id] = record

        # Persist hash to SQLite (never raw key to disk)
        key_hash = hashlib.sha256(session_key).hexdigest()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO network_sessions
                    (session_id, device_id, session_key_hash, created_at, expires_at, last_activity, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        device_id,
                        key_hash,
                        now.isoformat(),
                        record.expires_at.isoformat(),
                        now.isoformat(),
                        "active",
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning(f"Failed to persist session to DB: {e}")

        log.info(f"Registered secure network session [{session_id}] for device [{device_id}] (TTL {ttl}s).")
        return record

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Fetch session and verify validity."""
        rec = self._sessions.get(session_id)
        if not rec or rec.status != "active":
            return None

        now = datetime.now(timezone.utc)
        # Check hard TTL expiration
        if now > rec.expires_at:
            rec.status = "expired"
            self._update_db_status(session_id, "expired")
            return None

        # Check inactivity timeout
        if (now - rec.last_activity).total_seconds() > self.inactivity_timeout:
            rec.status = "expired"
            self._update_db_status(session_id, "expired")
            return None

        return rec

    def touch_session(self, session_id: str) -> bool:
        """Update last_activity timestamp on active traffic."""
        rec = self.get_session(session_id)
        if not rec:
            return False

        rec.last_activity = datetime.now(timezone.utc)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE network_sessions SET last_activity = ? WHERE session_id = ?",
                    (rec.last_activity.isoformat(), session_id),
                )
                conn.commit()
        except Exception:
            pass
        return True

    def revoke_session(self, session_id: str) -> bool:
        """Immediately terminate an active session."""
        rec = self._sessions.get(session_id)
        if not rec:
            return False

        rec.status = "revoked"
        self._update_db_status(session_id, "revoked")
        log.warning(f"Revoked secure network session [{session_id}].")
        return True

    def revoke_all_device_sessions(self, device_id: str) -> int:
        """Revoke all sessions belonging to a specific device."""
        count = 0
        for sid, rec in list(self._sessions.items()):
            if rec.device_id == device_id and rec.status == "active":
                self.revoke_session(sid)
                count += 1
        return count

    def cleanup_expired_sessions(self) -> int:
        """Prune expired sessions from memory."""
        now = datetime.now(timezone.utc)
        expired_count = 0
        for sid, rec in list(self._sessions.items()):
            if rec.status != "active" or now > rec.expires_at or (now - rec.last_activity).total_seconds() > self.inactivity_timeout:
                rec.status = "expired"
                self._update_db_status(sid, "expired")
                del self._sessions[sid]
                expired_count += 1
        return expired_count

    def _update_db_status(self, session_id: str, status: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE network_sessions SET status = ? WHERE session_id = ?", (status, session_id))
                conn.commit()
        except Exception:
            pass


session_manager = NetworkSessionManager()
