"""
JARVIS AI — Token Authenticator & Device Session Manager
========================================================
Handles cryptographic token verification, session lifecycle,
heartbeat telemetry, and instantaneous revocation for authorized devices.
"""

import os
import hashlib
import sqlite3
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.logger import log


class DeviceSession(BaseModel):
    device_id: str
    token_hash: str
    ip_address: Optional[str] = None
    status: str = "online"  # "online", "offline", "revoked"
    battery_level: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TokenAuthenticator:
    """Manages secure token hashing, verification, and session timeouts."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from config.settings import BASE_DIR
            self.db_path = str(BASE_DIR / "jarvis.db")
        else:
            self.db_path = db_path
            
        self._sessions: Dict[str, DeviceSession] = {}  # device_id -> DeviceSession
        self._token_to_device: Dict[str, str] = {}    # token_hash -> device_id
        self._init_db()

    def _init_db(self):
        """Create authorized_devices table in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS authorized_devices (
                        device_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        device_type TEXT NOT NULL,
                        auth_token_hash TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'offline',
                        battery_level REAL,
                        capabilities TEXT,
                        permissions TEXT,
                        paired_at TEXT NOT NULL,
                        last_seen TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning(f"TokenAuthenticator: SQLite init notice ({e}).")

    @staticmethod
    def hash_token(raw_token: str) -> str:
        """Compute SHA-256 hex digest of raw bearer token."""
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def register_device(
        self,
        device_id: str,
        raw_token: str,
        name: str,
        device_type: str,
        ip_address: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> str:
        """Register a newly paired device session and persist to DB."""
        token_hash = self.hash_token(raw_token)
        now = datetime.now(timezone.utc)

        session = DeviceSession(
            device_id=device_id,
            token_hash=token_hash,
            ip_address=ip_address,
            status="online",
            last_seen=now,
        )

        self._sessions[device_id] = session
        self._token_to_device[token_hash] = device_id

        # Persist to SQLite
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO authorized_devices 
                    (device_id, name, device_type, auth_token_hash, status, battery_level, capabilities, permissions, paired_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        device_id,
                        name,
                        device_type,
                        token_hash,
                        "online",
                        None,
                        json.dumps(capabilities or []),
                        json.dumps([]),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning(f"Failed to persist device to DB: {e}")

        log.info(f"Registered authorized session for device [{device_id}] ({name}).")
        return token_hash

    def authenticate_token(self, raw_token: str) -> Optional[str]:
        """
        Verify raw bearer token against registered sessions.
        Returns device_id if valid and active, None otherwise.
        """
        if not raw_token:
            return None
        token_hash = self.hash_token(raw_token)
        device_id = self._token_to_device.get(token_hash)
        if not device_id:
            return None

        session = self._sessions.get(device_id)
        if not session or session.status == "revoked":
            return None

        # Update last seen
        session.last_seen = datetime.now(timezone.utc)
        return device_id

    def update_heartbeat(
        self,
        device_id: str,
        battery_level: Optional[float] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Record device heartbeat telemetry."""
        session = self._sessions.get(device_id)
        if not session or session.status == "revoked":
            return False

        session.last_seen = datetime.now(timezone.utc)
        session.status = "online"
        if battery_level is not None:
            session.battery_level = battery_level
        if ip_address:
            session.ip_address = ip_address

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE authorized_devices SET status = 'online', battery_level = ?, last_seen = ? WHERE device_id = ?",
                    (session.battery_level, session.last_seen.isoformat(), device_id),
                )
                conn.commit()
        except Exception:
            pass

        return True

    def revoke_device(self, device_id: str) -> bool:
        """Immediately revoke device token, terminating access."""
        session = self._sessions.get(device_id)
        if not session:
            return False

        session.status = "revoked"
        if session.token_hash in self._token_to_device:
            del self._token_to_device[session.token_hash]
        del self._sessions[device_id]

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM authorized_devices WHERE device_id = ?", (device_id,))
                conn.commit()
        except Exception as e:
            log.warning(f"Error revoking device in DB: {e}")

        log.warning(f"Device [{device_id}] authentication revoked permanently.")
        return True

    def get_session(self, device_id: str) -> Optional[DeviceSession]:
        """Fetch session information for a device."""
        return self._sessions.get(device_id)

    def check_timeouts(self, timeout_seconds: int = 60):
        """Mark devices offline if heartbeat exceeded timeout."""
        now = datetime.now(timezone.utc)
        for session in self._sessions.values():
            if session.status == "online":
                elapsed = (now - session.last_seen).total_seconds()
                if elapsed > timeout_seconds:
                    session.status = "offline"


token_authenticator = TokenAuthenticator()
