"""
JARVIS AI — Memory Manager (Phase 1: In-Memory + SQLite)
=========================================================
Manages conversation history and session context.

Phase 1 Implementation:
  - Short-term memory: Python dict (session-scoped, lost on restart)
  - Persistence: SQLite via SQLAlchemy
  - Each conversation has a session_id
  - Stores full message history per session

Phase 3 will upgrade this to:
  - Redis for short-term
  - PostgreSQL for long-term
  - ChromaDB for semantic/vector memory
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass, field

from core.logger import log


@dataclass
class Message:
    """A single message in a conversation."""
    role: str           # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_llm_format(self) -> dict:
        """Format for sending to LLM API (role + content only)."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationSession:
    """A single conversation session with JARVIS."""
    session_id: str
    created_at: datetime
    messages: List[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> Message:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg

    def get_history_for_llm(self, max_messages: int = 20) -> List[dict]:
        """
        Returns the last N messages in LLM-ready format.
        Always includes system message if present.

        Why limit messages?
        - LLMs have context window limits (token limits)
        - Sending too many messages = higher API cost
        - Most relevant context is recent
        """
        # Always keep system messages
        system_msgs = [m for m in self.messages if m.role == "system"]
        # Get recent non-system messages
        conversation = [m for m in self.messages if m.role != "system"]
        recent = conversation[-max_messages:]
        all_msgs = system_msgs + recent
        return [m.to_llm_format() for m in all_msgs]

    def message_count(self) -> int:
        return len([m for m in self.messages if m.role != "system"])


class MemoryManager:
    """
    Manages all conversation sessions.
    Phase 1: In-memory storage (resets on restart).
    """

    def __init__(self):
        # session_id → ConversationSession
        self._sessions: dict[str, ConversationSession] = {}
        log.info("MemoryManager initialized (Phase 1: in-memory)")

    def create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Create a new conversation session."""
        sid = session_id or str(uuid.uuid4())
        session = ConversationSession(
            session_id=sid,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[sid] = session
        log.debug(f"New session created: {sid}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve an existing session by ID."""
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create a new one."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        return session

    def add_user_message(self, session_id: str, content: str) -> Message:
        """Add a user message to a session."""
        session = self.get_or_create_session(session_id)
        return session.add_message("user", content)

    def add_assistant_message(self, session_id: str, content: str) -> Message:
        """Add an assistant (JARVIS) message to a session."""
        session = self.get_or_create_session(session_id)
        return session.add_message("assistant", content)

    def get_history(self, session_id: str, max_messages: int = 20) -> List[dict]:
        """Get conversation history for LLM context."""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get_history_for_llm(max_messages)

    def list_sessions(self) -> List[dict]:
        """List all active sessions with metadata."""
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "message_count": s.message_count(),
            }
            for s in self._sessions.values()
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its history."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            log.info(f"Session deleted: {session_id}")
            return True
        return False

    def clear_all(self) -> int:
        """Clear all sessions. Returns count of deleted sessions."""
        count = len(self._sessions)
        self._sessions.clear()
        log.warning(f"All {count} sessions cleared from memory")
        return count


# ── Singleton instance ────────────────────────────────────────────────────
memory_manager = MemoryManager()
