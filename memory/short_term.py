"""J.A.R.V.I.S. Short-Term Working Memory (Phase 8).

Manages active in-memory session dialogue and transient scratchpad variables:
- Session-isolated short-term message lists
- Sliding-window context cropping
- Fast in-process state retrieval
- Ephemeral scratchpad buffers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ShortTermMessage:
    """An individual message in working memory."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ShortTermMemory:
    """Manages active session-scoped conversations and working memory."""

    def __init__(self, max_messages_per_session: int = 30):
        self.max_messages = max_messages_per_session
        self._sessions: Dict[str, List[ShortTermMessage]] = {}
        self._scratchpads: Dict[str, Dict[str, Any]] = {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShortTermMessage:
        """Append message to session history with sliding window limit."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self._scratchpads[session_id] = {}

        msg = ShortTermMessage(role=role, content=content, metadata=metadata or {})
        self._sessions[session_id].append(msg)

        # Enforce sliding window bounds
        if len(self._sessions[session_id]) > self.max_messages:
            self._sessions[session_id] = self._sessions[session_id][-self.max_messages :]

        return msg

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[ShortTermMessage]:
        """Retrieve recent messages for a session."""
        msgs = self._sessions.get(session_id, [])
        return msgs[-limit:] if limit else msgs[:]

    def get_context_for_prompt(self, session_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """Return messages in standard role/content format."""
        recent = self.get_messages(session_id, limit=max_turns)
        return [{"role": m.role, "content": m.content} for m in recent]

    def set_scratchpad(self, session_id: str, key: str, value: Any) -> None:
        """Store transient variable for active reasoning/agent coordination."""
        if session_id not in self._scratchpads:
            self._scratchpads[session_id] = {}
        self._scratchpads[session_id][key] = value

    def get_scratchpad(self, session_id: str, key: str, default: Any = None) -> Any:
        return self._scratchpads.get(session_id, {}).get(key, default)

    def clear_session(self, session_id: str) -> bool:
        """Clear short-term memory for a given session."""
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._scratchpads.pop(session_id, None)
        return existed

    def list_active_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# Default short-term memory manager
short_term_memory = ShortTermMemory()

__all__ = ["ShortTermMemory", "ShortTermMessage", "short_term_memory"]
