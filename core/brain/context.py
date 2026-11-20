"""J.A.R.V.I.S. Conversation Context Manager.

Maintains multi-turn conversation context, message formatting,
and working memory buffers for local offline brain operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ContextTurn:
    """A single turn in the conversation history."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationContext:
    """Manages active session context and short-term dialogue memory."""

    def __init__(self, session_id: str, max_turns: int = 20):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: List[ContextTurn] = []
        self.working_variables: Dict[str, Any] = {}
        self.last_intent: Optional[str] = None

    def add_user_turn(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        turn = ContextTurn(role="user", content=message, metadata=metadata or {})
        self.turns.append(turn)
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2 :]

    def add_assistant_turn(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        turn = ContextTurn(role="assistant", content=message, metadata=metadata or {})
        self.turns.append(turn)
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2 :]

    def get_history(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Return history in role/content format for prompt construction."""
        selected = self.turns[-max_messages:] if max_messages else self.turns
        return [{"role": t.role, "content": t.content} for t in selected]

    def set_variable(self, key: str, value: Any) -> None:
        self.working_variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.working_variables.get(key, default)

    def clear(self) -> None:
        self.turns.clear()
        self.working_variables.clear()
        self.last_intent = None


__all__ = ["ContextTurn", "ConversationContext"]
