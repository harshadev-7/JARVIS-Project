"""
JARVIS AI — Base Agent Interface
=================================
Abstract foundation for all 15 specialized agents in the multi-agent system.
Enforces defined responsibility, tool registration, access permissions,
and structured result reporting.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.logger import log


class AgentMessage(BaseModel):
    """Inter-agent communication message."""
    sender: str
    recipient: str
    message_type: str  # "task_request", "data_share", "task_result", "error"
    content: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentTaskResult(BaseModel):
    """Standardized response from an agent execution."""
    agent_name: str
    task_id: str
    success: bool
    data: Any = None
    summary: str
    error: Optional[str] = None
    duration_sec: float = 0.0


class BaseAgent(ABC):
    """Abstract base class for all JARVIS specialized agents."""

    def __init__(
        self,
        name: str,
        role: str,
        capabilities: List[str],
        required_autonomy_level: int = 2,
    ):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.required_autonomy_level = required_autonomy_level
        self._inbox: List[AgentMessage] = []

    def receive_message(self, message: AgentMessage):
        """Receive communication from another agent."""
        self._inbox.append(message)
        log.debug(f"[{self.name}] Received message from [{message.sender}] type: {message.message_type}")

    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute assigned domain task."""
        pass
