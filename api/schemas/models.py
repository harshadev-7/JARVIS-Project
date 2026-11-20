"""
JARVIS AI — API Schemas (Pydantic Models)
==========================================
Defines the data structures for all API requests and responses.
"""

from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


# ── Chat ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for sending a message to JARVIS."""
    message: str = Field(..., min_length=1, max_length=10000, description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity. Auto-generated if not provided.")
    stream: bool = Field(False, description="Enable streaming response via Server-Sent Events")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Hey Jarvis, how are you?",
                "session_id": "user-session-001",
                "stream": False,
            }
        }
    )


class ChatResponse(BaseModel):
    """Response from JARVIS for a chat message."""
    reply: str = Field(..., description="JARVIS's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    provider: str = Field(..., description="AI provider used (openai/gemini/etc)")
    model: str = Field(..., description="Specific model used")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed (if available)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reply": "Good evening, sir. How may I assist you?",
                "session_id": "user-session-001",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "tokens_used": 150,
            }
        }
    )


# ── Session ───────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    """Information about a conversation session."""
    session_id: str
    created_at: datetime
    message_count: int


class SessionListResponse(BaseModel):
    """List of all active sessions."""
    sessions: List[SessionInfo]
    total: int


class MessageHistory(BaseModel):
    """A message in the conversation history."""
    role: str
    content: str
    timestamp: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: List[MessageHistory]
    total_messages: int


# ── Health ────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """System health status."""
    status: str = "operational"
    version: str
    provider: str
    model: str
    environment: str
    uptime_seconds: Optional[float] = None


# ── Error ─────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
