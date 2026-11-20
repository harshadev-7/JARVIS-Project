"""
JARVIS AI — Chat API Routes
============================
Handles all /chat endpoints: single messages and conversation history.

Endpoints:
  POST /chat/message      → Send a message, get JARVIS response
  GET  /chat/history/{id} → Get conversation history
  DELETE /chat/session/{id} → Delete a session
"""

import uuid
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from api.schemas.models import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    MessageHistory,
    ErrorResponse,
)
from core.brain.jarvis_brain import brain
from core.memory.memory_manager import memory_manager
from core.logger import log

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a message to JARVIS",
    description="Send a message and receive JARVIS's intelligent response. Include a session_id to maintain conversation context.",
)
async def send_message(request: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Create/retrieve conversation session
    2. Store user message in memory
    3. Retrieve conversation history (for context)
    4. Send to AI Brain
    5. Store JARVIS response in memory
    6. Return structured response
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    log.info(f"Chat request | session={session_id} | message_len={len(request.message)}")

    # Store user message
    memory_manager.add_user_message(session_id, request.message)

    # Get conversation history for LLM context (last 20 messages)
    history = memory_manager.get_history(session_id, max_messages=20)
    # Remove the last message (user just added it; we'll pass it separately to brain)
    history_without_current = history[:-1] if history else []

    # Get JARVIS response from AI Brain
    response = await brain.think(
        user_message=request.message,
        conversation_history=history_without_current,
    )

    if not response.success:
        log.error(f"Brain error: {response.error}")
        raise HTTPException(
            status_code=503,
            detail=f"AI Brain error: {response.error}. Check your API key and provider settings.",
        )

    # Store JARVIS response
    memory_manager.add_assistant_message(session_id, response.content)

    log.info(f"Response generated | session={session_id} | tokens={response.tokens_used}")

    return ChatResponse(
        reply=response.content,
        session_id=session_id,
        provider=response.provider,
        model=response.model,
        tokens_used=response.tokens_used,
    )


@router.get(
    "/history/{session_id}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
)
async def get_history(session_id: str):
    """Retrieve the full conversation history for a session."""
    session = memory_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    messages = [
        MessageHistory(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp.isoformat(),
        )
        for msg in session.messages
        if msg.role != "system"  # Don't expose system prompt to user
    ]

    return ConversationHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages),
    )


@router.delete(
    "/session/{session_id}",
    summary="Delete a conversation session",
)
async def delete_session(session_id: str):
    """Delete a session and all its conversation history."""
    deleted = memory_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"message": f"Session '{session_id}' deleted successfully."}
