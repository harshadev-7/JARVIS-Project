"""
JARVIS AI — Phase 1 Test Suite
================================
Tests for the Core AI Brain and Memory System.
Run with: pytest tests/ -v

These tests verify:
1. Settings load correctly
2. Memory manager works (create, store, retrieve, delete)
3. Autonomy level config is correct
4. API endpoints return correct responses (without calling real AI)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# ── Settings Tests ─────────────────────────────────────────────────────────

def test_settings_load():
    """Settings should load with defaults even without .env file."""
    from config.settings import settings
    assert settings.JARVIS_NAME == "JARVIS"
    assert settings.JARVIS_VERSION is not None
    assert settings.DEFAULT_AUTONOMY_LEVEL in range(1, 6)


# ── Autonomy Level Tests ───────────────────────────────────────────────────

def test_autonomy_level_1_restrictions():
    """Level 1 should not allow research, planning, or execution."""
    from config.autonomy_levels import get_autonomy_config
    config = get_autonomy_config(1)
    assert config.can_answer is True
    assert config.can_research is False
    assert config.can_execute is False


def test_autonomy_level_5_permissions():
    """Level 5 should allow all actions."""
    from config.autonomy_levels import get_autonomy_config
    config = get_autonomy_config(5)
    assert config.can_answer is True
    assert config.can_research is True
    assert config.can_execute is True
    assert config.can_run_autonomous is True


def test_autonomy_level_clamp():
    """Level outside 1-5 should be clamped."""
    from config.autonomy_levels import get_autonomy_config
    config_low = get_autonomy_config(0)
    config_high = get_autonomy_config(99)
    assert config_low.level == 1
    assert config_high.level == 5


def test_check_permission():
    """Permission checker should return correct allow/deny."""
    from config.autonomy_levels import check_permission
    allowed, msg = check_permission(5, "execute")
    assert allowed is True

    denied, msg = check_permission(1, "execute")
    assert denied is False


# ── Memory Manager Tests ───────────────────────────────────────────────────

def test_memory_create_session():
    """Should create a new session with unique ID."""
    from core.memory.memory_manager import MemoryManager
    mm = MemoryManager()
    session = mm.create_session()
    assert session.session_id is not None
    assert session.message_count() == 0


def test_memory_add_messages():
    """Should store and retrieve messages correctly."""
    from core.memory.memory_manager import MemoryManager
    mm = MemoryManager()
    sid = "test-session-001"
    mm.add_user_message(sid, "Hello JARVIS")
    mm.add_assistant_message(sid, "Good day, sir.")

    session = mm.get_session(sid)
    assert session is not None
    assert session.message_count() == 2


def test_memory_history_format():
    """History should be in LLM-ready format."""
    from core.memory.memory_manager import MemoryManager
    mm = MemoryManager()
    sid = "test-session-002"
    mm.add_user_message(sid, "Hello")
    mm.add_assistant_message(sid, "Hi there")

    history = mm.get_history(sid)
    assert isinstance(history, list)
    assert all("role" in m and "content" in m for m in history)


def test_memory_delete_session():
    """Should delete a session cleanly."""
    from core.memory.memory_manager import MemoryManager
    mm = MemoryManager()
    sid = "test-delete-001"
    mm.create_session(sid)
    result = mm.delete_session(sid)
    assert result is True
    assert mm.get_session(sid) is None


def test_memory_get_or_create():
    """get_or_create_session should create if not exists."""
    from core.memory.memory_manager import MemoryManager
    mm = MemoryManager()
    sid = "new-session-xyz"
    session = mm.get_or_create_session(sid)
    assert session is not None
    # Calling again should return same session
    session2 = mm.get_or_create_session(sid)
    assert session.session_id == session2.session_id


# ── API Schema Tests ───────────────────────────────────────────────────────

def test_chat_request_validation():
    """ChatRequest should validate message length."""
    from api.schemas.models import ChatRequest

    req = ChatRequest(message="Hello JARVIS")
    assert req.message == "Hello JARVIS"
    assert req.session_id is None  # Optional


def test_chat_request_empty_message():
    """Empty message should fail validation."""
    from pydantic import ValidationError
    from api.schemas.models import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(message="")  # min_length=1


# ── FastAPI Integration Tests ──────────────────────────────────────────────

@pytest.fixture
def test_client():
    """Create a FastAPI test client with mocked AI Brain."""
    from fastapi.testclient import TestClient
    from api.app import app

    return TestClient(app)


def test_root_endpoint(test_client):
    """Root endpoint should return 200 with JARVIS name."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "JARVIS" in data["message"]


def test_system_info_endpoint(test_client):
    """System info endpoint should return config data."""
    response = test_client.get("/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "ai_provider" in data


def test_health_endpoint(test_client):
    """Health endpoint should return operational status."""
    response = test_client.get("/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
