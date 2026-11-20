"""Unit Tests for J.A.R.V.I.S. Memory System (Phase 8).

Validates:
1. Database schema initialization in SQLite
2. Short-term working memory, sliding window, and scratchpad
3. Long-term conversation persistence
4. User preference storage and retrieval
5. Sensitive data security protection (passwords/API keys blocked without user approval)
6. Knowledge item indexing and search
7. Query context retrieval for prompt augmentation
8. Selective deletion and session history wiping
9. Online database backup creation
"""

from pathlib import Path
import pytest

from memory.database import DatabaseManager
from memory.long_term import LongTermMemory, SensitiveDataError
from memory.retrieval import MemoryRetrieval
from memory.short_term import ShortTermMemory


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_memory.db"
    mgr = DatabaseManager(db_path=db_file)
    ltm = LongTermMemory(manager=mgr)
    stm = ShortTermMemory(max_messages_per_session=5)
    retrieval = MemoryRetrieval(long_term=ltm, short_term=stm, db=mgr)
    return mgr, ltm, stm, retrieval


def test_database_initialization(test_db):
    mgr, _, _, _ = test_db
    with mgr.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]

    assert "conversation_history" in tables
    assert "user_preferences" in tables
    assert "knowledge_items" in tables
    assert "task_execution_history" in tables


def test_short_term_memory(test_db):
    _, _, stm, _ = test_db
    session = "test_sess_01"

    # Add 7 messages to a session with limit 5
    for i in range(7):
        stm.add_message(session, "user" if i % 2 == 0 else "assistant", f"Message {i}")

    msgs = stm.get_messages(session)
    assert len(msgs) == 5  # Sliding window maintained
    assert msgs[-1].content == "Message 6"

    # Scratchpad
    stm.set_scratchpad(session, "step", 3)
    assert stm.get_scratchpad(session, "step") == 3

    # Clear
    stm.clear_session(session)
    assert len(stm.get_messages(session)) == 0


def test_long_term_persistence(test_db):
    _, ltm, _, _ = test_db
    session = "long_term_sess"

    t1 = ltm.store_turn(session, "user", "What is my current project?")
    t2 = ltm.store_turn(session, "assistant", "You are building JARVIS.")

    assert t1 > 0
    assert t2 > t1

    history = ltm.get_history(session)
    assert len(history) == 2
    assert history[0]["content"] == "What is my current project?"
    assert history[1]["content"] == "You are building JARVIS."


def test_user_preferences(test_db):
    _, ltm, _, _ = test_db

    ltm.set_preference("editor", "VS Code", category="tools")
    pref = ltm.get_preference("editor")

    assert pref is not None
    assert pref["value"] == "VS Code"
    assert pref["category"] == "tools"

    # Update
    ltm.set_preference("editor", "Antigravity IDE", category="tools")
    updated = ltm.get_preference("editor")
    assert updated["value"] == "Antigravity IDE"


def test_sensitive_data_security_rejection(test_db):
    _, ltm, _, _ = test_db

    # Storing password or secret key without approval must raise SensitiveDataError
    with pytest.raises(SensitiveDataError):
        ltm.store_turn("sess", "user", "My password: SuperSecretPassword123!")

    with pytest.raises(SensitiveDataError):
        ltm.store_turn("sess", "user", "Here is my key: sk-abcdef123456789012345678")

    # With explicit allow_sensitive=True, storage succeeds
    saved_id = ltm.store_turn(
        "sess", "user", "Here is my key: sk-abcdef123456789012345678", allow_sensitive=True
    )
    assert saved_id > 0


def test_knowledge_search_and_context_retrieval(test_db):
    _, ltm, _, retrieval = test_db

    ltm.store_knowledge(
        title="Offline Mode",
        content="JARVIS operates completely offline without external APIs.",
        category="Architecture",
    )

    matches = retrieval.search_memory("offline mode architecture")
    assert len(matches) > 0
    assert matches[0]["title"] == "Offline Mode"

    context = retrieval.retrieve_context_for_query("offline mode")
    assert "JARVIS operates completely offline" in context


def test_memory_deletion(test_db):
    _, ltm, _, retrieval = test_db
    session = "delete_sess"

    ltm.store_turn(session, "user", "Temporary message")
    assert len(ltm.get_history(session)) == 1

    deleted_count = retrieval.delete_session_history(session)
    assert deleted_count == 1
    assert len(ltm.get_history(session)) == 0


def test_database_backup(test_db, tmp_path):
    mgr, _, _, _ = test_db
    backup_dir = tmp_path / "backups"

    backup_path = mgr.backup_database(backup_dir=backup_dir)
    assert backup_path.exists()
    assert backup_path.stat().st_size > 0
