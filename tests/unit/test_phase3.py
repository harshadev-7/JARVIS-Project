"""
JARVIS AI — Phase 3 Test Suite
================================
Tests for Long-Term Memory, Knowledge Base, and Vector Semantic Search:
- Vector Space cosine similarity and semantic search
- Persistent User Preference tracking
- Task Execution History & Mistake Avoidance
- Knowledge Storage & Retrieval
- FastAPI Memory Endpoints
"""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from core.memory.vector_memory import SemanticVectorStore
from core.memory.persistent_memory import PersistentMemoryManager
from api.app import app


# ── 1. Semantic Vector Store Tests ─────────────────────────────────────────

def test_semantic_vector_store():
    """Vector store should calculate cosine similarity and return top matches."""
    store = SemanticVectorStore()
    store.add_document("doc-1", "FastAPI is a modern, high-performance web framework for building APIs with Python.")
    store.add_document("doc-2", "PostgreSQL is a powerful, open source object-relational database system.")
    store.add_document("doc-3", "Docker containers package software and its dependencies together.")

    results = store.search("Python web framework", top_k=2)
    assert len(results) > 0
    top_doc, score = results[0]
    assert top_doc.doc_id == "doc-1"
    assert score > 0.1


def test_vector_store_deletion():
    """Deleted document should no longer appear in search results."""
    store = SemanticVectorStore()
    store.add_document("temp-1", "Temporary document to delete")
    assert store.count() == 1

    deleted = store.delete_document("temp-1")
    assert deleted is True
    assert store.count() == 0


# ── 2. Persistent Memory Manager Tests ─────────────────────────────────────

@pytest.mark.asyncio
async def test_persistent_preferences(tmp_path):
    """User preferences should persist and be retrievable."""
    db_file = tmp_path / "test_memory.db"
    pm = PersistentMemoryManager(database_url=f"sqlite+aiosqlite:///{db_file}")
    await pm.initialize()

    await pm.set_preference("editor", "VSCode", category="developer")
    val = await pm.get_preference("editor")
    assert val == "VSCode"

    prefs = await pm.list_preferences(category="developer")
    assert len(prefs) == 1
    assert prefs[0]["key"] == "editor"


@pytest.mark.asyncio
async def test_persistent_knowledge_and_semantic_search(tmp_path):
    """Knowledge items should be saved in SQL and indexed in vector space."""
    db_file = tmp_path / "test_knowledge.db"
    pm = PersistentMemoryManager(database_url=f"sqlite+aiosqlite:///{db_file}")
    await pm.initialize()

    await pm.store_knowledge(
        title="Microservices Architecture",
        content="Microservices structure an application as a collection of loosely coupled services.",
        category="architecture",
    )

    results = await pm.search_knowledge("loosely coupled application services", top_k=1)
    assert len(results) == 1
    assert "Microservices" in results[0]["text"]


@pytest.mark.asyncio
async def test_task_history_and_mistake_avoidance(tmp_path):
    """Failed workflows should be retrievable to prevent repeating past mistakes."""
    db_file = tmp_path / "test_tasks.db"
    pm = PersistentMemoryManager(database_url=f"sqlite+aiosqlite:///{db_file}")
    await pm.initialize()

    await pm.log_task_execution(
        task_id="task-99",
        goal="Direct deploy to production server without staging verification",
        status="failed",
        error_summary="Deployment rejected due to schema mismatch",
        lessons_learned="Always run schema migrations on staging prior to production deploy",
    )

    mistakes = await pm.get_mistakes_to_avoid("deploy to production server")
    assert len(mistakes) > 0
    assert "schema mismatch" in mistakes[0] or "staging" in mistakes[0]


# ── 3. FastAPI Memory Endpoints Tests ───────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_preferences_flow(client):
    """POST /memory/preferences then GET /memory/preferences."""
    post_res = client.post(
        "/memory/preferences",
        json={"key": "assistant_tone", "value": "concise", "category": "personality"},
    )
    assert post_res.status_code == 200

    get_res = client.get("/memory/preferences?category=personality")
    assert get_res.status_code == 200
    data = get_res.json()
    assert any(p["key"] == "assistant_tone" for p in data)


def test_api_knowledge_flow(client):
    """POST /memory/knowledge then GET /memory/knowledge/search."""
    post_res = client.post(
        "/memory/knowledge",
        json={
            "title": "Quantum Supremacy",
            "content": "Quantum supremacy is the demonstration that a programmable quantum device can solve a problem that no classical computer can solve in any feasible amount of time.",
            "category": "physics",
        },
    )
    assert post_res.status_code == 200
    item_id = post_res.json()["id"]

    search_res = client.get("/memory/knowledge/search?q=quantum+classical+computer")
    assert search_res.status_code == 200
    results = search_res.json()["results"]
    assert len(results) > 0
