"""Unit tests for J.A.R.V.I.S. Knowledge Base & RAG Engine."""

import pytest
from knowledge.knowledge_base import knowledge_base
from knowledge.kb_ingestion import kb_ingestion
from knowledge.rag_engine import rag_engine


@pytest.mark.asyncio
async def test_knowledge_base_crud():
    await knowledge_base.initialize()

    # Add knowledge
    item_id = await knowledge_base.add_knowledge(
        title="Java Virtual Threads",
        content="Virtual threads were finalized in Java 21 under JEP 444 for high-throughput concurrency.",
        category="java",
        source="test_suite",
    )
    assert isinstance(item_id, int)
    assert item_id > 0

    # Search knowledge
    results = await knowledge_base.search("virtual threads high throughput", top_k=2)
    assert len(results) > 0
    assert any("Virtual" in r.title or "threads" in r.content for r in results)

    # Clean up
    deleted = await knowledge_base.delete(item_id)
    assert deleted is True


@pytest.mark.asyncio
async def test_kb_ingestion_chunking():
    sample_text = (
        "Operating systems coordinate CPU execution and memory frames.\n\n"
        "Deadlock requires mutual exclusion, hold and wait, no preemption, and circular wait.\n\n"
        "Banker algorithm guarantees safe state prior to resource allocation."
    )
    ids = await kb_ingestion.ingest_document(
        title="OS Concurrency Notes",
        text=sample_text,
        category="operating_systems",
        source="unit_test",
    )
    assert len(ids) == 3

    # Clean up
    for item_id in ids:
        await knowledge_base.delete(item_id)


@pytest.mark.asyncio
async def test_rag_engine_retrieval():
    item_id = await knowledge_base.add_knowledge(
        title="Dijkstra Shortest Path",
        content="Dijkstra algorithm calculates shortest path in graphs with non-negative edge weights.",
        category="algorithms",
        source="test_rag",
    )

    context = await rag_engine.retrieve("shortest path in graphs", top_k=2)
    assert context.has_relevant_knowledge is True
    assert len(context.entries) > 0
    assert "Dijkstra" in context.augmented_prompt

    # Clean up
    await knowledge_base.delete(item_id)
