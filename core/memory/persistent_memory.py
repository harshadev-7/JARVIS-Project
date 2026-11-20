"""
JARVIS AI — Persistent Memory & Learning System
================================================
Coordinates all long-term memory categories:
- Short-term session memory
- Long-term episodic memory
- User preference memory
- Task execution history (workflow learning & mistake avoidance)
- Knowledge database with semantic vector search
- Project tracking

Provides full CRUD inspection and deletion of all memory records.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete

from database.models.memory_models import (
    Base,
    ConversationEpisode,
    UserPreference,
    TaskExecutionHistory,
    KnowledgeItem,
    ProjectRecord,
)
from core.memory.vector_memory import semantic_vector_store
from config.settings import settings
from core.logger import log


class PersistentMemoryManager:
    """
    Unified manager for structured SQL storage and semantic vector retrieval.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.db_url = database_url or settings.DATABASE_URL
        self.engine = create_async_engine(self.db_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self._initialized = False

    async def initialize(self):
        """Create tables and initialize vector indexes."""
        if self._initialized:
            return
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._initialized = True
        log.info("Persistent Memory & Knowledge Database initialized successfully.")

    # ── User Preferences ───────────────────────────────────────────────────────

    async def set_preference(self, key: str, value: str, category: str = "general", confidence: float = 1.0) -> UserPreference:
        """Store or update a user preference."""
        await self.initialize()
        async with self.session_factory() as session:
            async with session.begin():
                stmt = select(UserPreference).where(UserPreference.key == key)
                result = await session.execute(stmt)
                pref = result.scalar_one_or_none()

                if pref:
                    pref.value = value
                    pref.category = category
                    pref.confidence = confidence
                    pref.updated_at = datetime.now(timezone.utc)
                else:
                    pref = UserPreference(
                        key=key,
                        value=value,
                        category=category,
                        confidence=confidence,
                    )
                    session.add(pref)
            await session.commit()
            return pref

    async def get_preference(self, key: str) -> Optional[str]:
        """Fetch preference value by key."""
        await self.initialize()
        async with self.session_factory() as session:
            stmt = select(UserPreference).where(UserPreference.key == key)
            result = await session.execute(stmt)
            pref = result.scalar_one_or_none()
            return pref.value if pref else None

    async def list_preferences(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List stored user preferences."""
        await self.initialize()
        async with self.session_factory() as session:
            stmt = select(UserPreference)
            if category:
                stmt = stmt.where(UserPreference.category == category)
            result = await session.execute(stmt)
            prefs = result.scalars().all()
            return [
                {
                    "id": p.id,
                    "key": p.key,
                    "value": p.value,
                    "category": p.category,
                    "confidence": p.confidence,
                    "updated_at": p.updated_at.isoformat(),
                }
                for p in prefs
            ]

    # ── Task Execution History & Mistake Avoidance ─────────────────────────────

    async def log_task_execution(
        self,
        task_id: str,
        goal: str,
        status: str,
        steps_count: int = 1,
        duration_sec: float = 0.0,
        error_summary: Optional[str] = None,
        lessons_learned: Optional[str] = None,
    ):
        """Record task outcome for workflow improvement and learning."""
        await self.initialize()
        async with self.session_factory() as session:
            record = TaskExecutionHistory(
                task_id=task_id,
                goal=goal,
                status=status,
                steps_count=steps_count,
                duration_sec=duration_sec,
                error_summary=error_summary,
                lessons_learned=lessons_learned,
            )
            session.add(record)
            await session.commit()

        # Index lesson in vector memory if failed or informative
        if error_summary or lessons_learned:
            semantic_vector_store.add_document(
                doc_id=f"task-history-{task_id}",
                text=f"Task Goal: {goal}. Status: {status}. Error: {error_summary or 'None'}. Lesson: {lessons_learned or 'None'}",
                metadata={"category": "task_lesson", "task_id": task_id, "status": status},
            )

    async def get_mistakes_to_avoid(self, current_goal: str) -> List[str]:
        """
        Query vector memory for past failed workflows matching the current goal
        to prevent repeating prior mistakes.
        """
        matches = semantic_vector_store.search(current_goal, top_k=3)
        lessons: List[str] = []
        for doc, score in matches:
            if doc.metadata.get("status") == "failed" or "Error:" in doc.text:
                lessons.append(doc.text)
        return lessons

    # ── Knowledge Base with Semantic Search ────────────────────────────────────

    async def store_knowledge(
        self,
        title: str,
        content: str,
        category: str = "general",
        source: str = "user",
    ) -> int:
        """Add verified item to SQL knowledge base and vector index."""
        await self.initialize()
        async with self.session_factory() as session:
            item = KnowledgeItem(
                title=title,
                content=content,
                category=category,
                source=source,
                is_verified=True,
            )
            session.add(item)
            await session.commit()
            item_id = item.id

        # Index in vector search
        semantic_vector_store.add_document(
            doc_id=f"knowledge-{item_id}",
            text=f"{title}: {content}",
            metadata={"category": category, "source": source, "id": item_id},
        )
        return item_id

    async def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search across knowledge base."""
        results = semantic_vector_store.search(query, top_k=top_k)
        return [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "score": score,
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]

    # ── Deletion and Inspection ────────────────────────────────────────────────

    async def delete_item(self, table_name: str, item_id: int) -> bool:
        """Allow user to inspect, update, and delete stored information."""
        await self.initialize()
        model_map = {
            "preferences": (UserPreference, None),
            "knowledge": (KnowledgeItem, f"knowledge-{item_id}"),
            "tasks": (TaskExecutionHistory, f"task-history-{item_id}"),
        }
        if table_name not in model_map:
            raise ValueError(f"Unknown memory category: {table_name}")

        model, vector_id = model_map[table_name]

        async with self.session_factory() as session:
            stmt = delete(model).where(model.id == item_id)
            result = await session.execute(stmt)
            await session.commit()

        if vector_id:
            semantic_vector_store.delete_document(vector_id)

        return result.rowcount > 0


persistent_memory = PersistentMemoryManager()
