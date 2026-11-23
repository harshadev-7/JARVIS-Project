"""J.A.R.V.I.S. Adaptive Learning Registry.

Coordinates SQLite persistence for:
- Learning records (user feedback, corrections, root-cause diagnosis, approval states)
- Synthesized training examples for optional neural fine-tuning
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.memory_models import (
    Base,
    LearningRecord,
    LearningTrainingExample,
)
from core.memory.persistent_memory import persistent_memory
from core.logger import log


class LearningRegistry:
    """Manages audited storage of learning records and candidate training pairs."""

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """Ensure database tables are created."""
        if self._initialized:
            return
        await persistent_memory.initialize()
        self._initialized = True

    async def record_feedback(
        self,
        session_id: str,
        original_query: str,
        original_response: str,
        user_correction: str,
        root_cause: str = "KNOWLEDGE_GAP",
        status: str = "pending_approval",
    ) -> int:
        """Create a new audited learning record."""
        await self.initialize()
        async with persistent_memory.session_factory() as session:
            record = LearningRecord(
                session_id=session_id,
                original_query=original_query,
                original_response=original_response,
                user_correction=user_correction,
                root_cause=root_cause,
                status=status,
                created_at=datetime.now(timezone.utc),
            )
            session.add(record)
            await session.commit()
            record_id = record.id

        log.info(f"LearningRegistry created record #{record_id} [{root_cause}] for session {session_id}")
        return record_id

    async def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific learning record by ID."""
        await self.initialize()
        async with persistent_memory.session_factory() as session:
            stmt = select(LearningRecord).where(LearningRecord.id == record_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
            if not record:
                return None
            return {
                "id": record.id,
                "session_id": record.session_id,
                "original_query": record.original_query,
                "original_response": record.original_response,
                "user_correction": record.user_correction,
                "root_cause": record.root_cause,
                "status": record.status,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "approved_at": record.approved_at.isoformat() if record.approved_at else None,
            }

    async def list_records(
        self, status: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List learning records with optional status filter."""
        await self.initialize()
        async with persistent_memory.session_factory() as session:
            stmt = select(LearningRecord)
            if status:
                stmt = stmt.where(LearningRecord.status == status)
            stmt = stmt.order_by(LearningRecord.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            records = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "original_query": r.original_query,
                    "original_response": r.original_response,
                    "user_correction": r.user_correction,
                    "root_cause": r.root_cause,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "approved_at": r.approved_at.isoformat() if r.approved_at else None,
                }
                for r in records
            ]

    async def update_status(self, record_id: int, new_status: str) -> bool:
        """Update record status ('approved', 'rejected', 'committed')."""
        await self.initialize()
        async with persistent_memory.session_factory() as session:
            stmt = select(LearningRecord).where(LearningRecord.id == record_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
            if not record:
                return False
            record.status = new_status
            if new_status in ("approved", "committed") and not record.approved_at:
                record.approved_at = datetime.now(timezone.utc)
            await session.commit()
            return True

    async def add_training_example(
        self,
        prompt: str,
        completion: str,
        category: str = "knowledge",
        record_id: Optional[int] = None,
        is_staged: bool = True,
    ) -> int:
        """Store synthesized training example."""
        await self.initialize()
        async with persistent_memory.session_factory() as session:
            example = LearningTrainingExample(
                record_id=record_id,
                prompt=prompt,
                completion=completion,
                category=category,
                is_staged=is_staged,
                created_at=datetime.now(timezone.utc),
            )
            session.add(example)
            await session.commit()
            example_id = example.id

        log.info(f"LearningRegistry stored training example #{example_id} [staged={is_staged}]")
        return example_id

    async def list_staged_examples(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all currently staged training examples."""
        await self.initialize()
        async with persistent_memory.session_factory() as session:
            stmt = (
                select(LearningTrainingExample)
                .where(LearningTrainingExample.is_staged == True)
                .order_by(LearningTrainingExample.created_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            examples = result.scalars().all()
            return [
                {
                    "id": ex.id,
                    "record_id": ex.record_id,
                    "prompt": ex.prompt,
                    "completion": ex.completion,
                    "category": ex.category,
                    "created_at": ex.created_at.isoformat() if ex.created_at else None,
                }
                for ex in examples
            ]

    async def unstage_examples(self, example_ids: List[int]) -> int:
        """Mark examples as un-staged after successful training."""
        await self.initialize()
        if not example_ids:
            return 0
        async with persistent_memory.session_factory() as session:
            stmt = (
                update(LearningTrainingExample)
                .where(LearningTrainingExample.id.in_(example_ids))
                .values(is_staged=False)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount


learning_registry = LearningRegistry()
__all__ = ["LearningRegistry", "learning_registry"]
