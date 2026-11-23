"""
JARVIS AI — Memory Database Models
===================================
SQLAlchemy 2.0 Models for all memory categories:
- Episodic Memory (Conversation episodes & sessions)
- User Preferences (Explicit & inferred habits/settings)
- Task Execution History (Workflow logs, success/failure metrics, root causes)
- Knowledge Base (Structured facts, references, and verified knowledge)
- Project History (Ongoing projects, deliverables, milestones)
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Text, DateTime, Integer, Float, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ConversationEpisode(Base):
    """Episodic memory: Specific dialogue turns and historical interactions."""
    __tablename__ = "conversation_episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )


class UserPreference(Base):
    """User preference memory: Habits, preferred tools, tone, and system settings."""
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    value: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32), default="general")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )


class TaskExecutionHistory(Base):
    """Task history: Tracks successful and failed workflows to learn from feedback."""
    __tablename__ = "task_execution_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    goal: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))  # "completed", "failed", "aborted"
    steps_count: Mapped[int] = mapped_column(Integer, default=1)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    error_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lessons_learned: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class KnowledgeItem(Base):
    """Knowledge database: Facts, domain knowledge, and reference documentation."""
    __tablename__ = "knowledge_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), default="general", index=True)
    source: Mapped[str] = mapped_column(String(128), default="user")
    is_verified: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class ProjectRecord(Base):
    """Project history: Tracks status, architecture, and objectives of ongoing projects."""
    __tablename__ = "project_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="active")  # active, paused, completed
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )


class LearningRecord(Base):
    """Adaptive Learning Engine: Audited record of user corrections, feedback, and root causes."""
    __tablename__ = "learning_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    original_query: Mapped[str] = mapped_column(Text)
    original_response: Mapped[str] = mapped_column(Text)
    user_correction: Mapped[str] = mapped_column(Text)
    root_cause: Mapped[str] = mapped_column(String(64), default="KNOWLEDGE_GAP")
    status: Mapped[str] = mapped_column(String(32), default="pending_approval")  # pending_approval, approved, rejected, committed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )


class LearningTrainingExample(Base):
    """Adaptive Learning Engine: Training prompt-completion pairs synthesized from approved feedback."""
    __tablename__ = "learning_training_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    prompt: Mapped[str] = mapped_column(Text)
    completion: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), default="knowledge")
    is_staged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

