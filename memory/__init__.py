"""J.A.R.V.I.S. Memory Package (Phase 8).

Modular Persistent Memory Architecture:
- database.py: SQLite connection manager & online backups
- short_term.py: Session-based in-memory working buffer
- long_term.py: Persistent SQLite storage (conversations, preferences, knowledge, tasks)
- retrieval.py: Memory search, context augmentation & deletion
"""

from .database import DatabaseManager, db_manager
from .long_term import LongTermMemory, SensitiveDataError, long_term_memory
from .retrieval import MemoryRetrieval, memory_retrieval
from .short_term import ShortTermMemory, ShortTermMessage, short_term_memory

__all__ = [
    "DatabaseManager",
    "LongTermMemory",
    "MemoryRetrieval",
    "SensitiveDataError",
    "ShortTermMemory",
    "ShortTermMessage",
    "db_manager",
    "long_term_memory",
    "memory_retrieval",
    "short_term_memory",
]
