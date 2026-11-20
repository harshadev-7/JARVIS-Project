"""
JARVIS AI — Logger
==================
Centralized logging using Loguru.
Writes to both console (with colors) and log file.

Why Loguru?
- Much simpler than Python's built-in logging
- Beautiful colored output in terminal
- Automatic file rotation
- Structured logging support
"""

import sys
from pathlib import Path
from loguru import logger
from config.settings import settings

# ── Log directory ─────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Remove default Loguru handler ─────────────────────────────────────────
logger.remove()

# ── Console handler (colored, human-readable) ─────────────────────────────
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ── File handler (JSON-structured for future log analysis) ────────────────
logger.add(
    LOG_DIR / "jarvis.log",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    rotation="10 MB",       # Rotate when file hits 10MB
    retention="30 days",    # Keep logs for 30 days
    compression="zip",      # Compress old logs
    enqueue=True,           # Thread-safe async logging
)

# ── Convenience re-export ─────────────────────────────────────────────────
# Usage: from core.logger import log
log = logger
