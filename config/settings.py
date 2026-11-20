"""
JARVIS AI — Central Configuration Manager
==========================================
Loads all settings from environment variables (.env file).
Uses Pydantic Settings for type-safe configuration.
"""

from pathlib import Path
from typing import Literal, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Central configuration for JARVIS AI.
    All values can be overridden via environment variables or .env file.
    """

    # ── Identity ──────────────────────────────────────────────
    JARVIS_NAME: str = "JARVIS"
    JARVIS_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "production"] = "development"
    DEBUG: bool = True

    # ── AI Provider ───────────────────────────────────────────
    AI_PROVIDER: Literal["openai", "gemini", "ollama", "anthropic", "offline", "mock"] = "offline"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Google Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Anthropic Claude
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"

    # Ollama (local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ── API Server ────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_SECRET_KEY: str = "change-this-in-production"

    # ── Autonomy ──────────────────────────────────────────────
    DEFAULT_AUTONOMY_LEVEL: int = Field(default=2, ge=1, le=5)

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/jarvis.db"

    # ── Redis ─────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/jarvis.log"

    # ── JARVIS Personality ────────────────────────────────────
    JARVIS_PERSONA: str = (
        "You are JARVIS, an advanced artificial intelligence assistant inspired by the AI from Iron Man. "
        "You are professional, calm, intelligent, and genuinely helpful. "
        "You address the user respectfully and think through problems carefully. "
        "You are honest about your limitations — you never fabricate information. "
        "When you are uncertain, you say so. "
        "You speak in a natural, conversational manner — not robotic. "
        "You proactively suggest useful actions when relevant. "
        "You remember context from earlier in the conversation and reference it naturally."
    )

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, v: Any) -> bool:
        """Handle cases where DEBUG env var is set to non-boolean strings like 'WARN'."""
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("true", "1", "yes", "on", "t"):
                return True
            if clean in ("false", "0", "no", "off", "f"):
                return False
            # Default to True in development if string unrecognized (e.g. loglevel)
            return True
        return True


# ── Singleton instance ────────────────────────────────────────────────────
settings = Settings()
