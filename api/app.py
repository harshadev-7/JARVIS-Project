"""
JARVIS AI — FastAPI Application
=================================
The main API server for JARVIS.
Registers all routes, middleware, and startup/shutdown events.

Why FastAPI?
- Async by default (handles many concurrent requests)
- Auto-generates OpenAPI documentation at /docs
- Built-in Pydantic validation
- WebSocket support (needed for streaming in later phases)
- Production-grade with uvicorn
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import BASE_DIR, settings
from core.brain.jarvis_brain import brain
from core.memory.persistent_memory import persistent_memory
from core.logger import log
from api.routes import chat, system, reasoning, memory, research, voice, automation, devices, programming, agents, social, security
from api.websocket import router as websocket_router


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Use this to initialize heavy resources (AI clients, DB connections, etc.)
    """
    # ── STARTUP ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  {settings.JARVIS_NAME} AI v{settings.JARVIS_VERSION} starting up")
    log.info(f"  Environment : {settings.ENVIRONMENT}")
    log.info(f"  AI Provider : {settings.AI_PROVIDER}")
    log.info(f"  Autonomy    : Level {settings.DEFAULT_AUTONOMY_LEVEL}")
    log.info("=" * 60)

    # Initialize the Persistent Memory and Knowledge Database
    try:
        await persistent_memory.initialize()
        log.info("Persistent Memory & Vector Store online.")
    except Exception as e:
        log.error(f"Persistent memory initialization warning: {e}")

    # Initialize the AI Brain (graceful if key not yet entered)
    try:
        await brain.initialize()
        log.info("JARVIS Brain online.")
    except Exception as e:
        log.warning(f"JARVIS Brain running in standby mode (Key pending): {e}")

    yield  # Server is running here

    # ── SHUTDOWN ─────────────────────────────────────────────
    log.info("JARVIS shutting down. Goodbye, sir.")


# ── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="JARVIS AI",
    description=(
        "Advanced Autonomous Multi-Agent Artificial Intelligence System. "
        "Phase 1: Core AI Brain & Conversational Interface."
    ),
    version=settings.JARVIS_VERSION,
    docs_url="/docs",        # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",      # ReDoc UI at http://localhost:8000/redoc
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ───────────────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(system.router)
app.include_router(reasoning.router)
app.include_router(memory.router)
app.include_router(research.router)
app.include_router(voice.router)
app.include_router(automation.router)
app.include_router(devices.router)
app.include_router(programming.router)
app.include_router(agents.router)
app.include_router(social.router)
app.include_router(security.router)
app.include_router(websocket_router)


# ── Root & HUD Endpoints ───────────────────────────────────────────────────
@app.get("/hud", tags=["Interface"])
async def hud():
    """Futuristic Iron Man JARVIS Holographic HUD Interface."""
    hud_path = BASE_DIR / "ui" / "web" / "index.html"
    return FileResponse(hud_path)


@app.get("/health", tags=["System"])
async def health_alias():
    """Canonical health endpoint for monitoring and quick status checks."""
    return await system.health_check()


@app.get("/status", tags=["System"])
async def status_alias():
    """Alias used by monitoring and lightweight operational dashboards."""
    return await system.health_check()


@app.get("/metrics", tags=["System"])
async def metrics_alias():
    """Alias for the machine metrics route used by dashboards and monitoring."""
    return await automation.get_metrics()


@app.get("/ready", tags=["System"])
async def ready_alias():
    """Readiness endpoint for orchestrators and deployment checks."""
    payload = await system.health_check()
    data = payload.model_dump()
    data["ready"] = True
    return data


@app.get("/", tags=["Root"])
async def root():
    """JARVIS API root — confirms the server is running."""
    return {
        "message": f"Good day. I am {settings.JARVIS_NAME}, version {settings.JARVIS_VERSION}.",
        "hud": "/hud",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "status_alias": "/status",
        "metrics": "/metrics",
        "ready": "/ready",
        "system_health": "/system/health",
        "system_metrics": "/automation/system/metrics",
    }
