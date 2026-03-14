"""
FastAPI application factory.

Start the local dashboard server:
    uvicorn attention_drift_detector.api.app:app --host 127.0.0.1 --port 8765 --reload

Or via CLI:
    attention-drift-detector serve

Dashboard: http://127.0.0.1:8765/
API docs:  http://127.0.0.1:8765/docs
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes.sessions import router as sessions_router
from .stream import router as stream_router

_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Attention Drift Detector API",
    description=(
        "Local REST API for session management and reporting. "
        "All data is stored on-device — nothing leaves your machine."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow localhost origins (e.g., Obsidian local plugin, VS Code webview)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
app.include_router(stream_router)

# Serve static assets (dashboard.html etc.)
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "attention-drift-detector", "version": "0.2.0"}


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the live web dashboard at the root URL."""
    return FileResponse(str(_STATIC / "dashboard.html"))
