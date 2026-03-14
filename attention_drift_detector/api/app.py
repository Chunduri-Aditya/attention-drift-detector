"""
FastAPI application factory.

Start the local dashboard server:
    uvicorn attention_drift_detector.api.app:app --host 127.0.0.1 --port 8765 --reload

Or via CLI (once wired):
    attention-drift-detector serve
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.sessions import router as sessions_router

app = FastAPI(
    title="Attention Drift Detector API",
    description=(
        "Local REST API for session management and reporting. "
        "All data is stored on-device — nothing leaves your machine."
    ),
    version="0.1.0",
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "attention-drift-detector"}
