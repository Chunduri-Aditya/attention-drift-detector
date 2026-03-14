"""
FastAPI routes for session management.

Endpoints
---------
GET  /sessions                  List sessions (optional ?user_id= filter)
GET  /sessions/{session_id}     Get a single session's metadata
GET  /sessions/{session_id}/events  Get full attention event timeline
GET  /sessions/{session_id}/report  Generate & return HTML report (auto-opens)
DELETE /sessions/{session_id}   Hard delete (privacy — removes all data)
"""
from __future__ import annotations

import webbrowser
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ...common.enums import AttentionState
from ...common.types import SessionRecord
from ...session.session_db import SessionDB
from ...session.session_report import SessionReportGenerator

router = APIRouter(prefix="/sessions", tags=["sessions"])

_db = SessionDB()
_report_gen = SessionReportGenerator()


# ------------------------------------------------------------------
# Pydantic response models
# ------------------------------------------------------------------

class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    started_at: str
    ended_at: Optional[str] = None
    focus_ratio: Optional[float] = None
    drift_events: int

    @classmethod
    def from_record(cls, r: SessionRecord) -> "SessionSummary":
        return cls(
            session_id=r.session_id,
            user_id=r.user_id,
            started_at=r.started_at.isoformat(),
            ended_at=r.ended_at.isoformat() if r.ended_at else None,
            focus_ratio=round(r.focus_ratio, 4) if r.focus_ratio is not None else None,
            drift_events=r.drift_events,
        )


class AttentionEventOut(BaseModel):
    timestamp: str
    state: AttentionState
    confidence: float


# ------------------------------------------------------------------
# Route handlers
# ------------------------------------------------------------------

@router.get("", response_model=list[SessionSummary])
def list_sessions(user_id: Optional[str] = None, limit: int = 50) -> list[SessionSummary]:
    """List the most recent sessions, optionally filtered by user."""
    records = _db.list_sessions(user_id=user_id, limit=limit)
    return [SessionSummary.from_record(r) for r in records]


@router.get("/{session_id}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    record = _db.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SessionSummary.from_record(record)


@router.get("/{session_id}/events", response_model=list[AttentionEventOut])
def get_events(session_id: str) -> list[AttentionEventOut]:
    """Return the full attention event timeline for a session."""
    # Verify session exists first
    if _db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    events = _db.get_events(session_id)
    return [
        AttentionEventOut(
            timestamp=e.timestamp.isoformat(),
            state=e.state,
            confidence=e.confidence,
        )
        for e in events
    ]


@router.get("/{session_id}/report", response_class=HTMLResponse)
def get_report(session_id: str, open_browser: bool = False) -> HTMLResponse:
    """Generate (or regenerate) a session HTML report and return it inline."""
    record = _db.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    events = _db.get_events(session_id)
    report_path = _report_gen.generate(record, events, auto_open=open_browser)
    html_content = report_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content, status_code=200)


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    """Permanently delete all data for a session (privacy hard-delete)."""
    if _db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    _db.delete_session(session_id)
