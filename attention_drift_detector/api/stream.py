"""
Live state endpoint for the web dashboard.

The realtime pipeline calls `update_live_state()` from its worker thread.
The SSE endpoint at /api/live polls this shared dict and streams JSON to
the browser every second.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# ── shared mutable state (pipeline writes, SSE reads) ─────────────────────────
_live: dict[str, Any] = {
    "running": False,
    "session_id": None,
    "state": None,
    "confidence": None,
    "focus_ratio": None,
    "drift_alerts": 0,
    "elapsed_seconds": 0,
    "calibrated": False,
    "updated_at": None,
}


def update_live_state(
    *,
    session_id: str,
    state: str,
    confidence: float,
    focus_ratio: float,
    drift_alerts: int,
    elapsed_seconds: float,
    calibrated: bool,
) -> None:
    """Called from the pipeline thread on every processed frame."""
    _live.update(
        running=True,
        session_id=session_id,
        state=state,
        confidence=round(confidence, 3),
        focus_ratio=round(focus_ratio, 1),
        drift_alerts=drift_alerts,
        elapsed_seconds=int(elapsed_seconds),
        calibrated=calibrated,
        updated_at=datetime.utcnow().isoformat(),
    )


def clear_live_state() -> None:
    """Called when the pipeline ends."""
    _live["running"] = False
    _live["updated_at"] = datetime.utcnow().isoformat()


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/live")
async def get_live() -> dict:
    """Snapshot of the current pipeline state. Poll this every ~1s."""
    return dict(_live)


@router.get("/api/live/stream")
async def sse_stream() -> StreamingResponse:
    """
    Server-Sent Events stream.  Connect with:
        const es = new EventSource('/api/live/stream');
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    async def _generator():
        while True:
            payload = json.dumps(_live)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
