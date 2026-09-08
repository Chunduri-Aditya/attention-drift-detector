from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Literal, Optional

from .enums import AttentionState


@dataclass(slots=True)
class FrameFeatures:
    """
    Per-frame features derived from tracking components.
    Raw frames are never stored; only these features are persisted.
    """

    timestamp: datetime
    head_angles_deg: Tuple[float, float, float]  # (pitch, yaw, roll)
    gaze_vector: Tuple[float, float, float]  # (gaze_x, gaze_y, gaze_z) unit vector
    blink: bool
    face_visibility: float  # MediaPipe visibility/confidence score


@dataclass(slots=True)
class AttentionEvent:
    """
    Attention state at a point in time (after temporal smoothing).
    """

    timestamp: datetime
    state: AttentionState
    confidence: float
    source: str = "rule"  # "rule" (calibrated thresholds) or "ml" (trained model)


@dataclass(slots=True)
class DriftAlert:
    """
    Represents a nudge emitted due to sustained attention drift.
    """

    timestamp: datetime
    session_id: str
    reason: Literal["sustained_drift"]
    cooldown_remaining_s: float


@dataclass(slots=True)
class SessionRecord:
    """
    High-level session metadata stored in SQLite and reports.
    """

    session_id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    focus_ratio: Optional[float] = None
    drift_events: int = 0

