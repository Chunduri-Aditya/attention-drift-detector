from __future__ import annotations

from ..common.enums import AttentionState
from ..common.types import AttentionEvent


class DriftEventDetector:
    """
    Detect sustained drift events over consecutive frames.
    """

    def __init__(self, drift_seconds_threshold: float, fps: int) -> None:
        self._threshold_frames = max(1, int(drift_seconds_threshold * max(1, fps)))
        self._consecutive_drifting = 0
        self._active_event = False

    def update(self, event: AttentionEvent) -> bool:
        if event.state == AttentionState.DRIFTING:
            self._consecutive_drifting += 1
        else:
            self._consecutive_drifting = 0
            self._active_event = False
            return False

        if self._consecutive_drifting >= self._threshold_frames and not self._active_event:
            self._active_event = True
            return True
        return False

