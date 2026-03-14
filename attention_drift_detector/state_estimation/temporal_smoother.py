from __future__ import annotations

from collections import deque

from ..common.enums import AttentionState
from ..common.types import AttentionEvent


class TemporalSmoother:
    """
    Sliding-window majority smoother over frame-level attention states.
    """

    def __init__(self, window_size: int) -> None:
        self._window_size = max(1, window_size)
        self._states: deque[AttentionEvent] = deque(maxlen=self._window_size)

    def push(self, event: AttentionEvent) -> AttentionEvent:
        self._states.append(event)
        counts = {
            AttentionState.FOCUSED: 0,
            AttentionState.DRIFTING: 0,
            AttentionState.ABSENT: 0,
        }
        total_confidence = 0.0
        for item in self._states:
            counts[item.state] += 1
            total_confidence += item.confidence

        smoothed_state = max(counts, key=counts.get)
        mean_confidence = total_confidence / float(len(self._states))
        return AttentionEvent(
            timestamp=event.timestamp,
            state=smoothed_state,
            confidence=mean_confidence,
        )

