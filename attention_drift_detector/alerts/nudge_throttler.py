from __future__ import annotations

import time


class NudgeThrottler:
    def __init__(self, cooldown_s: float, max_consecutive_nudges: int) -> None:
        self._cooldown_s = cooldown_s
        self._max_consecutive_nudges = max_consecutive_nudges
        self._last_nudge_ts = 0.0
        self._nudges_since_recovery = 0

    def mark_recovery(self) -> None:
        self._nudges_since_recovery = 0

    def can_nudge(self) -> tuple[bool, float]:
        now = time.monotonic()
        elapsed = now - self._last_nudge_ts
        if elapsed < self._cooldown_s:
            return False, self._cooldown_s - elapsed
        return True, 0.0

    def mark_nudged(self) -> bool:
        """
        Returns True when break suggestion should be shown.
        """
        self._last_nudge_ts = time.monotonic()
        self._nudges_since_recovery += 1
        return self._nudges_since_recovery >= self._max_consecutive_nudges

