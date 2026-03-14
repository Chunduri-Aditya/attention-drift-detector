from __future__ import annotations

from datetime import datetime

from ..common.types import DriftAlert
from .notifier import send_break_suggestion, send_nudge
from .nudge_throttler import NudgeThrottler


class NudgeSystem:
    def __init__(self, throttler: NudgeThrottler) -> None:
        self._throttler = throttler

    def on_recovery(self) -> None:
        self._throttler.mark_recovery()

    def maybe_nudge(self, session_id: str) -> tuple[DriftAlert | None, bool]:
        allowed, cooldown_remaining = self._throttler.can_nudge()
        if not allowed:
            return None, False

        should_suggest_break = self._throttler.mark_nudged()

        if should_suggest_break:
            send_break_suggestion()
        else:
            send_nudge()

        alert = DriftAlert(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            reason="sustained_drift",
            cooldown_remaining_s=cooldown_remaining,
        )
        return alert, should_suggest_break

