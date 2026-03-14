from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from ..common.enums import AttentionState
from ..common.types import AttentionEvent


@dataclass(slots=True)
class SessionTracker:
    user_id: str
    session_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=datetime.utcnow)
    events: list[AttentionEvent] = field(default_factory=list)
    drift_alerts: int = 0

    def record_event(self, event: AttentionEvent) -> None:
        self.events.append(event)

    def record_drift_alert(self) -> None:
        self.drift_alerts += 1

    def focus_ratio(self) -> float:
        if not self.events:
            return 0.0
        focused = sum(1 for e in self.events if e.state == AttentionState.FOCUSED)
        return focused / float(len(self.events))

