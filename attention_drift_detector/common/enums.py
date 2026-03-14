from __future__ import annotations

from enum import Enum


class AttentionState(str, Enum):
    FOCUSED = "focused"
    DRIFTING = "drifting"
    ABSENT = "absent"


