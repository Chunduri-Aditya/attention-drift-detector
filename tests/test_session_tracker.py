"""
Tests for SessionTracker — in-memory session state and focus ratio.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from attention_drift_detector.common.enums import AttentionState
from attention_drift_detector.common.types import AttentionEvent
from attention_drift_detector.session.session_tracker import SessionTracker


def _event(state: AttentionState, confidence: float = 0.8) -> AttentionEvent:
    return AttentionEvent(timestamp=datetime.utcnow(), state=state, confidence=confidence)


class TestSessionTracker:
    def test_session_id_is_uuid(self) -> None:
        import uuid
        tracker = SessionTracker(user_id="test_user")
        uuid.UUID(tracker.session_id)  # raises ValueError if invalid

    def test_focus_ratio_empty_session(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        assert tracker.focus_ratio() == 0.0

    def test_focus_ratio_all_focused(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        for _ in range(10):
            tracker.record_event(_event(AttentionState.FOCUSED))
        assert tracker.focus_ratio() == 1.0

    def test_focus_ratio_half_focused(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        for _ in range(5):
            tracker.record_event(_event(AttentionState.FOCUSED))
        for _ in range(5):
            tracker.record_event(_event(AttentionState.DRIFTING))
        assert abs(tracker.focus_ratio() - 0.5) < 1e-9

    def test_focus_ratio_no_focused_frames(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        for _ in range(10):
            tracker.record_event(_event(AttentionState.DRIFTING))
        assert tracker.focus_ratio() == 0.0

    def test_drift_alerts_starts_at_zero(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        assert tracker.drift_alerts == 0

    def test_drift_alerts_increments(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        tracker.record_drift_alert()
        tracker.record_drift_alert()
        assert tracker.drift_alerts == 2

    def test_events_list_grows(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        for i in range(5):
            tracker.record_event(_event(AttentionState.FOCUSED))
        assert len(tracker.events) == 5

    def test_two_trackers_have_different_session_ids(self) -> None:
        t1 = SessionTracker(user_id="u")
        t2 = SessionTracker(user_id="u")
        assert t1.session_id != t2.session_id

    def test_absent_frames_do_not_count_as_focused(self) -> None:
        tracker = SessionTracker(user_id="test_user")
        tracker.record_event(_event(AttentionState.ABSENT))
        tracker.record_event(_event(AttentionState.FOCUSED))
        assert abs(tracker.focus_ratio() - 0.5) < 1e-9
