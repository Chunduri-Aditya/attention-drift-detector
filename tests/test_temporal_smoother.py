"""
Tests for TemporalSmoother — sliding-window majority vote.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from attention_drift_detector.common.enums import AttentionState
from attention_drift_detector.common.types import AttentionEvent
from attention_drift_detector.state_estimation.temporal_smoother import TemporalSmoother


def _event(state: AttentionState, confidence: float = 0.8) -> AttentionEvent:
    return AttentionEvent(timestamp=datetime.utcnow(), state=state, confidence=confidence)


class TestTemporalSmoother:
    def test_single_event_passes_through(self) -> None:
        smoother = TemporalSmoother(window_size=5)
        result = smoother.push(_event(AttentionState.FOCUSED))
        assert result.state == AttentionState.FOCUSED

    def test_majority_wins(self) -> None:
        smoother = TemporalSmoother(window_size=5)
        # Push 3 drifting, 2 focused → majority should be drifting
        for _ in range(3):
            smoother.push(_event(AttentionState.DRIFTING))
        for _ in range(2):
            result = smoother.push(_event(AttentionState.FOCUSED))
        assert result.state == AttentionState.DRIFTING

    def test_window_evicts_old_events(self) -> None:
        smoother = TemporalSmoother(window_size=3)
        # Fill with drifting
        for _ in range(3):
            smoother.push(_event(AttentionState.DRIFTING))
        # Now push 3 focused — old drifting should be evicted
        for _ in range(3):
            result = smoother.push(_event(AttentionState.FOCUSED))
        assert result.state == AttentionState.FOCUSED

    def test_mean_confidence_is_averaged(self) -> None:
        smoother = TemporalSmoother(window_size=2)
        smoother.push(_event(AttentionState.FOCUSED, confidence=0.6))
        result = smoother.push(_event(AttentionState.FOCUSED, confidence=1.0))
        assert abs(result.confidence - 0.8) < 1e-6

    def test_window_size_one_acts_as_passthrough(self) -> None:
        smoother = TemporalSmoother(window_size=1)
        smoother.push(_event(AttentionState.DRIFTING))
        result = smoother.push(_event(AttentionState.FOCUSED))
        assert result.state == AttentionState.FOCUSED

    def test_zero_window_size_clamped_to_one(self) -> None:
        # window_size must be at least 1; should not crash
        smoother = TemporalSmoother(window_size=0)
        result = smoother.push(_event(AttentionState.ABSENT))
        assert result.state == AttentionState.ABSENT

    def test_absent_beats_drifting_when_majority(self) -> None:
        smoother = TemporalSmoother(window_size=5)
        for _ in range(3):
            smoother.push(_event(AttentionState.ABSENT))
        for _ in range(2):
            result = smoother.push(_event(AttentionState.DRIFTING))
        assert result.state == AttentionState.ABSENT

    def test_timestamp_comes_from_latest_push(self) -> None:
        from datetime import timedelta

        smoother = TemporalSmoother(window_size=3)
        t1 = datetime(2024, 1, 1, 0, 0, 0)
        t2 = t1 + timedelta(seconds=1)
        smoother.push(AttentionEvent(timestamp=t1, state=AttentionState.FOCUSED, confidence=0.8))
        result = smoother.push(AttentionEvent(timestamp=t2, state=AttentionState.FOCUSED, confidence=0.8))
        assert result.timestamp == t2
