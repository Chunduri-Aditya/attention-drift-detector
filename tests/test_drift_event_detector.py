"""
Tests for DriftEventDetector — sustained drift threshold logic.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from attention_drift_detector.common.enums import AttentionState
from attention_drift_detector.common.types import AttentionEvent
from attention_drift_detector.state_estimation.drift_event_detector import DriftEventDetector


def _event(state: AttentionState) -> AttentionEvent:
    return AttentionEvent(timestamp=datetime.utcnow(), state=state, confidence=0.8)


class TestDriftEventDetector:
    def test_no_event_before_threshold(self) -> None:
        detector = DriftEventDetector(drift_seconds_threshold=5.0, fps=1)
        # threshold = 5 frames; push 4 — should not fire
        for _ in range(4):
            fired = detector.update(_event(AttentionState.DRIFTING))
        assert not fired

    def test_fires_exactly_at_threshold(self) -> None:
        detector = DriftEventDetector(drift_seconds_threshold=5.0, fps=1)
        fired = False
        for _ in range(5):
            fired = detector.update(_event(AttentionState.DRIFTING))
        assert fired

    def test_fires_only_once_per_sustained_event(self) -> None:
        detector = DriftEventDetector(drift_seconds_threshold=3.0, fps=1)
        fires = []
        for _ in range(10):
            fires.append(detector.update(_event(AttentionState.DRIFTING)))
        assert fires.count(True) == 1

    def test_recovery_resets_counter(self) -> None:
        detector = DriftEventDetector(drift_seconds_threshold=3.0, fps=1)
        for _ in range(2):
            detector.update(_event(AttentionState.DRIFTING))
        detector.update(_event(AttentionState.FOCUSED))  # recovery
        fires = []
        for _ in range(3):
            fires.append(detector.update(_event(AttentionState.DRIFTING)))
        assert fires[-1] is True  # new event fires after recovery

    def test_absent_also_resets_drift_counter(self) -> None:
        detector = DriftEventDetector(drift_seconds_threshold=3.0, fps=1)
        for _ in range(2):
            detector.update(_event(AttentionState.DRIFTING))
        detector.update(_event(AttentionState.ABSENT))
        fires = []
        for _ in range(3):
            fires.append(detector.update(_event(AttentionState.DRIFTING)))
        assert fires[-1] is True

    def test_threshold_zero_frames_clamped_to_one(self) -> None:
        # drift_seconds=0, fps=1 → threshold_frames = max(1, 0) = 1
        detector = DriftEventDetector(drift_seconds_threshold=0.0, fps=1)
        fired = detector.update(_event(AttentionState.DRIFTING))
        assert fired

    def test_fps_scaling(self) -> None:
        # 5 seconds × 30 fps = 150 frames; verify 149 don't fire, 150th does
        detector = DriftEventDetector(drift_seconds_threshold=5.0, fps=30)
        fires = [detector.update(_event(AttentionState.DRIFTING)) for _ in range(150)]
        assert fires[148] is False
        assert fires[149] is True

    def test_no_fire_on_focused_frames(self) -> None:
        detector = DriftEventDetector(drift_seconds_threshold=3.0, fps=1)
        for _ in range(100):
            fired = detector.update(_event(AttentionState.FOCUSED))
        assert not fired
