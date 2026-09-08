"""
Tests for AttentionClassifier rule-based logic.
Covers all three output states: FOCUSED, DRIFTING (each drift axis), ABSENT.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from attention_drift_detector.common.enums import AttentionState
from attention_drift_detector.common.types import FrameFeatures
from attention_drift_detector.config.settings import AppConfig
from attention_drift_detector.state_estimation.attention_classifier import AttentionClassifier


def _make_features(
    *,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    gaze_x: float = 0.0,
    gaze_y: float = 0.0,
    face_visibility: float = 1.0,
) -> FrameFeatures:
    return FrameFeatures(
        timestamp=datetime.utcnow(),
        head_angles_deg=(pitch, yaw, roll),
        gaze_vector=(gaze_x, gaze_y, 1.0),
        blink=False,
        face_visibility=face_visibility,
    )


@pytest.fixture()
def clf() -> AttentionClassifier:
    return AttentionClassifier(config=AppConfig())


class TestDefaultIsRuleBased:
    """The rule-based path is the shipped, supported classifier. If the default
    flips back to ML, every test in this file silently starts exercising the
    trained model instead — so pin the invariant explicitly.
    """

    def test_default_construction_loads_no_ml_model(self) -> None:
        assert AttentionClassifier(config=AppConfig())._ml_model is None

    def test_default_predictions_are_tagged_rule(self, clf: AttentionClassifier) -> None:
        event = clf.predict(_make_features(pitch=0.0, yaw=0.0))
        assert event.source == "rule"


class TestAbsent:
    def test_low_visibility_returns_absent(self, clf: AttentionClassifier) -> None:
        features = _make_features(face_visibility=0.2)
        event = clf.predict(features)
        assert event.state == AttentionState.ABSENT

    def test_exactly_at_threshold_is_not_absent(self, clf: AttentionClassifier) -> None:
        # face_visibility_threshold is 0.7 — at exactly 0.7 it passes the absent check
        features = _make_features(face_visibility=0.7)
        event = clf.predict(features)
        assert event.state != AttentionState.ABSENT

    def test_absent_confidence_is_high(self, clf: AttentionClassifier) -> None:
        features = _make_features(face_visibility=0.0)
        event = clf.predict(features)
        assert event.confidence >= 0.95


class TestFocused:
    def test_centered_gaze_and_pose_is_focused(self, clf: AttentionClassifier) -> None:
        features = _make_features(pitch=0.0, yaw=0.0, gaze_x=0.0, gaze_y=0.0)
        event = clf.predict(features)
        assert event.state == AttentionState.FOCUSED

    def test_focused_confidence_is_reasonable(self, clf: AttentionClassifier) -> None:
        features = _make_features()
        event = clf.predict(features)
        assert 0.7 <= event.confidence <= 1.0

    def test_small_yaw_still_focused(self, clf: AttentionClassifier) -> None:
        # yaw threshold is 25 deg — 10 deg should be fine
        features = _make_features(yaw=10.0)
        event = clf.predict(features)
        assert event.state == AttentionState.FOCUSED


class TestDrifting:
    def test_high_yaw_is_drifting(self, clf: AttentionClassifier) -> None:
        features = _make_features(yaw=30.0)
        event = clf.predict(features)
        assert event.state == AttentionState.DRIFTING

    def test_negative_high_yaw_is_drifting(self, clf: AttentionClassifier) -> None:
        features = _make_features(yaw=-30.0)
        event = clf.predict(features)
        assert event.state == AttentionState.DRIFTING

    def test_high_pitch_up_is_drifting(self, clf: AttentionClassifier) -> None:
        # pitch_up threshold is 20 deg
        features = _make_features(pitch=25.0)
        event = clf.predict(features)
        assert event.state == AttentionState.DRIFTING

    def test_high_pitch_down_is_drifting(self, clf: AttentionClassifier) -> None:
        # pitch_down threshold is -30 deg
        features = _make_features(pitch=-35.0)
        event = clf.predict(features)
        assert event.state == AttentionState.DRIFTING

    def test_large_gaze_x_is_drifting(self, clf: AttentionClassifier) -> None:
        features = _make_features(gaze_x=0.35)
        event = clf.predict(features)
        assert event.state == AttentionState.DRIFTING

    def test_large_gaze_y_is_drifting(self, clf: AttentionClassifier) -> None:
        features = _make_features(gaze_y=0.30)
        event = clf.predict(features)
        assert event.state == AttentionState.DRIFTING

    def test_confidence_scales_with_drift_votes(self, clf: AttentionClassifier) -> None:
        """More axes in drift → higher confidence."""
        one_vote = clf.predict(_make_features(yaw=30.0))
        three_votes = clf.predict(_make_features(yaw=30.0, pitch=25.0, gaze_x=0.35))
        assert three_votes.confidence > one_vote.confidence

    def test_max_drift_confidence_capped(self, clf: AttentionClassifier) -> None:
        event = clf.predict(_make_features(yaw=30.0, pitch=25.0, gaze_x=0.35))
        assert event.confidence <= 0.95

    def test_timestamp_preserved(self, clf: AttentionClassifier) -> None:
        ts = datetime(2024, 6, 1, 12, 0, 0)
        features = FrameFeatures(
            timestamp=ts,
            head_angles_deg=(0.0, 0.0, 0.0),
            gaze_vector=(0.0, 0.0, 1.0),
            blink=False,
            face_visibility=1.0,
        )
        event = clf.predict(features)
        assert event.timestamp == ts
