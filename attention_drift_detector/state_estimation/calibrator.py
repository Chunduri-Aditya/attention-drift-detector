"""
Per-user calibration system.

During the first N seconds of a session the user sits normally and looks
at their screen.  The calibrator records every FrameFeatures sample, then
computes personal baselines (mean + k·std) for each tracking axis.

The resulting CalibrationResult is passed to AttentionClassifier so it
uses YOUR natural head position as the reference instead of hardcoded
population-average thresholds.

Why this matters
----------------
Every person sits at a different distance and angle from their camera.
A universal yaw threshold of ±25° may be fine for some setups but too
tight or too loose for others.  Calibration shifts the zero-point and
scales the tolerance to the individual.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ..common.types import FrameFeatures


@dataclass
class CalibrationResult:
    """
    Personal baselines derived from a calibration window.
    All angles in degrees; gaze values are normalised units.
    """
    # Baseline means (your natural resting position)
    mean_yaw: float = 0.0
    mean_pitch: float = 0.0
    mean_gaze_x: float = 0.0
    mean_gaze_y: float = 0.0

    # Adaptive thresholds (mean ± k·std, clamped to sensible ranges)
    yaw_threshold: float = 25.0
    pitch_up_threshold: float = 20.0
    pitch_down_threshold: float = -30.0
    gaze_x_threshold: float = 0.28
    gaze_y_threshold: float = 0.26

    # Metadata
    n_samples: int = 0
    completed: bool = False
    completed_at: datetime | None = None

    def summary(self) -> str:
        return (
            f"Calibration complete ({self.n_samples} samples)  "
            f"baseline yaw={self.mean_yaw:+.1f}°  pitch={self.mean_pitch:+.1f}°  "
            f"gaze=({self.mean_gaze_x:+.3f}, {self.mean_gaze_y:+.3f})  "
            f"thresholds yaw=±{self.yaw_threshold:.1f}°  "
            f"pitch=[{self.pitch_down_threshold:.1f}°, +{self.pitch_up_threshold:.1f}°]"
        )


class Calibrator:
    """
    Collects FrameFeatures for `window_seconds` then computes personal
    thresholds.  Call `update()` each frame; `is_done` goes True when
    the window closes.
    """

    # Multiplier on std dev to set adaptive thresholds (wider = more tolerant)
    _K_STD: float = 2.5
    # Hard ceilings / floors so calibration can't produce absurd thresholds
    _MAX_YAW: float = 40.0
    _MIN_YAW: float = 10.0
    _MAX_PITCH_UP: float = 35.0
    _MIN_PITCH_UP: float = 10.0
    _MAX_PITCH_DOWN: float = -10.0
    _MIN_PITCH_DOWN: float = -45.0
    _MAX_GAZE: float = 0.40
    _MIN_GAZE: float = 0.12

    def __init__(self, window_seconds: float = 30.0, fps: int = 30) -> None:
        self._required = max(10, int(window_seconds * max(1, fps)))
        self._samples: list[FrameFeatures] = []
        self._result: CalibrationResult = CalibrationResult()

    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self._result.completed

    @property
    def result(self) -> CalibrationResult:
        return self._result

    @property
    def progress(self) -> float:
        """0.0 → 1.0 progress through calibration window."""
        if self._result.completed:
            return 1.0
        return min(1.0, len(self._samples) / self._required)

    @property
    def seconds_remaining(self) -> float:
        if self._result.completed:
            return 0.0
        frames_left = max(0, self._required - len(self._samples))
        return frames_left / max(1, self._required) * (self._required / 30.0)

    def update(self, features: FrameFeatures) -> bool:
        """
        Feed one frame.  Returns True the moment calibration completes.
        Only counts frames where the face is visible.
        """
        if self._result.completed:
            return False

        if features.face_visibility >= 0.7:
            self._samples.append(features)

        if len(self._samples) >= self._required:
            self._compute()
            return True
        return False

    # ------------------------------------------------------------------

    def _compute(self) -> None:
        samples = self._samples

        yaws   = [f.head_angles_deg[1] for f in samples]
        pitches = [f.head_angles_deg[0] for f in samples]
        gxs    = [f.gaze_vector[0] for f in samples]
        gys    = [f.gaze_vector[1] for f in samples]

        mean_yaw   = statistics.mean(yaws)
        mean_pitch = statistics.mean(pitches)
        mean_gx    = statistics.mean(gxs)
        mean_gy    = statistics.mean(gys)

        std_yaw   = statistics.stdev(yaws)   if len(yaws) > 1 else 5.0
        std_pitch = statistics.stdev(pitches) if len(pitches) > 1 else 5.0
        std_gx    = statistics.stdev(gxs)    if len(gxs) > 1 else 0.05
        std_gy    = statistics.stdev(gys)    if len(gys) > 1 else 0.05

        yaw_thresh = _clamp(self._K_STD * std_yaw, self._MIN_YAW, self._MAX_YAW)
        pitch_up   = _clamp(self._K_STD * std_pitch, self._MIN_PITCH_UP, self._MAX_PITCH_UP)
        pitch_down = _clamp(-self._K_STD * std_pitch, self._MIN_PITCH_DOWN, self._MAX_PITCH_DOWN)
        gaze_x     = _clamp(self._K_STD * std_gx + abs(mean_gx), self._MIN_GAZE, self._MAX_GAZE)
        gaze_y     = _clamp(self._K_STD * std_gy + abs(mean_gy), self._MIN_GAZE, self._MAX_GAZE)

        self._result = CalibrationResult(
            mean_yaw=mean_yaw,
            mean_pitch=mean_pitch,
            mean_gaze_x=mean_gx,
            mean_gaze_y=mean_gy,
            yaw_threshold=yaw_thresh,
            pitch_up_threshold=pitch_up,
            pitch_down_threshold=pitch_down,
            gaze_x_threshold=gaze_x,
            gaze_y_threshold=gaze_y,
            n_samples=len(samples),
            completed=True,
            completed_at=datetime.utcnow(),
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
