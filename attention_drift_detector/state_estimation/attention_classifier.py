from __future__ import annotations

from ..common.enums import AttentionState
from ..common.types import AttentionEvent, FrameFeatures
from ..config.settings import AppConfig
from .calibrator import CalibrationResult


class AttentionClassifier:
    """
    Hybrid classifier: uses a trained ML model when available (models_store/attention_model.joblib),
    otherwise falls back to calibrated rule-based logic.

    ML model is loaded once at construction time. Rule-based fallback supports
    optional personal calibration for adaptive per-user thresholds.
    """

    def __init__(self, config: AppConfig, use_ml: bool = True) -> None:
        self._config = config
        self._calibration: CalibrationResult | None = None
        self._ml_model = None
        if use_ml:
            try:
                from ..models.mlp_classifier import MLAttentionClassifier
                self._ml_model = MLAttentionClassifier.load()
                if self._ml_model is not None:
                    print(f"[classifier] ML model loaded from {self._ml_model.model_path.name}")
            except Exception:  # noqa: BLE001
                pass  # silently fall through to rule-based

    def apply_calibration(self, result: CalibrationResult) -> None:
        """Hot-swap in personal thresholds mid-session (called after calibration)."""
        self._calibration = result

    def predict(self, features: FrameFeatures) -> AttentionEvent:
        # ── ML path ───────────────────────────────────────────────────────────
        if self._ml_model is not None:
            return self._ml_model.predict(features)

        # ── Rule-based path ───────────────────────────────────────────────────
        if features.face_visibility < self._config.face_visibility_threshold:
            return AttentionEvent(
                timestamp=features.timestamp,
                state=AttentionState.ABSENT,
                confidence=0.98,
            )

        pitch, yaw, _roll = features.head_angles_deg
        gaze_x, gaze_y, _gaze_z = features.gaze_vector

        if self._calibration is not None:
            # Correct for personal baseline before applying thresholds
            c = self._calibration
            adj_yaw   = yaw   - c.mean_yaw
            adj_pitch = pitch - c.mean_pitch
            adj_gx    = gaze_x - c.mean_gaze_x
            adj_gy    = gaze_y - c.mean_gaze_y

            yaw_thresh   = c.yaw_threshold
            pitch_up     = c.pitch_up_threshold
            pitch_down   = c.pitch_down_threshold
            gaze_x_thresh = c.gaze_x_threshold
            gaze_y_thresh = c.gaze_y_threshold
        else:
            adj_yaw   = yaw
            adj_pitch = pitch
            adj_gx    = gaze_x
            adj_gy    = gaze_y

            yaw_thresh   = self._config.head_pose.yaw_deg
            pitch_up     = self._config.head_pose.pitch_up_deg
            pitch_down   = self._config.head_pose.pitch_down_deg
            gaze_x_thresh = 0.28
            gaze_y_thresh = 0.26

        drift_votes = 0

        if abs(adj_yaw) > yaw_thresh:
            drift_votes += 1
        if adj_pitch > pitch_up or adj_pitch < pitch_down:
            drift_votes += 1
        if abs(adj_gx) > gaze_x_thresh or abs(adj_gy) > gaze_y_thresh:
            drift_votes += 1

        if drift_votes > 0:
            confidence = min(0.95, 0.65 + (0.10 * drift_votes))
            return AttentionEvent(
                timestamp=features.timestamp,
                state=AttentionState.DRIFTING,
                confidence=confidence,
            )

        return AttentionEvent(
            timestamp=features.timestamp,
            state=AttentionState.FOCUSED,
            confidence=0.82,
        )
