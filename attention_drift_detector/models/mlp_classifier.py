"""
Thin wrapper around the trained sklearn pipeline saved by trainer.py.

The AttentionClassifier checks for a trained model at startup and
automatically uses it when available, falling back to rule-based logic
if the model file is missing or fails to load.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.types import AttentionEvent, FrameFeatures

from ..config.paths import MODELS_DIR

_DEFAULT_MODEL_PATH = MODELS_DIR / "attention_model.joblib"

FEATURE_ORDER = [
    "yaw", "pitch", "roll",
    "gaze_x", "gaze_y", "gaze_z",
    "blink", "face_visibility",
]
LABEL_NAMES = ["focused", "drifting", "absent"]


class MLAttentionClassifier:
    """
    Wraps a joblib-serialized sklearn Pipeline (scaler + classifier).

    Usage:
        clf = MLAttentionClassifier.load()  # returns None if no model
        if clf:
            event = clf.predict(features)
    """

    def __init__(self, pipeline: object, model_path: Path) -> None:
        self._pipeline = pipeline
        self.model_path = model_path

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_path: Path = _DEFAULT_MODEL_PATH) -> "MLAttentionClassifier | None":
        """Return a loaded classifier, or None if no model exists."""
        if not model_path.exists():
            return None
        try:
            import joblib
            pipeline = joblib.load(model_path)
            return cls(pipeline, model_path)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"Failed to load ML model from {model_path}: {exc}. "
                "Falling back to rule-based classifier.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    @classmethod
    def is_available(cls, model_path: Path = _DEFAULT_MODEL_PATH) -> bool:
        return model_path.exists()

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, features: "FrameFeatures") -> "AttentionEvent":
        """Return an AttentionEvent using the trained ML model."""
        import numpy as np

        from ..common.enums import AttentionState
        from ..common.types import AttentionEvent

        pitch, yaw, roll = features.head_angles_deg
        gaze_x, gaze_y, gaze_z = features.gaze_vector

        # Order must match FEATURE_ORDER / the trained model's feature_cols.
        vec = np.array([[
            yaw,
            pitch,
            roll,
            gaze_x,
            gaze_y,
            gaze_z,
            float(features.blink),
            features.face_visibility,
        ]], dtype=np.float32)

        label_idx = int(self._pipeline.predict(vec)[0])
        proba = self._pipeline.predict_proba(vec)[0]
        confidence = float(proba[label_idx])

        state_map = {
            0: AttentionState.FOCUSED,
            1: AttentionState.DRIFTING,
            2: AttentionState.ABSENT,
        }

        return AttentionEvent(
            timestamp=features.timestamp,
            state=state_map.get(label_idx, AttentionState.DRIFTING),
            confidence=confidence,
            source="ml",
        )
