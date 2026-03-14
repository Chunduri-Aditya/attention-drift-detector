from __future__ import annotations

import numpy as np


class GazeEstimator:
    """
    Lightweight iris-based gaze proxy.
    Produces an approximate unit vector in camera space.
    """

    _LEFT_IRIS_CENTER = 468
    _RIGHT_IRIS_CENTER = 473
    _LEFT_EYE_OUTER = 33
    _RIGHT_EYE_OUTER = 263
    _LEFT_EYE_INNER = 133
    _RIGHT_EYE_INNER = 362

    def estimate(self, landmarks_px: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float]:
        w, h = image_size
        left_iris = landmarks_px[self._LEFT_IRIS_CENTER]
        right_iris = landmarks_px[self._RIGHT_IRIS_CENTER]
        iris_center = (left_iris + right_iris) * 0.5

        eye_center = (
            landmarks_px[self._LEFT_EYE_OUTER]
            + landmarks_px[self._LEFT_EYE_INNER]
            + landmarks_px[self._RIGHT_EYE_OUTER]
            + landmarks_px[self._RIGHT_EYE_INNER]
        ) * 0.25

        dx = (iris_center[0] - eye_center[0]) / max(1.0, w * 0.12)
        dy = (iris_center[1] - eye_center[1]) / max(1.0, h * 0.12)
        gaze = np.array([dx, dy, 1.0], dtype=np.float32)
        norm = float(np.linalg.norm(gaze))
        if norm == 0.0:
            return 0.0, 0.0, 1.0
        gaze /= norm
        return float(gaze[0]), float(gaze[1]), float(gaze[2])

