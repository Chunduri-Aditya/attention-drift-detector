from __future__ import annotations

import numpy as np


class BlinkDetector:
    """
    Blink detector based on Eye Aspect Ratio (EAR).
    """

    # (p1, p2, p3, p4, p5, p6): horizontal at p1-p4.
    _LEFT_EYE = (33, 160, 158, 133, 153, 144)
    _RIGHT_EYE = (362, 385, 387, 263, 373, 380)

    @staticmethod
    def _ear(landmarks: np.ndarray, idxs: tuple[int, int, int, int, int, int]) -> float:
        p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in idxs)
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal == 0:
            return 0.0
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def detect(self, landmarks_px: np.ndarray, threshold: float) -> tuple[bool, float]:
        left_ear = self._ear(landmarks_px, self._LEFT_EYE)
        right_ear = self._ear(landmarks_px, self._RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        return ear < threshold, ear

