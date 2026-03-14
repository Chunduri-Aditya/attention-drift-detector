from __future__ import annotations

import cv2
import numpy as np


class HeadPoseEstimator:
    """
    Estimate pitch, yaw, roll from facial landmarks via solvePnP.
    """

    # MediaPipe landmark indices for basic pose anchors.
    _NOSE_TIP = 1
    _CHIN = 152
    _LEFT_EYE_OUTER = 33
    _RIGHT_EYE_OUTER = 263
    _LEFT_MOUTH = 61
    _RIGHT_MOUTH = 291

    def estimate(
        self, landmarks_px: np.ndarray, image_size: tuple[int, int]
    ) -> tuple[float, float, float]:
        w, h = image_size
        image_points = np.array(
            [
                landmarks_px[self._NOSE_TIP],
                landmarks_px[self._CHIN],
                landmarks_px[self._LEFT_EYE_OUTER],
                landmarks_px[self._RIGHT_EYE_OUTER],
                landmarks_px[self._LEFT_MOUTH],
                landmarks_px[self._RIGHT_MOUTH],
            ],
            dtype=np.float64,
        )

        # Generic 3D face model points (mm-ish scale).
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # nose tip
                (0.0, -63.6, -12.5),  # chin
                (-43.3, 32.7, -26.0),  # left eye outer
                (43.3, 32.7, -26.0),  # right eye outer
                (-28.9, -28.9, -24.1),  # left mouth
                (28.9, -28.9, -24.1),  # right mouth
            ],
            dtype=np.float64,
        )

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        projection_matrix = np.hstack((rotation_matrix, np.zeros((3, 1), dtype=np.float64)))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])
        return pitch, yaw, roll

