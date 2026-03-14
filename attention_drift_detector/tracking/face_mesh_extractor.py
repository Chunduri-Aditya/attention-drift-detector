from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(slots=True)
class FaceMeshResult:
    landmarks_px: np.ndarray | None
    face_visibility: float
    image_size: tuple[int, int]


class FaceMeshExtractor:
    """
    MediaPipe Face Mesh wrapper.
    Returns 2D landmark coordinates in pixels for the first face.
    """

    def __init__(self) -> None:
        face_mesh_cls = self._resolve_face_mesh_class()
        self._face_mesh = face_mesh_cls(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    @staticmethod
    def _resolve_face_mesh_class() -> type:
        """
        Support all MediaPipe package layouts across versions.

        MediaPipe ≥ 0.10.14 removed the legacy Solutions API entirely.
        We try every known path and give a clear install hint on failure.
        """
        # Path 1: `from ... import` style — goes through __getattr__ lazy loader
        # differently than attribute access or direct import; best shot on 0.10.x
        try:
            from mediapipe.solutions import face_mesh as _fm
            return _fm.FaceMesh
        except (ImportError, AttributeError):
            pass

        # Path 2: attribute chain — triggers __getattr__ on solutions namespace
        try:
            return mp.solutions.face_mesh.FaceMesh
        except AttributeError:
            pass

        # Path 3: direct submodule import
        try:
            import mediapipe.solutions.face_mesh as _fm2
            return _fm2.FaceMesh
        except ImportError:
            pass

        # Path 4: legacy mediapipe.python.solutions (pre-0.10)
        try:
            from mediapipe.python.solutions import face_mesh as _fm3
            return _fm3.FaceMesh
        except ImportError:
            pass

        version = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            f"\n\nMediaPipe FaceMesh not found (installed version: {version}).\n"
            "The legacy Solutions API was removed in MediaPipe ≥ 0.10.14.\n"
            "Fix: downgrade to a compatible version:\n\n"
            "    pip install 'mediapipe==0.10.13'\n"
        )

    def extract(self, frame_bgr: np.ndarray) -> FaceMeshResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(frame_rgb)
        h, w = frame_bgr.shape[:2]

        if not result.multi_face_landmarks:
            return FaceMeshResult(
                landmarks_px=None,
                face_visibility=0.0,
                image_size=(w, h),
            )

        face_landmarks = result.multi_face_landmarks[0].landmark
        points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks], dtype=np.float32)
        return FaceMeshResult(landmarks_px=points, face_visibility=1.0, image_size=(w, h))

    def close(self) -> None:
        self._face_mesh.close()

