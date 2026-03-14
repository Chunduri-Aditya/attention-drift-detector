from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class WebcamCapture:
    """
    OpenCV camera wrapper that yields in-memory frames only.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "WebcamCapture":
        self._capture = cv2.VideoCapture(self.camera_index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if not self._capture.isOpened():
            raise RuntimeError("Failed to open webcam. Check camera permission/device index.")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def frames(self) -> Iterator[np.ndarray]:
        if self._capture is None:
            raise RuntimeError("WebcamCapture must be entered before reading frames.")

        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

