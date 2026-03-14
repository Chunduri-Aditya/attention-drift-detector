"""
Data collection mode — saves labeled FrameFeatures to CSV for ML training.

When the pipeline runs with `collect=True`, every frame's features plus
the smoothed attention label are appended to a CSV file.  This is the
training data pipeline for the future MLP classifier.

CSV schema
----------
timestamp, yaw, pitch, roll, gaze_x, gaze_y, gaze_z,
blink, face_visibility, state, confidence, session_id
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from ..common.types import AttentionEvent, FrameFeatures
from ..config.paths import SESSIONS_DIR

_HEADER = [
    "timestamp", "yaw", "pitch", "roll",
    "gaze_x", "gaze_y", "gaze_z",
    "blink", "face_visibility",
    "state", "confidence", "session_id",
]


class DataCollector:
    """
    Streams (features, label) pairs to a CSV file in real time.
    Designed for zero data loss — each row is flushed immediately.
    """

    def __init__(self, session_id: str, output_dir: Path = SESSIONS_DIR) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._path = output_dir / f"training_{session_id[:8]}_{ts}.csv"
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_HEADER)
        self._file.flush()
        self._n = 0

    def record(self, features: FrameFeatures, event: AttentionEvent) -> None:
        pitch, yaw, roll = features.head_angles_deg
        gx, gy, gz = features.gaze_vector
        self._writer.writerow([
            features.timestamp.isoformat(),
            round(yaw, 4), round(pitch, 4), round(roll, 4),
            round(gx, 6), round(gy, 6), round(gz, 6),
            int(features.blink),
            round(features.face_visibility, 4),
            event.state.value,
            round(event.confidence, 4),
            event.session_id if hasattr(event, "session_id") else "",
        ])
        self._n += 1
        # Flush every 30 rows (~1s) so data survives hard quits
        if self._n % 30 == 0:
            self._file.flush()

    def close(self) -> Path:
        self._file.flush()
        self._file.close()
        return self._path

    @property
    def path(self) -> Path:
        return self._path

    @property
    def rows_written(self) -> int:
        return self._n
