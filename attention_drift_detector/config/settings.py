from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeadPoseThresholds:
    yaw_deg: float = 25.0
    pitch_up_deg: float = 20.0
    pitch_down_deg: float = -30.0


@dataclass(frozen=True, slots=True)
class DriftDetectionConfig:
    temporal_window_s: float = 1.0  # 30 frames @ 30fps
    drift_seconds_threshold: float = 5.0
    nudge_cooldown_s: float = 180.0
    max_consecutive_nudges: int = 3


@dataclass(frozen=True, slots=True)
class BlinkConfig:
    ear_threshold: float = 0.25


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    persist_raw_frames: bool = False
    local_storage_only: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    target_fps: int = 30
    camera_index: int = 0
    face_visibility_threshold: float = 0.7
    show_preview: bool = True
    head_pose: HeadPoseThresholds = HeadPoseThresholds()
    drift: DriftDetectionConfig = DriftDetectionConfig()
    blink: BlinkConfig = BlinkConfig()
    privacy: PrivacyConfig = PrivacyConfig()


settings = AppConfig()

