from __future__ import annotations

import time
from datetime import datetime
from typing import Final

import numpy as np

from ..alerts.nudge_system import NudgeSystem
from ..alerts.nudge_throttler import NudgeThrottler
from ..common.enums import AttentionState
from ..common.types import AttentionEvent, FrameFeatures
from ..config.settings import settings
from ..session.data_collector import DataCollector
from ..session.session_db import SessionDB
from ..session.session_report import SessionReportGenerator
from ..session.session_tracker import SessionTracker
from ..state_estimation.attention_classifier import AttentionClassifier
from ..state_estimation.calibrator import Calibrator
from ..state_estimation.drift_event_detector import DriftEventDetector
from ..state_estimation.temporal_smoother import TemporalSmoother

STATUS_COLOR: Final[dict[AttentionState, tuple[int, int, int]]] = {
    AttentionState.FOCUSED: (60, 180, 75),
    AttentionState.DRIFTING: (50, 120, 230),
    AttentionState.ABSENT: (40, 40, 200),
}
CALIB_COLOR: Final[tuple[int, int, int]] = (255, 200, 0)  # amber during calibration


def _draw_overlay(
    frame: np.ndarray,
    event: AttentionEvent,
    tracker: SessionTracker,
    fps: float,
    calibrator: Calibrator | None,
    elapsed_s: float,
    collecting: bool,
) -> None:
    import cv2

    h, w = frame.shape[:2]

    # ── Calibration phase banner ────────────────────────────────────
    if calibrator is not None and not calibrator.is_done:
        pct = int(calibrator.progress * 100)
        secs = int(calibrator.seconds_remaining) + 1
        bar_w = int(w * calibrator.progress)
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), CALIB_COLOR, -1)
        cv2.putText(
            frame,
            f"CALIBRATING — sit naturally and look at screen  {pct}%  ({secs}s)",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, CALIB_COLOR, 2, cv2.LINE_AA,
        )
        return  # skip normal overlay during calibration

    # ── Normal overlay ───────────────────────────────────────────────
    color = STATUS_COLOR[event.state]
    label = event.state.value.upper()

    # State pill background
    cv2.rectangle(frame, (10, 8), (320, 44), (20, 20, 20), -1)
    cv2.putText(
        frame, f"{label}  {event.confidence:.2f}",
        (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA,
    )

    # Stats row
    m, s = divmod(int(elapsed_s), 60)
    time_str = f"{m:02d}:{s:02d}"
    stats = (
        f"focus {tracker.focus_ratio() * 100:.0f}%   "
        f"alerts {tracker.drift_alerts}   "
        f"fps {fps:.0f}   {time_str}"
    )
    cv2.putText(
        frame, stats,
        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Calibration indicator (top-right)
    calib_label = "CAL ✓" if (calibrator is not None and calibrator.is_done) else "NO CAL"
    calib_col = (60, 200, 60) if (calibrator is not None and calibrator.is_done) else (120, 120, 120)
    cv2.putText(
        frame, calib_label,
        (w - 110, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, calib_col, 1, cv2.LINE_AA,
    )

    # Data collection indicator
    if collecting:
        cv2.circle(frame, (w - 20, 20), 7, (0, 0, 220), -1)  # red dot


def _start_dashboard_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Boot uvicorn in a daemon thread so the pipeline loop stays in main thread."""
    import threading
    try:
        import uvicorn
    except ImportError:
        print("[dashboard] uvicorn not installed — run: pip install uvicorn")
        return

    config = uvicorn.Config(
        "attention_drift_detector.api.app:app",
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    # Small delay so the server is ready before the first frame
    import time as _time; _time.sleep(1.2)
    print(f"[dashboard] http://{host}:{port}/   (Ctrl+C to stop pipeline)")


def run_realtime_pipeline(
    user_id: str = "default_user",
    max_seconds: int | None = None,
    show_preview: bool | None = None,
    use_dummy: bool = False,
    save_session: bool = True,
    generate_report: bool = True,
    calibrate: bool = True,
    calibration_seconds: float = 30.0,
    collect: bool = False,
    use_ml: bool = False,
    serve_dashboard: bool = False,
    dashboard_port: int = 8765,
) -> None:
    """
    Real-time webcam attention loop.
    No raw frames are written to disk; all frame data is transient in memory.

    Parameters
    ----------
    calibrate:           Run a personal calibration window at session start.
    calibration_seconds: Length of the calibration window (default 30s).
    collect:             Save labeled FrameFeatures to CSV for ML training.
    use_ml:              Use the trained model when one is present; False forces rule-based.
    save_session:        Persist session metadata and events to SQLite.
    generate_report:     Write an HTML report after the session ends.
    serve_dashboard:     Boot the web dashboard in a background thread.
    dashboard_port:      Port for the dashboard server (default 8765).
    """
    if serve_dashboard:
        _start_dashboard_server(port=dashboard_port)

    window_frames = int(settings.drift.temporal_window_s * settings.target_fps)
    smoother = TemporalSmoother(window_size=max(1, window_frames))
    classifier = AttentionClassifier(config=settings, use_ml=use_ml)
    drift_detector = DriftEventDetector(
        drift_seconds_threshold=settings.drift.drift_seconds_threshold,
        fps=settings.target_fps,
    )
    nudge_system = NudgeSystem(
        NudgeThrottler(
            cooldown_s=settings.drift.nudge_cooldown_s,
            max_consecutive_nudges=settings.drift.max_consecutive_nudges,
        )
    )
    tracker = SessionTracker(user_id=user_id)
    calibrator: Calibrator | None = Calibrator(
        window_seconds=calibration_seconds,
        fps=settings.target_fps,
    ) if calibrate and not use_dummy else None

    # Storage
    db: SessionDB | None = None
    if save_session:
        db = SessionDB()
        db.create_session(
            session_id=tracker.session_id,
            user_id=user_id,
            started_at=tracker.started_at,
        )

    data_collector: DataCollector | None = None
    if collect and not use_dummy:
        data_collector = DataCollector(session_id=tracker.session_id)
        print(f"[collect] saving training data → {data_collector.path}")

    face_mesh = None
    head_pose = None
    gaze = None
    blink = None

    if not use_dummy:
        from ..tracking.blink_detector import BlinkDetector
        from ..tracking.face_mesh_extractor import FaceMeshExtractor
        from ..tracking.gaze_estimator import GazeEstimator
        from ..tracking.head_pose_estimator import HeadPoseEstimator

        face_mesh = FaceMeshExtractor()
        head_pose = HeadPoseEstimator()
        gaze = GazeEstimator()
        blink = BlinkDetector()

    started_monotonic = time.monotonic()
    last_print = 0.0
    preview_enabled = settings.show_preview if show_preview is None else show_preview

    if calibrate and not use_dummy:
        print(f"[calibrate] Sit naturally, look at your screen — {calibration_seconds:.0f}s calibration window starting...")

    # Live state push helper (no-op when dashboard not running)
    def _push_live(event: AttentionEvent, elapsed: float, calibrated: bool) -> None:
        if not serve_dashboard:
            return
        try:
            from ..api.stream import update_live_state
            update_live_state(
                session_id=tracker.session_id,
                state=event.state.value,
                confidence=event.confidence,
                focus_ratio=tracker.focus_ratio() * 100,
                drift_alerts=tracker.drift_alerts,
                elapsed_seconds=elapsed,
                calibrated=calibrated,
            )
        except Exception:  # noqa: BLE001
            pass

    # ── Dummy mode ─────────────────────────────────────────────────────
    if use_dummy:
        while True:
            now = datetime.utcnow()
            features = FrameFeatures(
                timestamp=now,
                head_angles_deg=(0.0, 0.0, 0.0),
                gaze_vector=(0.0, 0.0, 1.0),
                blink=False,
                face_visibility=1.0,
            )
            raw_event = classifier.predict(features)
            event = smoother.push(raw_event)
            tracker.record_event(event)
            elapsed = time.monotonic() - started_monotonic
            _push_live(event, elapsed, False)
            time.sleep(1.0 / max(1, settings.target_fps))

            if time.monotonic() - last_print >= 1.0:
                print(
                    f"[{event.timestamp.isoformat()}] "
                    f"state={event.state.value} conf={event.confidence:.2f} "
                    f"focus_ratio={tracker.focus_ratio() * 100:.1f}%"
                )
                last_print = time.monotonic()
            if max_seconds is not None and elapsed >= max_seconds:
                break

    # ── Live webcam mode ────────────────────────────────────────────────
    else:
        from ..capture.webcam_capture import WebcamCapture
        import cv2

        with WebcamCapture(camera_index=settings.camera_index) as capture:
            for frame in capture.frames():
                frame_start = time.perf_counter()
                now = datetime.utcnow()
                elapsed = time.monotonic() - started_monotonic

                assert face_mesh is not None
                mesh = face_mesh.extract(frame)
                if mesh.landmarks_px is None:
                    features = FrameFeatures(
                        timestamp=now,
                        head_angles_deg=(0.0, 0.0, 0.0),
                        gaze_vector=(0.0, 0.0, 1.0),
                        blink=False,
                        face_visibility=mesh.face_visibility,
                    )
                else:
                    assert head_pose is not None and gaze is not None and blink is not None
                    pitch, yaw, roll = head_pose.estimate(mesh.landmarks_px, mesh.image_size)
                    gaze_x, gaze_y, gaze_z = gaze.estimate(mesh.landmarks_px, mesh.image_size)
                    blink_flag, _ = blink.detect(mesh.landmarks_px, settings.blink.ear_threshold)
                    features = FrameFeatures(
                        timestamp=now,
                        head_angles_deg=(pitch, yaw, roll),
                        gaze_vector=(gaze_x, gaze_y, gaze_z),
                        blink=blink_flag,
                        face_visibility=mesh.face_visibility,
                    )

                # ── Calibration phase ──────────────────────────────────
                if calibrator is not None and not calibrator.is_done:
                    just_done = calibrator.update(features)
                    if just_done:
                        classifier.apply_calibration(calibrator.result)
                        print(f"\n[calibrate] {calibrator.result.summary()}\n")
                    # During calibration, still classify but don't track/nudge
                    raw_event = classifier.predict(features)
                    event = smoother.push(raw_event)
                    if preview_enabled:
                        _draw_overlay(frame, event, tracker, 0.0, calibrator, elapsed, collect)
                        cv2.imshow("Attention Drift Detector — Bunny", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                # ── Normal tracking phase ─────────────────────────────
                raw_event = classifier.predict(features)
                event = smoother.push(raw_event)
                tracker.record_event(event)

                if data_collector is not None:
                    data_collector.record(features, event)

                _push_live(event, elapsed, calibrator is not None and calibrator.is_done)

                if event.state == AttentionState.FOCUSED:
                    nudge_system.on_recovery()

                if drift_detector.update(event):
                    alert, _ = nudge_system.maybe_nudge(session_id=tracker.session_id)
                    if alert is not None:
                        tracker.record_drift_alert()

                frame_elapsed = max(1e-6, time.perf_counter() - frame_start)
                fps = 1.0 / frame_elapsed

                if time.monotonic() - last_print >= 1.0:
                    print(
                        f"[{event.timestamp.isoformat()}] "
                        f"state={event.state.value} conf={event.confidence:.2f} "
                        f"focus_ratio={tracker.focus_ratio() * 100:.1f}%"
                    )
                    last_print = time.monotonic()

                if preview_enabled:
                    _draw_overlay(frame, event, tracker, fps, calibrator, elapsed, collect)
                    cv2.imshow("Attention Drift Detector — Bunny", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_seconds is not None and elapsed >= max_seconds:
                    break

    # ── Cleanup ─────────────────────────────────────────────────────────
    if serve_dashboard:
        try:
            from ..api.stream import clear_live_state
            clear_live_state()
        except Exception:  # noqa: BLE001
            pass

    if face_mesh is not None:
        face_mesh.close()
    if preview_enabled:
        import cv2
        cv2.destroyAllWindows()

    if data_collector is not None:
        path = data_collector.close()
        print(f"[collect] {data_collector.rows_written} rows saved → {path}")

    ended_at = datetime.utcnow()
    focus_ratio = tracker.focus_ratio()

    print(
        f"\nsession_end  id={tracker.session_id[:8]}  "
        f"focus={focus_ratio * 100:.1f}%  "
        f"alerts={tracker.drift_alerts}  "
        f"calibrated={'yes' if calibrator and calibrator.is_done else 'no'}"
    )

    if db is not None:
        db.end_session(
            session_id=tracker.session_id,
            ended_at=ended_at,
            focus_ratio=focus_ratio,
            drift_events=tracker.drift_alerts,
        )
        _DOWNSAMPLE_EVERY = max(1, settings.target_fps)
        sampled = tracker.events[::_DOWNSAMPLE_EVERY]
        db.bulk_insert_events(session_id=tracker.session_id, events=sampled)

        if generate_report and sampled:
            session_record = db.get_session(tracker.session_id)
            if session_record is not None:
                report_gen = SessionReportGenerator()
                report_path = report_gen.generate(
                    session=session_record,
                    events=sampled,
                    auto_open=True,
                )
                print(f"report saved → {report_path}")
