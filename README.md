# Attention Drift Detector

A local, privacy-first webcam tool that monitors your attention in real time during study and deep work sessions. It tracks your head pose and gaze using your laptop camera, classifies your attention state frame-by-frame, nudges you when you've been drifting, and generates a full HTML session report when you're done.

No video is ever recorded. No data leaves your machine. Only derived numbers (head angles, gaze vectors) and session labels are stored locally in SQLite.

---

## What it detects

| State | Meaning |
|---|---|
| **Focused** | Head and gaze centered on screen |
| **Drifting** | Head turned, looking away, or gaze offset |
| **Absent** | Face not visible in frame |

Detection uses MediaPipe Face Mesh (478 landmarks) → head pose via solvePnP → iris-based gaze estimation → rule-based classifier with temporal smoothing over a 1-second sliding window. After 5 continuous seconds of drift, an OS notification fires.

---

## Quick start

### 1. Install

```bash
cd attention-drift-detector
pip install -e .
```

### 2. Test without a webcam (dummy mode)

Runs the full pipeline with synthetic frames — no camera required. Confirms everything is wired up end-to-end and generates a report.

```bash
python -m attention_drift_detector.cli.main realtime --dummy --max-seconds 10
```

### 3. Run live with your webcam

```bash
python -m attention_drift_detector.cli.main realtime --user-id yourname
```

A window opens showing your face with a live overlay: attention state, focus ratio, and drift alert count. Press **`q`** to end the session.

When you quit, two things happen automatically:
- Session is saved to `sessions/attention_sessions.db`
- An HTML report opens in your browser from `reports/`

---

## CLI reference

```bash
python -m attention_drift_detector.cli.main realtime [OPTIONS]

Options:
  --user-id TEXT        Local namespace for your sessions (default: default_user)
  --max-seconds INT     Auto-stop after N seconds (useful for testing)
  --no-preview          Disable the OpenCV overlay window (headless mode)
  --dummy               Synthetic frames — no webcam or MediaPipe required
```

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

50 tests across 5 modules — runs in under 1 second. No webcam or network required.

```
tests/test_attention_classifier.py    (14 tests)
tests/test_temporal_smoother.py       (8 tests)
tests/test_drift_event_detector.py    (8 tests)
tests/test_nudge_throttler.py         (9 tests)
tests/test_session_tracker.py         (10 tests)
```

---

## Session reports

After every session, a self-contained HTML report is generated in `reports/`. It contains:

- **Attention timeline** — state color-coded over the full session (green = focused, blue = drifting, red = absent)
- **State distribution** — bar chart showing % time in each state
- **Key stats** — total duration, focus %, longest focus streak, drift alert count

Reports open automatically in your browser. They work offline and have no external dependencies.

---

## Local REST API

A FastAPI server exposes your session history over localhost. Useful for reading your attention data from other tools (VS Code extensions, Obsidian plugins, dashboards).

```bash
uvicorn attention_drift_detector.api.app:app --host 127.0.0.1 --port 8765
```

| Endpoint | Description |
|---|---|
| `GET /sessions` | List all sessions |
| `GET /sessions/{id}` | Session metadata |
| `GET /sessions/{id}/events` | Full attention event timeline |
| `GET /sessions/{id}/report` | Generate and return HTML report |
| `DELETE /sessions/{id}` | Hard delete (removes all data) |

Interactive docs at `http://127.0.0.1:8765/docs`.

---

## Project structure

```
attention_drift_detector/
  capture/          # OpenCV webcam loop
  tracking/         # Face mesh, head pose, gaze, blink detection
  state_estimation/ # Classifier, temporal smoother, drift detector
  alerts/           # Nudge system, OS notifications, throttler
  session/          # SessionTracker, SQLite DB, HTML report generator
  pipelines/        # Main realtime loop
  api/              # FastAPI app and session routes
  config/           # All thresholds and settings in one place
  common/           # Shared types and enums
tests/              # 50 pytest-compatible unit tests
reports/            # Generated HTML session reports
sessions/           # SQLite database (attention_sessions.db)
```

---

## Privacy guarantees

- **No video recording** — webcam stream is processed in memory and discarded per frame
- **No raw frames on disk** — only numerical features (head angles, gaze vectors) and derived labels are stored
- **Local only** — SQLite database lives entirely on your machine
- **Hard delete** — `DELETE /sessions/{id}` permanently removes all data for a session
- **Pause anytime** — quit with `q` at any point; partial sessions are saved cleanly

---

## Configuration

All thresholds are in `attention_drift_detector/config/settings.py`. Key defaults:

| Setting | Default | Meaning |
|---|---|---|
| `head_pose.yaw_deg` | 25° | Max sideways head turn before drift |
| `head_pose.pitch_up_deg` | 20° | Max upward head tilt |
| `head_pose.pitch_down_deg` | -30° | Max downward head tilt (phone check) |
| `drift.drift_seconds_threshold` | 5s | Sustained drift before nudge fires |
| `drift.nudge_cooldown_s` | 180s | Minimum time between nudges (3 min) |
| `drift.temporal_window_s` | 1.0s | Smoothing window (30 frames @ 30fps) |
| `blink.ear_threshold` | 0.25 | Eye aspect ratio threshold for blink |

---

## Compatibility note

Requires **MediaPipe ≥ 0.10.0**. If you hit a FaceMesh import error, confirm your version:

```bash
python -c "import mediapipe; print(mediapipe.__version__)"
```

NumPy must be < 2.0 (MediaPipe constraint):

```bash
pip install "numpy<2.0"
```

---

## What's next

The `AttentionClassifier` is currently rule-based and explicitly designed as a drop-in placeholder. The interface is already defined — replacing it with a trained MLP requires:

1. Running real sessions to collect labeled `FrameFeatures` data (stored in SQLite)
2. Training a lightweight MLP on `(yaw, pitch, gaze_x, gaze_y, blink, face_visibility)` → `AttentionState`
3. Swapping the classifier without changing anything else in the pipeline

The storage layer and session report infrastructure built here generate exactly the training data needed for that upgrade.
