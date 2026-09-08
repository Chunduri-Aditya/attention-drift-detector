# Attention Drift Detector — showcase

One-paragraph version, a demo that needs no webcam, and an honest read on what is and is not finished.

## What it is

A local, privacy-first webcam tool that tracks your attention during deep work. It reads head pose (solvePnP over MediaPipe Face Mesh, 478 landmarks) and iris-based gaze, classifies each frame as focused / drifting / absent with a 1-second temporal smoother, fires an OS notification after 5 continuous seconds of drift, and writes a self-contained HTML report at session end.

No video is recorded. No raw frame touches disk. Only derived numbers (head angles, gaze vectors) and state labels are stored, in a local SQLite file.

The design commitment: **rule-based classification over a trained model.** The classifier is a face-visibility gate plus drift votes across yaw, pitch, and gaze offset, with an optional 30-second calibration window that recentres the thresholds on your neutral pose. It is transparent, needs no training data, and is what all 50 unit tests exercise.

## Demo (no webcam, no MediaPipe)

```bash
cd ~/Desktop/Projects/attention-drift-detector
pip install -e '.[dev]'

# full pipeline on synthetic frames — classify, track drift, write a report
python -m attention_drift_detector.cli.main realtime --dummy --max-seconds 10 --no-preview --no-calibrate
# → reports/session_<id>_<timestamp>.html opens in the browser

# session history over localhost
uvicorn attention_drift_detector.api.app:app --port 8765   # docs at /docs
```

Live with a webcam: `python -m attention_drift_detector.cli.main realtime --user-id you` (press `q` to end).

## Tests

```bash
python -m pytest tests/ -v
```

Last run 2026-09-08: **52 passed** in 0.1 s. No webcam, no network.

| File | Tests | Covers |
|---|---|---|
| `test_attention_classifier.py` | 17 | rule-based state logic on every drift axis, confidence scaling, absent gate |
| `test_session_tracker.py` | 10 | session lifecycle, focus ratio, SQLite persistence |
| `test_nudge_throttler.py` | 9 | nudge cooldown, max-consecutive cap |
| `test_drift_event_detector.py` | 8 | sustained-drift threshold, event boundaries |
| `test_temporal_smoother.py` | 8 | sliding-window majority vote |

## Done

- Real-time capture → face mesh → head pose → gaze → rule-based classify → temporal smooth → drift detect → nudge → HTML report, end to end
- Dummy-frame mode as a first-class entry point (CI-friendly, no camera)
- Personal calibration: 30 s window, per-user threshold recentring, hot-swapped mid-session
- Local FastAPI over session history (`/sessions`, `/events`, `/report`, hard `DELETE`)
- Live web dashboard (`--serve-dashboard`) and a `train` CLI + RandomForest/MLP trainer scaffold
- Privacy invariant enforced in code: no frame written to disk, features only

## Known gaps

- **The trained-model path (`--use-ml`) is wiring, not a result.** `models/mlp_classifier.py`'s `predict()` had been written against feature attributes that `FrameFeatures` does not expose — it had never run. That is fixed now (it builds the vector from `head_angles_deg` / `gaze_vector` in the model's `feature_cols` order) and `--use-ml` runs end to end, but: no model ships in the repo (`models_store/` is Git ignored), and the one on this machine was trained on 1732 self-collected samples in a single sitting with no held-out evaluation against the rule-based baseline. On synthetic input it disagrees with the rule-based classifier in ways that suggest the training distribution was narrow. Do not cite its `cv_f1` 0.9365. Rule-based is the supported classifier.
- The bundled demo model was pickled with scikit-learn 1.7.2; loading it under a newer sklearn prints an `InconsistentVersionWarning`. It only affects `--use-ml` on this machine.
- No CI workflow committed. `.github/` exists but the test run is by hand.
- Live webcam accuracy has not been measured against ground truth; the numbers in a report are the classifier's own labels, not validated against a human rater.
- Only two commits of history — the repo was built in two passes and sat untouched from 2026-03 until this audit.

## Layout

`~/Desktop/Projects/attention-drift-detector` · branch `main` · <https://github.com/Chunduri-Aditya/attention_drift_detector>
