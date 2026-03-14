"""
ML training pipeline for the attention classifier.

Usage:
    attention-drift-detector train --csv data/labeled_frames.csv
    attention-drift-detector train --csv data/ --model-out models_store/my_model.joblib

Input CSV columns (produced by DataCollector):
    timestamp, yaw, pitch, roll, gaze_x, gaze_y, gaze_z,
    blink, face_visibility, state, confidence, session_id

The trainer:
  1. Loads + cleans the CSV(s)
  2. Encodes labels (focused=0, drifting=1, absent=2)
  3. Standardizes features
  4. Trains a RandomForest baseline and MLP, picks the better one
  5. Cross-validates and prints classification report
  6. Saves the winning model + scaler to `models_store/`
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# ── feature columns (must match DataCollector CSV header) ────────────────────
FEATURE_COLS = ["yaw", "pitch", "roll", "gaze_x", "gaze_y", "gaze_z",
                "blink", "face_visibility"]
LABEL_COL = "state"
LABEL_MAP = {"focused": 0, "drifting": 1, "absent": 2}
LABEL_NAMES = ["focused", "drifting", "absent"]


def load_csvs(csv_path: str) -> "tuple[np.ndarray, np.ndarray]":
    """Load one CSV file or all CSVs in a directory. Returns (X, y)."""
    import pandas as pd

    p = Path(csv_path)
    if p.is_dir():
        files = list(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {p}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(files)} CSV(s) → {len(df):,} rows")
    else:
        df = pd.read_csv(p)
        print(f"Loaded {p.name} → {len(df):,} rows")

    # Drop rows with missing features or unknown labels
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    df = df[df[LABEL_COL].isin(LABEL_MAP)]

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].map(LABEL_MAP).values.astype(np.int64)
    return X, y


def train(csv_path: str, model_out: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    import joblib

    X, y = load_csvs(csv_path)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {LABEL_NAMES[u]:<10} {c:>6,} ({c / len(y) * 100:.1f}%)")

    if len(X) < 50:
        print(
            "\n⚠  Only {len(X)} samples — collect more data with `realtime --collect`.\n"
            "   Training anyway for demonstration."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # ── candidate models ──────────────────────────────────────────────────────
    candidates = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                random_state=42,
            )),
        ]),
    }

    print("\n── Training candidates ────────────────────────────────────────────")
    best_name, best_pipeline, best_score = None, None, -1.0

    for name, pipeline in candidates.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_weighted")
        mean_f1 = cv_scores.mean()
        print(f"  {name:<14} CV F1 = {mean_f1:.3f}  (±{cv_scores.std():.3f})")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_name = name
            best_pipeline = pipeline

    # ── final fit on full train split ─────────────────────────────────────────
    print(f"\n✓ Best model: {best_name}  (CV F1 = {best_score:.3f})")
    best_pipeline.fit(X_train, y_train)

    # ── test set report ───────────────────────────────────────────────────────
    y_pred = best_pipeline.predict(X_test)
    print("\n── Test-set classification report ─────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))

    # ── save ──────────────────────────────────────────────────────────────────
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_out)

    meta = {
        "model_type": best_name,
        "cv_f1": round(best_score, 4),
        "feature_cols": FEATURE_COLS,
        "label_map": LABEL_MAP,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    meta_path = model_out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n✓ Model saved → {model_out}")
    print(f"  Metadata   → {meta_path}")
