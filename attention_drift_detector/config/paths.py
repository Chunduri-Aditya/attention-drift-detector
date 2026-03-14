from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

SESSIONS_DIR = BASE_DIR / "sessions"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_STORE_DIR = BASE_DIR / "models_store"

# Single SQLite database for all sessions (WAL mode, privacy-safe)
DB_PATH = SESSIONS_DIR / "attention_sessions.db"


def session_db_path(user_id: str, session_id: str) -> Path:
    return SESSIONS_DIR / user_id / f"{session_id}.db"


def session_report_path(session_id: str, timestamp_str: str) -> Path:
    return REPORTS_DIR / f"{session_id}_{timestamp_str}.json"


def attention_model_weights(version: str = "1") -> Path:
    return MODELS_STORE_DIR / f"attention_classifier_v{version}.pt"


