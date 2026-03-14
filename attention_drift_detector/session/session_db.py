"""
SQLite session storage.

Schema
------
sessions
  session_id     TEXT PRIMARY KEY
  user_id        TEXT NOT NULL
  started_at     TEXT NOT NULL    -- ISO-8601 UTC
  ended_at       TEXT            -- NULL until session ends
  focus_ratio    REAL            -- NULL until session ends
  drift_events   INTEGER NOT NULL DEFAULT 0

attention_events
  id             INTEGER PRIMARY KEY AUTOINCREMENT
  session_id     TEXT NOT NULL REFERENCES sessions(session_id)
  timestamp      TEXT NOT NULL
  state          TEXT NOT NULL    -- focused / drifting / absent
  confidence     REAL NOT NULL

Privacy guarantee: no raw frames or images are ever stored.
Only derived numerical features and labels reach this layer.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Sequence

from ..common.enums import AttentionState
from ..common.types import AttentionEvent, SessionRecord
from ..config.paths import DB_PATH


class SessionDB:
    """
    Thin SQLite wrapper.  All writes are auto-committed; reads return plain
    Python objects so callers never need to import sqlite3.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    started_at   TEXT NOT NULL,
                    ended_at     TEXT,
                    focus_ratio  REAL,
                    drift_events INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS attention_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
                    timestamp   TEXT NOT NULL,
                    state       TEXT NOT NULL,
                    confidence  REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_session
                    ON attention_events(session_id);
                """
            )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, session_id: str, user_id: str, started_at: datetime) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_id, started_at) VALUES (?, ?, ?)",
                (session_id, user_id, started_at.isoformat()),
            )

    def end_session(
        self,
        session_id: str,
        ended_at: datetime,
        focus_ratio: float,
        drift_events: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET ended_at=?, focus_ratio=?, drift_events=?
                WHERE session_id=?
                """,
                (ended_at.isoformat(), focus_ratio, drift_events, session_id),
            )

    # ------------------------------------------------------------------
    # Event batching
    # ------------------------------------------------------------------

    def bulk_insert_events(
        self, session_id: str, events: Sequence[AttentionEvent]
    ) -> None:
        """Write a batch of attention events in a single transaction."""
        rows = [
            (session_id, e.timestamp.isoformat(), e.state.value, e.confidence)
            for e in events
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO attention_events (session_id, timestamp, state, confidence) VALUES (?, ?, ?, ?)",
                rows,
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_session(row)

    def list_sessions(self, user_id: str | None = None, limit: int = 50) -> list[SessionRecord]:
        query = "SELECT * FROM sessions"
        params: list = []
        if user_id is not None:
            query += " WHERE user_id=?"
            params.append(user_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_session(r) for r in rows]

    def get_events(self, session_id: str) -> list[AttentionEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, state, confidence FROM attention_events WHERE session_id=? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        return [
            AttentionEvent(
                timestamp=datetime.fromisoformat(r["timestamp"]),
                state=AttentionState(r["state"]),
                confidence=r["confidence"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Privacy: hard delete
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str) -> None:
        """Permanently remove a session and all its events."""
        with self._conn() as conn:
            conn.execute("DELETE FROM attention_events WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        user_id=row["user_id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
        focus_ratio=row["focus_ratio"],
        drift_events=row["drift_events"] or 0,
    )
