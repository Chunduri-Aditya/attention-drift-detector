"""
`sessions` CLI command — prints a table of past attention sessions from SQLite.
"""
from __future__ import annotations

from argparse import Namespace


def list_sessions(args: Namespace) -> None:
    from ...session.session_db import SessionDB

    db = SessionDB()
    sessions = db.list_sessions(user_id=args.user_id, limit=args.limit)

    if not sessions:
        print("No sessions found.")
        return

    # ── header ────────────────────────────────────────────────────────────────
    col_w = [8, 14, 24, 8, 7, 8]
    headers = ["ID", "User", "Started", "Dur(m)", "Focus%", "Alerts"]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print("\n" + header_line)
    print("─" * len(header_line))

    # ── rows ──────────────────────────────────────────────────────────────────
    for s in sessions:
        short_id = s.session_id[:8]

        started = s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "—"

        if s.ended_at and s.started_at:
            dur_s = (s.ended_at - s.started_at).total_seconds()
            dur_str = f"{dur_s / 60:.1f}"
        else:
            dur_str = "—"

        focus_str = f"{s.focus_ratio * 100:.0f}%" if s.focus_ratio is not None else "—"
        alerts_str = str(s.drift_events) if s.drift_events is not None else "—"

        row = [short_id, s.user_id or "—", started, dur_str, focus_str, alerts_str]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))

    print(f"\n{len(sessions)} session(s) shown.\n")
