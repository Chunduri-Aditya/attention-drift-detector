from __future__ import annotations

import argparse
from typing import Callable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attention-drift-detector",
        description="Webcam-based attention drift detector (local-only, privacy-first).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── realtime ──────────────────────────────────────────────────────────────
    realtime = subparsers.add_parser(
        "realtime", help="Run real-time attention monitoring."
    )
    realtime.add_argument(
        "--user-id", default="default_user",
        help="Local user/session namespace. (default: default_user)",
    )
    realtime.add_argument(
        "--max-seconds", type=int, default=None,
        help="Auto-stop after N seconds (useful for testing).",
    )
    realtime.add_argument(
        "--no-preview", action="store_true",
        help="Disable OpenCV overlay window (headless mode).",
    )
    realtime.add_argument(
        "--dummy", action="store_true",
        help="Synthetic frames — no webcam or MediaPipe required.",
    )
    realtime.add_argument(
        "--no-calibrate", action="store_true",
        help="Skip the personal calibration window (use static thresholds).",
    )
    realtime.add_argument(
        "--calibration-seconds", type=float, default=30.0,
        help="Length of the calibration window in seconds. (default: 30)",
    )
    realtime.add_argument(
        "--collect", action="store_true",
        help="Save labeled FrameFeatures to CSV for ML training.",
    )
    realtime.add_argument(
        "--serve-dashboard", action="store_true",
        help="Boot the live web dashboard in the background (http://127.0.0.1:8765/).",
    )
    realtime.add_argument(
        "--dashboard-port", type=int, default=8765,
        help="Port for the dashboard server. (default: 8765)",
    )
    realtime.set_defaults(func=_run_realtime)

    # ── sessions ──────────────────────────────────────────────────────────────
    sessions = subparsers.add_parser(
        "sessions", help="List and inspect past attention sessions."
    )
    sessions.add_argument(
        "--user-id", default=None,
        help="Filter sessions by user ID.",
    )
    sessions.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of sessions to show. (default: 20)",
    )
    sessions.set_defaults(func=_list_sessions)

    # ── analytics ─────────────────────────────────────────────────────────────
    analytics = subparsers.add_parser(
        "analytics", help="Cross-session trends, heatmaps, and weekly summary.",
    )
    analytics.add_argument(
        "--user-id", default=None,
        help="Filter by user ID.",
    )
    analytics.add_argument(
        "--open", action="store_true",
        help="Auto-open the HTML report in your browser.",
    )
    analytics.add_argument(
        "--no-report", action="store_true",
        help="Print terminal summary only, skip HTML report.",
    )
    analytics.set_defaults(func=_analytics)

    # ── train ─────────────────────────────────────────────────────────────────
    train = subparsers.add_parser(
        "train", help="Train the ML attention classifier from collected CSV data.",
    )
    train.add_argument(
        "--csv", required=True,
        help="Path to a labeled CSV file or directory of CSVs (from `realtime --collect`).",
    )
    train.add_argument(
        "--model-out", default=None,
        help="Output path for the saved model (default: models_store/attention_model.joblib).",
    )
    train.set_defaults(func=_train)

    # ── serve ─────────────────────────────────────────────────────────────────
    serve = subparsers.add_parser(
        "serve", help="Start the local REST API + dashboard server (FastAPI + uvicorn).",
    )
    serve.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host. (default: 127.0.0.1 — localhost only)",
    )
    serve.add_argument(
        "--port", type=int, default=8765,
        help="Bind port. (default: 8765)",
    )
    serve.add_argument(
        "--reload", action="store_true",
        help="Enable hot-reload (development mode).",
    )
    serve.set_defaults(func=_serve)

    return parser


def app() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    func: Callable[..., None] = getattr(args, "func")
    func(args)


def _run_realtime(args: argparse.Namespace) -> None:
    from .commands.run_realtime import run_realtime
    run_realtime(args)


def _list_sessions(args: argparse.Namespace) -> None:
    from .commands.list_sessions import list_sessions
    list_sessions(args)


def _analytics(args: argparse.Namespace) -> None:
    from .commands.analytics import analytics
    analytics(args)


def _train(args: argparse.Namespace) -> None:
    from .commands.train import train
    train(args)


def _serve(args: argparse.Namespace) -> None:
    from .commands.serve import serve
    serve(args)


if __name__ == "__main__":
    app()
