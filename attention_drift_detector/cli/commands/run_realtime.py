from __future__ import annotations

from argparse import Namespace

from ...pipelines.realtime_pipeline import run_realtime_pipeline


def run_realtime(args: Namespace) -> None:
    """
    Entry point for the `realtime` CLI command.
    """

    run_realtime_pipeline(
        user_id=args.user_id,
        max_seconds=args.max_seconds,
        show_preview=not args.no_preview,
        use_dummy=args.dummy,
        calibrate=not args.no_calibrate,
        calibration_seconds=args.calibration_seconds,
        collect=args.collect,
        serve_dashboard=args.serve_dashboard,
        dashboard_port=args.dashboard_port,
    )


