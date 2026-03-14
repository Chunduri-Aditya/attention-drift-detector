"""
`serve` CLI command — starts the local FastAPI REST API server via uvicorn.
"""
from __future__ import annotations

from argparse import Namespace


def serve(args: Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            "\nuvicorn is not installed.  Install it with:\n\n"
            "    pip install uvicorn\n"
        )
        return

    print(
        f"\nStarting attention-drift-detector API server at "
        f"http://{args.host}:{args.port}\n"
        f"  Docs: http://{args.host}:{args.port}/docs\n"
        f"  Press Ctrl+C to stop.\n"
    )

    uvicorn.run(
        "attention_drift_detector.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
