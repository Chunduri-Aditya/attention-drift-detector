"""
`train` CLI command — trains the ML attention classifier from collected CSV data.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def train(args: Namespace) -> None:
    from ...config.paths import MODELS_DIR
    from ...models.trainer import train as run_training

    model_out = Path(args.model_out) if args.model_out else MODELS_DIR / "attention_model.joblib"

    try:
        run_training(csv_path=args.csv, model_out=model_out)
    except ImportError:
        print(
            "\nsklearn is not installed.  Install it with:\n\n"
            "    pip install scikit-learn joblib\n"
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Collect training data first:\n\n    attention-drift-detector realtime --collect\n")
