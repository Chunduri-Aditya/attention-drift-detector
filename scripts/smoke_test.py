from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    run([sys.executable, "-m", "attention_drift_detector.cli.main", "--help"])
    run(
        [
            sys.executable,
            "-m",
            "attention_drift_detector.cli.main",
            "realtime",
            "--dummy",
            "--max-seconds",
            "1",
            "--no-preview",
        ]
    )
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

