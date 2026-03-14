from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform project bootstrap.")
    parser.add_argument("--dev", action="store_true", help="Install optional development dependencies.")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required.")

    print(f"Detected OS: {platform.system()} {platform.release()}")
    print(f"Using project root: {ROOT}")

    if not VENV_DIR.exists():
        _run([sys.executable, "-m", "venv", str(VENV_DIR)])

    vpy = _venv_python()
    _run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run([str(vpy), "-m", "pip", "install", "-e", str(ROOT)])
    if args.dev:
        _run([str(vpy), "-m", "pip", "install", "-e", f"{ROOT}[dev]"])

    if os.name == "nt":
        activate = ".venv\\Scripts\\Activate.ps1"
    else:
        activate = "source .venv/bin/activate"

    print("\nBootstrap complete.")
    print("Activate venv:", activate)
    print("Run CLI: attention-drift-detector --help")


if __name__ == "__main__":
    main()

