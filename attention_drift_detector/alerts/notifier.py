"""
Cross-platform OS notification backend.

Priority order for each platform:
  macOS  → osascript (built-in, zero deps)
  Linux  → notify-send (libnotify, common on most distros)
  Windows → win10toast if installed, else ctypes MessageBox fallback
  Any    → terminal bell + print (always works, final fallback)

Design rationale: no new package dependencies by default.  We shell out
to platform tools that are almost certainly already present.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Literal


_SYSTEM = platform.system()  # "Darwin" | "Linux" | "Windows"

_NUDGE_MESSAGES = [
    "Take a breath — bring your gaze back to the screen.",
    "You drifted. Refocus — what were you working on?",
    "Gentle reminder: attention drift detected.",
    "Mind wandering? Ground yourself and come back.",
    "Check in: are you still on task?",
]

_BREAK_MESSAGE = (
    "You've drifted several times this session. "
    "A 5-minute break might help you come back sharper."
)

_message_index = 0


def _next_nudge_message() -> str:
    global _message_index
    msg = _NUDGE_MESSAGES[_message_index % len(_NUDGE_MESSAGES)]
    _message_index += 1
    return msg


def _notify_macos(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _notify_linux(title: str, message: str) -> None:
    try:
        subprocess.run(
            ["notify-send", "--urgency=normal", "--expire-time=5000", title, message],
            check=False,
            capture_output=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _notify_terminal(title: str, message: str) -> None:
    sys.stdout.write(f"\a[{title}] {message}\n")
    sys.stdout.flush()


def send_notification(
    title: str,
    message: str,
    urgency: Literal["normal", "critical"] = "normal",
) -> None:
    """
    Send an OS notification.  Falls back gracefully on any platform.
    Always prints to stdout as well so headless runs capture the nudge.
    """
    print(f"[{title}] {message}")

    if _SYSTEM == "Darwin":
        _notify_macos(title, message)
    elif _SYSTEM == "Linux":
        _notify_linux(title, message)
    else:
        # Windows or unknown: terminal bell is the safe fallback
        _notify_terminal(title, message)


def send_nudge() -> None:
    send_notification(
        title="Attention Drift Detector",
        message=_next_nudge_message(),
        urgency="normal",
    )


def send_break_suggestion() -> None:
    send_notification(
        title="Attention Drift Detector — Break Suggested",
        message=_BREAK_MESSAGE,
        urgency="critical",
    )
