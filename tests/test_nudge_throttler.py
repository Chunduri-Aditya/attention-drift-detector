"""
Tests for NudgeThrottler — cooldown and consecutive-nudge logic.
"""
from __future__ import annotations

import time

import pytest

from attention_drift_detector.alerts.nudge_throttler import NudgeThrottler


class TestNudgeThrottler:
    def test_first_nudge_is_allowed(self) -> None:
        throttler = NudgeThrottler(cooldown_s=60.0, max_consecutive_nudges=3)
        allowed, cooldown_remaining = throttler.can_nudge()
        assert allowed is True
        assert cooldown_remaining == 0.0

    def test_nudge_blocked_within_cooldown(self) -> None:
        throttler = NudgeThrottler(cooldown_s=60.0, max_consecutive_nudges=3)
        throttler.mark_nudged()
        allowed, cooldown_remaining = throttler.can_nudge()
        assert allowed is False
        assert cooldown_remaining > 0.0

    def test_cooldown_remaining_is_approximate(self) -> None:
        throttler = NudgeThrottler(cooldown_s=10.0, max_consecutive_nudges=3)
        throttler.mark_nudged()
        _, remaining = throttler.can_nudge()
        assert 9.0 <= remaining <= 10.0

    def test_nudge_allowed_after_cooldown(self) -> None:
        throttler = NudgeThrottler(cooldown_s=0.05, max_consecutive_nudges=3)
        throttler.mark_nudged()
        time.sleep(0.06)
        allowed, _ = throttler.can_nudge()
        assert allowed is True

    def test_break_suggestion_at_max_consecutive(self) -> None:
        throttler = NudgeThrottler(cooldown_s=0.0, max_consecutive_nudges=2)
        throttler.mark_nudged()
        should_break = throttler.mark_nudged()
        assert should_break is True

    def test_no_break_suggestion_before_max(self) -> None:
        throttler = NudgeThrottler(cooldown_s=0.0, max_consecutive_nudges=3)
        r1 = throttler.mark_nudged()
        r2 = throttler.mark_nudged()
        assert r1 is False
        assert r2 is False

    def test_recovery_resets_consecutive_count(self) -> None:
        throttler = NudgeThrottler(cooldown_s=0.0, max_consecutive_nudges=2)
        throttler.mark_nudged()
        throttler.mark_nudged()  # should_break = True
        throttler.mark_recovery()
        # After recovery, nudge count resets — next two nudges should not break
        r1 = throttler.mark_nudged()
        assert r1 is False

    def test_multiple_recoveries_safe(self) -> None:
        throttler = NudgeThrottler(cooldown_s=0.0, max_consecutive_nudges=3)
        throttler.mark_recovery()
        throttler.mark_recovery()
        allowed, _ = throttler.can_nudge()
        assert allowed is True

    def test_zero_cooldown_always_allows(self) -> None:
        throttler = NudgeThrottler(cooldown_s=0.0, max_consecutive_nudges=10)
        for _ in range(5):
            throttler.mark_nudged()
        allowed, _ = throttler.can_nudge()
        assert allowed is True
