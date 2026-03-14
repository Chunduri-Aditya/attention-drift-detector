"""
`analytics` CLI command — cross-session trends, heatmaps, weekly summary.
"""
from __future__ import annotations

from argparse import Namespace


def analytics(args: Namespace) -> None:
    from ...session.analytics import SessionAnalytics

    sa = SessionAnalytics()
    summary = sa.weekly_summary(user_id=args.user_id)
    all_s = sa.all_completed(user_id=args.user_id)

    # ── terminal summary ───────────────────────────────────────────────────────
    print("\n── Weekly Summary (last 7 days) ────────────────────────────────────")
    if summary.get("n_sessions", 0) == 0:
        print("  No completed sessions yet. Run the detector and come back!")
    else:
        print(f"  Sessions:          {summary['n_sessions']}")
        print(f"  Total time:        {summary['total_minutes']:.0f} min")
        print(f"  Avg focus:         {summary['avg_focus_pct']:.0f}%")
        print(f"  Best session:      {summary['best_focus_pct']:.0f}%")
        print(f"  Worst session:     {summary['worst_focus_pct']:.0f}%")
        print(f"  Total alerts:      {summary['total_alerts']}")

    print(f"\n  All-time sessions: {len(all_s)}")

    # ── best / worst ──────────────────────────────────────────────────────────
    best = sa.best_sessions(n=3, user_id=args.user_id)
    if best:
        print("\n── Top 3 Sessions ──────────────────────────────────────────────────")
        for s in best:
            started = s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "—"
            print(f"  {s.session_id[:8]}  {started}  focus={s.focus_ratio * 100:.0f}%")

    # ── hour heatmap (terminal) ────────────────────────────────────────────────
    by_hour = sa.focus_by_hour(user_id=args.user_id)
    if by_hour:
        print("\n── Focus by Hour of Day ────────────────────────────────────────────")
        for row in by_hour:
            if row["avg_focus"] is None:
                continue
            bar_len = int(row["avg_focus"] / 5)
            bar = "█" * bar_len
            print(f"  {row['hour']:02d}:00  {bar:<20} {row['avg_focus']:.0f}%  ({row['n_sessions']} sessions)")

    # ── HTML report ───────────────────────────────────────────────────────────
    if not args.no_report:
        report_path = sa.generate_report(user_id=args.user_id, auto_open=args.open)
        print(f"\n✓ Full analytics report → {report_path}\n")
    else:
        print()
