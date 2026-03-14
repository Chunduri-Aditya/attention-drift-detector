"""
Cross-session analytics: trends, heatmaps, weekly summaries.

Reads from SessionDB and produces both data structures and
a self-contained HTML analytics report (dark mode, Plotly CDN).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .session_db import SessionDB, SessionRecord
from ..config.paths import REPORTS_DIR


class SessionAnalytics:
    def __init__(self, db: SessionDB | None = None) -> None:
        self._db = db or SessionDB()

    # ── data methods ──────────────────────────────────────────────────────────

    def all_completed(self, user_id: str | None = None, limit: int = 500) -> list[SessionRecord]:
        """Return sessions that have a recorded focus_ratio (i.e. ended cleanly)."""
        sessions = self._db.list_sessions(user_id=user_id, limit=limit)
        return [s for s in sessions if s.focus_ratio is not None and s.ended_at is not None]

    def focus_trend(self, n: int = 14, user_id: str | None = None) -> list[dict]:
        """Daily average focus% over the last N days."""
        sessions = self.all_completed(user_id=user_id)
        cutoff = datetime.utcnow() - timedelta(days=n)
        recent = [s for s in sessions if s.started_at and s.started_at >= cutoff]

        by_day: dict[date, list[float]] = defaultdict(list)
        for s in recent:
            day = s.started_at.date()
            by_day[day].append(s.focus_ratio * 100)

        # Fill in all N days (including zeros for missing days)
        result = []
        today = datetime.utcnow().date()
        for i in range(n - 1, -1, -1):
            d = today - timedelta(days=i)
            vals = by_day.get(d, [])
            result.append({
                "date": d.isoformat(),
                "avg_focus": round(sum(vals) / len(vals), 1) if vals else None,
                "n_sessions": len(vals),
            })
        return result

    def focus_by_hour(self, user_id: str | None = None) -> list[dict]:
        """Average focus% by hour of day (0–23)."""
        sessions = self.all_completed(user_id=user_id)
        by_hour: dict[int, list[float]] = defaultdict(list)
        for s in sessions:
            if s.started_at:
                by_hour[s.started_at.hour].append(s.focus_ratio * 100)

        return [
            {
                "hour": h,
                "avg_focus": round(sum(vals) / len(vals), 1) if vals else None,
                "n_sessions": len(vals),
            }
            for h, vals in sorted(by_hour.items())
        ]

    def best_sessions(self, n: int = 5, user_id: str | None = None) -> list[SessionRecord]:
        sessions = self.all_completed(user_id=user_id)
        return sorted(sessions, key=lambda s: s.focus_ratio or 0, reverse=True)[:n]

    def worst_sessions(self, n: int = 5, user_id: str | None = None) -> list[SessionRecord]:
        sessions = self.all_completed(user_id=user_id)
        return sorted(sessions, key=lambda s: s.focus_ratio or 1)[:n]

    def weekly_summary(self, user_id: str | None = None) -> dict[str, Any]:
        """Stats for the last 7 days."""
        sessions = self.all_completed(user_id=user_id)
        cutoff = datetime.utcnow() - timedelta(days=7)
        week = [s for s in sessions if s.started_at and s.started_at >= cutoff]

        if not week:
            return {"n_sessions": 0}

        durations = []
        for s in week:
            if s.ended_at and s.started_at:
                durations.append((s.ended_at - s.started_at).total_seconds() / 60)

        focus_vals = [s.focus_ratio * 100 for s in week]
        alerts = [s.drift_events or 0 for s in week]

        return {
            "n_sessions": len(week),
            "total_minutes": round(sum(durations), 1),
            "avg_focus_pct": round(sum(focus_vals) / len(focus_vals), 1),
            "best_focus_pct": round(max(focus_vals), 1),
            "worst_focus_pct": round(min(focus_vals), 1),
            "total_alerts": sum(alerts),
            "avg_alerts_per_session": round(sum(alerts) / len(alerts), 1),
        }

    # ── report ────────────────────────────────────────────────────────────────

    def generate_report(
        self,
        user_id: str | None = None,
        auto_open: bool = False,
    ) -> Path:
        """Generate a self-contained HTML analytics report."""
        trend = self.focus_trend(n=14, user_id=user_id)
        by_hour = self.focus_by_hour(user_id=user_id)
        summary = self.weekly_summary(user_id=user_id)
        best = self.best_sessions(n=5, user_id=user_id)
        worst = self.worst_sessions(n=5, user_id=user_id)
        all_s = self.all_completed(user_id=user_id)

        html = _render_report(trend, by_hour, summary, best, worst, all_s, user_id)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = f"_{user_id}" if user_id else ""
        out = REPORTS_DIR / f"analytics{slug}_{ts}.html"
        out.write_text(html, encoding="utf-8")

        if auto_open:
            import webbrowser
            webbrowser.open(out.as_uri())

        return out


# ── HTML renderer ─────────────────────────────────────────────────────────────

def _fmt_session(s: SessionRecord) -> str:
    started = s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "—"
    focus = f"{s.focus_ratio * 100:.0f}%" if s.focus_ratio is not None else "—"
    if s.ended_at and s.started_at:
        dur = f"{(s.ended_at - s.started_at).total_seconds() / 60:.1f}m"
    else:
        dur = "—"
    return f"<tr><td>{s.session_id[:8]}</td><td>{started}</td><td>{dur}</td><td>{focus}</td></tr>"


def _render_report(
    trend: list[dict],
    by_hour: list[dict],
    summary: dict,
    best: list[SessionRecord],
    worst: list[SessionRecord],
    all_sessions: list[SessionRecord],
    user_id: str | None,
) -> str:
    title = f"Attention Analytics — {user_id or 'All Users'}"
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Trend chart data
    trend_dates = [t["date"] for t in trend]
    trend_focus = [t["avg_focus"] for t in trend]

    # Hour chart data
    hours = [h["hour"] for h in by_hour]
    hour_labels = [f"{h:02d}:00" for h in hours]
    hour_focus = [h["avg_focus"] for h in by_hour]

    # All-sessions scatter (focus% vs duration)
    scatter_x, scatter_y, scatter_text = [], [], []
    for s in all_sessions:
        if s.ended_at and s.started_at:
            dur = (s.ended_at - s.started_at).total_seconds() / 60
            scatter_x.append(round(dur, 2))
            scatter_y.append(round((s.focus_ratio or 0) * 100, 1))
            scatter_text.append(s.session_id[:8])

    # Summary cards
    n = summary.get("n_sessions", 0)
    cards = [
        ("Sessions (7d)", str(n)),
        ("Total time (7d)", f"{summary.get('total_minutes', 0):.0f}m"),
        ("Avg focus (7d)", f"{summary.get('avg_focus_pct', 0):.0f}%"),
        ("Best session", f"{summary.get('best_focus_pct', 0):.0f}%"),
        ("Total alerts (7d)", str(summary.get("total_alerts", 0))),
        ("All sessions", str(len(all_sessions))),
    ]

    cards_html = "".join(
        f'<div class="card"><div class="card-val">{v}</div>'
        f'<div class="card-lbl">{l}</div></div>'
        for l, v in cards
    )

    best_rows = "".join(_fmt_session(s) for s in best)
    worst_rows = "".join(_fmt_session(s) for s in worst)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; color: #58a6ff; }}
  .meta {{ font-size: 0.8rem; color: #8b949e; margin-bottom: 2rem; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem 1.5rem; min-width: 130px; }}
  .card-val {{ font-size: 2rem; font-weight: 700; color: #58a6ff; }}
  .card-lbl {{ font-size: 0.75rem; color: #8b949e; margin-top: 0.25rem; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-full {{ margin-bottom: 2rem; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem; }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #c9d1d9; margin-bottom: 0.75rem; }}
  .table-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; color: #8b949e; font-weight: 500; padding: 0.4rem 0.6rem; border-bottom: 1px solid #30363d; }}
  td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #21262d; }}
  .footer {{ font-size: 0.75rem; color: #484f58; margin-top: 2rem; text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Generated {generated} · All data stored locally · Never transmitted</div>

<div class="cards">{cards_html}</div>

<div class="chart-full">
  <div class="chart-box">
    <h2>14-Day Focus Trend</h2>
    <div id="trend"></div>
  </div>
</div>

<div class="chart-row">
  <div class="chart-box">
    <h2>Focus by Hour of Day</h2>
    <div id="byhour"></div>
  </div>
  <div class="chart-box">
    <h2>Session Duration vs Focus%</h2>
    <div id="scatter"></div>
  </div>
</div>

<div class="table-row">
  <div class="chart-box">
    <h2>Best Sessions</h2>
    <table>
      <tr><th>ID</th><th>Date</th><th>Dur</th><th>Focus</th></tr>
      {best_rows}
    </table>
  </div>
  <div class="chart-box">
    <h2>Worst Sessions</h2>
    <table>
      <tr><th>ID</th><th>Date</th><th>Dur</th><th>Focus</th></tr>
      {worst_rows}
    </table>
  </div>
</div>

<div class="footer">Attention Drift Detector · Privacy-first · Data never leaves your device</div>

<script>
const DARK = {{ paper_bgcolor:"#161b22", plot_bgcolor:"#161b22",
  font:{{color:"#c9d1d9"}}, xaxis:{{gridcolor:"#21262d",linecolor:"#30363d"}},
  yaxis:{{gridcolor:"#21262d",linecolor:"#30363d"}}, margin:{{t:10,b:40,l:50,r:10}} }};

const trend_dates = {json.dumps(trend_dates)};
const trend_focus = {json.dumps(trend_focus)};
Plotly.newPlot("trend", [{{
  x: trend_dates, y: trend_focus, type:"scatter", mode:"lines+markers",
  line:{{color:"#58a6ff",width:2}}, marker:{{color:"#58a6ff",size:6}},
  connectgaps: true,
  hovertemplate:"%{{x}}<br>Focus: %{{y:.1f}}%<extra></extra>"
}}], {{...DARK, yaxis:{{...DARK.yaxis, range:[0,100], title:"Focus %"}}}}, {{responsive:true}});

const hours = {json.dumps(hour_labels)};
const hour_focus = {json.dumps(hour_focus)};
Plotly.newPlot("byhour", [{{
  x: hours, y: hour_focus, type:"bar",
  marker:{{color:"#3fb950"}},
  hovertemplate:"%{{x}}<br>Avg Focus: %{{y:.1f}}%<extra></extra>"
}}], {{...DARK, yaxis:{{...DARK.yaxis, range:[0,100], title:"Avg Focus %"}}}}, {{responsive:true}});

const sx = {json.dumps(scatter_x)};
const sy = {json.dumps(scatter_y)};
const st = {json.dumps(scatter_text)};
Plotly.newPlot("scatter", [{{
  x: sx, y: sy, text: st, type:"scatter", mode:"markers",
  marker:{{color:sy, colorscale:[["0","#f85149"],["0.5","#d29922"],["1","#3fb950"]],
    size:9, showscale:true, cmin:0, cmax:100,
    colorbar:{{title:"Focus%", thickness:12, len:0.8}}}},
  hovertemplate:"Session %{{text}}<br>Dur: %{{x:.1f}}m<br>Focus: %{{y:.0f}}%<extra></extra>"
}}], {{...DARK, xaxis:{{...DARK.xaxis, title:"Duration (min)"}},
  yaxis:{{...DARK.yaxis, range:[0,100], title:"Focus %"}}}}, {{responsive:true}});
</script>
</body>
</html>"""
