"""
Post-session HTML report generator.

Produces a self-contained HTML file with:
  - Attention timeline (state over time, color-coded)
  - State distribution bar chart
  - Key stats: total time, focus %, longest focus streak, drift alerts
  - Privacy note at the bottom

Usage
-----
    from attention_drift_detector.session.session_report import SessionReportGenerator
    from attention_drift_detector.session.session_db import SessionDB

    db = SessionDB()
    events = db.get_events(session_id)
    session = db.get_session(session_id)
    gen = SessionReportGenerator()
    path = gen.generate(session, events)
"""
from __future__ import annotations

import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from ..common.enums import AttentionState
from ..common.types import AttentionEvent, SessionRecord
from ..config.paths import REPORTS_DIR

# Map states to display-friendly colors (consistent with overlay palette)
_STATE_COLORS: dict[AttentionState, str] = {
    AttentionState.FOCUSED: "#3CB44B",   # green
    AttentionState.DRIFTING: "#4578E6",  # blue
    AttentionState.ABSENT: "#E63946",    # red
}

_STATE_LABELS: dict[AttentionState, str] = {
    AttentionState.FOCUSED: "Focused",
    AttentionState.DRIFTING: "Drifting",
    AttentionState.ABSENT: "Absent",
}


class SessionReportGenerator:
    def __init__(self, reports_dir: Path = REPORTS_DIR) -> None:
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        session: SessionRecord,
        events: Sequence[AttentionEvent],
        auto_open: bool = False,
    ) -> Path:
        """
        Build the HTML report and return its path.
        Optionally open in the default browser.
        """
        stats = _compute_stats(events, session)
        html = _render_html(session, events, stats)

        ts = session.started_at.strftime("%Y%m%d_%H%M%S")
        out_path = self._reports_dir / f"session_{session.session_id[:8]}_{ts}.html"
        out_path.write_text(html, encoding="utf-8")

        if auto_open:
            webbrowser.open(out_path.as_uri())

        return out_path


# ------------------------------------------------------------------
# Stats helpers
# ------------------------------------------------------------------

def _compute_stats(
    events: Sequence[AttentionEvent], session: SessionRecord
) -> dict:
    if not events:
        return {
            "total_seconds": 0,
            "focus_pct": 0.0,
            "drift_pct": 0.0,
            "absent_pct": 0.0,
            "longest_focus_streak_s": 0.0,
            "drift_alerts": session.drift_events,
        }

    counts = {s: 0 for s in AttentionState}
    for e in events:
        counts[e.state] += 1

    total = len(events)

    # Estimate duration from timestamps if possible, else from frame count
    if session.ended_at and session.started_at:
        total_seconds = (session.ended_at - session.started_at).total_seconds()
    else:
        total_seconds = (events[-1].timestamp - events[0].timestamp).total_seconds()

    # Longest focus streak (approximate, in seconds)
    # We estimate per-event duration as total_seconds / total frames
    seconds_per_frame = total_seconds / max(1, total)
    max_streak = 0
    current_streak = 0
    for e in events:
        if e.state == AttentionState.FOCUSED:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return {
        "total_seconds": total_seconds,
        "focus_pct": counts[AttentionState.FOCUSED] / total * 100,
        "drift_pct": counts[AttentionState.DRIFTING] / total * 100,
        "absent_pct": counts[AttentionState.ABSENT] / total * 100,
        "longest_focus_streak_s": max_streak * seconds_per_frame,
        "drift_alerts": session.drift_events,
    }


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


# ------------------------------------------------------------------
# HTML renderer (self-contained, no external deps at runtime)
# ------------------------------------------------------------------

def _render_html(
    session: SessionRecord,
    events: Sequence[AttentionEvent],
    stats: dict,
) -> str:
    # Build timeline data for Plotly
    timestamps = [e.timestamp.isoformat() for e in events]
    states_numeric = {
        AttentionState.FOCUSED: 1,
        AttentionState.DRIFTING: 0,
        AttentionState.ABSENT: -1,
    }
    y_values = [states_numeric[e.state] for e in events]
    colors = [_STATE_COLORS[e.state] for e in events]
    hover_text = [
        f"{_STATE_LABELS[e.state]}<br>conf: {e.confidence:.2f}<br>{e.timestamp.strftime('%H:%M:%S')}"
        for e in events
    ]

    # Distribution bar data
    bar_labels = ["Focused", "Drifting", "Absent"]
    bar_values = [stats["focus_pct"], stats["drift_pct"], stats["absent_pct"]]
    bar_colors = [
        _STATE_COLORS[AttentionState.FOCUSED],
        _STATE_COLORS[AttentionState.DRIFTING],
        _STATE_COLORS[AttentionState.ABSENT],
    ]

    session_start = session.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    total_duration = _fmt_duration(stats["total_seconds"])
    longest_streak = _fmt_duration(stats["longest_focus_streak_s"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Attention Session Report — {session.session_id[:8]}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *{{ box-sizing: border-box; margin:0; padding:0; }}
    body{{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background:#0f1117; color:#e0e0e0; padding:32px 40px; }}
    h1{{ font-size:1.6rem; font-weight:600; color:#fff; margin-bottom:4px; }}
    .subtitle{{ font-size:0.85rem; color:#888; margin-bottom:32px; }}
    .stat-grid{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                 gap:16px; margin-bottom:32px; }}
    .stat-card{{ background:#1a1d27; border-radius:10px; padding:18px 20px;
                 border:1px solid #2a2d3a; }}
    .stat-label{{ font-size:0.75rem; color:#888; text-transform:uppercase;
                  letter-spacing:0.05em; margin-bottom:6px; }}
    .stat-value{{ font-size:1.6rem; font-weight:700; color:#fff; }}
    .stat-value.green{{ color:#3CB44B; }}
    .stat-value.blue{{ color:#4578E6; }}
    .stat-value.red{{ color:#E63946; }}
    .chart-wrap{{ background:#1a1d27; border-radius:10px; padding:20px;
                  border:1px solid #2a2d3a; margin-bottom:24px; }}
    .chart-title{{ font-size:0.9rem; font-weight:600; color:#aaa;
                   text-transform:uppercase; letter-spacing:0.06em; margin-bottom:12px; }}
    .privacy-note{{ font-size:0.75rem; color:#555; margin-top:32px; text-align:center; }}
    .privacy-note span{{ color:#777; }}
  </style>
</head>
<body>
  <h1>Attention Session Report</h1>
  <div class="subtitle">Session {session.session_id[:8]} &bull; {session_start} &bull; User: {session.user_id}</div>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-label">Total Duration</div>
      <div class="stat-value">{total_duration}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Focus Time</div>
      <div class="stat-value green">{stats["focus_pct"]:.1f}%</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Drift Time</div>
      <div class="stat-value blue">{stats["drift_pct"]:.1f}%</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Absent</div>
      <div class="stat-value red">{stats["absent_pct"]:.1f}%</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Longest Focus Streak</div>
      <div class="stat-value">{longest_streak}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Drift Alerts</div>
      <div class="stat-value">{stats["drift_alerts"]}</div>
    </div>
  </div>

  <div class="chart-wrap">
    <div class="chart-title">Attention Timeline</div>
    <div id="timeline"></div>
  </div>

  <div class="chart-wrap">
    <div class="chart-title">State Distribution</div>
    <div id="distribution"></div>
  </div>

  <div class="privacy-note">
    <span>Privacy guarantee:</span> No raw video or images were captured or stored.
    Only numerical features (head pose, gaze vector) and derived labels are persisted.
    All data is stored locally on your machine.
  </div>

  <script>
    // Timeline chart
    const timelineData = [{{
      x: {timestamps},
      y: {y_values},
      mode: 'markers',
      type: 'scatter',
      marker: {{
        color: {colors},
        size: 4,
        opacity: 0.85,
      }},
      text: {hover_text},
      hoverinfo: 'text',
    }}];
    const timelineLayout = {{
      paper_bgcolor: '#1a1d27',
      plot_bgcolor: '#1a1d27',
      font: {{ color: '#aaa', size: 11 }},
      height: 220,
      margin: {{ t:10, b:40, l:60, r:20 }},
      xaxis: {{ gridcolor:'#2a2d3a', title:'Time' }},
      yaxis: {{
        gridcolor:'#2a2d3a',
        tickvals: [-1, 0, 1],
        ticktext: ['Absent', 'Drifting', 'Focused'],
        range: [-1.6, 1.6],
      }},
    }};
    Plotly.newPlot('timeline', timelineData, timelineLayout, {{displayModeBar: false}});

    // Distribution chart
    const distData = [{{
      x: {bar_labels},
      y: {bar_values},
      type: 'bar',
      marker: {{ color: {bar_colors} }},
      text: {bar_values}.map(v => v.toFixed(1) + '%'),
      textposition: 'auto',
    }}];
    const distLayout = {{
      paper_bgcolor: '#1a1d27',
      plot_bgcolor: '#1a1d27',
      font: {{ color: '#aaa', size: 12 }},
      height: 200,
      margin: {{ t:10, b:40, l:50, r:20 }},
      yaxis: {{ gridcolor:'#2a2d3a', range:[0,100], title:'%' }},
      xaxis: {{ gridcolor:'#2a2d3a' }},
    }};
    Plotly.newPlot('distribution', distData, distLayout, {{displayModeBar: false}});
  </script>
</body>
</html>"""
