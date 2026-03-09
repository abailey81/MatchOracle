#!/usr/bin/env python3
"""
MatchOracle — Dashboard Generator
===================================
Reads dashboard_data.json and generates a self-contained HTML dashboard.
All values are dynamically sourced from the pipeline output — nothing is hardcoded.

Usage:
    python dashboard.py              # Generate and open dashboard
    python dashboard.py --no-open    # Generate without opening
    python dashboard.py --serve      # Start HTTP server for remote access
    python dashboard.py --serve --port 8080  # Custom port
"""

import json
import os
import sys
import webbrowser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = Path(__file__).resolve().parent / "dashboard.html"


def generate_dashboard():
    """Generate a self-contained HTML dashboard from dashboard_data.json."""
    json_path = DATA_DIR / "dashboard_data.json"
    if not json_path.exists():
        print("  ERROR: dashboard_data.json not found.")
        print("  Run the prediction pipeline first:")
        print("    python predict.py")
        print("  or:")
        print("    python models/run_pipeline.py")
        return None

    with open(json_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    model_results = data.get("model_results", {})
    team_ratings = data.get("team_ratings", [])
    upcoming = data.get("upcoming_predictions", [])
    feature_importance = data.get("feature_importance", {})
    dc_params = data.get("dc_params", {})
    backtest = data.get("backtest_results", [])
    calibration = data.get("calibration_curve", [])
    model_weights = data.get("model_weights", {})
    data_sources = data.get("data_sources", {})
    confidence_tiers = data.get("confidence_tiers", {})
    ensemble_method = data.get("ensemble_method", "Stacking")

    n_features = metrics.get('total_features', 0)
    n_learners = metrics.get('n_base_learners', 0)
    n_models = len(model_weights)
    test_season = metrics.get('test_season', '?')
    n_matches = metrics.get('total_matches', 0)
    n_seasons = metrics.get('train_seasons', 0)
    rps = metrics.get('final_rps', 0)
    acc = metrics.get('final_accuracy', 0)
    ece = metrics.get('final_ece', 0)
    logloss = metrics.get('final_logloss', 0)
    home_mae = metrics.get('home_goals_mae', 0)
    away_mae = metrics.get('away_goals_mae', 0)
    skill_baseline = metrics.get('rps_skill_vs_baseline', 0)
    skill_market = metrics.get('rps_skill_vs_market', 0)

    data_json = json.dumps(data, indent=None)

    # Build confidence tiers HTML from data
    tier_rows = ""
    for tier_name, tier_data in confidence_tiers.items():
        if tier_name == "ALL":
            continue
        t_acc = tier_data.get("accuracy", 0)
        t_n = tier_data.get("n", 0)
        color = "var(--green)" if t_acc > 0.65 else ("var(--accent)" if t_acc > 0.50 else "var(--amber)")
        tier_rows += f'<tr><td style="font-weight:600">{tier_name}</td><td class="mono" style="color:{color}">{t_acc:.1%}</td><td class="mono">{t_n}</td></tr>'

    # Build data sources HTML from data
    sources_html = ""
    for k, v in data_sources.items():
        label = k.replace("_", " ").title()
        sources_html += f'<div style="padding:10px 14px;background:var(--bg);border-radius:8px;border:1px solid var(--border)"><span style="font-size:12px;font-weight:700;color:var(--accent)">{label}:</span> <span style="font-size:12px;color:var(--text2)">{v}</span></div>'

    # Build model names for pipeline description
    model_names = ", ".join(model_weights.keys()) if model_weights else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MatchOracle Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg: #0a0e17; --card: #111827; --border: #1e293b;
  --accent: #22d3ee; --green: #10b981; --red: #ef4444;
  --amber: #f59e0b; --purple: #a78bfa; --blue: #3b82f6; --pink: #f472b6;
  --text: #f1f5f9; --text2: #94a3b8; --text3: #64748b;
}}
body {{ background: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, sans-serif; min-height: 100vh; }}
.mono {{ font-family: 'JetBrains Mono', monospace; }}
.container {{ max-width: 1000px; margin: 0 auto; padding: 24px; }}
.header {{ display: flex; align-items: center; gap: 12px; padding: 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px; flex-wrap: wrap; }}
.header h1 {{ font-size: 24px; font-weight: 800; background: linear-gradient(135deg, var(--text), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.header .sub {{ font-size: 11px; color: var(--text3); }}
.badges {{ margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{ padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; }}
.badge-green {{ background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: var(--green); }}
.badge-cyan {{ background: rgba(34,211,238,0.15); border: 1px solid rgba(34,211,238,0.3); color: var(--accent); }}
.badge-amber {{ background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); color: var(--amber); }}

.tabs {{ display: flex; gap: 0; margin-bottom: 24px; border-bottom: 1px solid var(--border); overflow-x: auto; }}
.tab {{ padding: 10px 16px; background: none; border: none; border-bottom: 2px solid transparent; color: var(--text3); font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap; }}
.tab.active {{ border-bottom-color: var(--accent); color: var(--accent); }}

.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 24px; }}
.metric-card {{ background: var(--card); border-radius: 12px; padding: 14px 18px; border: 1px solid var(--border); }}
.metric-card .label {{ font-size: 10px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }}
.metric-card .value {{ font-size: 24px; font-weight: 700; line-height: 1; }}
.metric-card .sub {{ font-size: 11px; color: var(--text2); margin-top: 4px; }}

.match-card {{ background: var(--card); border-radius: 14px; border: 1px solid var(--border); margin-bottom: 12px; overflow: hidden; cursor: pointer; transition: box-shadow 0.2s; }}
.match-card:hover {{ box-shadow: 0 0 0 1px rgba(34,211,238,0.25); }}
.match-card .top {{ padding: 16px 20px; }}
.match-card .teams {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }}
.match-card .team {{ font-size: 16px; font-weight: 700; flex: 1; }}
.match-card .team.fav {{ color: var(--accent); }}
.match-card .score-badge {{ padding: 6px 14px; background: rgba(34,211,238,0.1); border-radius: 8px; border: 1px solid rgba(34,211,238,0.2); font-size: 15px; font-weight: 800; color: var(--accent); }}
.match-card .conf {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
.match-card .xg {{ font-size: 11px; color: var(--text3); }}

.prob-bar {{ display: flex; border-radius: 6px; overflow: hidden; height: 32px; font-size: 11px; font-weight: 700; }}
.prob-bar .h {{ background: linear-gradient(135deg, var(--green), #059669); }}
.prob-bar .d {{ background: linear-gradient(135deg, var(--amber), #d97706); }}
.prob-bar .a {{ background: linear-gradient(135deg, var(--blue), #2563eb); }}
.prob-bar > div {{ display: flex; align-items: center; justify-content: center; color: #fff; transition: width 0.5s; }}
.prob-labels {{ display: flex; justify-content: space-between; margin-top: 8px; font-size: 11px; color: var(--text3); }}

.details {{ border-top: 1px solid var(--border); padding: 16px 20px; background: rgba(10,14,23,0.5); display: none; }}
.match-card.expanded .details {{ display: block; }}
.stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }}
.stat-box {{ background: var(--bg); border-radius: 8px; padding: 10px 14px; border: 1px solid var(--border); }}
.stat-box .label {{ font-size: 10px; color: var(--text3); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
.stat-box .val {{ font-size: 18px; font-weight: 700; }}
.scorelines {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.scoreline {{ padding: 6px 10px; border-radius: 6px; background: var(--bg); border: 1px solid var(--border); font-size: 12px; }}
.scoreline.top {{ background: rgba(34,211,238,0.1); border-color: rgba(34,211,238,0.3); }}
.scoreline .s {{ font-weight: 700; }}
.scoreline .p {{ color: var(--text3); margin-left: 6px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ padding: 12px 14px; text-align: left; color: var(--text3); font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--border); }}
td {{ padding: 10px 14px; border-bottom: 1px solid rgba(30,41,59,0.3); }}
tr:nth-child(even) {{ background: rgba(10,14,23,0.3); }}

.section {{ background: var(--card); border-radius: 12px; padding: 20px; border: 1px solid var(--border); margin-bottom: 20px; }}
.section-title {{ font-size: 12px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }}

.bar-chart {{ display: flex; flex-direction: column; gap: 6px; }}
.bar-row {{ display: flex; align-items: center; gap: 10px; }}
.bar-row .name {{ width: 160px; font-size: 12px; color: var(--text2); text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bar-row .bar {{ height: 22px; border-radius: 4px; transition: width 0.5s; display: flex; align-items: center; padding-left: 6px; font-size: 10px; color: #fff; font-weight: 600; min-width: 2px; }}

.pipeline {{ display: flex; flex-direction: column; gap: 0; }}
.pipe-step {{ display: flex; gap: 14px; align-items: flex-start; }}
.pipe-dot {{ display: flex; flex-direction: column; align-items: center; min-width: 20px; }}
.pipe-dot .dot {{ width: 12px; height: 12px; border-radius: 50%; border: 2px solid var(--bg); z-index: 1; }}
.pipe-dot .line {{ width: 2px; height: 36px; opacity: 0.3; }}
.pipe-step .info {{ padding-bottom: 16px; }}
.pipe-step .info .name {{ font-size: 14px; font-weight: 700; }}
.pipe-step .info .desc {{ font-size: 12px; color: var(--text2); margin-top: 2px; }}

.filter-bar {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
.filter-btn {{ padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; border: 1px solid var(--border); background: var(--card); color: var(--text2); }}
.filter-btn.active {{ background: rgba(34,211,238,0.15); border-color: rgba(34,211,238,0.3); color: var(--accent); }}

.model-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }}
.model-mini {{ background: var(--bg); border-radius: 6px; padding: 8px 12px; border: 1px solid var(--border); font-size: 11px; }}
.model-mini .mname {{ font-weight: 600; color: var(--text2); }}
.model-mini .mvals {{ display: flex; gap: 8px; margin-top: 4px; }}
.model-mini .mv {{ color: var(--text3); }}

.backtest-table {{ overflow-x: auto; }}
@media (max-width: 640px) {{
  .container {{ padding: 12px; }}
  .match-card .team {{ font-size: 14px; }}
  .metrics {{ grid-template-columns: repeat(2, 1fr); }}
  .stats-grid {{ grid-template-columns: 1fr; }}
  .model-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="header">
  <div>
    <h1>MatchOracle</h1>
    <div class="sub">5-Layer Deep Ensemble | {n_features} Features | {n_learners} Base Learners | {n_seasons} Seasons | {n_matches:,} Matches</div>
  </div>
  <div class="badges">
    <span class="badge badge-green mono">RPS {rps:.4f}</span>
    <span class="badge badge-cyan mono">{acc:.1%} ACC</span>
    <span class="badge badge-amber mono">vs Mkt {skill_market:+.1%}</span>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <button class="tab active" onclick="switchTab('predictions', this)">Predictions</button>
  <button class="tab" onclick="switchTab('models', this)">Models</button>
  <button class="tab" onclick="switchTab('ratings', this)">Ratings</button>
  <button class="tab" onclick="switchTab('features', this)">Features</button>
  <button class="tab" onclick="switchTab('backtest', this)">Backtest</button>
  <button class="tab" onclick="switchTab('calibration', this)">Calibration</button>
  <button class="tab" onclick="switchTab('system', this)">System</button>
</div>

<!-- Predictions Tab -->
<div id="tab-predictions" class="tab-content active">
  <h3 style="font-size:18px;font-weight:700;margin-bottom:4px">Match Predictions</h3>
  <p style="font-size:13px;color:var(--text2);margin-bottom:12px">
    {len(upcoming)} upcoming EPL fixture{'s' if len(upcoming) != 1 else ''} — calibrated {ensemble_method.lower()} ensemble with {n_learners} base learners + Dixon-Coles.
  </p>
  <div class="filter-bar" id="conf-filters"></div>
  <div id="match-cards"></div>
  {'<p style="color:var(--text3);font-size:13px;margin-top:16px">No upcoming predictions available. Run: python predict.py</p>' if not upcoming else ''}
</div>

<!-- Models Tab -->
<div id="tab-models" class="tab-content">
  <h3 style="font-size:18px;font-weight:700;margin-bottom:16px">Model Performance ({test_season} Test Season)</h3>
  <div class="metrics">
    <div class="metric-card"><div class="label">Best RPS</div><div class="value mono" style="color:var(--green)">{rps:.4f}</div><div class="sub">Calibrated Ensemble</div></div>
    <div class="metric-card"><div class="label">Accuracy</div><div class="value mono" style="color:var(--accent)">{acc:.1%}</div><div class="sub">{test_season}</div></div>
    <div class="metric-card"><div class="label">ECE</div><div class="value mono" style="color:var(--purple)">{ece:.4f}</div><div class="sub">Calibration error</div></div>
    <div class="metric-card"><div class="label">Log Loss</div><div class="value mono" style="color:var(--blue)">{logloss:.4f}</div><div class="sub">Information</div></div>
    <div class="metric-card"><div class="label">vs Baseline</div><div class="value mono" style="color:var(--green)">{skill_baseline:+.1%}</div><div class="sub">RPS skill</div></div>
    <div class="metric-card"><div class="label">vs Market</div><div class="value mono" style="color:var(--green)">{skill_market:+.1%}</div><div class="sub">RPS skill</div></div>
    <div class="metric-card"><div class="label">Goals MAE</div><div class="value mono" style="color:var(--amber)">{(home_mae + away_mae) / 2:.3f}</div><div class="sub">Home {home_mae:.3f} Away {away_mae:.3f}</div></div>
    <div class="metric-card"><div class="label">Ensemble</div><div class="value mono" style="color:var(--pink)">{ensemble_method}</div><div class="sub">{n_learners} base + DC</div></div>
  </div>
  <div class="section">
    <div class="section-title">Ranked Probability Score by Model (lower = better)</div>
    <div class="bar-chart" id="rps-chart"></div>
  </div>
  <div class="section">
    <div class="section-title">Ensemble Weights (Inverse-RPS)</div>
    <div class="bar-chart" id="weight-chart"></div>
  </div>
  <div class="section">
    <div class="section-title">Confidence Tier Accuracy (Historical)</div>
    <table>
      <thead><tr><th>Tier</th><th>Accuracy</th><th>Matches</th></tr></thead>
      <tbody>{tier_rows}</tbody>
    </table>
  </div>
</div>

<!-- Ratings Tab -->
<div id="tab-ratings" class="tab-content">
  <h3 style="font-size:18px;font-weight:700;margin-bottom:4px">Dixon-Coles Team Ratings</h3>
  <p style="font-size:13px;color:var(--text2);margin-bottom:16px">
    Fitted on {n_seasons} seasons. Home advantage: {dc_params.get('home_advantage', 0):.4f} | Rho: {dc_params.get('rho', 0):.4f}
  </p>
  <div class="section" style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>#</th><th>Team</th><th>Attack</th><th>Defense</th><th>xG (Home)</th><th>xG (Away)</th><th>Overall</th>
      </tr></thead>
      <tbody id="ratings-table"></tbody>
    </table>
  </div>
</div>

<!-- Features Tab -->
<div id="tab-features" class="tab-content">
  <h3 style="font-size:18px;font-weight:700;margin-bottom:4px">Feature Importance</h3>
  <p style="font-size:13px;color:var(--text2);margin-bottom:20px">
    Top 30 features from {n_features} engineered features, ranked by HGB model importance.
  </p>
  <div class="section">
    <div class="bar-chart" id="fi-chart"></div>
    {'<p style="color:var(--text3);font-size:13px">Feature importance data not available. Run a full training pipeline (not cached) to generate.</p>' if not feature_importance else ''}
  </div>
</div>

<!-- Backtest Tab -->
<div id="tab-backtest" class="tab-content">
  <h3 style="font-size:18px;font-weight:700;margin-bottom:4px">Walk-Forward Backtest</h3>
  <p style="font-size:13px;color:var(--text2);margin-bottom:16px">
    Season-by-season out-of-sample evaluation. Model retrained on all prior data for each test season.
  </p>
  <div class="section backtest-table">
    <table>
      <thead><tr><th>Season</th><th>Matches</th><th>Accuracy</th><th>RPS</th><th>Market Acc</th><th>Market RPS</th><th>vs Market</th><th>High-Conf Acc</th></tr></thead>
      <tbody id="backtest-table"></tbody>
    </table>
    {'<p style="color:var(--text3);font-size:13px;margin-top:12px">No backtest data available.</p>' if not backtest else ''}
  </div>
</div>

<!-- Calibration Tab -->
<div id="tab-calibration" class="tab-content">
  <h3 style="font-size:18px;font-weight:700;margin-bottom:4px">Probability Calibration</h3>
  <p style="font-size:13px;color:var(--text2);margin-bottom:16px">
    How well predicted probabilities match actual outcomes. ECE: {ece:.4f}
  </p>
  <div class="section">
    <div id="cal-chart" style="position:relative;height:300px;margin-bottom:12px"></div>
    <div style="display:flex;gap:16px;font-size:11px;color:var(--text3);justify-content:center">
      <span><span style="color:var(--green)">&#9679;</span> Home</span>
      <span><span style="color:var(--amber)">&#9679;</span> Draw</span>
      <span><span style="color:var(--blue)">&#9679;</span> Away</span>
      <span style="border-left:1px solid var(--border);padding-left:16px">&#9473; Perfect calibration</span>
    </div>
  </div>
  {'<p style="color:var(--text3);font-size:13px">No calibration data available.</p>' if not calibration else ''}
</div>

<!-- System Tab -->
<div id="tab-system" class="tab-content">
  <h3 style="font-size:18px;font-weight:700;margin-bottom:16px">System Architecture</h3>
  <div class="metrics">
    <div class="metric-card"><div class="label">Features</div><div class="value mono" style="color:var(--accent)">{n_features}</div><div class="sub">Engineered</div></div>
    <div class="metric-card"><div class="label">Seasons</div><div class="value mono" style="color:var(--blue)">{n_seasons}</div><div class="sub">Training</div></div>
    <div class="metric-card"><div class="label">Matches</div><div class="value mono" style="color:var(--green)">{n_matches:,}</div><div class="sub">Total dataset</div></div>
    <div class="metric-card"><div class="label">Base Models</div><div class="value mono" style="color:var(--purple)">{n_learners}</div><div class="sub">+ Dixon-Coles</div></div>
    <div class="metric-card"><div class="label">Home Adv</div><div class="value mono" style="color:var(--amber)">{dc_params.get('home_advantage', 0):.4f}</div><div class="sub">Dixon-Coles</div></div>
    <div class="metric-card"><div class="label">DC Rho</div><div class="value mono" style="color:var(--pink)">{dc_params.get('rho', 0):.4f}</div><div class="sub">Goal correlation</div></div>
  </div>
  <div class="section">
    <div class="section-title">5-Layer Pipeline Architecture</div>
    <div class="pipeline">
      <div class="pipe-step"><div class="pipe-dot"><div class="dot" style="background:var(--blue)"></div><div class="line" style="background:var(--blue)"></div></div><div class="info"><div class="name" style="color:var(--blue)">Layer 0: Data Acquisition ({len(data_sources)} sources)</div><div class="desc">{', '.join(k.replace('_', ' ').title() for k in data_sources.keys())}</div></div></div>
      <div class="pipe-step"><div class="pipe-dot"><div class="dot" style="background:var(--accent)"></div><div class="line" style="background:var(--accent)"></div></div><div class="info"><div class="name" style="color:var(--accent)">Layer 1: Feature Engineering ({n_features} features)</div><div class="desc">Elo, Pi-ratings, rolling form, H2H, market odds, sentiment, injuries, momentum, weather, referee, table position</div></div></div>
      <div class="pipe-step"><div class="pipe-dot"><div class="dot" style="background:var(--green)"></div><div class="line" style="background:var(--green)"></div></div><div class="info"><div class="name" style="color:var(--green)">Layer 2: Base Learners ({n_models} models)</div><div class="desc">{model_names}</div></div></div>
      <div class="pipe-step"><div class="pipe-dot"><div class="dot" style="background:var(--purple)"></div><div class="line" style="background:var(--purple)"></div></div><div class="info"><div class="name" style="color:var(--purple)">Layer 3: {ensemble_method} Meta-Learner</div><div class="desc">Meta-LR + Meta-HGB on OOF predictions, binary classifier augmentation, inverse-RPS weighting</div></div></div>
      <div class="pipe-step"><div class="pipe-dot"><div class="dot" style="background:var(--pink)"></div></div><div class="info"><div class="name" style="color:var(--pink)">Layer 4: Calibration + Confidence</div><div class="desc">Isotonic calibration, confidence tiers (ELITE/VERY HIGH/HIGH/MEDIUM/LOW)</div></div></div>
    </div>
  </div>
  <div class="section">
    <div class="section-title">Data Sources</div>
    <div style="display:grid;gap:8px">{sources_html}</div>
  </div>
</div>

</div>

<script>
const DATA = {data_json};
const pct = v => (v * 100).toFixed(1) + '%';

function switchTab(name, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}

// Confidence filter
let activeFilter = 'ALL';
function renderFilters() {{
  const bar = document.getElementById('conf-filters');
  if (!DATA.upcoming_predictions || !DATA.upcoming_predictions.length) return;
  const tiers = ['ALL', 'ELITE', 'VERY HIGH', 'HIGH', 'MEDIUM', 'LOW'];
  const counts = {{}};
  DATA.upcoming_predictions.forEach(m => {{ counts[m.confidence] = (counts[m.confidence] || 0) + 1; }});
  counts['ALL'] = DATA.upcoming_predictions.length;
  tiers.forEach(t => {{
    if (!counts[t] && t !== 'ALL') return;
    const btn = document.createElement('button');
    btn.className = 'filter-btn' + (t === 'ALL' ? ' active' : '');
    btn.textContent = t + ' (' + (counts[t] || 0) + ')';
    btn.onclick = () => {{
      activeFilter = t;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderMatches();
    }};
    bar.appendChild(btn);
  }});
}}

function renderMatches() {{
  const container = document.getElementById('match-cards');
  container.innerHTML = '';
  if (!DATA.upcoming_predictions || !DATA.upcoming_predictions.length) return;

  const filtered = activeFilter === 'ALL' ? DATA.upcoming_predictions : DATA.upcoming_predictions.filter(m => m.confidence === activeFilter);

  filtered.forEach((m, i) => {{
    const conf = m.confidence || 'LOW';
    const confColor = conf === 'ELITE' ? 'var(--green)' : conf === 'VERY HIGH' ? 'var(--accent)' : conf === 'HIGH' ? 'var(--blue)' : conf === 'MEDIUM' ? 'var(--amber)' : 'var(--text3)';
    const maxP = Math.max(m.home_win, m.draw, m.away_win);
    const fav = m.home_win >= m.away_win && m.home_win >= m.draw ? 'home' : (m.away_win > m.home_win && m.away_win >= m.draw ? 'away' : 'draw');
    const total = m.home_win + m.draw + m.away_win;
    const hPct = m.home_win / total * 100;
    const dPct = m.draw / total * 100;
    const aPct = m.away_win / total * 100;

    let scoresHtml = '';
    if (m.top_scorelines) {{
      m.top_scorelines.slice(0, 10).forEach((s, j) => {{
        scoresHtml += `<div class="scoreline ${{j===0?'top':''}}"><span class="s">${{s.score}}</span><span class="p">${{pct(s.prob)}}</span></div>`;
      }});
    }}

    // Model breakdown
    let breakdownHtml = '';
    if (m.model_breakdown) {{
      const models = Object.entries(m.model_breakdown);
      breakdownHtml = '<div class="model-grid">';
      models.forEach(([name, probs]) => {{
        const mFav = probs.H >= probs.A && probs.H >= probs.D ? 'H' : (probs.A > probs.H && probs.A >= probs.D ? 'A' : 'D');
        breakdownHtml += `<div class="model-mini"><div class="mname">${{name}}</div><div class="mvals"><span class="mv" style="color:${{mFav==='H'?'var(--green)':'var(--text3)'}}">H ${{(probs.H*100).toFixed(0)}}%</span><span class="mv" style="color:${{mFav==='D'?'var(--amber)':'var(--text3)'}}">D ${{(probs.D*100).toFixed(0)}}%</span><span class="mv" style="color:${{mFav==='A'?'var(--blue)':'var(--text3)'}}">A ${{(probs.A*100).toFixed(0)}}%</span></div></div>`;
      }});
      breakdownHtml += '</div>';
    }}

    // Sentiment
    let sentHtml = '';
    if (m.live_sentiment) {{
      const hs = m.live_sentiment.home || {{}};
      const as_ = m.live_sentiment.away || {{}};
      const sentColor = v => v > 0.1 ? 'var(--green)' : v < -0.1 ? 'var(--red)' : 'var(--text2)';
      sentHtml = `
        <div style="font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin:14px 0 8px">Live Sentiment</div>
        <div class="stats-grid">
          <div class="stat-box"><div class="label">${{m.home_team}} Sentiment</div><div class="val mono" style="color:${{sentColor(hs.sentiment||0)}}">${{(hs.sentiment||0) > 0 ? '+' : ''}}${{((hs.sentiment||0)*100).toFixed(0)}}%</div></div>
          <div class="stat-box"><div class="label">${{m.away_team}} Sentiment</div><div class="val mono" style="color:${{sentColor(as_.sentiment||0)}}">${{(as_.sentiment||0) > 0 ? '+' : ''}}${{((as_.sentiment||0)*100).toFixed(0)}}%</div></div>
        </div>`;
    }}

    const dateStr = m.date ? `<span style="font-size:11px;color:var(--text3)">${{m.date}}</span>` : '';

    const card = document.createElement('div');
    card.className = 'match-card';
    card.onclick = (e) => {{ if (!e.target.closest('.filter-btn')) card.classList.toggle('expanded'); }};
    card.innerHTML = `
      <div class="top">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:${{confColor}}"></div>
            <span class="conf" style="color:${{confColor}}">${{conf}}</span>
            ${{dateStr}}
          </div>
          <div style="font-size:11px;color:var(--text3)">Margin: ${{pct(m.prediction_margin || 0)}}</div>
        </div>
        <div class="teams">
          <div class="team${{fav==='home'?' fav':''}}" style="text-align:left">${{m.home_team}}</div>
          <div class="score-badge mono">${{m.predicted_score || '?-?'}}</div>
          <div class="team${{fav==='away'?' fav':''}}" style="text-align:right">${{m.away_team}}</div>
        </div>
        <div class="prob-bar">
          <div class="h" style="width:${{hPct}}%;${{hPct>10?'':'font-size:0'}}">${{hPct>10?pct(m.home_win):''}}</div>
          <div class="d" style="width:${{dPct}}%;${{dPct>10?'':'font-size:0'}}">${{dPct>10?pct(m.draw):''}}</div>
          <div class="a" style="width:${{aPct}}%;${{aPct>10?'':'font-size:0'}}">${{aPct>10?pct(m.away_win):''}}</div>
        </div>
        <div class="prob-labels">
          <span style="color:var(--green)">Home ${{pct(m.home_win)}}</span>
          <span style="color:var(--amber)">Draw ${{pct(m.draw)}}</span>
          <span style="color:var(--blue)">Away ${{pct(m.away_win)}}</span>
        </div>
      </div>
      <div class="details">
        <div class="stats-grid">
          <div class="stat-box"><div class="label">Over 2.5 Goals</div><div class="val mono" style="color:${{(m.over_2_5||0)>0.55?'var(--green)':'var(--text)'}}">${{pct(m.over_2_5||0)}}</div></div>
          <div class="stat-box"><div class="label">Both Teams Score</div><div class="val mono" style="color:${{(m.btts||0)>0.55?'var(--green)':'var(--text)'}}">${{pct(m.btts||0)}}</div></div>
          <div class="stat-box"><div class="label">xG Home (Dixon-Coles)</div><div class="val mono" style="color:var(--accent)">${{(m.xg_home||0).toFixed(2)}}</div></div>
          <div class="stat-box"><div class="label">xG Away (Dixon-Coles)</div><div class="val mono" style="color:var(--text2)">${{(m.xg_away||0).toFixed(2)}}</div></div>
          <div class="stat-box"><div class="label">xG Home (ML)</div><div class="val mono" style="color:var(--accent)">${{(m.ml_xg_home||0).toFixed(2)}}</div></div>
          <div class="stat-box"><div class="label">xG Away (ML)</div><div class="val mono" style="color:var(--text2)">${{(m.ml_xg_away||0).toFixed(2)}}</div></div>
        </div>
        <div style="font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Scoreline Distribution</div>
        <div class="scorelines">${{scoresHtml}}</div>
        ${{sentHtml}}
        <div style="font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin:14px 0 8px">Individual Model Predictions</div>
        ${{breakdownHtml}}
      </div>`;
    container.appendChild(card);
  }});
}}

function renderRPSChart() {{
  const chart = document.getElementById('rps-chart');
  if (!DATA.model_results) return;
  const models = Object.entries(DATA.model_results)
    .map(([n, m]) => ({{ name: n, rps: m.rps, acc: m.accuracy, ll: m.log_loss }}))
    .sort((a, b) => a.rps - b.rps);
  const maxRps = Math.max(...models.map(m => m.rps));
  models.forEach((m, i) => {{
    const color = i < 2 ? 'var(--green)' : i < 5 ? 'var(--accent)' : i < 10 ? 'var(--blue)' : 'var(--text3)';
    const row = document.createElement('div');
    row.className = 'bar-row';
    const accStr = m.acc ? ` | ${{(m.acc*100).toFixed(1)}}%` : '';
    row.innerHTML = `<div class="name">${{m.name}}</div><div class="bar" style="width:${{(m.rps/maxRps*100).toFixed(1)}}%;background:${{color}}">${{m.rps.toFixed(4)}}${{accStr}}</div>`;
    chart.appendChild(row);
  }});
}}

function renderWeightChart() {{
  const chart = document.getElementById('weight-chart');
  if (!DATA.model_weights) return;
  const weights = Object.entries(DATA.model_weights).sort((a, b) => b[1] - a[1]);
  const maxW = Math.max(...weights.map(w => w[1]));
  const colors = ['var(--green)', 'var(--accent)', 'var(--blue)', 'var(--purple)', 'var(--amber)', 'var(--pink)'];
  weights.forEach(([name, w], i) => {{
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<div class="name">${{name}}</div><div class="bar" style="width:${{(w/maxW*100).toFixed(1)}}%;background:${{colors[i % colors.length] || 'var(--text3)'}}">${{(w*100).toFixed(1)}}%</div>`;
    chart.appendChild(row);
  }});
}}

function renderRatings() {{
  const tbody = document.getElementById('ratings-table');
  if (!DATA.team_ratings) return;
  DATA.team_ratings.forEach((t, i) => {{
    const tr = document.createElement('tr');
    const atkColor = t.attack > 0 ? 'var(--green)' : 'var(--red)';
    const defColor = t.defense < 0 ? 'var(--green)' : 'var(--red)';
    const overall = (t.attack - t.defense).toFixed(3);
    const ovrColor = overall > 0 ? 'var(--green)' : 'var(--red)';
    tr.innerHTML = `
      <td class="mono" style="color:var(--text3);font-weight:700">${{i+1}}</td>
      <td style="font-weight:600">${{t.team}}</td>
      <td class="mono" style="color:${{atkColor}}">${{t.attack > 0 ? '+' : ''}}${{t.attack.toFixed(3)}}</td>
      <td class="mono" style="color:${{defColor}}">${{t.defense > 0 ? '+' : ''}}${{t.defense.toFixed(3)}}</td>
      <td class="mono" style="color:var(--accent)">${{t.expected_home_goals.toFixed(2)}}</td>
      <td class="mono" style="color:var(--text2)">${{t.expected_away_goals.toFixed(2)}}</td>
      <td class="mono" style="color:${{ovrColor}};font-weight:700">${{overall > 0 ? '+' : ''}}${{overall}}</td>`;
    tbody.appendChild(tr);
  }});
}}

function renderFeatures() {{
  const chart = document.getElementById('fi-chart');
  if (!DATA.feature_importance || !Object.keys(DATA.feature_importance).length) return;
  const features = Object.entries(DATA.feature_importance).slice(0, 30);
  const maxImp = Math.max(...features.map(f => f[1]));
  features.forEach(([name, imp], i) => {{
    const row = document.createElement('div');
    row.className = 'bar-row';
    const color = i < 3 ? 'var(--green)' : i < 8 ? 'var(--accent)' : i < 15 ? 'var(--blue)' : 'var(--text3)';
    const label = name.replace(/_/g, ' ');
    row.innerHTML = `<div class="name" title="${{name}}">${{label}}</div><div class="bar" style="width:${{(imp/maxImp*100).toFixed(1)}}%;background:${{color}}">${{imp.toFixed(4)}}</div>`;
    chart.appendChild(row);
  }});
}}

function renderBacktest() {{
  const tbody = document.getElementById('backtest-table');
  if (!DATA.backtest_results || !DATA.backtest_results.length) return;
  DATA.backtest_results.forEach(bt => {{
    const tr = document.createElement('tr');
    const vsColor = bt.accuracy > (bt.market_accuracy || 0) ? 'var(--green)' : 'var(--red)';
    const diff = bt.market_accuracy ? ((bt.accuracy - bt.market_accuracy) * 100).toFixed(1) : '—';
    const hcAcc = bt.high_conf_accuracy ? (bt.high_conf_accuracy * 100).toFixed(1) + '%' : '—';
    const hcN = bt.high_conf_n ? ` (${{bt.high_conf_n}})` : '';
    tr.innerHTML = `
      <td style="font-weight:600">${{bt.test_season || bt.season || '?'}}</td>
      <td class="mono">${{bt.n_matches || 380}}</td>
      <td class="mono" style="color:var(--accent)">${{(bt.accuracy*100).toFixed(1)}}%</td>
      <td class="mono">${{bt.rps ? bt.rps.toFixed(4) : '—'}}</td>
      <td class="mono">${{bt.market_accuracy ? (bt.market_accuracy*100).toFixed(1)+'%' : '—'}}</td>
      <td class="mono">${{bt.market_rps ? bt.market_rps.toFixed(4) : '—'}}</td>
      <td class="mono" style="color:${{vsColor}}">${{diff !== '—' ? (diff > 0 ? '+' : '') + diff + '%' : '—'}}</td>
      <td class="mono">${{hcAcc}}${{hcN}}</td>`;
    tbody.appendChild(tr);
  }});
}}

function renderCalibration() {{
  const container = document.getElementById('cal-chart');
  if (!DATA.calibration_curve || !DATA.calibration_curve.length) {{
    container.innerHTML = '<p style="color:var(--text3);font-size:13px;text-align:center;padding-top:120px">No calibration data</p>';
    return;
  }}
  const w = container.clientWidth || 600;
  const h = 300;
  const pad = 40;
  let svg = `<svg width="${{w}}" height="${{h}}" style="width:100%;height:100%">`;
  // Grid
  for (let i = 0; i <= 10; i++) {{
    const x = pad + (w - 2*pad) * i / 10;
    const y = pad + (h - 2*pad) * (1 - i/10);
    svg += `<line x1="${{pad}}" y1="${{y}}" x2="${{w-pad}}" y2="${{y}}" stroke="var(--border)" stroke-width="0.5"/>`;
    svg += `<text x="${{x}}" y="${{h-10}}" fill="var(--text3)" font-size="9" text-anchor="middle">${{(i*10)}}%</text>`;
    svg += `<text x="${{pad-6}}" y="${{y+3}}" fill="var(--text3)" font-size="9" text-anchor="end">${{(i*10)}}%</text>`;
  }}
  // Perfect line
  svg += `<line x1="${{pad}}" y1="${{h-pad}}" x2="${{w-pad}}" y2="${{pad}}" stroke="var(--text3)" stroke-width="1" stroke-dasharray="4"/>`;
  // Points
  const colors = {{'Home': 'var(--green)', 'Draw': 'var(--amber)', 'Away': 'var(--blue)'}};
  DATA.calibration_curve.forEach(pt => {{
    const cx = pad + (w - 2*pad) * pt.predicted;
    const cy = pad + (h - 2*pad) * (1 - pt.actual);
    const r = Math.min(8, Math.max(3, pt.count / 10));
    const col = colors[pt.outcome] || 'var(--text3)';
    svg += `<circle cx="${{cx}}" cy="${{cy}}" r="${{r}}" fill="${{col}}" opacity="0.8"><title>${{pt.outcome}}: pred=${{(pt.predicted*100).toFixed(0)}}% actual=${{(pt.actual*100).toFixed(0)}}% (n=${{pt.count}})</title></circle>`;
  }});
  svg += '</svg>';
  container.innerHTML = svg;
}}

// Initialize
renderFilters();
renderMatches();
renderRPSChart();
renderWeightChart();
renderRatings();
renderFeatures();
renderBacktest();
renderCalibration();
</script>
</body>
</html>"""

    OUTPUT_PATH.write_text(html)
    return OUTPUT_PATH


def serve_dashboard(port=8000):
    """Start an HTTP server to share the dashboard over the network."""
    import http.server
    import socket
    import threading

    dashboard_dir = str(OUTPUT_PATH.parent)

    # Get local IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dashboard_dir, **kwargs)
        def log_message(self, format, *args):
            print(f"  [{self.client_address[0]}] {args[0]}")

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)

    print(f"\n  MatchOracle Dashboard Server")
    print(f"  {'=' * 50}")
    print(f"  Local:   http://localhost:{port}/dashboard.html")
    print(f"  Network: http://{local_ip}:{port}/dashboard.html")
    print(f"  {'=' * 50}")
    print(f"  Share the Network URL with others on the same network.")
    print(f"  For internet access, use: ngrok http {port}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MatchOracle — Dashboard Generator")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server for remote access")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP server (default: 8000)")
    args = parser.parse_args()

    print("\n  Generating dashboard ...")
    path = generate_dashboard()

    if path:
        print(f"  Dashboard saved to: {path}")
        if args.serve:
            serve_dashboard(args.port)
        elif not args.no_open:
            print("  Opening in browser ...")
            webbrowser.open(f"file://{path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
