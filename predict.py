#!/usr/bin/env python3
"""
MatchOracle — Interactive Match Predictor CLI
===============================================
A rich, interactive terminal UI for predicting EPL matches.

Features:
  - Arrow-key / keyboard-driven fixture selection with checkboxes
  - Real-time fixture fetching from 3 sources
  - Full 5-layer deep ensemble pipeline execution in-process
  - Rich prediction output with accuracy, confidence, model breakdown
  - Auto-generated HTML dashboard

Usage:
    python predict.py
    python predict.py --fdo-key KEY --apif-key KEY --news-key KEY
"""

import argparse
import io
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))


def load_env() -> None:
    """Load API keys from .env file if it exists."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value


load_env()

# ─────────────────────────────────────────────────────────────────────
# Imports (with graceful fallbacks)
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich.rule import Rule
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("ERROR: 'rich' is required. Install: pip install rich")
    sys.exit(1)

from data.generator import normalise_team

console = Console()

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


# ─────────────────────────────────────────────────────────────────────
# Interactive fixture selector (keyboard-driven)
# ─────────────────────────────────────────────────────────────────────
class FixtureSelector:
    """Arrow-key driven fixture selector with toggle checkboxes."""

    def __init__(self, fixtures_df, max_show=15):
        self.fixtures = fixtures_df.head(max_show).reset_index(drop=True)
        self.n = len(self.fixtures)
        self.cursor = 0
        self.selected = set()

    def _render(self):
        table = Table(
            title="[bold cyan]UPCOMING EPL FIXTURES[/bold cyan]",
            box=box.HEAVY_EDGE,
            title_style="bold cyan",
            border_style="cyan",
            show_lines=False,
            pad_edge=True,
            padding=(0, 1),
        )
        table.add_column("", width=3, justify="center")
        table.add_column("#", width=3, justify="right", style="dim")
        table.add_column("Date", width=14)
        table.add_column("KO", width=6, style="dim")
        table.add_column("Home", width=20, justify="right")
        table.add_column("", width=3, justify="center")
        table.add_column("Away", width=20)

        for i in range(self.n):
            row = self.fixtures.iloc[i]
            is_cursor = (i == self.cursor)
            is_selected = (i in self.selected)

            # Checkbox
            if is_selected:
                check = "[green bold][X][/green bold]"
            else:
                check = "[dim][ ][/dim]"

            # Highlight current row
            if is_cursor:
                style = "bold white on grey23"
                num = f"[cyan bold]{i+1}[/cyan bold]"
                date_str = row["date"].strftime("%a %d %b") if pd.notna(row["date"]) else "TBD"
                time_str = row.get("time", "") or ""
                home = f"[bold white]{row['home_team']}[/bold white]"
                vs = "[cyan]vs[/cyan]"
                away = f"[bold white]{row['away_team']}[/bold white]"
            else:
                style = ""
                num = f"[dim]{i+1}[/dim]"
                date_str = row["date"].strftime("%a %d %b") if pd.notna(row["date"]) else "TBD"
                time_str = row.get("time", "") or ""
                home = row["home_team"]
                vs = "[dim]vs[/dim]"
                away = row["away_team"]

            table.add_row(check, num, date_str, time_str, home, vs, away, style=style)

        return table

    def _controls_panel(self):
        controls = (
            "[cyan bold]CONTROLS[/cyan bold]\n"
            "[green]SPACE[/green]  toggle match    "
            "[green]a[/green]  select all    "
            "[green]n[/green]  select none\n"
            "[green]ENTER[/green]  confirm         "
            "[green]q[/green]  quit"
        )
        n_sel = len(self.selected)
        status = f"[bold green]{n_sel} match{'es' if n_sel != 1 else ''} selected[/bold green]" if n_sel > 0 else "[dim]No matches selected[/dim]"
        return Panel(
            f"{controls}\n\n{status}",
            border_style="dim",
            padding=(0, 2),
        )

    def run(self):
        """Run the interactive selector. Returns list of indices or None."""
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)

            while True:
                # Clear and render
                sys.stdout.write("\033[2J\033[H")  # clear screen
                sys.stdout.flush()

                # Render with rich
                output = io.StringIO()
                temp_console = Console(file=output, force_terminal=True, width=console.width)
                temp_console.print()
                temp_console.print(
                    Align.center(
                        Panel(
                            "[bold white]MATCHORACLE[/bold white]  [dim]|[/dim]  "
                            "[cyan]5-Layer Deep Ensemble[/cyan]  [dim]|[/dim]  "
                            "[cyan]376+ Features[/cyan]  [dim]|[/dim]  "
                            "[cyan]13 Base Learners[/cyan]",
                            border_style="cyan",
                            padding=(0, 2),
                        )
                    )
                )
                temp_console.print()
                temp_console.print(Align.center(self._render()))
                temp_console.print()
                temp_console.print(Align.center(self._controls_panel()))

                sys.stdout.write(output.getvalue())
                sys.stdout.flush()

                # Read key
                ch = sys.stdin.read(1)

                if ch == '\x1b':  # escape sequence
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A':  # up
                            self.cursor = (self.cursor - 1) % self.n
                        elif ch3 == 'B':  # down
                            self.cursor = (self.cursor + 1) % self.n
                    elif ch2 == '\x1b':  # double escape = quit
                        return None
                elif ch == ' ':  # toggle
                    if self.cursor in self.selected:
                        self.selected.discard(self.cursor)
                    else:
                        self.selected.add(self.cursor)
                    self.cursor = min(self.cursor + 1, self.n - 1)
                elif ch == 'a':  # select all
                    self.selected = set(range(self.n))
                elif ch == 'n':  # select none
                    self.selected.clear()
                elif ch in ('\r', '\n'):  # enter = confirm
                    if self.selected:
                        return sorted(self.selected)
                    # flash message — need at least 1
                elif ch == 'q':
                    return None
                elif ch == 'j':  # vim down
                    self.cursor = (self.cursor + 1) % self.n
                elif ch == 'k':  # vim up
                    self.cursor = (self.cursor - 1) % self.n

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────
# Fetch fixtures
# ─────────────────────────────────────────────────────────────────────
def fetch_fixtures(fdo_key=None, apif_key=None) -> pd.DataFrame:
    """Fetch upcoming EPL fixtures from up to 3 sources."""
    fixtures = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        TextColumn("{task.fields[status]}"),
        console=console,
    ) as progress:

        # Source 1
        t1 = progress.add_task("football-data.co.uk", total=1, status="[dim]checking...[/dim]")
        try:
            resp = requests.get(FIXTURES_URL, timeout=15)
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip",
                              encoding_errors="replace")
            raw.columns = [c.strip().replace("\ufeff", "").replace("\xef\xbb\xbf", "")
                           for c in raw.columns]
            div_col = [c for c in raw.columns if "div" in c.lower() or c == "Div"]
            if div_col:
                raw = raw[raw[div_col[0]].str.strip() == "E0"]
            if not raw.empty:
                rename = {"HomeTeam": "home_team", "AwayTeam": "away_team",
                          "Date": "date", "Time": "time"}
                raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
                if "date" in raw.columns:
                    raw["date"] = pd.to_datetime(raw["date"], dayfirst=True, format="mixed")
                raw["home_team"] = raw["home_team"].apply(normalise_team)
                raw["away_team"] = raw["away_team"].apply(normalise_team)
                for _, row in raw.iterrows():
                    fixtures.append({
                        "date": row.get("date", pd.NaT),
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "time": str(row.get("time", ""))[:5] if pd.notna(row.get("time")) else "",
                    })
                progress.update(t1, completed=1, status=f"[green]{len(fixtures)} fixtures[/green]")
            else:
                progress.update(t1, completed=1, status="[dim]no fixtures[/dim]")
        except Exception as e:
            progress.update(t1, completed=1, status=f"[red]failed[/red]")

        # Source 2
        t2 = progress.add_task("Football-Data.org API", total=1, status="[dim]...[/dim]")
        if not fixtures and fdo_key:
            try:
                from data.api_client import FootballDataOrgClient
                client = FootballDataOrgClient(fdo_key)
                data = client.get_matches(status="SCHEDULED")
                if data and "matches" in data:
                    for m in data["matches"]:
                        ht = normalise_team(m.get("homeTeam", {}).get("name", ""))
                        at = normalise_team(m.get("awayTeam", {}).get("name", ""))
                        dt = m.get("utcDate", "")
                        if ht and at:
                            fixtures.append({
                                "date": pd.to_datetime(dt[:10]) if dt else pd.NaT,
                                "home_team": ht, "away_team": at,
                                "time": dt[11:16] if len(dt) > 16 else "",
                            })
                progress.update(t2, completed=1,
                                status=f"[green]{len(fixtures)} fixtures[/green]" if fixtures else "[dim]none[/dim]")
            except Exception as e:
                progress.update(t2, completed=1, status=f"[red]failed: {e}[/red]")
        else:
            status = "[dim]skipped[/dim]" if fixtures else "[dim]no key[/dim]"
            progress.update(t2, completed=1, status=status)

        # Source 3
        t3 = progress.add_task("API-Football", total=1, status="[dim]...[/dim]")
        if not fixtures and apif_key:
            try:
                from data.api_client import APIFootballClient
                client = APIFootballClient(apif_key)
                from datetime import datetime as _dt
                _cur_yr = _dt.now().year if _dt.now().month >= 8 else _dt.now().year - 1
                for season in [_cur_yr, _cur_yr - 1]:
                    data = client.get_fixtures(season)
                    if data and "response" in data:
                        for m in data["response"]:
                            status_code = m.get("fixture", {}).get("status", {}).get("short", "")
                            if status_code in ("NS", "TBD", "PST"):
                                ht = normalise_team(m.get("teams", {}).get("home", {}).get("name", ""))
                                at = normalise_team(m.get("teams", {}).get("away", {}).get("name", ""))
                                dt = m.get("fixture", {}).get("date", "")
                                if ht and at:
                                    fixtures.append({
                                        "date": pd.to_datetime(dt[:10]) if dt else pd.NaT,
                                        "home_team": ht, "away_team": at,
                                        "time": dt[11:16] if len(dt) > 16 else "",
                                    })
                        if fixtures:
                            break
                progress.update(t3, completed=1,
                                status=f"[green]{len(fixtures)} fixtures[/green]" if fixtures else "[dim]none[/dim]")
            except Exception as e:
                progress.update(t3, completed=1, status=f"[red]failed: {e}[/red]")
        else:
            status = "[dim]skipped[/dim]" if fixtures else "[dim]no key[/dim]"
            progress.update(t3, completed=1, status=status)

    if not fixtures:
        return pd.DataFrame()

    df = pd.DataFrame(fixtures)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["home_team", "away_team"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# Smart data readiness check — auto-fetch/build whatever is missing
# ─────────────────────────────────────────────────────────────────────
DATA_MAX_AGE_DAYS = 2  # Rebuild data if older than 2 days — freshness is critical


def _check_data_status():
    """Diagnose exactly what data exists, what's missing, and what's stale."""
    matches_path = DATA_DIR / "epl_matches.parquet"
    featured_path = DATA_DIR / "epl_featured.parquet"
    extra_path = DATA_DIR / "extra_data.json"

    status = {
        "matches_exist": matches_path.exists(),
        "featured_exist": featured_path.exists(),
        "extra_exist": extra_path.exists(),
        "matches_age": None,
        "featured_age": None,
        "has_xg": False,
        "has_elo": False,
        "has_weather": False,
        "has_sentiment": False,
        "has_injuries": False,
        "n_matches": 0,
        "n_features": 0,
        "needs_data_rebuild": False,
        "needs_feature_rebuild": False,
        "reasons": [],
    }

    if matches_path.exists():
        status["matches_age"] = (time.time() - matches_path.stat().st_mtime) / 86400
        try:
            # Read Parquet metadata efficiently
            df_sample = pd.read_parquet(matches_path, engine="pyarrow")
            status["n_matches"] = len(df_sample)
            status["has_elo"] = any("elo" in c.lower() for c in df_sample.columns)
            status["has_weather"] = "temperature" in df_sample.columns

            # Check xG and weather coverage in recent data
            if "xg_home" in df_sample.columns:
                recent = df_sample.tail(1000)
                status["has_xg"] = recent["xg_home"].notna().sum() > 100
            if "temperature" in df_sample.columns:
                recent = df_sample.tail(1000)
                status["has_weather"] = recent["temperature"].notna().sum() > 100
        except Exception:
            pass

    if featured_path.exists():
        status["featured_age"] = (time.time() - featured_path.stat().st_mtime) / 86400
        try:
            df_f = pd.read_parquet(featured_path, engine="pyarrow")
            status["n_features"] = len(df_f.columns)
        except Exception:
            pass

    if extra_path.exists():
        try:
            import json
            with open(extra_path) as f:
                extra = json.load(f)
            status["has_sentiment"] = bool(extra.get("sentiment", {}))
            status["has_injuries"] = bool(extra.get("injuries", {}))
        except Exception:
            pass

    # Decide what needs rebuilding
    if not status["matches_exist"]:
        status["needs_data_rebuild"] = True
        status["reasons"].append("No match data found")
    elif status["matches_age"] and status["matches_age"] > DATA_MAX_AGE_DAYS:
        status["needs_data_rebuild"] = True
        status["reasons"].append(f"Match data is {status['matches_age']:.0f} days old (>{DATA_MAX_AGE_DAYS}d)")

    # xG data from Understat (only available 2014+, so ~4000/7600 is full coverage)
    if not status["has_xg"] and status["matches_exist"]:
        status["needs_data_rebuild"] = True
        status["reasons"].append("Missing xG data (Understat) — will fetch via match pages")

    # Weather data should cover home matches
    if not status["has_weather"] and status["matches_exist"]:
        status["needs_data_rebuild"] = True
        status["reasons"].append("Missing weather data (Open-Meteo)")

    if not status["featured_exist"]:
        status["needs_feature_rebuild"] = True
        status["reasons"].append("No featured dataset found")
    elif status["needs_data_rebuild"]:
        status["needs_feature_rebuild"] = True
        status["reasons"].append("Features need rebuild after data update")
    elif (status["featured_age"] is not None and status["matches_age"] is not None
          and status["matches_age"] < status["featured_age"] - 0.01):
        # Matches are newer than features — features need rebuild
        status["needs_feature_rebuild"] = True
        status["reasons"].append("Features are stale (older than match data)")

    return status


def _display_data_status(status):
    """Show a rich status panel of what data exists."""
    rows = []

    def _icon(ok):
        return "[green]OK[/green]" if ok else "[red]MISSING[/red]"

    def _age(days):
        if days is None:
            return ""
        if days < 1:
            return f"[dim]({days*24:.0f}h ago)[/dim]"
        return f"[dim]({days:.1f}d ago)[/dim]"

    rows.append(f"  Match data:    {_icon(status['matches_exist'])}  {_age(status['matches_age'])}  [dim]{status['n_matches']} matches[/dim]")
    rows.append(f"  Featured data: {_icon(status['featured_exist'])}  {_age(status['featured_age'])}  [dim]{status['n_features']} features[/dim]")
    xg_status = "[green]OK[/green]" if status["has_xg"] else "[red]MISSING[/red]"
    rows.append(f"  xG (Understat):{xg_status}     Elo: {_icon(status['has_elo'])}     Weather: {_icon(status['has_weather'])}")
    rows.append(f"  Sentiment:     {_icon(status['has_sentiment'])}     Injuries: {_icon(status['has_injuries'])}")

    content = "\n".join(rows)

    if status["reasons"]:
        content += "\n\n  [yellow bold]Action needed:[/yellow bold]"
        for r in status["reasons"]:
            content += f"\n    [yellow]> {r}[/yellow]"

    border = "green" if not status["reasons"] else "yellow"
    console.print(Panel(content, title="[bold]Data Status[/bold]", border_style=border))


def ensure_data_ready(fdo_key=None, apif_key=None, news_key=None) -> bool:
    """Smart data check — diagnose, display status, auto-build whatever is missing."""
    console.print()
    console.print(Rule("[bold cyan]Checking Data[/bold cyan]", style="cyan"))
    console.print()

    status = _check_data_status()
    _display_data_status(status)

    if not status["needs_data_rebuild"] and not status["needs_feature_rebuild"]:
        console.print(f"\n  [green bold]All data ready.[/green bold]")
        return True

    # Auto-build what's needed
    if status["needs_data_rebuild"]:
        console.print(
            Panel(
                "[yellow bold]Building/updating data pipeline[/yellow bold]\n"
                "[dim]Fetching from 8 sources (20 seasons). First run ~10-15 min, cached after.[/dim]",
                border_style="yellow",
            )
        )

        cmd = [sys.executable, str(DATA_DIR / "generator.py")]
        if fdo_key:
            cmd += ["--fdo-key", fdo_key]
        if apif_key:
            cmd += ["--apif-key", apif_key]
        if news_key:
            cmd += ["--news-key", news_key]

        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            console.print("[red bold]Data generation failed.[/red bold]")
            return False
        status["needs_feature_rebuild"] = True  # Always rebuild features after data

    if status["needs_feature_rebuild"]:
        console.print("\n  [bold]Running feature engineering (376+ features) ...[/bold]")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "features" / "engine.py")],
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            console.print("[red bold]Feature engineering failed.[/red bold]")
            return False

    # Verify
    final_status = _check_data_status()
    if final_status["featured_exist"] and final_status["matches_exist"]:
        console.print(f"\n  [green bold]Data pipeline complete.[/green bold]")
        _display_data_status(final_status)
        return True
    else:
        console.print("[red]Data build completed but files not found. Check errors above.[/red]")
        return False


# ─────────────────────────────────────────────────────────────────────
# Live data refresh — fetch latest results before prediction
# ─────────────────────────────────────────────────────────────────────
def _fetch_latest_results(fdo_key=None):
    """Fetch latest match results from football-data.org to ensure current season data is fresh."""
    if not fdo_key:
        return

    try:
        import requests
        console.print("  [dim]Fetching latest match results ...[/dim]")
        headers = {"X-Auth-Token": fdo_key}
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/PL/matches",
            params={"status": "FINISHED", "limit": 50},
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            matches = data.get("matches", [])
            console.print(f"  [dim]Got {len(matches)} recent results from API[/dim]")
            # Store for pipeline to use
            import json
            live_path = DATA_DIR / "live_results.json"
            with open(live_path, "w") as f:
                json.dump({"matches": matches, "fetched_at": time.time()}, f)
    except Exception as e:
        console.print(f"  [dim]Could not fetch live results: {e}[/dim]")


def _fetch_latest_standings(fdo_key=None):
    """Fetch current league standings for table position features."""
    if not fdo_key:
        return

    try:
        import requests
        console.print("  [dim]Fetching current standings ...[/dim]")
        headers = {"X-Auth-Token": fdo_key}
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/PL/standings",
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            import json
            standings_path = DATA_DIR / "live_standings.json"
            with open(standings_path, "w") as f:
                json.dump(data, f)
            console.print(f"  [dim]Standings updated[/dim]")
    except Exception as e:
        console.print(f"  [dim]Could not fetch standings: {e}[/dim]")


# ─────────────────────────────────────────────────────────────────────
# Run predictions
# ─────────────────────────────────────────────────────────────────────
def run_predictions(selected_df, fdo_key=None, apif_key=None, news_key=None) -> None:
    if not ensure_data_ready(fdo_key, apif_key, news_key):
        console.print("[red]Could not build data.[/red]")
        console.print("[dim]Run: python data/generator.py && python features/engine.py[/dim]")
        return

    # Fetch latest live data before running predictions
    console.print("\n  [bold]Refreshing live data ...[/bold]")
    _fetch_latest_results(fdo_key)
    _fetch_latest_standings(fdo_key)

    # Save selected fixtures
    upcoming_path = DATA_DIR / "upcoming_fixtures.parquet"
    selected_df[["date", "home_team", "away_team"]].to_parquet(upcoming_path, index=False, engine="pyarrow")

    n = len(selected_df)

    # Show what we're predicting
    match_list = "  ".join(
        f"[bold]{r['home_team']}[/bold] vs [bold]{r['away_team']}[/bold]"
        for _, r in selected_df.iterrows()
    )
    console.print(
        Panel(
            f"[bold cyan]RUNNING FULL 5-LAYER PREDICTION PIPELINE[/bold cyan]\n\n"
            f"[dim]13 base learners | 4 meta-learners | binary boosting | calibration | live NLP[/dim]\n"
            f"[dim]Predicting {n} match{'es' if n != 1 else ''}:[/dim]\n\n"
            f"{match_list}",
            border_style="cyan",
            padding=(1, 3),
        )
    )

    t0 = time.time()

    # Run pipeline in-process
    from models.run_pipeline import main as pipeline_main
    pipeline_main()

    elapsed = time.time() - t0

    console.print()
    console.print(f"  [green bold]Pipeline completed in {elapsed:.1f}s[/green bold]")

    # Dashboard
    dashboard_path = PROJECT_ROOT / "dashboard.html"
    if dashboard_path.exists():
        console.print(f"  [green]Dashboard:[/green] {dashboard_path}")
        if Confirm.ask("\n  Open dashboard in browser?", default=True):
            webbrowser.open(f"file://{dashboard_path}")
            console.print("  [green]Opened![/green]")


# ─────────────────────────────────────────────────────────────────────
# Fallback text-based selector (if terminal doesn't support raw mode)
# ─────────────────────────────────────────────────────────────────────
def fallback_selector(fixtures_df, max_show=15):
    """Simple text-based selector as fallback."""
    n = min(len(fixtures_df), max_show)

    table = Table(
        title="[bold cyan]UPCOMING EPL FIXTURES[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("#", width=3, justify="right", style="cyan bold")
    table.add_column("Date", width=14)
    table.add_column("KO", width=6, style="dim")
    table.add_column("Home", width=20, justify="right", style="bold")
    table.add_column("", width=3, justify="center", style="dim")
    table.add_column("Away", width=20, style="bold")

    for i in range(n):
        row = fixtures_df.iloc[i]
        date_str = row["date"].strftime("%a %d %b") if pd.notna(row["date"]) else "TBD"
        time_str = row.get("time", "") or ""
        table.add_row(str(i+1), date_str, time_str, row["home_team"], "vs", row["away_team"])

    console.print()
    console.print(table)
    console.print()

    console.print(
        Panel(
            "[cyan]1,3,5[/cyan]  specific matches    "
            "[cyan]1-5[/cyan]  a range    "
            "[cyan]all[/cyan]  all matches    "
            "[cyan]q[/cyan]  quit",
            title="[bold]Select matches[/bold]",
            border_style="dim",
        )
    )

    while True:
        choice = Prompt.ask("[cyan]>>>[/cyan]").strip().lower()

        if choice in ('q', 'quit', 'exit'):
            return None
        if choice == '':
            continue
        if choice == 'all':
            return list(range(n))

        try:
            indices = []
            for part in choice.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    indices.extend(range(int(a) - 1, int(b)))
                else:
                    indices.append(int(part) - 1)

            invalid = [i + 1 for i in indices if i < 0 or i >= n]
            if invalid:
                console.print(f"[red]Invalid: {invalid}. Enter 1-{n}.[/red]")
                continue
            return sorted(set(indices))
        except ValueError:
            console.print("[red]Invalid. Try: 1,3,5 or 1-5 or all or q[/red]")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MatchOracle — Interactive EPL Match Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py
  python predict.py --fdo-key YOUR_KEY --apif-key YOUR_KEY --news-key YOUR_KEY
  python predict.py --max 20
  python predict.py --text          # Use text-based selector (no arrow keys)

Environment variables (alternative to flags):
  FDO_KEY    Football-Data.org API key
  APIF_KEY   API-Football key
  NEWS_KEY   NewsAPI key
        """
    )
    parser.add_argument("--fdo-key", type=str, default=os.environ.get("FDO_KEY"))
    parser.add_argument("--apif-key", type=str, default=os.environ.get("APIF_KEY"))
    parser.add_argument("--news-key", type=str, default=os.environ.get("NEWS_KEY"))
    parser.add_argument("--max", type=int, default=15, help="Max fixtures to display")
    parser.add_argument("--text", action="store_true", help="Use text-based selector (no arrow keys)")
    args = parser.parse_args()

    # ── Banner ──
    console.print()
    console.print(
        Align.center(
            Panel(
                "[bold white]M A T C H O R A C L E[/bold white]\n\n"
                "[cyan]5-Layer Deep Ensemble[/cyan]  [dim]|[/dim]  "
                "[cyan]13 Base Learners[/cyan]  [dim]|[/dim]  "
                "[cyan]376+ Features[/cyan]\n"
                "[dim]8 Data Sources  |  20 Seasons  |  ~7,600 Matches[/dim]",
                border_style="cyan",
                padding=(1, 4),
                title="[bold cyan]EPL Match Predictor[/bold cyan]",
                subtitle="[dim]Interactive Terminal UI[/dim]",
            )
        )
    )

    # ── Interactive loop ──
    while True:
        # Fetch fixtures
        console.print()
        console.print(Rule("[bold cyan]Fetching Upcoming Fixtures[/bold cyan]", style="cyan"))
        console.print()
        fixtures_df = fetch_fixtures(fdo_key=args.fdo_key, apif_key=args.apif_key)

        if fixtures_df.empty:
            console.print(
                Panel(
                    "[yellow bold]No upcoming EPL fixtures found.[/yellow bold]\n\n"
                    "[dim]This happens between seasons (June-August).[/dim]\n\n"
                    "[bold]You can still run validation:[/bold]\n"
                    "  [cyan]python models/run_pipeline.py[/cyan]",
                    border_style="yellow",
                )
            )
            return

        # Select fixtures
        use_interactive = not args.text
        if use_interactive:
            try:
                import termios  # noqa: F401
                import tty      # noqa: F401
            except ImportError:
                use_interactive = False

            # Also check if stdin is a real terminal
            if not sys.stdin.isatty():
                use_interactive = False

        if use_interactive:
            try:
                selector = FixtureSelector(fixtures_df, max_show=args.max)
                selected_indices = selector.run()
            except Exception:
                # Fall back to text mode on any terminal issue
                selected_indices = fallback_selector(fixtures_df, max_show=args.max)
        else:
            selected_indices = fallback_selector(fixtures_df, max_show=args.max)

        if selected_indices is None:
            console.print("\n  [dim]Goodbye.[/dim]\n")
            return

        selected = fixtures_df.iloc[selected_indices].copy()

        # Confirm selection
        console.print()
        sel_table = Table(
            title=f"[bold green]Selected {len(selected)} Match{'es' if len(selected) != 1 else ''}[/bold green]",
            box=box.SIMPLE,
            border_style="green",
        )
        sel_table.add_column("Date", width=14)
        sel_table.add_column("Home", width=20, justify="right", style="bold")
        sel_table.add_column("", width=3, justify="center", style="dim")
        sel_table.add_column("Away", width=20, style="bold")

        for _, row in selected.iterrows():
            date_str = row["date"].strftime("%a %d %b") if pd.notna(row["date"]) else "TBD"
            sel_table.add_row(date_str, row["home_team"], "vs", row["away_team"])

        console.print(sel_table)
        console.print()

        if not Confirm.ask("  Start prediction pipeline?", default=True):
            console.print("  [dim]Cancelled.[/dim]")
            if Confirm.ask("\n  Pick different matches?", default=True):
                continue
            else:
                console.print("\n  [dim]Goodbye.[/dim]\n")
                return

        # Run pipeline
        console.print()
        run_predictions(
            selected,
            fdo_key=args.fdo_key,
            apif_key=args.apif_key,
            news_key=args.news_key,
        )

        # Again?
        console.print()
        if Confirm.ask("  Predict more matches?", default=False):
            console.print()
            continue
        else:
            console.print("\n  [dim]Goodbye.[/dim]\n")
            return


if __name__ == "__main__":
    main()
