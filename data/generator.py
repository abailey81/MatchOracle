"""
MatchOracle Real Data Pipeline
===============================
Fetches 20 seasons of real EPL match data from multiple free sources:

  1. football-data.co.uk  — results, stats, odds, referee       (CSV, no key)
  2. Understat             — match-level xG / xGA                (scraping, no key)
  3. Club Elo              — historical team Elo ratings          (REST API, no key)
  4. Open-Meteo            — historical weather at stadium coords (REST API, no key)

Coverage:
  - football-data.co.uk:  2005-06 to 2024-25 (20 seasons, ~7600 matches)
  - Understat xG:         2014-15 to 2024-25 (11 seasons)
  - Club Elo:             full 20-season coverage
  - Open-Meteo weather:   full 20-season coverage

Usage:
    python data/generator.py
"""

import asyncio
import io
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent


def _load_env():
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


_load_env()

# ---------------------------------------------------------------------------
# Season configuration — 20 seasons
# ---------------------------------------------------------------------------
SEASONS_CONFIG = [
    ("0506", "2005-06"), ("0607", "2006-07"), ("0708", "2007-08"),
    ("0809", "2008-09"), ("0910", "2009-10"), ("1011", "2010-11"),
    ("1112", "2011-12"), ("1213", "2012-13"), ("1314", "2013-14"),
    ("1415", "2014-15"), ("1516", "2015-16"), ("1617", "2016-17"),
    ("1718", "2017-18"), ("1819", "2018-19"), ("1920", "2019-20"),
    ("2021", "2020-21"), ("2122", "2021-22"), ("2223", "2022-23"),
    ("2324", "2023-24"), ("2425", "2024-25"),
]

SEASON_CODES = [s[0] for s in SEASONS_CONFIG]
SEASON_LABELS = [s[1] for s in SEASONS_CONFIG]

# Understat only has data from 2014-15 onward (start year = 2014)
UNDERSTAT_SEASONS = list(range(2014, 2025))  # 2014 .. 2024

# ---------------------------------------------------------------------------
# 1. football-data.co.uk
# ---------------------------------------------------------------------------
FOOTBALLDATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# Core column mapping
FDCO_COL_MAP = {
    "Date": "date", "Time": "kickoff_time",
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "goals_home", "FTAG": "goals_away", "FTR": "result",
    "HTHG": "ht_goals_home", "HTAG": "ht_goals_away", "HTR": "ht_result",
    "Referee": "referee",
    "HS": "shots_home", "AS": "shots_away",
    "HST": "sot_home", "AST": "sot_away",
    "HC": "corners_home", "AC": "corners_away",
    "HF": "fouls_home", "AF": "fouls_away",
    "HY": "yellows_home", "AY": "yellows_away",
    "HR": "reds_home", "AR": "reds_away",
}

# Odds — we pull multiple bookmakers for robustness
ODDS_COL_MAP = {
    # Bet365
    "B365H": "odds_b365_home", "B365D": "odds_b365_draw", "B365A": "odds_b365_away",
    # Pinnacle (open)
    "PSH": "odds_pin_home", "PSD": "odds_pin_draw", "PSA": "odds_pin_away",
    # Pinnacle (close)
    "PSCH": "odds_pin_home_close", "PSCD": "odds_pin_draw_close", "PSCA": "odds_pin_away_close",
    # Market max
    "MaxH": "odds_max_home", "MaxD": "odds_max_draw", "MaxA": "odds_max_away",
    # Market average
    "AvgH": "odds_avg_home", "AvgD": "odds_avg_draw", "AvgA": "odds_avg_away",
    # Over/Under 2.5
    "Avg>2.5": "odds_avg_over25", "Avg<2.5": "odds_avg_under25",
    "B365>2.5": "odds_b365_over25", "B365<2.5": "odds_b365_under25",
    # Bet365 close
    "B365CH": "odds_b365_home_close", "B365CD": "odds_b365_draw_close",
    "B365CA": "odds_b365_away_close",
    # Market average close
    "AvgCH": "odds_avg_home_close", "AvgCD": "odds_avg_draw_close",
    "AvgCA": "odds_avg_away_close",
    # Interwetten
    "IWH": "odds_iw_home", "IWD": "odds_iw_draw", "IWA": "odds_iw_away",
    # William Hill
    "WHH": "odds_wh_home", "WHD": "odds_wh_draw", "WHA": "odds_wh_away",
}


def fetch_footballdata(season_codes: List[str], labels: List[str]) -> pd.DataFrame:
    """Download EPL CSVs from football-data.co.uk for all 20 seasons (parallel)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_season(code, label):
        url = FOOTBALLDATA_URL.format(season=code)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    raw = pd.read_csv(io.StringIO(resp.content.decode(enc)),
                                      on_bad_lines="skip")
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                return label, None

            raw = raw.dropna(how="all")
            if raw.empty:
                return label, None

            if "HG" in raw.columns and "FTHG" not in raw.columns:
                raw = raw.rename(columns={"HG": "FTHG", "AG": "FTAG", "Res": "FTR"})

            rename = {}
            for src, dst in {**FDCO_COL_MAP, **ODDS_COL_MAP}.items():
                if src in raw.columns:
                    rename[src] = dst
            df = raw.rename(columns=rename)
            keep = [c for c in rename.values() if c in df.columns]
            df = df[keep].copy()
            df["season"] = label

            if "goals_home" in df.columns:
                df = df.dropna(subset=["goals_home", "goals_away"])
                df["goals_home"] = df["goals_home"].astype(int)
                df["goals_away"] = df["goals_away"].astype(int)

            return label, df
        except Exception as e:
            print(f"    WARNING: Failed to fetch {label}: {e}")
            return label, None

    frames = []
    print(f"    Fetching {len(season_codes)} seasons in parallel ...")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_season, c, l): l
                   for c, l in zip(season_codes, labels)}
        for f in as_completed(futures):
            label, df = f.result()
            if df is not None:
                frames.append((label, df))
                print(f"    [{label}] {len(df)} matches")

    if not frames:
        raise RuntimeError("Could not fetch any season data from football-data.co.uk")

    # Sort by season label to maintain order
    frames.sort(key=lambda x: x[0])
    combined = pd.concat([f[1] for f in frames], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, format="mixed")
    combined = combined.sort_values("date").reset_index(drop=True)
    print(f"  => Total: {len(combined)} matches across {len(frames)} seasons")
    return combined


# ---------------------------------------------------------------------------
# 2. Understat xG
# ---------------------------------------------------------------------------
def fetch_understat_xg(seasons: List[int]) -> pd.DataFrame:
    """Fetch match-level xG from Understat (2014-15 onwards). No API key.

    Understat moved to client-side rendering so league pages no longer contain
    inline data. However, individual /match/{id} pages still expose match_info
    via a JSON.parse() call. We scan the ID range with heavy parallelisation
    (20 workers) to fetch EPL matches efficiently.

    Strategy:
      1. Probe the ID space to find EPL (league_id=1) boundaries per season
      2. Dense-scan within those boundaries with 20 parallel workers
      3. Filter to requested seasons
    """
    import re
    import json as _json
    import codecs
    from concurrent.futures import ThreadPoolExecutor, as_completed

    HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    EPL_LEAGUE_ID = "1"

    def _fetch_match(match_id: int):
        """Fetch xG data from a single Understat match page."""
        try:
            resp = requests.get(
                f"https://understat.com/match/{match_id}",
                timeout=10, headers=HEADERS,
            )
            if resp.status_code != 200:
                return match_id, None
            m = re.search(
                r"var\s+match_info\s*=\s*JSON\.parse\(\s*'(.+?)'\s*\)",
                resp.text, re.DOTALL,
            )
            if not m:
                return match_id, None
            decoded = codecs.decode(m.group(1), 'unicode_escape')
            info = _json.loads(decoded)
            return match_id, info
        except Exception:
            return match_id, None

    # ── Phase 1: Probe to find ID ranges per season ──
    # Understat IDs are sequential across all leagues (~1900 IDs/season).
    # EPL seasons 2014-2024 span roughly IDs 1 - 27500.
    print("    Phase 1: Probing ID space for EPL match boundaries ...")
    probe_ids = list(range(1, 28500, 200))  # ~142 probes

    season_id_ranges = {}  # season -> (min_id, max_id)
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_fetch_match, mid): mid for mid in probe_ids}
        for f in as_completed(futures):
            mid, info = f.result()
            if info and str(info.get("league_id")) == EPL_LEAGUE_ID:
                ssn = int(info.get("season", 0))
                if ssn not in season_id_ranges:
                    season_id_ranges[ssn] = [mid, mid]
                else:
                    season_id_ranges[ssn][0] = min(season_id_ranges[ssn][0], mid)
                    season_id_ranges[ssn][1] = max(season_id_ranges[ssn][1], mid)

    if not season_id_ranges:
        print("    WARNING: Could not find any EPL matches on Understat.")
        print("    Pipeline will use goals + shots data instead.")
        return pd.DataFrame()

    # Expand ranges to ensure we capture all matches (add buffer)
    for ssn in season_id_ranges:
        lo, hi = season_id_ranges[ssn]
        season_id_ranges[ssn] = [max(1, lo - 250), hi + 250]

    requested = set(seasons)
    target_seasons = {s: r for s, r in season_id_ranges.items() if s in requested}

    if not target_seasons:
        print(f"    No matching seasons found. Available: {sorted(season_id_ranges.keys())}")
        return pd.DataFrame()

    # ── Phase 2: Dense scan within each season's range ──
    print(f"    Phase 2: Fetching xG for {len(target_seasons)} seasons "
          f"({sum(r[1]-r[0]+1 for r in target_seasons.values())} IDs) ...")

    all_ids = []
    for ssn, (lo, hi) in sorted(target_seasons.items()):
        all_ids.extend(range(lo, hi + 1))
        print(f"      Season {ssn}/{ssn+1}: scanning IDs {lo}-{hi}")

    all_matches = []
    done = 0
    total = len(all_ids)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_fetch_match, mid): mid for mid in all_ids}
        for f in as_completed(futures):
            done += 1
            mid, info = f.result()
            if info and str(info.get("league_id")) == EPL_LEAGUE_ID:
                ssn = int(info.get("season", 0))
                if ssn in requested:
                    all_matches.append({
                        "understat_id": str(mid),
                        "date_us": info.get("date", "")[:10],
                        "home_team_us": info.get("team_h", ""),
                        "away_team_us": info.get("team_a", ""),
                        "xg_home": float(info.get("h_xg", 0)),
                        "xg_away": float(info.get("a_xg", 0)),
                    })
            # Progress update every 500 IDs
            if done % 500 == 0:
                print(f"      ... {done}/{total} IDs scanned, "
                      f"{len(all_matches)} EPL matches found")

    if not all_matches:
        print("    WARNING: No EPL xG data retrieved.")
        return pd.DataFrame()

    # Count per season
    season_counts = {}
    for m in all_matches:
        yr = m["date_us"][:4]
        season_counts[yr] = season_counts.get(yr, 0) + 1

    for yr in sorted(season_counts):
        n = season_counts[yr]
        # Map calendar year to season label
        print(f"    {yr}: {n} matches with xG")

    df = pd.DataFrame(all_matches)
    df["date_us"] = pd.to_datetime(df["date_us"])
    print(f"  => {len(df)} total matches with xG data across {len(target_seasons)} seasons")
    return df


# ---------------------------------------------------------------------------
# 3. Club Elo ratings
# ---------------------------------------------------------------------------
CLUB_ELO_URL = "http://api.clubelo.com/{date}"


def fetch_club_elo(date_strings: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch Elo ratings at monthly intervals across the date range."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique_months = sorted(set(d[:7] for d in date_strings))
    sample_dates = [f"{m}-01" for m in unique_months]

    def _parse_elo_response(d: str) -> Tuple[str, Dict[str, float]]:
        """Fetch and parse one date's Elo ratings."""
        try:
            resp = requests.get(CLUB_ELO_URL.format(date=d), timeout=15)
            if resp.status_code != 200:
                return d, {}
            lines = resp.text.strip().split("\n")
            if not lines:
                return d, {}
            header = lines[0].split(",")
            # Detect format: newer has Rank,Club,Country,Level,Elo
            country_idx = 2 if "Country" in header or len(header) > 4 else 1
            elo_idx = 4 if "Elo" in header or len(header) > 4 else 3
            club_idx = 1 if "Club" in header or len(header) > 4 else 0
            teams = {}
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > max(country_idx, elo_idx, club_idx):
                    if parts[country_idx].strip() != "ENG":
                        continue
                    team = normalise_team(parts[club_idx].strip())
                    try:
                        teams[team] = float(parts[elo_idx])
                    except ValueError:
                        continue
            return d, teams
        except Exception:
            return d, {}

    elo_data = {}
    total = len(sample_dates)
    print(f"  Fetching Club Elo for {total} monthly snapshots (parallel) ...")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_parse_elo_response, d): d for d in sample_dates}
        done = 0
        for future in as_completed(futures):
            d, teams = future.result()
            if teams:
                elo_data[d] = teams
            done += 1
            if done % 48 == 0 or done == total:
                print(f"  Club Elo progress: {done}/{total} months")

    print(f"  => Elo data for {len(elo_data)} monthly snapshots")
    return elo_data


def merge_elo(df: pd.DataFrame, elo_data: Dict) -> pd.DataFrame:
    """Merge closest pre-match Elo rating into each row."""
    if not elo_data:
        df["elo_pre_home"] = np.nan
        df["elo_pre_away"] = np.nan
        return df

    elo_dates = sorted(elo_data.keys())

    # Build a fast lookup using bisect
    import bisect

    def _get_elo(team: str, match_date_str: str) -> float:
        idx = bisect.bisect_right(elo_dates, match_date_str) - 1
        if idx >= 0 and elo_dates[idx] <= match_date_str:
            return elo_data[elo_dates[idx]].get(team, np.nan)
        return np.nan

    date_strs = df["date"].dt.strftime("%Y-%m-%d")
    df["elo_pre_home"] = [_get_elo(ht, ds) for ht, ds in zip(df["home_team"], date_strs)]
    df["elo_pre_away"] = [_get_elo(at, ds) for at, ds in zip(df["away_team"], date_strs)]
    matched = df["elo_pre_home"].notna().sum()
    print(f"  => Matched Elo for {matched}/{len(df)} matches")
    return df


# ---------------------------------------------------------------------------
# 4. Open-Meteo historical weather
# ---------------------------------------------------------------------------
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Stadium coordinates for all EPL teams (2005-2025)
STADIUM_COORDS = {
    "Arsenal": (51.5549, -0.1084),
    "Aston Villa": (52.5092, -1.8846),
    "Birmingham": (52.4758, -1.8681),
    "Blackburn": (53.7286, -2.4893),
    "Blackpool": (53.8046, -3.0484),
    "Bolton": (53.5808, -2.5356),
    "Bournemouth": (50.7352, -1.8388),
    "Brentford": (51.4907, -0.2888),
    "Brighton": (50.8616, -0.0837),
    "Burnley": (53.7890, -2.2302),
    "Cardiff": (51.4728, -3.2030),
    "Charlton": (51.4865, 0.0367),
    "Chelsea": (51.4817, -0.1910),
    "Crystal Palace": (51.3983, -0.0855),
    "Derby": (52.9147, -1.4473),
    "Everton": (53.4389, -2.9664),
    "Fulham": (51.4749, -0.2217),
    "Hull": (53.7460, -0.3680),
    "Huddersfield": (53.6543, -1.7685),
    "Ipswich": (52.0545, 1.1447),
    "Leeds": (53.7771, -1.5722),
    "Leicester": (52.6204, -1.1422),
    "Liverpool": (53.4308, -2.9609),
    "Luton": (51.8842, -0.4316),
    "Manchester City": (53.4831, -2.2004),
    "Manchester United": (53.4631, -2.2913),
    "Middlesbrough": (54.5782, -1.2170),
    "Newcastle": (54.9756, -1.6217),
    "Norwich": (52.6222, 1.3093),
    "Nottm Forest": (52.9399, -1.1325),
    "Portsmouth": (50.7964, -1.0636),
    "QPR": (51.5093, -0.2322),
    "Reading": (51.4222, -0.9828),
    "Sheffield Utd": (53.3703, -1.4710),
    "Southampton": (50.9058, -1.3910),
    "Stoke": (52.9884, -2.1755),
    "Sunderland": (54.9146, -1.3884),
    "Swansea": (51.6428, -3.9350),
    "Tottenham": (51.6042, -0.0662),
    "Watford": (51.6498, -0.4014),
    "West Brom": (52.5090, -1.9638),
    "West Ham": (51.5387, 0.0166),
    "Wigan": (53.5474, -2.6540),
    "Wolves": (52.5901, -2.1306),
}


def fetch_weather_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch historical weather per team per season range from Open-Meteo.
    Uses parallel requests (10 workers) for speed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    teams_in_data = df["home_team"].unique()

    # Build list of (team, lat, lon, start_date, end_date) jobs
    jobs = []
    for team in teams_in_data:
        coords = STADIUM_COORDS.get(team)
        if not coords:
            continue
        team_matches = df[df["home_team"] == team]
        dates = sorted(team_matches["date"].dt.date.unique())
        if not dates:
            continue
        lat, lon = coords
        years = sorted(set(d.year for d in dates))
        for year in years:
            year_dates = [d for d in dates if d.year == year]
            jobs.append((team, lat, lon, str(min(year_dates)), str(max(year_dates))))

    def _fetch_wx(job):
        team, lat, lon, start, end = job
        results = []
        try:
            resp = requests.get(OPEN_METEO_URL, params={
                "latitude": lat, "longitude": lon,
                "start_date": start, "end_date": end,
                "daily": "temperature_2m_mean,relative_humidity_2m_mean,"
                         "wind_speed_10m_max,precipitation_sum",
                "timezone": "Europe/London",
            }, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get("daily", {})
                if data and data.get("time"):
                    for i, t in enumerate(data["time"]):
                        results.append({
                            "home_team": team,
                            "wx_date": t,
                            "temperature": data.get("temperature_2m_mean", [None])[i],
                            "humidity": data.get("relative_humidity_2m_mean", [None])[i],
                            "wind_speed": data.get("wind_speed_10m_max", [None])[i],
                            "precipitation": data.get("precipitation_sum", [None])[i],
                        })
        except Exception:
            pass
        return results

    print(f"    Fetching weather for {len(teams_in_data)} teams, {len(jobs)} requests (parallel) ...")
    all_wx = []
    done = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_wx, j): j for j in jobs}
        for f in as_completed(futures):
            result = f.result()
            all_wx.extend(result)
            done += 1
            if done % 100 == 0:
                print(f"      ... {done}/{len(jobs)} weather requests done")

    if not all_wx:
        print("  => No weather data fetched")
        for col in ["temperature", "humidity", "wind_speed", "precipitation"]:
            df[col] = np.nan
        return df

    wx_df = pd.DataFrame(all_wx)
    wx_df["wx_date"] = pd.to_datetime(wx_df["wx_date"]).dt.normalize()

    df["_wx_date"] = df["date"].dt.normalize()
    df = df.merge(wx_df, left_on=["home_team", "_wx_date"],
                  right_on=["home_team", "wx_date"], how="left")
    df.drop(columns=["_wx_date", "wx_date"], errors="ignore", inplace=True)

    matched = df["temperature"].notna().sum()
    print(f"  => Weather for {matched}/{len(df)} matches")
    return df


# ---------------------------------------------------------------------------
# Team name normalisation (covers all ~45 EPL teams 2005-2025)
# ---------------------------------------------------------------------------
TEAM_NAME_MAP = {
    # Understat -> standard
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nottm Forest",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield Utd",
    "Norwich City": "Norwich",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Brighton and Hove Albion": "Brighton",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton": "Wolves",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    "Derby County": "Derby",
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Wigan Athletic": "Wigan",
    "Charlton Athletic": "Charlton",
    "Portsmouth": "Portsmouth",
    "Reading": "Reading",
    "Sunderland": "Sunderland",
    "Middlesbrough": "Middlesbrough",
    "Burnley": "Burnley",
    "Queens Park Rangers": "QPR",
    "Blackpool": "Blackpool",
    "Huddersfield Town": "Huddersfield",
    "Watford": "Watford",
    "Luton Town": "Luton",
    "Bournemouth": "Bournemouth",
    "Crystal Palace": "Crystal Palace",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "AFC Bournemouth": "Bournemouth",
    "Fulham": "Fulham",
    "Southampton": "Southampton",
    "Brentford": "Brentford",
    "Everton": "Everton",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Aston Villa": "Aston Villa",
    "Chelsea": "Chelsea",
    # Club Elo name variants
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottm Forest",
    "Nottingham Forest": "Nottm Forest",
    "Sheffield Utd": "Sheffield Utd",
    "West Brom": "West Brom",
    "West Ham": "West Ham",
    "Nott'ham Forest": "Nottm Forest",
    "Sheffield United": "Sheffield Utd",
    "West Bromwich": "West Brom",
    # Club Elo specific short names
    "Blackburn": "Blackburn",
    "Bolton": "Bolton",
    "Burnley": "Burnley",
    "Hull": "Hull",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "Stoke": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Watford": "Watford",
    "Wigan": "Wigan",
    "Wolves": "Wolves",
    "Middlesbrough": "Middlesbrough",
    "Birmingham": "Birmingham",
    "Derby": "Derby",
    "Cardiff": "Cardiff",
    "QPR": "QPR",
    "Reading": "Reading",
    "Blackpool": "Blackpool",
    "Charlton": "Charlton",
    "Portsmouth": "Portsmouth",
    "Huddersfield": "Huddersfield",
    "Luton": "Luton",
}


def normalise_team(name: str) -> str:
    """Normalise team name to a canonical form."""
    if not name or not isinstance(name, str):
        return name
    name = name.strip()
    # Direct lookup first
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # Strip common API suffixes and try again
    for suffix in (" FC", " AFC", " F.C.", " A.F.C."):
        if name.endswith(suffix):
            stripped = name[:-len(suffix)].strip()
            if stripped in TEAM_NAME_MAP:
                return TEAM_NAME_MAP[stripped]
    # Try replacing & with "and"
    alt = name.replace("&", "and").replace("  ", " ").strip()
    for suffix in ("", " FC", " AFC"):
        candidate = alt.removesuffix(suffix).strip() if suffix else alt
        if candidate in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[candidate]
    return name


# ---------------------------------------------------------------------------
# Derbies (comprehensive for 20 seasons)
# ---------------------------------------------------------------------------
DERBIES = {
    frozenset({"Arsenal", "Tottenham"}): "North London Derby",
    frozenset({"Arsenal", "Chelsea"}): "London Derby",
    frozenset({"Chelsea", "Tottenham"}): "London Derby",
    frozenset({"Chelsea", "Fulham"}): "West London Derby",
    frozenset({"Crystal Palace", "Brighton"}): "M23 Derby",
    frozenset({"Liverpool", "Everton"}): "Merseyside Derby",
    frozenset({"Manchester City", "Manchester United"}): "Manchester Derby",
    frozenset({"Wolves", "Aston Villa"}): "West Midlands Derby",
    frozenset({"Wolves", "West Brom"}): "Black Country Derby",
    frozenset({"Aston Villa", "Birmingham"}): "Second City Derby",
    frozenset({"Aston Villa", "West Brom"}): "West Midlands Derby",
    frozenset({"Nottm Forest", "Leicester"}): "East Midlands Derby",
    frozenset({"Nottm Forest", "Derby"}): "East Midlands Derby",
    frozenset({"Newcastle", "Sunderland"}): "Tyne-Wear Derby",
    frozenset({"Leeds", "Manchester United"}): "Roses Derby",
    frozenset({"Liverpool", "Manchester United"}): "North-West Derby",
    frozenset({"West Ham", "Tottenham"}): "London Derby",
    frozenset({"West Ham", "Chelsea"}): "London Derby",
    frozenset({"Arsenal", "West Ham"}): "London Derby",
    frozenset({"Crystal Palace", "Charlton"}): "South London Derby",
    frozenset({"Portsmouth", "Southampton"}): "South Coast Derby",
    frozenset({"Burnley", "Blackburn"}): "East Lancashire Derby",
}


# ---------------------------------------------------------------------------
# Upcoming fixtures
# ---------------------------------------------------------------------------
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def fetch_upcoming_fixtures() -> pd.DataFrame:
    """Fetch upcoming EPL fixtures from football-data.co.uk."""
    print("  Fetching upcoming fixtures ...")
    try:
        resp = requests.get(FIXTURES_URL, timeout=15)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip",
                          encoding_errors="replace")
        # Clean BOM from column names (handles both \ufeff and raw UTF-8 BOM bytes)
        raw.columns = [c.strip().replace("\ufeff", "").replace("ï»¿", "") for c in raw.columns]
        # Filter to EPL only (Div == "E0")
        div_col = [c for c in raw.columns if "div" in c.lower() or c == "Div"]
        if div_col:
            raw = raw[raw[div_col[0]].str.strip() == "E0"]
        if raw.empty:
            print("  => No upcoming EPL fixtures found")
            return pd.DataFrame()
        rename = {"HomeTeam": "home_team", "AwayTeam": "away_team", "Date": "date"}
        raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
        if "date" in raw.columns:
            raw["date"] = pd.to_datetime(raw["date"], dayfirst=True, format="mixed")
        raw["home_team"] = raw["home_team"].apply(normalise_team)
        raw["away_team"] = raw["away_team"].apply(normalise_team)
        print(f"  => {len(raw)} upcoming fixtures")
        return raw[["date", "home_team", "away_team"]].reset_index(drop=True)
    except Exception as e:
        print(f"  WARNING: Could not fetch fixtures: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all columns expected by the downstream pipeline."""

    # Result-derived
    df["home_points"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["away_points"] = df["result"].map({"H": 0, "D": 1, "A": 3})
    df["total_goals"] = df["goals_home"] + df["goals_away"]
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)
    df["btts"] = ((df["goals_home"] > 0) & (df["goals_away"] > 0)).astype(int)

    # Weather flags
    df["is_rain"] = (df["precipitation"].fillna(0) > 0.5).astype(int)

    # Derby flag
    df["is_derby"] = df.apply(
        lambda r: int(frozenset({r["home_team"], r["away_team"]}) in DERBIES), axis=1
    )

    # Venue (home team's stadium)
    venue_map = {team: team for team in df["home_team"].unique()}
    df["venue"] = df["home_team"].map(venue_map)

    # --- Gameweek (proper calculation) ---
    # Within each season, group matches by date clusters (~same matchday = same GW)
    gw_list = []
    for season, sdf in df.groupby("season"):
        dates_sorted = sorted(sdf["date"].unique())
        # Cluster dates within 3 days as the same gameweek
        gw_map = {}
        gw = 1
        prev_date = None
        for d in dates_sorted:
            if prev_date is not None and (d - prev_date).days > 3:
                gw += 1
            gw_map[d] = min(gw, 38)
            prev_date = d
        for idx, row in sdf.iterrows():
            gw_list.append((idx, gw_map.get(row["date"], 1)))

    gw_series = pd.Series(dict(gw_list))
    df["gameweek"] = gw_series

    # --- Rest days (computed from schedule) ---
    df = df.sort_values("date").reset_index(drop=True)
    last_match: Dict[str, pd.Timestamp] = {}
    rest_home = []
    rest_away = []
    for _, row in df.iterrows():
        ht, at, d = row["home_team"], row["away_team"], row["date"]
        rest_home.append(min((d - last_match[ht]).days, 30) if ht in last_match else 7)
        rest_away.append(min((d - last_match[at]).days, 30) if at in last_match else 7)
        last_match[ht] = d
        last_match[at] = d
    df["rest_days_home"] = rest_home
    df["rest_days_away"] = rest_away

    # --- Referee stats (incremental, no look-ahead leakage) ---
    ref_fouls = {}  # referee -> running list of total fouls
    ref_yellows = {}
    ref_home_wins = {}
    ref_total = {}
    ref_avg_fouls_list = []
    ref_avg_yellows_list = []
    ref_home_bias_list = []

    for _, row in df.iterrows():
        ref = row["referee"]
        if pd.isna(ref):
            ref_avg_fouls_list.append(np.nan)
            ref_avg_yellows_list.append(np.nan)
            ref_home_bias_list.append(np.nan)
            continue

        # Record PRE-match stats (before this game)
        if ref in ref_fouls and len(ref_fouls[ref]) > 0:
            ref_avg_fouls_list.append(np.mean(ref_fouls[ref]))
            ref_avg_yellows_list.append(np.mean(ref_yellows[ref]))
            hw_rate = ref_home_wins.get(ref, 0) / max(ref_total.get(ref, 1), 1)
            ref_home_bias_list.append(hw_rate - 0.46)  # baseline home win rate ~46%
        else:
            ref_avg_fouls_list.append(np.nan)
            ref_avg_yellows_list.append(np.nan)
            ref_home_bias_list.append(np.nan)

        # Update with this match
        total_fouls = (row.get("fouls_home", 0) or 0) + (row.get("fouls_away", 0) or 0)
        total_yellows = (row.get("yellows_home", 0) or 0) + (row.get("yellows_away", 0) or 0)
        ref_fouls.setdefault(ref, []).append(total_fouls)
        ref_yellows.setdefault(ref, []).append(total_yellows)
        ref_total[ref] = ref_total.get(ref, 0) + 1
        if row["result"] == "H":
            ref_home_wins[ref] = ref_home_wins.get(ref, 0) + 1

    df["referee_avg_fouls"] = ref_avg_fouls_list
    df["referee_avg_yellows"] = ref_avg_yellows_list
    df["referee_home_bias"] = ref_home_bias_list

    # --- Odds: map to pipeline-expected column names ---
    # Priority: Pinnacle > Bet365 close > Market avg close > Bet365 open > Market avg open
    for suffix in ["home", "draw", "away"]:
        # Open odds
        open_candidates = [
            f"odds_pin_{suffix}", f"odds_b365_{suffix}",
            f"odds_avg_{suffix}", f"odds_iw_{suffix}", f"odds_wh_{suffix}",
        ]
        df[f"odds_{suffix}_open"] = np.nan
        for col in open_candidates:
            if col in df.columns:
                df[f"odds_{suffix}_open"] = df[f"odds_{suffix}_open"].fillna(
                    pd.to_numeric(df[col], errors="coerce")
                )

        # Close odds
        close_candidates = [
            f"odds_pin_{suffix}_close", f"odds_b365_{suffix}_close",
            f"odds_avg_{suffix}_close",
        ]
        df[f"odds_{suffix}_close"] = np.nan
        for col in close_candidates:
            if col in df.columns:
                df[f"odds_{suffix}_close"] = df[f"odds_{suffix}_close"].fillna(
                    pd.to_numeric(df[col], errors="coerce")
                )
        # Ultimate fallback: use open odds as close
        df[f"odds_{suffix}_close"] = df[f"odds_{suffix}_close"].fillna(
            df[f"odds_{suffix}_open"]
        )

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_dataset(skip_weather: bool = False, skip_elo: bool = False,
                  skip_xg: bool = False, skip_apis: bool = False,
                  fdo_key: Optional[str] = None, apif_key: Optional[str] = None,
                  news_key: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Build the full real-data dataset from multiple sources.

    Returns:
        (historical_df, upcoming_df, extra_data) — matches, fixtures, and extra data dict
    """
    try:
        from data.api_client import UnifiedDataFetcher
    except ImportError:
        from api_client import UnifiedDataFetcher

    print("=" * 65)
    print("  MatchOracle — Real Data Pipeline (20 seasons, multi-source)")
    print("=" * 65)

    extra_data = {"sentiment": {}, "standings": {}, "injuries": []}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── Parallel Phase: Fetch independent data sources concurrently ──
    # Sources 1 (football-data.co.uk) must complete first as other merges need it.
    # Sources 2-4 (Understat, Elo, Weather) + 5-6 (APIs, fixtures) can overlap.
    print("\n[1/7] football-data.co.uk — results, stats, odds, referee")
    df = fetch_footballdata(SEASON_CODES, SEASON_LABELS)

    # Normalise team names
    df["home_team"] = df["home_team"].apply(normalise_team)
    df["away_team"] = df["away_team"].apply(normalise_team)

    # ── Launch parallel fetches for sources 2-6 ──
    print("\n  Launching parallel data fetches ...")
    parallel_results = {}

    def _fetch_xg():
        if skip_xg:
            return None
        print("\n[2/7] Understat — match xG (2014-15 onwards)")
        return fetch_understat_xg(UNDERSTAT_SEASONS)

    def _fetch_elo():
        if skip_elo:
            return None
        print("\n[3/7] Club Elo — pre-match team ratings")
        date_strs = df["date"].dt.strftime("%Y-%m-%d").tolist()
        return fetch_club_elo(date_strs)

    def _fetch_apis():
        if skip_apis or not any([fdo_key, apif_key, news_key]):
            return None
        print("\n[5/7] API integrations (Football-Data.org, API-Football, NewsAPI)")
        fetcher = UnifiedDataFetcher(fdo_key=fdo_key, apif_key=apif_key, news_key=news_key)
        api_extra = {"sentiment": {}, "standings": {}, "injuries": []}
        if fdo_key:
            try:
                seasons_to_fetch = list(range(2020, 2026))
                api_extra["standings"] = fetcher.fetch_fdo_standings(seasons_to_fetch)
            except Exception as e:
                print(f"    FDO standings error: {e}")
        if apif_key:
            try:
                api_extra["injuries"] = fetcher.fetch_injuries(2024)
            except Exception as e:
                print(f"    API-Football injuries error: {e}")
        if news_key:
            try:
                current_teams = sorted(set(
                    df[df["season"] == df["season"].iloc[-1]]["home_team"].unique()
                ))
                api_extra["sentiment"] = fetcher.fetch_team_sentiment(current_teams, days_back=7)
            except Exception as e:
                print(f"    NewsAPI sentiment error: {e}")
        fetcher.print_stats()
        return api_extra

    def _fetch_fixtures():
        print("\n[6/7] Upcoming fixtures")
        upcoming = fetch_upcoming_fixtures()
        if upcoming.empty and any([fdo_key, apif_key]):
            print("  Trying APIs for upcoming fixtures ...")
            upcoming = fetch_live_upcoming_from_api(fdo_key=fdo_key, apif_key=apif_key)
        return upcoming

    # Run all independent fetches in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_xg = pool.submit(_fetch_xg)
        future_elo = pool.submit(_fetch_elo)
        future_apis = pool.submit(_fetch_apis)
        future_fix = pool.submit(_fetch_fixtures)

        # Collect results as they complete
        xg_df = future_xg.result()
        elo_data = future_elo.result()
        api_result = future_apis.result()
        upcoming = future_fix.result()

    # ── Merge xG ──
    if xg_df is not None and not xg_df.empty:
        xg_df["home_team_us"] = xg_df["home_team_us"].apply(normalise_team)
        xg_df["away_team_us"] = xg_df["away_team_us"].apply(normalise_team)
        df["_merge_date"] = df["date"].dt.date
        xg_df["_merge_date"] = xg_df["date_us"].dt.date
        df = df.merge(
            xg_df[["_merge_date", "home_team_us", "away_team_us", "xg_home", "xg_away"]],
            left_on=["_merge_date", "home_team", "away_team"],
            right_on=["_merge_date", "home_team_us", "away_team_us"],
            how="left",
        )
        df.drop(columns=["home_team_us", "away_team_us"], errors="ignore", inplace=True)
        df.drop(columns=["_merge_date"], errors="ignore", inplace=True)
        print(f"  => xG matched: {df['xg_home'].notna().sum()}/{len(df)}")
    else:
        if skip_xg:
            print("\n[2/7] Understat — SKIPPED")
        df["xg_home"] = np.nan
        df["xg_away"] = np.nan

    # ── Merge Elo ──
    if elo_data is not None:
        df = merge_elo(df, elo_data)
    else:
        if skip_elo:
            print("\n[3/7] Club Elo — SKIPPED")
        df["elo_pre_home"] = np.nan
        df["elo_pre_away"] = np.nan

    # ── Weather (needs df rows, runs after core data is ready) ──
    if not skip_weather:
        print("\n[4/7] Open-Meteo — historical weather")
        df = fetch_weather_batch(df)
    else:
        print("\n[4/7] Open-Meteo — SKIPPED")
        for col in ["temperature", "humidity", "wind_speed", "precipitation"]:
            df[col] = np.nan

    # ── Merge API results ──
    if api_result:
        extra_data.update(api_result)
    elif not skip_apis:
        print("\n[5/7] API integrations — SKIPPED")

    # 7. Compute derived columns
    print("\n[7/7] Computing derived columns ...")
    df = compute_derived_columns(df)

    # Cleanup — drop raw intermediate odds columns
    raw_odds = [c for c in df.columns if c.startswith("odds_") and
                any(c.startswith(f"odds_{bk}") for bk in
                    ["b365", "pin", "avg", "max", "iw", "wh"])
                and c not in [f"odds_{s}_{t}" for s in ["home", "draw", "away"]
                              for t in ["open", "close"]]]
    df.drop(columns=[c for c in raw_odds if c in df.columns], inplace=True, errors="ignore")

    df = df.sort_values("date").reset_index(drop=True)

    return df, upcoming, extra_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def fetch_live_upcoming_from_api(fdo_key: Optional[str] = None,
                                 apif_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch upcoming fixtures from APIs when football-data.co.uk fixtures are stale."""
    try:
        from data.api_client import FootballDataOrgClient, APIFootballClient
    except ImportError:
        from api_client import FootballDataOrgClient, APIFootballClient

    frames = []

    # Try Football-Data.org first
    if fdo_key:
        try:
            client = FootballDataOrgClient(fdo_key)
            data = client.get_matches(status="SCHEDULED")
            if data and "matches" in data:
                for m in data["matches"]:
                    ht = normalise_team(m.get("homeTeam", {}).get("name", ""))
                    at = normalise_team(m.get("awayTeam", {}).get("name", ""))
                    dt = m.get("utcDate", "")[:10]
                    if ht and at:
                        frames.append({"date": dt, "home_team": ht, "away_team": at})
        except Exception as e:
            print(f"    FDO upcoming error: {e}")

    # Fallback to API-Football
    if not frames and apif_key:
        try:
            client = APIFootballClient(apif_key)
            from datetime import datetime as _dt
            _current_season = _dt.now().year if _dt.now().month >= 8 else _dt.now().year - 1
            data = client.get_fixtures(_current_season, round_="next")
            if data and "response" in data:
                for m in data["response"]:
                    ht = normalise_team(m.get("teams", {}).get("home", {}).get("name", ""))
                    at = normalise_team(m.get("teams", {}).get("away", {}).get("name", ""))
                    dt = m.get("fixture", {}).get("date", "")[:10]
                    if ht and at:
                        frames.append({"date": dt, "home_team": ht, "away_team": at})
        except Exception as e:
            print(f"    APIF upcoming error: {e}")

    if frames:
        df = pd.DataFrame(frames)
        df["date"] = pd.to_datetime(df["date"])
        print(f"  => {len(df)} upcoming fixtures from API")
        return df
    return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MatchOracle Real Data Pipeline")
    parser.add_argument("--skip-weather", action="store_true",
                        help="Skip weather data (faster)")
    parser.add_argument("--skip-elo", action="store_true",
                        help="Skip Club Elo data (faster)")
    parser.add_argument("--skip-xg", action="store_true",
                        help="Skip Understat xG data (faster)")
    parser.add_argument("--skip-apis", action="store_true",
                        help="Skip API integrations (Football-Data.org, API-Football, NewsAPI)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip all external APIs except football-data.co.uk")
    parser.add_argument("--fdo-key", type=str, default=os.environ.get("FDO_KEY"),
                        help="Football-Data.org API key (or set FDO_KEY in .env)")
    parser.add_argument("--apif-key", type=str, default=os.environ.get("APIF_KEY"),
                        help="API-Football API key (or set APIF_KEY in .env)")
    parser.add_argument("--news-key", type=str, default=os.environ.get("NEWS_KEY"),
                        help="NewsAPI API key (or set NEWS_KEY in .env)")
    args = parser.parse_args()

    if args.fast:
        args.skip_weather = args.skip_elo = args.skip_xg = args.skip_apis = True

    df, upcoming, extra_data = build_dataset(
        skip_weather=args.skip_weather,
        skip_elo=args.skip_elo,
        skip_xg=args.skip_xg,
        skip_apis=args.skip_apis,
        fdo_key=args.fdo_key,
        apif_key=args.apif_key,
        news_key=args.news_key,
    )

    # Save historical data (Parquet for performance + type preservation)
    out_path = DATA_DIR / "epl_matches.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # Save extra data (sentiment, standings, injuries)
    import json as _json
    extra_path = DATA_DIR / "extra_data.json"
    with open(extra_path, "w") as f:
        _json.dump(extra_data, f, indent=2, default=str)

    # Save upcoming fixtures (overwrite even if empty to clear stale data)
    upcoming_path = DATA_DIR / "upcoming_fixtures.parquet"
    if not upcoming.empty:
        upcoming.to_parquet(upcoming_path, index=False, engine="pyarrow")
    elif upcoming_path.exists():
        upcoming_path.unlink()

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  DONE — {len(df)} real matches saved to {out_path}")
    print(f"{'=' * 65}")
    print(f"  Seasons:  {df['season'].nunique()} ({df['season'].min()} to {df['season'].max()})")
    print(f"  Teams:    {df['home_team'].nunique()} unique teams")
    print(f"  Columns:  {len(df.columns)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\n  Data coverage:")
    print(f"    xG:        {df['xg_home'].notna().sum():>5}/{len(df)} matches (Understat, 2014+)")
    print(f"    Elo:       {df['elo_pre_home'].notna().sum():>5}/{len(df)} matches (Club Elo)")
    print(f"    Weather:   {df['temperature'].notna().sum():>5}/{len(df)} matches (Open-Meteo)")
    print(f"    Odds:      {df['odds_home_open'].notna().sum():>5}/{len(df)} matches")
    print(f"    Referee:   {df['referee'].notna().sum():>5}/{len(df)} matches")
    print(f"    Sentiment: {len(extra_data.get('sentiment', {}))} teams (NewsAPI)")
    print(f"    Injuries:  {len(extra_data.get('injuries', []))} entries (API-Football)")
    if not upcoming.empty:
        print(f"\n  Upcoming fixtures: {len(upcoming)}")
        print(upcoming.to_string(index=False))
    print(f"\n  Sample data:")
    print(df[["date", "season", "home_team", "away_team", "goals_home", "goals_away",
              "xg_home", "xg_away", "result"]].tail(10).to_string(index=False))
