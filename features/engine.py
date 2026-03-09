"""
MatchOracle Feature Engineering Pipeline
==========================================
Computes 376+ features across 24 categories from real EPL data.
Handles gracefully: missing xG (pre-2014), missing possession/pass accuracy,
missing weather, missing odds (older seasons).

All features are computed incrementally (no look-ahead leakage).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class FeatureEngine:
    """Compute all features for match prediction."""

    ROLLING_WINDOWS = [3, 5, 8, 10, 15, 20]

    def __init__(self, df: pd.DataFrame, sentiment_data: Optional[Dict] = None,
                 injury_data: Optional[List] = None):
        self.raw = df.sort_values("date").reset_index(drop=True)
        self.raw["date"] = pd.to_datetime(self.raw["date"])
        self.teams = sorted(set(self.raw["home_team"].unique()) |
                            set(self.raw["away_team"].unique()))
        self.sentiment_data = sentiment_data or {}
        self.injury_data = injury_data or []

        # Pre-compute injury counts per team
        self.injury_counts = {}
        for inj in self.injury_data:
            team = inj.get("team", "")
            self.injury_counts[team] = self.injury_counts.get(team, 0) + 1

        # Detect which optional columns are available
        self.has_xg = "xg_home" in self.raw.columns and self.raw["xg_home"].notna().any()
        self.has_possession = "possession_home" in self.raw.columns
        self.has_pass_accuracy = "pass_accuracy_home" in self.raw.columns
        self.has_weather = "temperature" in self.raw.columns
        self.has_odds = "odds_home_open" in self.raw.columns
        self.has_elo_pre = "elo_pre_home" in self.raw.columns
        self.has_sentiment = len(self.sentiment_data) > 0
        self.has_injuries = len(self.injury_counts) > 0

    def compute_all_features(self) -> pd.DataFrame:
        """Master feature computation pipeline."""
        df = self.raw.copy()

        print(f"  Computing features for {len(df)} matches, {len(self.teams)} teams ...")
        print(f"  Optional data: xG={'yes' if self.has_xg else 'no'}, "
              f"possession={'yes' if self.has_possession else 'no'}, "
              f"weather={'yes' if self.has_weather else 'no'}, "
              f"odds={'yes' if self.has_odds else 'no'}, "
              f"sentiment={'yes' if self.has_sentiment else 'no'}")

        # 1. Elo ratings (always computed from results — independent of external Elo)
        df = self._compute_elo_ratings(df)

        # 2. Pi-ratings
        df = self._compute_pi_ratings(df)

        # 3. Rolling form
        df = self._compute_rolling_form(df)

        # 4. Head-to-head
        df = self._compute_h2h(df)

        # 5. Contextual
        df = self._compute_contextual(df)

        # 6. Betting market
        if self.has_odds:
            df = self._compute_market_features(df)

        # 7. Interactions
        df = self._compute_interactions(df)

        # 8. League table (per-season)
        df = self._compute_table_features(df)

        # 9. Advanced stats
        df = self._compute_advanced_stats(df)

        # 10. External Elo features (if available from Club Elo)
        if self.has_elo_pre:
            df = self._compute_external_elo_features(df)

        # 11. NLP Sentiment features (if available from NewsAPI)
        if self.has_sentiment:
            df = self._compute_sentiment_features(df)

        # 12. Injury impact features (if available from API-Football)
        if self.has_injuries:
            df = self._compute_injury_features(df)

        # 13. Momentum & streaks (advanced)
        df = self._compute_momentum_features(df)

        # 14. Surprise factor (upset detection)
        df = self._compute_surprise_features(df)

        # 15. Glicko-2 ratings (rating + uncertainty + volatility)
        df = self._compute_glicko2_ratings(df)

        # 16. Shin-adjusted market probabilities (true odds)
        if self.has_odds:
            df = self._compute_shin_probabilities(df)

        # 17. Match sequence features (last N results encoding)
        df = self._compute_sequence_features(df)

        # 18. Managerial stability features
        df = self._compute_manager_features(df)

        # 19. GK quality proxy (clean sheet + goals conceded patterns)
        df = self._compute_gk_quality(df)

        # 20. Poisson goal expectation features
        df = self._compute_poisson_features(df)

        # 21. Expected points features (luck / regression indicator)
        df = self._compute_expected_points_features(df)

        # 22. Rest days features
        df = self._compute_rest_days_features(df)

        # 23. Scoring patterns (when goals are scored)
        df = self._compute_scoring_patterns(df)

        # 24. League position features (enhanced)
        df = self._compute_league_position_features(df)

        # Clean up infinities
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    # ------------------------------------------------------------------
    # 1. Elo Ratings (computed from results, independent of Club Elo)
    # ------------------------------------------------------------------
    def _compute_elo_ratings(self, df: pd.DataFrame, k: float = 20.0,
                              home_adv: float = 65.0,
                              season_regression: float = 0.1) -> pd.DataFrame:
        """Dynamic Elo with goal-margin scaling, home advantage, and
        season-boundary mean-reversion."""
        elo = {team: 1500.0 for team in self.teams}

        elo_home_list = []
        elo_away_list = []
        elo_diff_list = []
        elo_momentum_home = []
        elo_momentum_away = []
        recent_changes = {team: [] for team in self.teams}

        prev_season = None

        for idx, row in df.iterrows():
            season = row.get("season", "")

            # Season-boundary regression to mean
            if prev_season is not None and season != prev_season:
                for team in self.teams:
                    elo[team] = elo[team] * (1 - season_regression) + 1500.0 * season_regression

            prev_season = season
            ht, at = row["home_team"], row["away_team"]

            elo_home_list.append(elo[ht])
            elo_away_list.append(elo[at])
            elo_diff_list.append(elo[ht] - elo[at] + home_adv)

            # Momentum (avg change over last 5)
            elo_momentum_home.append(
                np.mean(recent_changes[ht][-5:]) if recent_changes[ht] else 0.0
            )
            elo_momentum_away.append(
                np.mean(recent_changes[at][-5:]) if recent_changes[at] else 0.0
            )

            # Expected scores
            exp_home = 1.0 / (1.0 + 10 ** (-(elo[ht] - elo[at] + home_adv) / 400))

            # Actual scores
            gh, ga = row["goals_home"], row["goals_away"]
            if gh > ga:
                actual_home = 1.0
            elif gh < ga:
                actual_home = 0.0
            else:
                actual_home = 0.5

            # Goal margin scaling
            goal_diff = abs(gh - ga)
            margin_scale = np.log(1 + goal_diff) * 0.7 + 1.0

            # Update
            change_home = k * margin_scale * (actual_home - exp_home)
            elo[ht] += change_home
            elo[at] -= change_home

            recent_changes[ht].append(change_home)
            recent_changes[at].append(-change_home)

        df["elo_home"] = elo_home_list
        df["elo_away"] = elo_away_list
        df["elo_diff"] = elo_diff_list
        df["elo_momentum_home"] = elo_momentum_home
        df["elo_momentum_away"] = elo_momentum_away
        df["elo_momentum_diff"] = df["elo_momentum_home"] - df["elo_momentum_away"]

        return df

    # ------------------------------------------------------------------
    # 2. Pi-Ratings
    # ------------------------------------------------------------------
    def _compute_pi_ratings(self, df: pd.DataFrame,
                             lr: float = 0.05, gamma: float = 0.6) -> pd.DataFrame:
        """Pi-ratings with separate home/away attack/defense (Constantinou & Fenton 2013)."""
        pi = {}
        for team in self.teams:
            pi[team] = {
                "home_attack": 1.0, "home_defense": 1.0,
                "away_attack": 1.0, "away_defense": 1.0,
            }

        cols = {k: [] for k in ["pi_home_attack", "pi_home_defense",
                                "pi_away_attack", "pi_away_defense",
                                "pi_rating_home", "pi_rating_away", "pi_diff"]}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            cols["pi_home_attack"].append(pi[ht]["home_attack"])
            cols["pi_home_defense"].append(pi[ht]["home_defense"])
            cols["pi_away_attack"].append(pi[at]["away_attack"])
            cols["pi_away_defense"].append(pi[at]["away_defense"])

            pi_home = pi[ht]["home_attack"] - pi[at]["away_defense"]
            pi_away = pi[at]["away_attack"] - pi[ht]["home_defense"]
            cols["pi_rating_home"].append(pi_home)
            cols["pi_rating_away"].append(pi_away)
            cols["pi_diff"].append(pi_home - pi_away)

            exp_home = max(0.1, 1.36 * np.exp(pi_home))
            exp_away = max(0.1, 1.36 * np.exp(pi_away))

            gh, ga = row["goals_home"], row["goals_away"]
            disc_home = (gh - exp_home) / max(exp_home, 0.5)
            disc_away = (ga - exp_away) / max(exp_away, 0.5)

            pi[ht]["home_attack"] += lr * disc_home
            pi[at]["away_defense"] -= lr * gamma * disc_home
            pi[at]["away_attack"] += lr * disc_away
            pi[ht]["home_defense"] -= lr * gamma * disc_away

        for k, v in cols.items():
            df[k] = v

        return df

    # ------------------------------------------------------------------
    # 3. Rolling form
    # ------------------------------------------------------------------
    def _compute_rolling_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling form features — handles missing xG/possession/pass_accuracy gracefully."""
        team_history = {team: [] for team in self.teams}
        feature_names = self._get_rolling_feature_names()
        feature_data = {f: [] for f in feature_names}

        for idx, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            for prefix, team, is_home in [("home", ht, True), ("away", at, False)]:
                history = team_history[team]

                for window in self.ROLLING_WINDOWS:
                    recent = history[-window:] if len(history) >= window else history
                    n = len(recent)
                    suffix = f"_{prefix}_l{window}"

                    if n == 0:
                        for metric in self._rolling_metrics():
                            feature_data[f"{metric}{suffix}"].append(np.nan)
                    else:
                        pts = [m["points"] for m in recent]
                        feature_data[f"ppg{suffix}"].append(np.mean(pts))
                        feature_data[f"goals_scored{suffix}"].append(
                            np.mean([m["goals_for"] for m in recent]))
                        feature_data[f"goals_conceded{suffix}"].append(
                            np.mean([m["goals_against"] for m in recent]))

                        # xG — safely handle NaN
                        xg_for = [m["xg_for"] for m in recent if not np.isnan(m["xg_for"])]
                        xg_against = [m["xg_against"] for m in recent if not np.isnan(m["xg_against"])]
                        feature_data[f"xg_for{suffix}"].append(
                            np.mean(xg_for) if xg_for else np.nan)
                        feature_data[f"xg_against{suffix}"].append(
                            np.mean(xg_against) if xg_against else np.nan)

                        # xG overperformance
                        xg_over = [m["goals_for"] - m["xg_for"] for m in recent
                                   if not np.isnan(m["xg_for"])]
                        feature_data[f"xg_overperformance{suffix}"].append(
                            np.mean(xg_over) if xg_over else np.nan)

                        feature_data[f"clean_sheets{suffix}"].append(
                            np.mean([1 if m["goals_against"] == 0 else 0 for m in recent]))
                        feature_data[f"btts_pct{suffix}"].append(
                            np.mean([1 if m["goals_for"] > 0 and m["goals_against"] > 0
                                     else 0 for m in recent]))
                        feature_data[f"over25_pct{suffix}"].append(
                            np.mean([1 if m["goals_for"] + m["goals_against"] > 2
                                     else 0 for m in recent]))
                        feature_data[f"win_pct{suffix}"].append(
                            np.mean([1 if m["points"] == 3 else 0 for m in recent]))
                        feature_data[f"draw_pct{suffix}"].append(
                            np.mean([1 if m["points"] == 1 else 0 for m in recent]))
                        feature_data[f"loss_pct{suffix}"].append(
                            np.mean([1 if m["points"] == 0 else 0 for m in recent]))

                        _sf = [m["shots_for"] for m in recent if not np.isnan(m["shots_for"])]
                        feature_data[f"shots_for{suffix}"].append(
                            np.mean(_sf) if _sf else np.nan)
                        _sa = [m["shots_against"] for m in recent if not np.isnan(m["shots_against"])]
                        feature_data[f"shots_against{suffix}"].append(
                            np.mean(_sa) if _sa else np.nan)

                        shots = [m["shots_for"] for m in recent if not np.isnan(m["shots_for"])]
                        sots = [m["sot_for"] for m in recent if not np.isnan(m["sot_for"])]
                        if shots and sum(shots) > 0:
                            feature_data[f"sot_pct{suffix}"].append(
                                sum(sots) / sum(shots) * 100)
                        else:
                            feature_data[f"sot_pct{suffix}"].append(np.nan)

                        # Possession — optional
                        poss = [m["possession"] for m in recent
                                if not np.isnan(m.get("possession", np.nan))]
                        feature_data[f"possession{suffix}"].append(
                            np.mean(poss) if poss else np.nan)

                        # Pass accuracy — optional
                        pa = [m["pass_accuracy"] for m in recent
                              if not np.isnan(m.get("pass_accuracy", np.nan))]
                        feature_data[f"pass_accuracy{suffix}"].append(
                            np.mean(pa) if pa else np.nan)

            # Add to team histories
            def _safe_get(r, col, default=np.nan):
                v = r.get(col, default)
                return v if pd.notna(v) else default

            team_history[ht].append({
                "points": row["home_points"],
                "goals_for": row["goals_home"],
                "goals_against": row["goals_away"],
                "xg_for": _safe_get(row, "xg_home"),
                "xg_against": _safe_get(row, "xg_away"),
                "shots_for": _safe_get(row, "shots_home"),
                "shots_against": _safe_get(row, "shots_away"),
                "sot_for": _safe_get(row, "sot_home"),
                "sot_against": _safe_get(row, "sot_away"),
                "possession": _safe_get(row, "possession_home"),
                "pass_accuracy": _safe_get(row, "pass_accuracy_home"),
                "is_home": True,
            })
            team_history[at].append({
                "points": row["away_points"],
                "goals_for": row["goals_away"],
                "goals_against": row["goals_home"],
                "xg_for": _safe_get(row, "xg_away"),
                "xg_against": _safe_get(row, "xg_home"),
                "shots_for": _safe_get(row, "shots_away"),
                "shots_against": _safe_get(row, "shots_home"),
                "sot_for": _safe_get(row, "sot_away"),
                "sot_against": _safe_get(row, "sot_home"),
                "possession": _safe_get(row, "possession_away"),
                "pass_accuracy": _safe_get(row, "pass_accuracy_away"),
                "is_home": False,
            })

        for k, v in feature_data.items():
            df[k] = v

        return df

    def _rolling_metrics(self) -> List[str]:
        return ["ppg", "goals_scored", "goals_conceded", "xg_for", "xg_against",
                "xg_overperformance", "clean_sheets", "btts_pct", "over25_pct",
                "win_pct", "draw_pct", "loss_pct", "shots_for", "shots_against",
                "sot_pct", "possession", "pass_accuracy"]

    def _get_rolling_feature_names(self) -> List[str]:
        names = []
        for m in self._rolling_metrics():
            for prefix in ["home", "away"]:
                for window in self.ROLLING_WINDOWS:
                    names.append(f"{m}_{prefix}_l{window}")
        return names

    # ------------------------------------------------------------------
    # 4. Head-to-head
    # ------------------------------------------------------------------
    def _compute_h2h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced head-to-head matchup analysis with venue dominance,
        goal patterns, xG matchups, style compatibility, and recency weighting."""
        # Full record: (goals_team1, goals_team2, team1_was_home, xg_t1, xg_t2, date)
        h2h_records: Dict[Tuple[str, str], list] = {}

        cols = {k: [] for k in [
            "h2h_home_win_pct", "h2h_draw_pct", "h2h_away_win_pct",
            "h2h_avg_goals", "h2h_matches",
            # Advanced pair features
            "h2h_venue_dominance",     # how much home team dominates AT HOME vs this opponent
            "h2h_goals_volatility",    # std of total goals in meetings
            "h2h_btts_rate",           # both teams score rate in H2H
            "h2h_over25_rate",         # over 2.5 goals rate in H2H
            "h2h_home_goals_avg",      # avg goals home team scores vs this opponent
            "h2h_away_goals_avg",      # avg goals away team scores vs this opponent
            "h2h_recent_shift",        # recent form shift (last 3 vs all H2H)
            "h2h_xg_dominance",        # xG advantage in H2H
            "h2h_home_scoring_trend",  # is home team scoring more/less vs this opponent recently?
            "h2h_clean_sheet_pct",     # how often home team keeps clean sheet vs opponent
        ]}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            key = (ht, at)
            rev_key = (at, ht)

            # Gather all meetings from home team's perspective
            all_meetings = []
            for rec in h2h_records.get(key, []):
                all_meetings.append((rec[0], rec[1], True, rec[3], rec[4]))
            for rec in h2h_records.get(rev_key, []):
                all_meetings.append((rec[1], rec[0], False, rec[4], rec[3]))

            if all_meetings:
                recent = all_meetings[-10:]
                n = len(recent)
                # From ht's perspective: g1=goals_for_ht, g2=goals_against_ht
                hw = sum(1 for g1, g2, *_ in recent if g1 > g2)
                dr = sum(1 for g1, g2, *_ in recent if g1 == g2)
                cols["h2h_home_win_pct"].append(hw / n)
                cols["h2h_draw_pct"].append(dr / n)
                cols["h2h_away_win_pct"].append(1 - hw / n - dr / n)
                total_goals = [g1 + g2 for g1, g2, *_ in recent]
                cols["h2h_avg_goals"].append(np.mean(total_goals))
                cols["h2h_matches"].append(n)

                # Venue dominance: win rate when THIS team is at home vs opponent
                home_meetings = [(g1, g2) for g1, g2, was_home, *_ in recent if was_home]
                if home_meetings:
                    cols["h2h_venue_dominance"].append(
                        sum(1 for g1, g2 in home_meetings if g1 > g2) / len(home_meetings))
                else:
                    cols["h2h_venue_dominance"].append(0.45)

                # Goal volatility
                cols["h2h_goals_volatility"].append(float(np.std(total_goals)) if len(total_goals) > 1 else 0.0)

                # BTTS and over 2.5 rates
                cols["h2h_btts_rate"].append(
                    np.mean([1 if g1 > 0 and g2 > 0 else 0 for g1, g2, *_ in recent]))
                cols["h2h_over25_rate"].append(
                    np.mean([1 if g1 + g2 > 2 else 0 for g1, g2, *_ in recent]))

                # Average goals per team
                cols["h2h_home_goals_avg"].append(np.mean([g1 for g1, g2, *_ in recent]))
                cols["h2h_away_goals_avg"].append(np.mean([g2 for g1, g2, *_ in recent]))

                # Recent shift: compare last 3 vs all (momentum in matchup)
                if n >= 4:
                    last3 = recent[-3:]
                    l3_wr = sum(1 for g1, g2, *_ in last3 if g1 > g2) / 3
                    all_wr = hw / n
                    cols["h2h_recent_shift"].append(l3_wr - all_wr)
                else:
                    cols["h2h_recent_shift"].append(0.0)

                # xG dominance
                xg_diffs = [xg1 - xg2 for g1, g2, _, xg1, xg2 in recent
                            if not np.isnan(xg1) and not np.isnan(xg2)]
                cols["h2h_xg_dominance"].append(np.mean(xg_diffs) if xg_diffs else 0.0)

                # Home scoring trend (is ht scoring more recently?)
                if n >= 4:
                    first_half = [g1 for g1, *_ in recent[:n // 2]]
                    second_half = [g1 for g1, *_ in recent[n // 2:]]
                    cols["h2h_home_scoring_trend"].append(
                        np.mean(second_half) - np.mean(first_half))
                else:
                    cols["h2h_home_scoring_trend"].append(0.0)

                # Clean sheet rate
                cols["h2h_clean_sheet_pct"].append(
                    np.mean([1 if g2 == 0 else 0 for g1, g2, *_ in recent]))

            else:
                # No history — use league averages
                cols["h2h_home_win_pct"].append(0.45)
                cols["h2h_draw_pct"].append(0.27)
                cols["h2h_away_win_pct"].append(0.28)
                cols["h2h_avg_goals"].append(2.7)
                cols["h2h_matches"].append(0)
                cols["h2h_venue_dominance"].append(0.45)
                cols["h2h_goals_volatility"].append(0.0)
                cols["h2h_btts_rate"].append(0.5)
                cols["h2h_over25_rate"].append(0.5)
                cols["h2h_home_goals_avg"].append(1.5)
                cols["h2h_away_goals_avg"].append(1.2)
                cols["h2h_recent_shift"].append(0.0)
                cols["h2h_xg_dominance"].append(0.0)
                cols["h2h_home_scoring_trend"].append(0.0)
                cols["h2h_clean_sheet_pct"].append(0.3)

            # Store this result
            xg_h = row.get("xg_home", np.nan)
            xg_a = row.get("xg_away", np.nan)
            if not isinstance(xg_h, (int, float)):
                xg_h = np.nan
            if not isinstance(xg_a, (int, float)):
                xg_a = np.nan
            h2h_records.setdefault(key, []).append(
                (row["goals_home"], row["goals_away"], True, xg_h, xg_a)
            )

        for k, v in cols.items():
            df[k] = v

        return df

    # ------------------------------------------------------------------
    # 5. Contextual
    # ------------------------------------------------------------------
    def _compute_contextual(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rest_diff"] = df.get("rest_days_home", 7) - df.get("rest_days_away", 7)
        df["is_midweek"] = df["date"].dt.dayofweek.isin([1, 2, 3]).astype(int)
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek

        gw = df.get("gameweek", pd.Series(19, index=df.index))
        df["season_progress"] = gw / 38.0
        df["is_early_season"] = (gw <= 5).astype(int)
        df["is_late_season"] = (gw >= 33).astype(int)
        df["is_run_in"] = (gw >= 30).astype(int)

        if self.has_weather:
            df["is_cold"] = (df["temperature"].fillna(10) < 5).astype(int)
            df["is_hot"] = (df["temperature"].fillna(10) > 25).astype(int)
            df["is_windy"] = (df["wind_speed"].fillna(10) > 25).astype(int)
        else:
            df["is_cold"] = 0
            df["is_hot"] = 0
            df["is_windy"] = 0

        return df

    # ------------------------------------------------------------------
    # 6. Betting market features
    # ------------------------------------------------------------------
    def _compute_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for prefix in ["open", "close"]:
            raw_h = 1.0 / pd.to_numeric(df.get(f"odds_home_{prefix}"), errors="coerce")
            raw_d = 1.0 / pd.to_numeric(df.get(f"odds_draw_{prefix}"), errors="coerce")
            raw_a = 1.0 / pd.to_numeric(df.get(f"odds_away_{prefix}"), errors="coerce")
            total = raw_h + raw_d + raw_a

            df[f"implied_prob_home_{prefix}"] = raw_h / total
            df[f"implied_prob_draw_{prefix}"] = raw_d / total
            df[f"implied_prob_away_{prefix}"] = raw_a / total
            df[f"market_overround_{prefix}"] = total

        # Odds movement
        df["odds_movement_home"] = (df.get("implied_prob_home_close", np.nan) -
                                    df.get("implied_prob_home_open", np.nan))
        df["odds_movement_draw"] = (df.get("implied_prob_draw_close", np.nan) -
                                    df.get("implied_prob_draw_open", np.nan))
        df["odds_movement_away"] = (df.get("implied_prob_away_close", np.nan) -
                                    df.get("implied_prob_away_open", np.nan))

        df["steam_move_flag"] = (df["odds_movement_home"].abs() > 0.05).astype(int)

        df["market_favorite"] = np.where(
            df.get("implied_prob_home_close", 0) > df.get("implied_prob_away_close", 0), 1,
            np.where(df.get("implied_prob_away_close", 0) > df.get("implied_prob_home_close", 0),
                     -1, 0)
        )

        return df

    # ------------------------------------------------------------------
    # 7. Interactions
    # ------------------------------------------------------------------
    def _compute_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df["elo_rest_interaction_home"] = df["elo_home"] * np.log1p(
            df.get("rest_days_home", 7))
        df["elo_rest_interaction_away"] = df["elo_away"] * np.log1p(
            df.get("rest_days_away", 7))

        if "ppg_home_l5" in df.columns:
            df["form_home_advantage"] = df["ppg_home_l5"].fillna(1.5) * 1.1
            df["form_away_disadvantage"] = df["ppg_away_l5"].fillna(1.5) * 0.9

        df["rating_concordance"] = (
            np.sign(df["elo_diff"]) == np.sign(df["pi_diff"])
        ).astype(int)

        df["derby_intensity"] = df.get("is_derby", 0) * df.get("season_progress", 0.5)

        if self.has_weather:
            df["weather_impact"] = (df["precipitation"].fillna(0) *
                                    df["wind_speed"].fillna(0) / 100)
        else:
            df["weather_impact"] = 0

        return df

    # ------------------------------------------------------------------
    # 8. League table (per-season, only teams active in that season)
    # ------------------------------------------------------------------
    def _compute_table_features(self, df: pd.DataFrame) -> pd.DataFrame:
        table_pos_home = []
        table_pos_away = []
        table_pts_home = []
        table_pts_away = []
        table_gd_home = []
        table_gd_away = []

        season_tables: Dict[str, Dict[str, Dict[str, int]]] = {}
        season_teams: Dict[str, set] = {}

        for _, row in df.iterrows():
            season = row["season"]
            ht, at = row["home_team"], row["away_team"]

            # Initialise season if new
            if season not in season_tables:
                season_tables[season] = {}
                season_teams[season] = set()

            # Add teams to season set if first appearance
            for t in [ht, at]:
                if t not in season_tables[season]:
                    season_tables[season][t] = {"pts": 0, "gd": 0, "gf": 0, "ga": 0}
                    season_teams[season].add(t)

            table = season_tables[season]

            # Sort only teams that have played this season
            active = season_teams[season]
            sorted_teams = sorted(
                active,
                key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]),
                reverse=True
            )
            positions = {t: i + 1 for i, t in enumerate(sorted_teams)}

            table_pos_home.append(positions.get(ht, 10))
            table_pos_away.append(positions.get(at, 10))
            table_pts_home.append(table[ht]["pts"])
            table_pts_away.append(table[at]["pts"])
            table_gd_home.append(table[ht]["gd"])
            table_gd_away.append(table[at]["gd"])

            # Update table
            table[ht]["pts"] += row["home_points"]
            table[at]["pts"] += row["away_points"]
            table[ht]["gd"] += row["goals_home"] - row["goals_away"]
            table[at]["gd"] += row["goals_away"] - row["goals_home"]
            table[ht]["gf"] += row["goals_home"]
            table[at]["gf"] += row["goals_away"]
            table[ht]["ga"] += row["goals_away"]
            table[at]["ga"] += row["goals_home"]

        df["table_pos_home"] = table_pos_home
        df["table_pos_away"] = table_pos_away
        df["table_pos_diff"] = np.array(table_pos_away) - np.array(table_pos_home)
        df["table_pts_home"] = table_pts_home
        df["table_pts_away"] = table_pts_away
        df["table_pts_diff"] = np.array(table_pts_home) - np.array(table_pts_away)
        df["table_gd_home"] = table_gd_home
        df["table_gd_away"] = table_gd_away

        n = len(df)
        df["home_in_top4"] = (np.array(table_pos_home) <= 4).astype(int)
        df["away_in_top4"] = (np.array(table_pos_away) <= 4).astype(int)
        df["home_in_relegation"] = (np.array(table_pos_home) >= 18).astype(int)
        df["away_in_relegation"] = (np.array(table_pos_away) >= 18).astype(int)

        return df

    # ------------------------------------------------------------------
    # 9. Advanced stats
    # ------------------------------------------------------------------
    def _compute_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in [5, 10]:
            col = f"ppg_home_l{window}"
            if col in df.columns:
                df[f"weighted_streak_home_l{window}"] = (
                    df[col] * (1 + df["elo_momentum_home"] / 20))
            col = f"ppg_away_l{window}"
            if col in df.columns:
                df[f"weighted_streak_away_l{window}"] = (
                    df[col] * (1 + df["elo_momentum_away"] / 20))

        df["attack_def_mismatch_home"] = df["pi_home_attack"] + df["pi_away_defense"]
        df["attack_def_mismatch_away"] = df["pi_away_attack"] + df["pi_home_defense"]

        df["combined_rating_diff"] = (
            df["elo_diff"] / 200 +
            df["pi_diff"] * 2 +
            df["table_pos_diff"] / 5
        ) / 3

        # ----- Style matchup features -----

        # Attack quality ratio: how attacking each team is (goals + xG) vs opponent defense
        for side, opp in [("home", "away"), ("away", "home")]:
            gs_col = f"goals_scored_{side}_l5"
            gc_col = f"goals_conceded_{opp}_l5"
            if gs_col in df.columns and gc_col in df.columns:
                df[f"attack_vs_defense_{side}"] = (
                    df[gs_col].fillna(1.3) / df[gc_col].clip(lower=0.3).fillna(1.3))

        # Goal expectation from both sides (combinatorial)
        if "goals_scored_home_l5" in df.columns and "goals_conceded_away_l5" in df.columns:
            df["expected_home_goals"] = (
                df["goals_scored_home_l5"].fillna(1.3) +
                df["goals_conceded_away_l5"].fillna(1.3)) / 2
        if "goals_scored_away_l5" in df.columns and "goals_conceded_home_l5" in df.columns:
            df["expected_away_goals"] = (
                df["goals_scored_away_l5"].fillna(1.1) +
                df["goals_conceded_home_l5"].fillna(1.1)) / 2

        # Form-Elo divergence (is form tracking Elo or diverging?)
        for side in ["home", "away"]:
            ppg = f"ppg_{side}_l5"
            if ppg in df.columns:
                elo_expected = 1 / (1 + 10 ** (-(df[f"elo_{side}"] - 1500) / 400))
                actual_form = df[ppg].fillna(1.5) / 3.0
                df[f"form_elo_divergence_{side}"] = actual_form - elo_expected

        # Table pressure (context of where team is in table)
        if "table_pos_home" in df.columns:
            df["position_pressure_home"] = np.where(
                df["table_pos_home"] <= 4, 1.0,
                np.where(df["table_pos_home"] >= 18, 1.0,
                         np.where(df["table_pos_home"].between(5, 7), 0.5, 0.0)))
            df["position_pressure_away"] = np.where(
                df["table_pos_away"] <= 4, 1.0,
                np.where(df["table_pos_away"] >= 18, 1.0,
                         np.where(df["table_pos_away"].between(5, 7), 0.5, 0.0)))

        # Defensive solidity indicator
        for side in ["home", "away"]:
            gc = f"goals_conceded_{side}_l5"
            cs = f"clean_sheets_{side}_l5"
            if gc in df.columns and cs in df.columns:
                df[f"defensive_solidity_{side}"] = (
                    df[cs].fillna(0.2) * 2 - df[gc].fillna(1.3) / 3)

        return df

    # ------------------------------------------------------------------
    # 10. External Elo features (from Club Elo API)
    # ------------------------------------------------------------------
    def _compute_external_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """If Club Elo pre-match ratings are available, use them as features."""
        df["ext_elo_diff"] = df["elo_pre_home"] - df["elo_pre_away"]
        # Agreement between internal Elo and external Elo
        df["elo_agreement"] = (
            np.sign(df["elo_diff"]) == np.sign(df["ext_elo_diff"].fillna(0))
        ).astype(int)
        return df

    # ------------------------------------------------------------------
    # 11. NLP Sentiment features
    # ------------------------------------------------------------------
    def _compute_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add NLP sentiment features from news data."""
        sentiment_home = []
        sentiment_away = []
        sentiment_diff = []
        volume_home = []
        volume_away = []
        consensus_home = []
        consensus_away = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            h_sent = self.sentiment_data.get(ht, {})
            a_sent = self.sentiment_data.get(at, {})

            h_val = h_sent.get("sentiment", 0.0)
            a_val = a_sent.get("sentiment", 0.0)
            sentiment_home.append(h_val)
            sentiment_away.append(a_val)
            sentiment_diff.append(h_val - a_val)
            volume_home.append(h_sent.get("volume", 0))
            volume_away.append(a_sent.get("volume", 0))
            consensus_home.append(h_sent.get("consensus", 0.0))
            consensus_away.append(a_sent.get("consensus", 0.0))

        df["sentiment_home"] = sentiment_home
        df["sentiment_away"] = sentiment_away
        df["sentiment_diff"] = sentiment_diff
        df["news_volume_home"] = volume_home
        df["news_volume_away"] = volume_away
        df["sentiment_consensus_home"] = consensus_home
        df["sentiment_consensus_away"] = consensus_away

        # Sentiment-form interaction
        if "ppg_home_l5" in df.columns:
            df["sentiment_form_home"] = df["sentiment_home"] * df["ppg_home_l5"].fillna(1.5)
            df["sentiment_form_away"] = df["sentiment_away"] * df["ppg_away_l5"].fillna(1.5)

        return df

    # ------------------------------------------------------------------
    # 12. Injury impact features
    # ------------------------------------------------------------------
    def _compute_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add injury burden features — teams with more injuries are weakened."""
        # Normalise injury counts (0-1 scale based on max)
        max_injuries = max(self.injury_counts.values()) if self.injury_counts else 1

        inj_home = []
        inj_away = []
        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            h_count = self.injury_counts.get(ht, 0) / max(max_injuries, 1)
            a_count = self.injury_counts.get(at, 0) / max(max_injuries, 1)
            inj_home.append(h_count)
            inj_away.append(a_count)

        df["injury_burden_home"] = inj_home
        df["injury_burden_away"] = inj_away
        df["injury_burden_diff"] = np.array(inj_home) - np.array(inj_away)

        return df

    # ------------------------------------------------------------------
    # 13. Momentum & Streaks (advanced)
    # ------------------------------------------------------------------
    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute advanced momentum features: streaks, acceleration, velocity,
        home/away specific form, goal trends, fixture congestion, xG trends."""
        # Track per-team: points, goals scored, goals conceded, dates, home/away flag
        team_history = {team: {"pts": [], "gf": [], "ga": [], "dates": [],
                               "is_home": [], "xg_for": [], "xg_against": []}
                        for team in self.teams}

        cols = {k: [] for k in [
            "streak_home", "streak_away",
            "volatility_home", "volatility_away",
            # Form acceleration (is form improving or declining?)
            "form_accel_home", "form_accel_away",
            # Home/away specific form (how they perform in this venue type)
            "home_form_at_home", "away_form_at_away",
            # Goal scoring trend (are they scoring more/less recently?)
            "goal_trend_home", "goal_trend_away",
            # Defensive trend (are they conceding more/less?)
            "defense_trend_home", "defense_trend_away",
            # Fixture congestion (games in last 14 days)
            "congestion_home", "congestion_away",
            # xG trend (improving/declining underlying performance)
            "xg_trend_home", "xg_trend_away",
            # Points velocity (points per game over last 3 vs last 8)
            "points_velocity_home", "points_velocity_away",
            # Unbeaten run length
            "unbeaten_run_home", "unbeaten_run_away",
            # Winless run length
            "winless_run_home", "winless_run_away",
        ]}

        def _current_streak(pts_history):
            if not pts_history:
                return 0
            last = pts_history[-1]
            count = 0
            for r in reversed(pts_history):
                if r == last:
                    count += 1
                else:
                    break
            return count if last == 3 else (-count if last == 0 else 0)

        def _unbeaten_run(pts_history):
            count = 0
            for r in reversed(pts_history):
                if r > 0:
                    count += 1
                else:
                    break
            return count

        def _winless_run(pts_history):
            count = 0
            for r in reversed(pts_history):
                if r < 3:
                    count += 1
                else:
                    break
            return count

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            h_hist = team_history[ht]
            a_hist = team_history[at]

            # Streaks
            cols["streak_home"].append(_current_streak(h_hist["pts"]))
            cols["streak_away"].append(_current_streak(a_hist["pts"]))

            # Unbeaten / winless runs
            cols["unbeaten_run_home"].append(_unbeaten_run(h_hist["pts"]))
            cols["unbeaten_run_away"].append(_unbeaten_run(a_hist["pts"]))
            cols["winless_run_home"].append(_winless_run(h_hist["pts"]))
            cols["winless_run_away"].append(_winless_run(a_hist["pts"]))

            # Volatility
            for side, hist in [("home", h_hist), ("away", a_hist)]:
                r = hist["pts"][-10:]
                cols[f"volatility_{side}"].append(float(np.std(r)) if len(r) >= 3 else 0.0)

            # Form acceleration: ppg_l3 - ppg_l8 (positive = improving)
            for side, hist in [("home", h_hist), ("away", a_hist)]:
                pts = hist["pts"]
                if len(pts) >= 8:
                    ppg3 = np.mean(pts[-3:]) / 3.0
                    ppg8 = np.mean(pts[-8:]) / 3.0
                    cols[f"form_accel_{side}"].append(ppg3 - ppg8)
                elif len(pts) >= 3:
                    cols[f"form_accel_{side}"].append(0.0)
                else:
                    cols[f"form_accel_{side}"].append(0.0)

            # Home/away specific form (how team performs specifically at home/away)
            home_pts_at_home = [p for p, ih in zip(h_hist["pts"], h_hist["is_home"]) if ih]
            away_pts_at_away = [p for p, ih in zip(a_hist["pts"], a_hist["is_home"]) if not ih]
            cols["home_form_at_home"].append(
                np.mean(home_pts_at_home[-5:]) / 3.0 if len(home_pts_at_home) >= 2 else 0.5)
            cols["away_form_at_away"].append(
                np.mean(away_pts_at_away[-5:]) / 3.0 if len(away_pts_at_away) >= 2 else 0.33)

            # Goal scoring trend (last 3 vs last 8 average)
            for side, hist in [("home", h_hist), ("away", a_hist)]:
                gf = hist["gf"]
                if len(gf) >= 8:
                    cols[f"goal_trend_{side}"].append(np.mean(gf[-3:]) - np.mean(gf[-8:]))
                else:
                    cols[f"goal_trend_{side}"].append(0.0)

            # Defensive trend
            for side, hist in [("home", h_hist), ("away", a_hist)]:
                ga = hist["ga"]
                if len(ga) >= 8:
                    cols[f"defense_trend_{side}"].append(np.mean(ga[-3:]) - np.mean(ga[-8:]))
                else:
                    cols[f"defense_trend_{side}"].append(0.0)

            # Fixture congestion (games in last 14 days)
            match_date = row["date"]
            for side, hist in [("home", h_hist), ("away", a_hist)]:
                if hist["dates"]:
                    recent_dates = [d for d in hist["dates"] if (match_date - d).days <= 14]
                    cols[f"congestion_{side}"].append(len(recent_dates))
                else:
                    cols[f"congestion_{side}"].append(0)

            # xG trend
            for side, hist in [("home", h_hist), ("away", a_hist)]:
                xgf = [x for x in hist["xg_for"] if not np.isnan(x)]
                if len(xgf) >= 5:
                    cols[f"xg_trend_{side}"].append(np.mean(xgf[-3:]) - np.mean(xgf[-5:]))
                else:
                    cols[f"xg_trend_{side}"].append(0.0)

            # Points velocity (recent vs medium-term)
            for side, hist in [("home", h_hist), ("away", a_hist)]:
                pts = hist["pts"]
                if len(pts) >= 8:
                    v3 = np.mean(pts[-3:])
                    v8 = np.mean(pts[-8:])
                    cols[f"points_velocity_{side}"].append(v3 - v8)
                else:
                    cols[f"points_velocity_{side}"].append(0.0)

            # Update team histories
            h_hist["pts"].append(row.get("home_points", 0))
            h_hist["gf"].append(row["goals_home"])
            h_hist["ga"].append(row["goals_away"])
            h_hist["dates"].append(match_date)
            h_hist["is_home"].append(True)
            h_hist["xg_for"].append(row.get("xg_home", np.nan) if pd.notna(row.get("xg_home")) else np.nan)
            h_hist["xg_against"].append(row.get("xg_away", np.nan) if pd.notna(row.get("xg_away")) else np.nan)

            a_hist["pts"].append(row.get("away_points", 0))
            a_hist["gf"].append(row["goals_away"])
            a_hist["ga"].append(row["goals_home"])
            a_hist["dates"].append(match_date)
            a_hist["is_home"].append(False)
            a_hist["xg_for"].append(row.get("xg_away", np.nan) if pd.notna(row.get("xg_away")) else np.nan)
            a_hist["xg_against"].append(row.get("xg_home", np.nan) if pd.notna(row.get("xg_home")) else np.nan)

        for k, v in cols.items():
            df[k] = v

        # Compute diffs
        df["streak_diff"] = df["streak_home"] - df["streak_away"]
        df["volatility_diff"] = df["volatility_home"] - df["volatility_away"]
        df["form_accel_diff"] = df["form_accel_home"] - df["form_accel_away"]
        df["congestion_diff"] = df["congestion_home"] - df["congestion_away"]

        return df

    # ------------------------------------------------------------------
    # 14. Surprise factor
    # ------------------------------------------------------------------
    def _compute_surprise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track upset tendency — teams that frequently cause or suffer upsets."""
        team_surprise_as_underdog = {team: [] for team in self.teams}
        team_surprise_as_favorite = {team: [] for team in self.teams}

        upset_home = []
        upset_away = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            # Use Elo diff as proxy for expectation
            elo_diff = row.get("elo_diff", 0)

            # Upset rate as underdog (last 20 matches)
            def _upset_rate(history, n=20):
                recent = history[-n:]
                if not recent:
                    return 0.0
                return np.mean(recent)

            upset_home.append(_upset_rate(team_surprise_as_underdog[ht]))
            upset_away.append(_upset_rate(team_surprise_as_underdog[at]))

            # Update: was this an upset?
            result = row.get("result", "D")
            if elo_diff > 50 and result == "A":  # Home was favorite, away won
                team_surprise_as_underdog[at].append(1.0)
                team_surprise_as_favorite[ht].append(1.0)
            elif elo_diff < -50 and result == "H":  # Away was favorite, home won
                team_surprise_as_underdog[ht].append(1.0)
                team_surprise_as_favorite[at].append(1.0)
            else:
                team_surprise_as_underdog[ht].append(0.0)
                team_surprise_as_underdog[at].append(0.0)

        df["upset_tendency_home"] = upset_home
        df["upset_tendency_away"] = upset_away

        return df

    # ------------------------------------------------------------------
    # 15. Glicko-2 Ratings (rating + uncertainty + volatility)
    # ------------------------------------------------------------------
    def _compute_glicko2_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Glicko-2: Bayesian rating with uncertainty (RD) and volatility.

        Unlike Elo, Glicko-2 tracks HOW CERTAIN we are about each team's
        strength (RD decreases with more games) and HOW VOLATILE their
        performance is (volatility captures consistency).
        """
        import math

        MU_INIT = 1500.0
        RD_INIT = 200.0
        VOL_INIT = 0.06
        TAU = 0.5  # System constant (constrains volatility changes)

        glicko = {team: {"mu": MU_INIT, "rd": RD_INIT, "vol": VOL_INIT}
                  for team in self.teams}

        cols = {k: [] for k in [
            "glicko_mu_home", "glicko_mu_away", "glicko_mu_diff",
            "glicko_rd_home", "glicko_rd_away", "glicko_rd_diff",
            "glicko_vol_home", "glicko_vol_away",
            # Confidence-weighted rating (strong team with low uncertainty = high confidence)
            "glicko_confidence_home", "glicko_confidence_away", "glicko_confidence_diff",
        ]}

        def _g(rd):
            return 1.0 / math.sqrt(1 + 3 * (rd / 173.7178) ** 2 / (math.pi ** 2))

        def _e(mu, mu_opp, rd_opp):
            return 1.0 / (1 + math.exp(-_g(rd_opp) * (mu - mu_opp) / 173.7178))

        prev_season = None
        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            season = row.get("season", "")

            # Season boundary: increase RD (uncertainty grows with inactivity)
            if prev_season is not None and season != prev_season:
                for team in self.teams:
                    glicko[team]["rd"] = min(RD_INIT,
                        math.sqrt(glicko[team]["rd"] ** 2 + glicko[team]["vol"] ** 2 * 100))
            prev_season = season

            h, a = glicko[ht], glicko[at]

            # Store pre-match ratings
            for k, v in [("glicko_mu_home", h["mu"]), ("glicko_mu_away", a["mu"]),
                         ("glicko_rd_home", h["rd"]), ("glicko_rd_away", a["rd"]),
                         ("glicko_vol_home", h["vol"]), ("glicko_vol_away", a["vol"])]:
                cols[k].append(v)
            cols["glicko_mu_diff"].append(h["mu"] - a["mu"])
            cols["glicko_rd_diff"].append(h["rd"] - a["rd"])
            # Confidence = rating / uncertainty (higher = more certain of strength)
            conf_h = h["mu"] / max(h["rd"], 10)
            conf_a = a["mu"] / max(a["rd"], 10)
            cols["glicko_confidence_home"].append(conf_h)
            cols["glicko_confidence_away"].append(conf_a)
            cols["glicko_confidence_diff"].append(conf_h - conf_a)

            # Compute result (1=win, 0.5=draw, 0=loss for home)
            gh, ga = row["goals_home"], row["goals_away"]
            s_home = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)

            # Update both teams
            for team, s, opp in [(ht, s_home, a), (at, 1.0 - s_home, h)]:
                g_val = _g(opp["rd"])
                e_val = _e(glicko[team]["mu"], opp["mu"], opp["rd"])
                v_inv = g_val ** 2 * e_val * (1 - e_val)
                v = 1.0 / max(v_inv, 1e-6)

                delta = v * g_val * (s - e_val)

                # Simplified volatility update
                a_val = math.log(glicko[team]["vol"] ** 2)
                phi = glicko[team]["rd"] / 173.7178
                delta_sq = delta ** 2

                # Iterative algorithm for new volatility (simplified)
                new_vol = glicko[team]["vol"]
                for _ in range(20):
                    ex = math.exp(a_val)
                    d = phi ** 2 + v + ex
                    h1 = -(a_val - math.log(new_vol ** 2)) / (TAU ** 2) + \
                         0.5 * ex * (delta_sq / (d ** 2) - 1) / d
                    h2 = -1.0 / (TAU ** 2) - 0.5 * ex * (phi ** 2 + v) / (d ** 2)
                    if abs(h2) < 1e-10:
                        break
                    a_val -= h1 / h2
                    new_vol = math.exp(a_val / 2)

                new_vol = max(0.01, min(0.2, new_vol))

                # Update RD
                rd_star = math.sqrt(glicko[team]["rd"] ** 2 + new_vol ** 2)
                new_rd = 1.0 / math.sqrt(1.0 / (rd_star ** 2) + v_inv)
                new_mu = glicko[team]["mu"] + new_rd ** 2 * g_val * (s - e_val) * 173.7178 / new_rd

                glicko[team]["mu"] = new_mu
                glicko[team]["rd"] = min(new_rd, RD_INIT)
                glicko[team]["vol"] = new_vol

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # 16. Shin-adjusted market probabilities (true odds from bookmaker margins)
    # ------------------------------------------------------------------
    def _compute_shin_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shin (1993) method to extract true probabilities from bookmaker odds.

        The Shin model assumes some fraction z of bettors are insiders.
        It produces more accurate probabilities than simple normalization.
        """
        shin_h = []
        shin_d = []
        shin_a = []

        for _, row in df.iterrows():
            try:
                oh = float(row.get("odds_home_open", 0))
                od = float(row.get("odds_draw_open", 0))
                oa = float(row.get("odds_away_open", 0))

                if oh > 1 and od > 1 and oa > 1:
                    # Implied probabilities
                    ip = [1.0 / oh, 1.0 / od, 1.0 / oa]
                    total = sum(ip)

                    # Shin parameter z (estimated from overround)
                    n = 3  # number of outcomes
                    z = (total - 1) / (n - 1) if total > 1 else 0.0
                    z = max(0.0, min(0.3, z))

                    # Shin-adjusted probabilities
                    true_probs = []
                    for p in ip:
                        # Shin formula: true_p = (sqrt(z^2 + 4*(1-z)*p^2/total) - z) / (2*(1-z))
                        discriminant = z ** 2 + 4 * (1 - z) * (p ** 2) / total
                        if discriminant > 0 and (1 - z) > 0:
                            true_p = (discriminant ** 0.5 - z) / (2 * (1 - z))
                        else:
                            true_p = p / total
                        true_probs.append(max(0.01, true_p))

                    # Normalize
                    tp_sum = sum(true_probs)
                    true_probs = [p / tp_sum for p in true_probs]

                    shin_h.append(true_probs[0])
                    shin_d.append(true_probs[1])
                    shin_a.append(true_probs[2])
                else:
                    shin_h.append(np.nan)
                    shin_d.append(np.nan)
                    shin_a.append(np.nan)
            except (ValueError, TypeError):
                shin_h.append(np.nan)
                shin_d.append(np.nan)
                shin_a.append(np.nan)

        df["shin_prob_home"] = shin_h
        df["shin_prob_draw"] = shin_d
        df["shin_prob_away"] = shin_a
        df["shin_home_draw_ratio"] = np.array(shin_h) / np.clip(shin_d, 0.05, None)
        df["shin_home_away_ratio"] = np.array(shin_h) / np.clip(shin_a, 0.05, None)
        df["shin_decisiveness"] = np.max([shin_h, shin_d, shin_a], axis=0) - \
                                   np.min([shin_h, shin_d, shin_a], axis=0)

        return df

    # ------------------------------------------------------------------
    # 17. Match sequence features (pattern encoding of last N results)
    # ------------------------------------------------------------------
    def _compute_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode last N match results as sequence features.

        Research shows treating recent form as a SEQUENCE (not just aggregate)
        captures patterns like 'collapse after a win' or 'bounce after loss'.
        """
        team_sequences = {team: [] for team in self.teams}

        cols = {k: [] for k in [
            # Last 5 results encoded (W=3, D=1, L=0)
            "seq_last1_home", "seq_last2_home", "seq_last3_home",
            "seq_last4_home", "seq_last5_home",
            "seq_last1_away", "seq_last2_away", "seq_last3_away",
            "seq_last4_away", "seq_last5_away",
            # Sequence pattern features
            "seq_wdl_pattern_home",    # Encoded W/D/L pattern (hash-like)
            "seq_wdl_pattern_away",
            "seq_bounce_back_home",    # Won after loss (resilience)
            "seq_bounce_back_away",
            "seq_collapse_home",       # Lost after win (fragility)
            "seq_collapse_away",
            "seq_consistency_home",    # How consistent are recent results
            "seq_consistency_away",
        ]}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            for side, team in [("home", ht), ("away", at)]:
                seq = team_sequences[team]

                # Last 5 individual results
                for i in range(1, 6):
                    key = f"seq_last{i}_{side}"
                    if len(seq) >= i:
                        cols[key].append(seq[-i])
                    else:
                        cols[key].append(np.nan)

                # Pattern encoding (weighted sum of last 5)
                if len(seq) >= 3:
                    pattern = sum(seq[-i-1] * (3 ** i) for i in range(min(5, len(seq))))
                    cols[f"seq_wdl_pattern_{side}"].append(pattern)
                else:
                    cols[f"seq_wdl_pattern_{side}"].append(0)

                # Bounce-back rate (W after L in last 10)
                if len(seq) >= 5:
                    recent = seq[-10:]
                    bounces = sum(1 for i in range(1, len(recent))
                                 if recent[i] == 3 and recent[i-1] == 0)
                    losses = sum(1 for r in recent[:-1] if r == 0)
                    cols[f"seq_bounce_back_{side}"].append(
                        bounces / max(losses, 1))
                else:
                    cols[f"seq_bounce_back_{side}"].append(0.5)

                # Collapse rate (L after W in last 10)
                if len(seq) >= 5:
                    recent = seq[-10:]
                    collapses = sum(1 for i in range(1, len(recent))
                                   if recent[i] == 0 and recent[i-1] == 3)
                    wins = sum(1 for r in recent[:-1] if r == 3)
                    cols[f"seq_collapse_{side}"].append(
                        collapses / max(wins, 1))
                else:
                    cols[f"seq_collapse_{side}"].append(0.5)

                # Consistency (inverse of std of last 10 points)
                if len(seq) >= 5:
                    cols[f"seq_consistency_{side}"].append(
                        1.0 / (1.0 + float(np.std(seq[-10:]))))
                else:
                    cols[f"seq_consistency_{side}"].append(0.5)

            # Update sequences
            team_sequences[ht].append(row.get("home_points", 0))
            team_sequences[at].append(row.get("away_points", 0))

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # 18. Managerial stability features
    # ------------------------------------------------------------------
    def _compute_manager_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track managerial tenure and detect new-manager bounce effect.

        Research shows teams get a short-term bounce after a managerial
        change, then normalize. We detect this from result patterns.
        """
        # Track consecutive managers via result pattern changes
        team_tenure = {team: 0 for team in self.teams}
        team_prev_results = {team: [] for team in self.teams}

        tenure_home = []
        tenure_away = []
        new_mgr_bounce_home = []
        new_mgr_bounce_away = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            season = row.get("season", "")

            for side, team in [("home", ht), ("away", at)]:
                prev = team_prev_results[team]

                # Heuristic: detect managerial change from sudden result swing
                # (5+ match losing streak followed by wins, or season boundary)
                if len(prev) >= 5:
                    last5 = prev[-5:]
                    # If team was losing badly (avg < 0.6 pts/game) then suddenly improves
                    if np.mean(last5) < 0.6:
                        team_tenure[team] = 0  # Likely new manager

                team_tenure[team] += 1

                if side == "home":
                    tenure_home.append(team_tenure[team])
                    new_mgr_bounce_home.append(
                        1.0 if team_tenure[team] <= 5 else 0.0)
                else:
                    tenure_away.append(team_tenure[team])
                    new_mgr_bounce_away.append(
                        1.0 if team_tenure[team] <= 5 else 0.0)

            # Update results
            team_prev_results[ht].append(row.get("home_points", 0))
            team_prev_results[at].append(row.get("away_points", 0))

        df["manager_tenure_home"] = tenure_home
        df["manager_tenure_away"] = tenure_away
        df["new_manager_bounce_home"] = new_mgr_bounce_home
        df["new_manager_bounce_away"] = new_mgr_bounce_away

        return df

    # ------------------------------------------------------------------
    # 19. GK quality proxy (clean sheet patterns + goals conceded quality)
    # ------------------------------------------------------------------
    def _compute_gk_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """GK quality indicators from save patterns and clean sheet streaks.

        Research shows GK stats are MORE predictive of goals conceded
        than attacker stats are of goals scored.
        """
        team_gk = {team: {"ga": [], "cs": []} for team in self.teams}

        cols = {k: [] for k in [
            "gk_quality_home", "gk_quality_away", "gk_quality_diff",
            "gk_cs_streak_home", "gk_cs_streak_away",
            "gk_consistency_home", "gk_consistency_away",
        ]}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            for side, team, ga in [("home", ht, row["goals_away"]),
                                    ("away", at, row["goals_home"])]:
                hist = team_gk[team]

                # GK quality: weighted combo of clean sheet rate + low concession rate
                if len(hist["ga"]) >= 5:
                    recent_ga = hist["ga"][-10:]
                    recent_cs = hist["cs"][-10:]
                    cs_rate = np.mean(recent_cs)
                    avg_ga = np.mean(recent_ga)
                    quality = cs_rate * 2 - avg_ga / 3 + 0.5
                    cols[f"gk_quality_{side}"].append(quality)
                    cols[f"gk_consistency_{side}"].append(
                        1.0 / (1.0 + float(np.std(recent_ga))))
                else:
                    cols[f"gk_quality_{side}"].append(0.5)
                    cols[f"gk_consistency_{side}"].append(0.5)

                # Clean sheet streak
                streak = 0
                for cs in reversed(hist["cs"]):
                    if cs == 1:
                        streak += 1
                    else:
                        break
                cols[f"gk_cs_streak_{side}"].append(streak)

            # Update
            team_gk[ht]["ga"].append(row["goals_away"])
            team_gk[ht]["cs"].append(1 if row["goals_away"] == 0 else 0)
            team_gk[at]["ga"].append(row["goals_home"])
            team_gk[at]["cs"].append(1 if row["goals_home"] == 0 else 0)

        cols["gk_quality_diff"] = [h - a for h, a in
                                    zip(cols["gk_quality_home"], cols["gk_quality_away"])]

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # 20. Poisson goal expectation features
    # ------------------------------------------------------------------
    def _compute_poisson_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Poisson-based goal expectation from team attack/defense strength.

        Models each team's scoring as Poisson(attack * opp_defense * league_avg).
        More principled than simple rolling averages.
        """
        team_attack = {team: [] for team in self.teams}
        team_defense = {team: [] for team in self.teams}

        cols = {k: [] for k in [
            "poisson_home_lambda", "poisson_away_lambda",
            "poisson_total_lambda", "poisson_diff",
            "poisson_home_win", "poisson_draw", "poisson_away_win",
            "poisson_over25", "poisson_btts",
        ]}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            ha = team_attack[ht][-10:]
            hd = team_defense[ht][-10:]
            aa = team_attack[at][-10:]
            ad = team_defense[at][-10:]

            if len(ha) >= 5 and len(aa) >= 5:
                # Home expected goals = home attack rate * away defense weakness
                h_att = np.mean(ha)  # goals scored per game
                a_def = np.mean(ad)  # goals conceded per game
                a_att = np.mean(aa)
                h_def = np.mean(hd)

                league_avg = 1.35  # EPL average ~1.35 goals per side
                home_lambda = max(0.3, h_att * a_def / league_avg * 1.1)  # home boost
                away_lambda = max(0.2, a_att * h_def / league_avg * 0.9)

                cols["poisson_home_lambda"].append(home_lambda)
                cols["poisson_away_lambda"].append(away_lambda)
                cols["poisson_total_lambda"].append(home_lambda + away_lambda)
                cols["poisson_diff"].append(home_lambda - away_lambda)

                # Compute Poisson probabilities
                from scipy.stats import poisson
                max_goals = 6
                h_win = d_prob = a_win = 0.0
                over25 = btts = 0.0
                for i in range(max_goals):
                    for j in range(max_goals):
                        p = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
                        if i > j:
                            h_win += p
                        elif i == j:
                            d_prob += p
                        else:
                            a_win += p
                        if i + j > 2:
                            over25 += p
                        if i > 0 and j > 0:
                            btts += p

                cols["poisson_home_win"].append(h_win)
                cols["poisson_draw"].append(d_prob)
                cols["poisson_away_win"].append(a_win)
                cols["poisson_over25"].append(over25)
                cols["poisson_btts"].append(btts)
            else:
                for k in cols:
                    cols[k].append(np.nan)

            # Update team stats
            team_attack[ht].append(row["goals_home"])
            team_defense[ht].append(row["goals_away"])
            team_attack[at].append(row["goals_away"])
            team_defense[at].append(row["goals_home"])

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # 21. Expected Points Features (luck / regression indicator)
    # ------------------------------------------------------------------
    def _compute_expected_points_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track expected points vs actual points as a luck/regression indicator.

        Uses xG-based Poisson model to compute expected points per match,
        then compares cumulative expected points to actual points.
        Positive surplus = team has been lucky and may regress.
        """
        from scipy.stats import poisson

        team_actual_pts = {team: [] for team in self.teams}
        team_expected_pts = {team: [] for team in self.teams}

        cols = {k: [] for k in [
            "xpts_surplus_home", "xpts_surplus_away",
            "xpts_trend_home", "xpts_trend_away",
        ]}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            # Compute surplus: actual cumulative pts - expected cumulative pts
            h_actual = team_actual_pts[ht]
            h_expected = team_expected_pts[ht]
            a_actual = team_actual_pts[at]
            a_expected = team_expected_pts[at]

            if len(h_actual) >= 3:
                surplus_h = sum(h_actual[-20:]) - sum(h_expected[-20:])
                cols["xpts_surplus_home"].append(surplus_h)
            else:
                cols["xpts_surplus_home"].append(0.0)

            if len(a_actual) >= 3:
                surplus_a = sum(a_actual[-20:]) - sum(a_expected[-20:])
                cols["xpts_surplus_away"].append(surplus_a)
            else:
                cols["xpts_surplus_away"].append(0.0)

            # 5-game rolling xpts surplus trend
            if len(h_actual) >= 5:
                recent_surplus_h = [h_actual[-5 + i] - h_expected[-5 + i] for i in range(5)]
                cols["xpts_trend_home"].append(np.mean(recent_surplus_h))
            else:
                cols["xpts_trend_home"].append(0.0)

            if len(a_actual) >= 5:
                recent_surplus_a = [a_actual[-5 + i] - a_expected[-5 + i] for i in range(5)]
                cols["xpts_trend_away"].append(np.mean(recent_surplus_a))
            else:
                cols["xpts_trend_away"].append(0.0)

            # After recording features, compute expected points for this match
            # and update histories
            gh = row["goals_home"]
            ga = row["goals_away"]

            # Actual points
            if gh > ga:
                act_h, act_a = 3, 0
            elif gh == ga:
                act_h, act_a = 1, 1
            else:
                act_h, act_a = 0, 3

            # Expected points from xG (Poisson model)
            xg_h = row.get("xg_home", np.nan) if self.has_xg else np.nan
            xg_a = row.get("xg_away", np.nan) if self.has_xg else np.nan

            if pd.notna(xg_h) and pd.notna(xg_a) and xg_h > 0 and xg_a > 0:
                # Compute win/draw/loss probabilities from xG using Poisson
                max_goals = 6
                p_hw = p_d = p_aw = 0.0
                for i in range(max_goals):
                    for j in range(max_goals):
                        p = poisson.pmf(i, xg_h) * poisson.pmf(j, xg_a)
                        if i > j:
                            p_hw += p
                        elif i == j:
                            p_d += p
                        else:
                            p_aw += p
                exp_h = 3 * p_hw + 1 * p_d
                exp_a = 3 * p_aw + 1 * p_d
            else:
                # Fallback: use league-average expected points (~1.33 per game)
                exp_h = 1.5  # slight home advantage
                exp_a = 1.2

            team_actual_pts[ht].append(act_h)
            team_expected_pts[ht].append(exp_h)
            team_actual_pts[at].append(act_a)
            team_expected_pts[at].append(exp_a)

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # 22. Rest Days Features
    # ------------------------------------------------------------------
    def _compute_rest_days_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track rest days between matches for each team.

        Computes days since last match, rest advantage, and binary
        freshness/fatigue flags.
        """
        team_last_date = {}

        cols = {k: [] for k in [
            "rest_days_home", "rest_days_away",
            "rest_advantage",
            "fresh_flag_home", "fresh_flag_away",
            "fatigue_flag_home", "fatigue_flag_away",
        ]}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            match_date = row["date"]

            # Rest days for home team
            if ht in team_last_date:
                delta_h = (match_date - team_last_date[ht]).days
                rest_h = max(0, delta_h)
            else:
                rest_h = 7  # default for first match of season

            # Rest days for away team
            if at in team_last_date:
                delta_a = (match_date - team_last_date[at]).days
                rest_a = max(0, delta_a)
            else:
                rest_a = 7

            cols["rest_days_home"].append(rest_h)
            cols["rest_days_away"].append(rest_a)
            cols["rest_advantage"].append(rest_h - rest_a)
            cols["fresh_flag_home"].append(1 if rest_h >= 6 else 0)
            cols["fresh_flag_away"].append(1 if rest_a >= 6 else 0)
            cols["fatigue_flag_home"].append(1 if rest_h <= 2 else 0)
            cols["fatigue_flag_away"].append(1 if rest_a <= 2 else 0)

            # Update last match date
            team_last_date[ht] = match_date
            team_last_date[at] = match_date

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # 23. Scoring Patterns (when goals are scored)
    # ------------------------------------------------------------------
    def _compute_scoring_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track scoring patterns: early goals, comeback rate, clean sheets,
        and first-goal rate.

        Uses half-time data as a proxy for timing of goals where available.
        """
        team_ht_goals = {team: [] for team in self.teams}
        team_ft_goals = {team: [] for team in self.teams}
        team_ht_conceded = {team: [] for team in self.teams}
        team_ft_conceded = {team: [] for team in self.teams}

        cols = {k: [] for k in [
            "early_goals_pct_home", "early_goals_pct_away",
            "comeback_rate_home", "comeback_rate_away",
            "clean_sheet_rate_home", "clean_sheet_rate_away",
            "first_goal_rate_home", "first_goal_rate_away",
        ]}

        has_ht = "ht_goals_home" in df.columns

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            h_ht_goals = team_ht_goals[ht]
            h_ft_goals = team_ft_goals[ht]
            h_ht_conc = team_ht_conceded[ht]
            h_ft_conc = team_ft_conceded[ht]
            a_ht_goals = team_ht_goals[at]
            a_ft_goals = team_ft_goals[at]
            a_ht_conc = team_ht_conceded[at]
            a_ft_conc = team_ft_conceded[at]

            n_h = len(h_ft_goals)
            n_a = len(a_ft_goals)

            # --- Early goals % (HT goals / FT goals as proxy) ---
            if n_h >= 5:
                total_ft_h = sum(h_ft_goals[-10:])
                total_ht_h = sum(h_ht_goals[-10:])
                cols["early_goals_pct_home"].append(
                    total_ht_h / total_ft_h if total_ft_h > 0 else 0.0
                )
            else:
                cols["early_goals_pct_home"].append(np.nan)

            if n_a >= 5:
                total_ft_a = sum(a_ft_goals[-10:])
                total_ht_a = sum(a_ht_goals[-10:])
                cols["early_goals_pct_away"].append(
                    total_ht_a / total_ft_a if total_ft_a > 0 else 0.0
                )
            else:
                cols["early_goals_pct_away"].append(np.nan)

            # --- Comeback rate (losing at HT but winning/drawing at FT) ---
            if n_h >= 5:
                comebacks_h = 0
                behind_h = 0
                window = min(10, n_h)
                for i in range(-window, 0):
                    if h_ht_goals[i] < h_ht_conc[i]:  # losing at HT
                        behind_h += 1
                        if h_ft_goals[i] >= h_ft_conc[i]:  # drew or won at FT
                            comebacks_h += 1
                cols["comeback_rate_home"].append(
                    comebacks_h / behind_h if behind_h > 0 else 0.0
                )
            else:
                cols["comeback_rate_home"].append(np.nan)

            if n_a >= 5:
                comebacks_a = 0
                behind_a = 0
                window = min(10, n_a)
                for i in range(-window, 0):
                    if a_ht_goals[i] < a_ht_conc[i]:
                        behind_a += 1
                        if a_ft_goals[i] >= a_ft_conc[i]:
                            comebacks_a += 1
                cols["comeback_rate_away"].append(
                    comebacks_a / behind_a if behind_a > 0 else 0.0
                )
            else:
                cols["comeback_rate_away"].append(np.nan)

            # --- Clean sheet rate (last 10 matches) ---
            if n_h >= 5:
                window = min(10, n_h)
                cs_h = sum(1 for c in h_ft_conc[-window:] if c == 0)
                cols["clean_sheet_rate_home"].append(cs_h / window)
            else:
                cols["clean_sheet_rate_home"].append(np.nan)

            if n_a >= 5:
                window = min(10, n_a)
                cs_a = sum(1 for c in a_ft_conc[-window:] if c == 0)
                cols["clean_sheet_rate_away"].append(cs_a / window)
            else:
                cols["clean_sheet_rate_away"].append(np.nan)

            # --- First goal rate (proxy: scored at HT while opponent didn't) ---
            if n_h >= 5:
                window = min(10, n_h)
                first_h = sum(
                    1 for i in range(-window, 0)
                    if h_ht_goals[i] > 0 and h_ht_conc[i] == 0
                )
                cols["first_goal_rate_home"].append(first_h / window)
            else:
                cols["first_goal_rate_home"].append(np.nan)

            if n_a >= 5:
                window = min(10, n_a)
                first_a = sum(
                    1 for i in range(-window, 0)
                    if a_ht_goals[i] > 0 and a_ht_conc[i] == 0
                )
                cols["first_goal_rate_away"].append(first_a / window)
            else:
                cols["first_goal_rate_away"].append(np.nan)

            # --- Update histories ---
            gh = row["goals_home"]
            ga = row["goals_away"]
            ht_gh = row.get("ht_goals_home", 0) if has_ht else 0
            ht_ga = row.get("ht_goals_away", 0) if has_ht else 0

            # Handle NaN half-time values
            if pd.isna(ht_gh):
                ht_gh = 0
            if pd.isna(ht_ga):
                ht_ga = 0

            team_ht_goals[ht].append(ht_gh)
            team_ft_goals[ht].append(gh)
            team_ht_conceded[ht].append(ht_ga)
            team_ft_conceded[ht].append(ga)

            team_ht_goals[at].append(ht_ga)
            team_ft_goals[at].append(ga)
            team_ht_conceded[at].append(ht_gh)
            team_ft_conceded[at].append(gh)

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # 24. League Position Features (enhanced)
    # ------------------------------------------------------------------
    def _compute_league_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced position-based features: position gap, top-6/bottom-3 flags,
        and big-match / relegation-battle indicators.

        Requires table_pos_home and table_pos_away to already be computed
        (from step 8 _compute_table_features).
        """
        cols = {k: [] for k in [
            "position_gap",
            "top6_home", "top6_away",
            "bottom3_home", "bottom3_away",
            "promotion_zone_match",
            "relegation_battle",
        ]}

        # Get number of teams per season for dynamic bottom-N thresholds
        season_team_counts = {}
        for _, row in df.iterrows():
            season = row["season"]
            if season not in season_team_counts:
                season_team_counts[season] = set()
            season_team_counts[season].add(row["home_team"])
            season_team_counts[season].add(row["away_team"])
        season_team_counts = {s: len(teams) for s, teams in season_team_counts.items()}

        for _, row in df.iterrows():
            h_pos = row.get("table_pos_home", 10)
            a_pos = row.get("table_pos_away", 10)
            season = row["season"]
            n_teams = season_team_counts.get(season, 20)

            # Position gap (absolute difference)
            cols["position_gap"].append(abs(h_pos - a_pos))

            # Top 6 flags
            cols["top6_home"].append(1 if h_pos <= 6 else 0)
            cols["top6_away"].append(1 if a_pos <= 6 else 0)

            # Bottom 3 flags
            bottom3_threshold = n_teams - 2  # e.g. 18th, 19th, 20th in a 20-team league
            cols["bottom3_home"].append(1 if h_pos >= bottom3_threshold else 0)
            cols["bottom3_away"].append(1 if a_pos >= bottom3_threshold else 0)

            # Top-6 clash: both teams competing for European qualification
            cols["promotion_zone_match"].append(
                1 if h_pos <= 6 and a_pos <= 6 else 0
            )

            # Relegation battle: both teams in bottom 6
            bottom6_threshold = n_teams - 5
            cols["relegation_battle"].append(
                1 if h_pos >= bottom6_threshold and a_pos >= bottom6_threshold else 0
            )

        for k, v in cols.items():
            df[k] = v
        return df

    # ------------------------------------------------------------------
    # Feature column selection
    # ------------------------------------------------------------------
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get all feature columns, excluding targets and metadata."""
        exclude = {
            "season", "date", "home_team", "away_team", "venue", "referee",
            "goals_home", "goals_away", "result", "home_points", "away_points",
            "total_goals", "over_2_5", "btts",
            "xg_home", "xg_away",  # raw xG is a target, rolling xG is a feature
            "ht_goals_home", "ht_goals_away", "ht_result",
            "kickoff_time", "understat_id",
            "elo_pre_home", "elo_pre_away",  # raw external Elo (derived features kept)
            # Close odds excluded as they may not be available at prediction time
            "odds_home_close", "odds_draw_close", "odds_away_close",
            "implied_prob_home_close", "implied_prob_draw_close", "implied_prob_away_close",
        }
        numeric_types = [np.float64, np.float32, np.int64, np.int32, float, int, np.bool_]
        return [c for c in df.columns
                if c not in exclude and df[c].dtype in numeric_types]


if __name__ == "__main__":
    import json as _json

    data_path = DATA_DIR / "epl_matches.parquet"
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path, engine="pyarrow")

    # Load extra data if available
    extra_path = DATA_DIR / "extra_data.json"
    sentiment_data = {}
    injury_data = []
    if extra_path.exists():
        try:
            extra = _json.load(open(extra_path))
            sentiment_data = extra.get("sentiment", {})
            injury_data = extra.get("injuries", [])
            print(f"  Extra data: {len(sentiment_data)} sentiments, {len(injury_data)} injuries")
        except Exception:
            pass

    engine = FeatureEngine(df, sentiment_data=sentiment_data, injury_data=injury_data)
    featured = engine.compute_all_features()
    feature_cols = engine.get_feature_columns(featured)

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Total rows: {len(featured)}")

    # Categorise features
    categories: Dict[str, List[str]] = {}
    for f in feature_cols:
        if "elo" in f.lower() and "ext" not in f:
            cat = "Elo Ratings"
        elif "ext_elo" in f or "elo_agreement" in f:
            cat = "External Elo"
        elif "pi_" in f:
            cat = "Pi Ratings"
        elif any(x in f for x in ["ppg", "goals_scored", "goals_conceded", "xg_for",
                                    "xg_against", "xg_over", "clean_sheets", "btts_pct",
                                    "over25", "win_pct", "draw_pct", "loss_pct", "shots_",
                                    "sot_", "possession_", "pass_accuracy_"]):
            cat = "Rolling Form"
        elif "h2h" in f:
            cat = "Head-to-Head"
        elif any(x in f for x in ["rest", "midweek", "month", "day_of",
                                    "season_", "early", "late", "run_in"]):
            cat = "Contextual"
        elif any(x in f for x in ["implied", "odds_", "market", "overround",
                                    "steam", "movement"]):
            cat = "Market"
        elif any(x in f for x in ["table", "top4", "relegation"]):
            cat = "Table Position"
        elif any(x in f for x in ["weather", "temp", "humid", "wind",
                                    "precip", "rain", "cold", "hot", "windy"]):
            cat = "Weather"
        elif any(x in f for x in ["derby", "interaction", "concordance",
                                    "mismatch", "combined", "weighted_streak"]):
            cat = "Interactions"
        elif "referee" in f:
            cat = "Referee"
        else:
            cat = "Other"
        categories.setdefault(cat, []).append(f)

    print("\nFeature categories:")
    for cat, feats in sorted(categories.items()):
        print(f"  {cat}: {len(feats)} features")

    out_path = DATA_DIR / "epl_featured.parquet"
    featured.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\nSaved to {out_path}")
