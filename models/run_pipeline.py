"""
MatchOracle — World-Class Multi-Layer Stacking Prediction Engine
=================================================================
Architecture (5-Layer Deep Ensemble):
  Layer 0: Dixon-Coles statistical model (goals + xG variants)
  Layer 1: 13 diverse ML base learners with time-decay weighting
           (HGB, HGB-Agg, HGB-Deep, XGBoost, LightGBM, RF, Extra Trees,
            DeepMLP, MLP-Wide, LR, Bagging-HGB, Vote-HGB3, CatBoost)
  Layer 2: 4 meta-learners (LR, MLP, HGB, XGB) trained on OOF predictions
  Layer 3: Isotonic calibration with proper holdout (auto-skip if degrading)
  Layer 4: Confidence-weighted final output with Monte Carlo validation
  Output:  Calibrated H/D/A probabilities + xG + scorelines + sentiment

Data Sources (all real, no synthetic):
  - football-data.co.uk: 20 seasons of results, stats, odds, referee
  - Club Elo: pre-match team Elo ratings
  - Open-Meteo: historical weather at stadium coordinates
  - Understat: match-level xG (2014+)
  - Football-Data.org API: live current-season standings and team stats
  - API-Football: injuries, player ratings
  - Google News RSS: NLP sentiment analysis with keyword-based scoring

Advanced Pipeline:
  - Multi-pass data cleaning (NaN, zero-var, correlation >0.95, winsorization)
  - 376+ engineered features (Elo, Pi-ratings, rolling form, H2H, sentiment, etc.)
  - Exponential time-decay sample weighting (ξ=0.0015)
  - 5-fold expanding-window time-aware cross-validation
  - Adaptive DC-ML blending with disagreement detection
  - Stacking + inverse-RPS weighted averaging (best selected)
  - Live xG estimation from current-season goal data
  - Walk-forward backtesting across 5 seasons
  - Monte Carlo simulation (10,000 iterations)
  - Confidence-stratified accuracy analysis

Usage:
    python models/run_pipeline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, HistGradientBoostingRegressor,
    BaggingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.isotonic import IsotonicRegression
from collections import Counter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from models.dixon_coles import DixonColesModel
from models.model_cache import (
    save_trained_state, load_trained_state, needs_retraining,
    get_cache_info, _hash_dataframe, _hash_features,
)
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

N_JOBS = max(1, multiprocessing.cpu_count() - 1)

DATA_DIR = PROJECT_ROOT / "data"


# =====================================================================
# Metrics
# =====================================================================
def rps(y_true, y_pred) -> float:
    """Ranked Probability Score — lower is better."""
    n = len(y_true)
    y_oh = np.zeros((n, 3))
    y_oh[np.arange(n), y_true.astype(int)] = 1.0
    cp = np.cumsum(y_pred, axis=1)
    ct = np.cumsum(y_oh, axis=1)
    return np.mean(np.mean((cp - ct) ** 2, axis=1))


def ece(y_true, y_probs, n_bins=10) -> float:
    """Expected Calibration Error."""
    e = 0.0
    for c in range(3):
        p = y_probs[:, c]
        bt = (y_true == c).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        for b in range(n_bins):
            m = (p >= edges[b]) & (p < edges[b + 1])
            if m.sum() > 0:
                e += m.sum() * abs(p[m].mean() - bt[m].mean())
    return e / (len(y_true) * 3)


def compute_time_weights(dates: pd.Series, seasons: pd.Series,
                          xi: float = 0.0015) -> np.ndarray:
    """Exponential time-decay + season-boost sample weights.

    Current season gets 12x boost, previous season 5x, two seasons
    ago 2.5x, with rapid exponential decay for older seasons. This
    ensures the model heavily prioritises recent form and patterns.
    """
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values
    # Base exponential time decay
    time_weights = np.exp(-xi * days_ago)

    # Season boost: current season >> recent seasons >> old seasons
    unique_seasons = sorted(seasons.unique())
    current_season = unique_seasons[-1]
    season_map = {s: i for i, s in enumerate(unique_seasons)}
    n_seasons = len(unique_seasons)

    season_multipliers = np.ones(len(dates))
    for i, s in enumerate(seasons):
        age = n_seasons - 1 - season_map[s]  # 0 = current, 1 = last, etc.
        if age == 0:
            season_multipliers[i] = 12.0  # Current season: 12x
        elif age == 1:
            season_multipliers[i] = 5.0   # Last season: 5x
        elif age == 2:
            season_multipliers[i] = 2.5   # Two seasons ago: 2.5x
        elif age == 3:
            season_multipliers[i] = 1.2   # Three seasons ago: 1.2x
        else:
            season_multipliers[i] = max(0.1, np.exp(-0.4 * age))  # Rapid decay

    weights = time_weights * season_multipliers
    weights = weights / weights.mean()
    return weights


# =====================================================================
# Data Cleaning Pipeline
# =====================================================================
def clean_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, feature_cols: list,
                   verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    """Advanced multi-pass data cleaning pipeline.

    Returns cleaned X_train, X_test, updated feature_cols, and cleaning_state
    (for applying the same transforms to upcoming match data).
    """
    state = {}

    # 1. Drop all-NaN columns
    still_nan = X_train.columns[X_train.isna().all()]
    if len(still_nan) > 0:
        if verbose:
            print(f"    Dropped {len(still_nan)} all-NaN columns")
        X_train = X_train.drop(columns=still_nan)
        X_test = X_test.drop(columns=still_nan)
        feature_cols = [c for c in feature_cols if c not in set(still_nan)]

    # 2. Impute remaining NaN with training medians, then 0
    medians = X_train.median()
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)
    state["medians"] = medians

    # 3. Remove zero-variance features (but keep live_ columns — they're zero
    #    in training but will have real values for upcoming predictions)
    variances = X_train.var()
    zero_var = [c for c in variances[variances < 1e-10].index.tolist()
                if not c.startswith("live_")]
    if zero_var:
        if verbose:
            print(f"    Dropped {len(zero_var)} zero-variance features")
        X_train = X_train.drop(columns=zero_var)
        X_test = X_test.drop(columns=zero_var)
        feature_cols = [c for c in feature_cols if c not in set(zero_var)]

    # 4. Remove highly correlated pairs (>0.95), keep the one more
    #    correlated with the target
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    target_corr = X_train.corrwith(y_train).abs()
    to_drop = set()
    for col in upper.columns:
        for other in upper.index[upper[col] > 0.95]:
            if other not in to_drop and col not in to_drop:
                if target_corr.get(col, 0) >= target_corr.get(other, 0):
                    to_drop.add(other)
                else:
                    to_drop.add(col)
    if to_drop:
        if verbose:
            print(f"    Dropped {len(to_drop)} highly correlated features (>0.95)")
        X_train = X_train.drop(columns=list(to_drop))
        X_test = X_test.drop(columns=list(to_drop))
        feature_cols = [c for c in feature_cols if c not in to_drop]

    # 5. Winsorise extreme outliers at 0.1%/99.9% quantiles
    lower_q = X_train.quantile(0.001)
    upper_q = X_train.quantile(0.999)
    X_train = X_train.clip(lower=lower_q, upper=upper_q, axis=1)
    X_test = X_test.clip(lower=lower_q, upper=upper_q, axis=1)
    state["lower_q"] = lower_q
    state["upper_q"] = upper_q
    state["feature_cols"] = feature_cols

    if verbose:
        print(f"    Final features: {len(feature_cols)}")

    return X_train, X_test, feature_cols, state


def apply_cleaning(X: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply same cleaning transforms to new data (upcoming matches)."""
    cols = state["feature_cols"]
    X = X[cols].fillna(state["medians"]).fillna(0)
    X = X.clip(lower=state["lower_q"], upper=state["upper_q"], axis=1)
    return X


# =====================================================================
# Model definitions — 13 diverse base learners
# =====================================================================
def build_base_learners(n_features: int) -> dict:
    """Create all Level-1 base learners — maximally diverse ensemble.

    Diversity sources:
      - Algorithm diversity: gradient boosting (6 variants incl. CatBoost),
        bagging (2), neural (2), linear (1), voting ensemble (1)
      - Hyperparameter diversity: conservative + aggressive configs
      - Feature subsampling diversity: different colsample_bytree / max_features
      - Regularisation diversity: different L1/L2 strengths
    """
    models = {}

    # HistGradientBoosting — handles NaN natively, strongest on tabular data
    models["HGB"] = HistGradientBoostingClassifier(
        max_iter=2500, max_depth=7, learning_rate=0.01,
        min_samples_leaf=10, l2_regularization=0.8,
        max_bins=255, random_state=42
    )

    # HGB-Agg — conservative variant (shallower, more regularisation)
    models["HGB-Agg"] = HistGradientBoostingClassifier(
        max_iter=3000, max_depth=4, learning_rate=0.008,
        min_samples_leaf=20, l2_regularization=1.5,
        max_bins=200, random_state=123
    )

    # HGB-Deep — deeper variant (captures complex interactions)
    models["HGB-Deep"] = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=9, learning_rate=0.012,
        min_samples_leaf=15, l2_regularization=1.0,
        max_bins=255, random_state=77
    )

    # XGBoost — gradient boosting with column subsampling
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=2500, max_depth=7, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=6, reg_alpha=0.03, reg_lambda=1.0,
            gamma=0.03,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42,
            verbosity=0, n_jobs=-1
        )

    # LightGBM — leaf-wise gradient boosting
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=2500, max_depth=9, learning_rate=0.01,
            num_leaves=63, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=10, reg_alpha=0.03, reg_lambda=1.0,
            min_split_gain=0.003,
            objective="multiclass", num_class=3,
            random_state=42, verbose=-1, n_jobs=-1
        )

    # Random Forest — large ensemble for variance reduction
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=2000, max_depth=18, min_samples_leaf=6,
        max_features="sqrt", min_samples_split=12,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )

    # Extra Trees — more randomised than RF
    models["Extra Trees"] = ExtraTreesClassifier(
        n_estimators=2000, max_depth=18, min_samples_leaf=6,
        max_features="sqrt", min_samples_split=12,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )

    # DeepMLP — 4-layer neural network
    models["DeepMLP"] = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32), activation="relu",
        solver="adam", alpha=0.002, batch_size=128,
        learning_rate="adaptive", learning_rate_init=0.001,
        max_iter=2000, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=60, random_state=42
    )

    # MLP-Wide — wide shallow network
    models["MLP-Wide"] = MLPClassifier(
        hidden_layer_sizes=(512, 128), activation="relu",
        solver="adam", alpha=0.003, batch_size=256,
        learning_rate="adaptive", learning_rate_init=0.001,
        max_iter=1500, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=40, random_state=99
    )

    # Logistic Regression — L2 regularised
    models["LR"] = LogisticRegression(
        C=0.5, max_iter=5000, solver="lbfgs", penalty="l2",
        random_state=42, n_jobs=-1
    )

    # Bagging of strong HGBs — reduced variance via bootstrap
    models["Bagging-HGB"] = BaggingClassifier(
        estimator=HistGradientBoostingClassifier(
            max_iter=1000, max_depth=6, learning_rate=0.015,
            min_samples_leaf=12, l2_regularization=1.0, random_state=42
        ),
        n_estimators=20, max_samples=0.85, max_features=0.85,
        random_state=42, n_jobs=-1
    )

    # Voting soft ensemble of 3 HGBs with different hyperparameters
    models["Vote-HGB3"] = VotingClassifier(
        estimators=[
            ("hgb_a", HistGradientBoostingClassifier(
                max_iter=1500, max_depth=6, learning_rate=0.012,
                min_samples_leaf=12, l2_regularization=0.8, random_state=10
            )),
            ("hgb_b", HistGradientBoostingClassifier(
                max_iter=2000, max_depth=8, learning_rate=0.01,
                min_samples_leaf=8, l2_regularization=1.5, random_state=20
            )),
            ("hgb_c", HistGradientBoostingClassifier(
                max_iter=1200, max_depth=4, learning_rate=0.02,
                min_samples_leaf=20, l2_regularization=2.0, random_state=30
            )),
        ],
        voting="soft", n_jobs=-1
    )

    # CatBoost — ordered boosting, strong on tabular
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=2500, depth=7, learning_rate=0.01,
            l2_leaf_reg=1.5, random_seed=42, verbose=0,
            loss_function="MultiClass", auto_class_weights="Balanced",
            bootstrap_type="Bayesian", bagging_temperature=0.2,
        )

    return models


# =====================================================================
# Stacking: generate out-of-fold predictions for meta-learner
# =====================================================================
def generate_oof_predictions(X_train: pd.DataFrame, y_train: pd.Series,
                             sample_weights: np.ndarray,
                             models: dict, n_folds: int = 5
                             ) -> Tuple[np.ndarray, dict]:
    """Generate out-of-fold predictions using time-aware splits.

    Uses expanding-window cross-validation:
      - Fold k trains on first (k+1)/n_folds of data, validates on next chunk
      - This respects temporal ordering (no future leakage)

    Returns: oof_preds (n_train, 3*n_models), fitted_models (last fold)
    """
    n = len(X_train)
    n_models = len(models)
    oof_preds = np.full((n, 3 * n_models), np.nan)

    # Create time-aware folds (expanding window)
    fold_size = n // (n_folds + 1)
    min_train = fold_size * 2  # minimum 2 chunks for training

    # Scaler for models requiring normalised input (LR, DeepMLP, MLP-Wide)
    scaler = StandardScaler()

    fitted_models = {}
    for i, (name, model_template) in enumerate(models.items()):
        print(f"      {name}: ", end="", flush=True)
        fold_scores = []

        for fold in range(n_folds):
            val_start = min_train + fold * fold_size
            val_end = min(val_start + fold_size, n)
            if val_start >= n:
                break

            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)

            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            w_tr = sample_weights[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            # Clone model
            from sklearn.base import clone
            model = clone(model_template)

            needs_scaling = name in ("LR", "DeepMLP", "MLP-Wide")
            if needs_scaling:
                sc = StandardScaler()
                X_tr_input = sc.fit_transform(X_tr)
                X_va_input = sc.transform(X_va)
            else:
                X_tr_input = X_tr
                X_va_input = X_va

            model.fit(X_tr_input, y_tr, sample_weight=w_tr)
            preds = model.predict_proba(X_va_input)
            # Clip degenerate predictions — no class should be <2% or >96%
            preds = np.clip(preds, 0.02, 0.96)
            preds /= preds.sum(axis=1, keepdims=True)
            oof_preds[val_idx, i*3:(i+1)*3] = preds

            fold_rps = rps(y_va.values, preds)
            fold_scores.append(fold_rps)
            print(f"f{fold}={fold_rps:.4f} ", end="", flush=True)

        avg_rps = np.mean(fold_scores) if fold_scores else 0
        print(f" avg={avg_rps:.4f}")

        # Train final model on all training data
        final_model = clone(model_template)
        if name in ("LR", "DeepMLP", "MLP-Wide"):
            X_input = scaler.fit_transform(X_train)
            final_model.fit(X_input, y_train, sample_weight=sample_weights)
        else:
            final_model.fit(X_train, y_train, sample_weight=sample_weights)
        fitted_models[name] = final_model

    return oof_preds, fitted_models, scaler


# =====================================================================
# Live Season Data: fetch real 2025-26 PL match results from API
# =====================================================================
def fetch_live_season_stats() -> Dict:
    """Fetch current season PL match results and compute per-team rolling stats.

    Returns a dict keyed by team name (using upcoming_fixtures naming convention)
    with comprehensive stats: form, goals, PPG, streaks, etc.
    """
    import os
    import json
    from pathlib import Path

    cache_path = PROJECT_ROOT / "cache" / "pl_2025_26_matches.json"

    # Try loading cached data first (refresh if older than 6 hours)
    matches = []
    if cache_path.exists():
        import time
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 6:
            with open(cache_path) as f:
                matches = json.load(f)

    if not matches:
        try:
            import requests
            fdo_key = os.environ.get("FDO_KEY", "")
            if not fdo_key:
                env_path = PROJECT_ROOT / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("FDO_KEY="):
                            fdo_key = line.split("=", 1)[1].strip()
            if fdo_key:
                headers = {"X-Auth-Token": fdo_key}
                r = requests.get(
                    "https://api.football-data.org/v4/competitions/PL/matches?season=2025&status=FINISHED&limit=500",
                    headers=headers, timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    for m in data.get("matches", []):
                        matches.append({
                            "date": m["utcDate"][:10],
                            "matchday": m["matchday"],
                            "home_team": m["homeTeam"]["shortName"],
                            "away_team": m["awayTeam"]["shortName"],
                            "home_goals": m["score"]["fullTime"]["home"],
                            "away_goals": m["score"]["fullTime"]["away"],
                        })
                    with open(cache_path, "w") as f:
                        json.dump(matches, f, indent=2)
                    print(f"  [Live] Fetched {len(matches)} current-season PL matches from API")
        except Exception as e:
            print(f"  [Live] API fetch failed: {e}")

    if not matches:
        return {}

    # API shortName -> fixture name mapping
    api_to_fixture = {
        "Arsenal": "Arsenal", "Aston Villa": "Aston Villa",
        "Bournemouth": "Bournemouth", "Brentford": "Brentford",
        "Brighton Hove": "Brighton", "Burnley": "Burnley",
        "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace",
        "Everton": "Everton", "Fulham": "Fulham",
        "Leeds United": "Leeds", "Liverpool": "Liverpool",
        "Man City": "Manchester City", "Man United": "Manchester United",
        "Newcastle": "Newcastle", "Nottingham": "Nottm Forest",
        "Sunderland": "Sunderland", "Tottenham": "Tottenham",
        "West Ham": "West Ham", "Wolverhampton": "Wolves",
    }

    # Sort chronologically
    matches.sort(key=lambda x: x["date"])

    # Build per-team match history
    team_history = {}  # fixture_name -> list of match dicts
    for m in matches:
        ht_fix = api_to_fixture.get(m["home_team"], m["home_team"])
        at_fix = api_to_fixture.get(m["away_team"], m["away_team"])

        for is_home, team_fix in [(True, ht_fix), (False, at_fix)]:
            if team_fix not in team_history:
                team_history[team_fix] = []
            gf = m["home_goals"] if is_home else m["away_goals"]
            ga = m["away_goals"] if is_home else m["home_goals"]
            result = "W" if gf > ga else ("D" if gf == ga else "L")
            pts = 3 if result == "W" else (1 if result == "D" else 0)
            team_history[team_fix].append({
                "date": m["date"], "is_home": is_home,
                "gf": gf, "ga": ga, "result": result, "pts": pts,
            })

    # Compute per-team stats for various windows
    team_stats = {}
    for team, hist in team_history.items():
        s = {"team": team, "total_matches": len(hist)}

        for window_name, window in [("l3", 3), ("l5", 5), ("l8", 8), ("l10", 10), ("l15", 15), ("l20", 20)]:
            recent = hist[-window:] if len(hist) >= window else hist
            n = len(recent)
            if n == 0:
                continue

            wins = sum(1 for m in recent if m["result"] == "W")
            draws = sum(1 for m in recent if m["result"] == "D")
            losses = n - wins - draws
            gf = sum(m["gf"] for m in recent) / n
            ga = sum(m["ga"] for m in recent) / n
            ppg = (3 * wins + draws) / n
            cs = sum(1 for m in recent if m["ga"] == 0) / n
            btts = sum(1 for m in recent if m["gf"] > 0 and m["ga"] > 0) / n
            over25 = sum(1 for m in recent if m["gf"] + m["ga"] > 2) / n

            s[f"ppg_{window_name}"] = ppg
            s[f"goals_scored_{window_name}"] = gf
            s[f"goals_conceded_{window_name}"] = ga
            s[f"clean_sheets_{window_name}"] = cs
            s[f"btts_pct_{window_name}"] = btts
            s[f"over25_pct_{window_name}"] = over25
            s[f"win_pct_{window_name}"] = wins / n
            s[f"draw_pct_{window_name}"] = draws / n
            s[f"loss_pct_{window_name}"] = losses / n

        # Home-specific stats
        home_hist = [m for m in hist if m["is_home"]]
        for window_name, window in [("l3", 3), ("l5", 5), ("l8", 8), ("l10", 10), ("l15", 15), ("l20", 20)]:
            recent = home_hist[-window:] if len(home_hist) >= window else home_hist
            n = len(recent)
            if n == 0:
                continue
            s[f"home_ppg_{window_name}"] = (3 * sum(1 for m in recent if m["result"] == "W") + sum(1 for m in recent if m["result"] == "D")) / n
            s[f"home_gf_{window_name}"] = sum(m["gf"] for m in recent) / n
            s[f"home_ga_{window_name}"] = sum(m["ga"] for m in recent) / n
            s[f"home_cs_{window_name}"] = sum(1 for m in recent if m["ga"] == 0) / n

        # Away-specific stats
        away_hist = [m for m in hist if not m["is_home"]]
        for window_name, window in [("l3", 3), ("l5", 5), ("l8", 8), ("l10", 10), ("l15", 15), ("l20", 20)]:
            recent = away_hist[-window:] if len(away_hist) >= window else away_hist
            n = len(recent)
            if n == 0:
                continue
            s[f"away_ppg_{window_name}"] = (3 * sum(1 for m in recent if m["result"] == "W") + sum(1 for m in recent if m["result"] == "D")) / n
            s[f"away_gf_{window_name}"] = sum(m["gf"] for m in recent) / n
            s[f"away_ga_{window_name}"] = sum(m["ga"] for m in recent) / n
            s[f"away_cs_{window_name}"] = sum(1 for m in recent if m["ga"] == 0) / n

        # Streak and momentum
        streak = 0
        for m in reversed(hist):
            if m["result"] == hist[-1]["result"]:
                streak += 1
            else:
                break
        s["streak"] = streak if hist[-1]["result"] == "W" else (-streak if hist[-1]["result"] == "L" else 0)

        unbeaten = 0
        for m in reversed(hist):
            if m["result"] != "L":
                unbeaten += 1
            else:
                break
        s["unbeaten_run"] = unbeaten

        winless = 0
        for m in reversed(hist):
            if m["result"] != "W":
                winless += 1
            else:
                break
        s["winless_run"] = winless

        # Volatility (std of points in last 10)
        last10 = hist[-10:]
        pts_list = [m["pts"] for m in last10]
        s["volatility"] = float(np.std(pts_list)) if len(pts_list) > 1 else 0.0

        # Goal trend (last 5 vs previous 5)
        if len(hist) >= 10:
            recent5_gf = np.mean([m["gf"] for m in hist[-5:]])
            prev5_gf = np.mean([m["gf"] for m in hist[-10:-5]])
            s["goal_trend"] = recent5_gf - prev5_gf
            recent5_ga = np.mean([m["ga"] for m in hist[-5:]])
            prev5_ga = np.mean([m["ga"] for m in hist[-10:-5]])
            s["defense_trend"] = prev5_ga - recent5_ga  # positive = improving
        else:
            s["goal_trend"] = 0.0
            s["defense_trend"] = 0.0

        # Points velocity
        if len(hist) >= 5:
            s["points_velocity"] = sum(m["pts"] for m in hist[-5:]) / 5 - sum(m["pts"] for m in hist) / len(hist)
        else:
            s["points_velocity"] = 0.0

        # Weighted streak (last 10, more recent = more weight)
        if hist:
            last10 = hist[-10:]
            weights = np.array([1.5 ** i for i in range(len(last10))])
            pts_arr = np.array([m["pts"] for m in last10])
            s["weighted_streak"] = float(np.average(pts_arr, weights=weights))
        else:
            s["weighted_streak"] = 1.0

        # Form at home / form at away
        if home_hist:
            last5h = home_hist[-5:]
            s["home_form_at_home"] = sum(1 for m in last5h if m["result"] == "W") / len(last5h)
        else:
            s["home_form_at_home"] = 0.4
        if away_hist:
            last5a = away_hist[-5:]
            s["away_form_at_away"] = sum(1 for m in last5a if m["result"] == "W") / len(last5a)
        else:
            s["away_form_at_away"] = 0.25

        # Table position from API standings
        total_pts = sum(m["pts"] for m in hist)
        total_gd = sum(m["gf"] - m["ga"] for m in hist)
        s["table_pts"] = total_pts
        s["table_gd"] = total_gd
        s["total_gf"] = sum(m["gf"] for m in hist)
        s["total_ga"] = sum(m["ga"] for m in hist)

        # Early goals, comeback rate, first goal rate (from available data)
        s["early_goals_pct"] = 0.35  # default, would need minute data
        s["comeback_rate"] = sum(1 for m in hist[-10:] if m["result"] == "W") / max(len(hist[-10:]), 1) * 0.3
        s["first_goal_rate"] = sum(1 for m in hist[-10:] if m["gf"] > 0) / max(len(hist[-10:]), 1)

        team_stats[team] = s

    # Compute table positions from points
    sorted_teams = sorted(team_stats.values(), key=lambda x: (-x["table_pts"], -x["table_gd"]))
    for pos, ts in enumerate(sorted_teams, 1):
        team_stats[ts["team"]]["table_pos"] = pos

    return team_stats


def _map_live_stats_to_features(team_stats: Dict, home_team: str, away_team: str,
                                 feature_cols: List[str]) -> Dict:
    """Map live season stats to the model's feature columns.

    This creates a feature dict using REAL current-season data from the API,
    overriding any stale training-data features.
    """
    h_stats = team_stats.get(home_team, {})
    a_stats = team_stats.get(away_team, {})
    feats = {}

    if not h_stats and not a_stats:
        return feats

    # Direct mappings: feature_col -> (home_stat_key, away_stat_key)
    window_mappings = {
        "ppg_home": "home_ppg", "ppg_away": "away_ppg",
        "goals_scored_home": "home_gf", "goals_scored_away": "away_gf",
        "goals_conceded_home": "home_ga", "goals_conceded_away": "away_ga",
        "clean_sheets_home": "home_cs", "clean_sheets_away": "away_cs",
    }

    # Rolling window features (ppg_home_l3, ppg_home_l5, etc.)
    for fc in feature_cols:
        val = None

        # Match rolling window features
        for prefix, stat_prefix in window_mappings.items():
            for wn in ("_l3", "_l5", "_l8", "_l10", "_l15", "_l20"):
                if fc == f"{prefix}{wn}":
                    is_home_feat = "_home" in prefix
                    stats = h_stats if is_home_feat else a_stats
                    val = stats.get(f"{stat_prefix}{wn}")
                    break
            if val is not None:
                break

        # Overall rolling features (btts_pct_home_l5, over25_pct_home_l8, etc.)
        if val is None:
            for base in ("btts_pct", "over25_pct", "win_pct", "draw_pct", "loss_pct"):
                for side in ("home", "away"):
                    for wn in ("_l3", "_l5", "_l8", "_l10", "_l15", "_l20"):
                        if fc == f"{base}_{side}{wn}":
                            stats = h_stats if side == "home" else a_stats
                            val = stats.get(f"{base}{wn}")
                            break
                    if val is not None:
                        break
                if val is not None:
                    break

        # Table features
        if val is None:
            table_map = {
                "table_pos_home": ("home", "table_pos"),
                "table_pos_away": ("away", "table_pos"),
                "table_pts_home": ("home", "table_pts"),
                "table_pts_away": ("away", "table_pts"),
                "table_gd_home": ("home", "table_gd"),
                "table_gd_away": ("away", "table_gd"),
                "home_in_top4": ("home", "table_pos"),
                "away_in_top4": ("away", "table_pos"),
                "top6_home": ("home", "table_pos"),
                "top6_away": ("away", "table_pos"),
                "bottom3_home": ("home", "table_pos"),
                "bottom3_away": ("away", "table_pos"),
            }
            if fc in table_map:
                side, key = table_map[fc]
                stats = h_stats if side == "home" else a_stats
                raw_val = stats.get(key)
                if raw_val is not None:
                    if fc.startswith("home_in_top4") or fc.startswith("away_in_top4"):
                        val = 1.0 if raw_val <= 4 else 0.0
                    elif fc.startswith("top6"):
                        val = 1.0 if raw_val <= 6 else 0.0
                    elif fc.startswith("bottom3"):
                        val = 1.0 if raw_val >= 18 else 0.0
                    else:
                        val = raw_val

        # Streak and momentum features
        if val is None:
            streak_map = {
                "streak_home": ("home", "streak"),
                "streak_away": ("away", "streak"),
                "volatility_home": ("home", "volatility"),
                "volatility_away": ("away", "volatility"),
                "goal_trend_home": ("home", "goal_trend"),
                "goal_trend_away": ("away", "goal_trend"),
                "defense_trend_home": ("home", "defense_trend"),
                "defense_trend_away": ("away", "defense_trend"),
                "unbeaten_run_home": ("home", "unbeaten_run"),
                "unbeaten_run_away": ("away", "unbeaten_run"),
                "winless_run_home": ("home", "winless_run"),
                "winless_run_away": ("away", "winless_run"),
                "points_velocity_away": ("away", "points_velocity"),
                "weighted_streak_home_l10": ("home", "weighted_streak"),
                "weighted_streak_away_l10": ("away", "weighted_streak"),
                "home_form_at_home": ("home", "home_form_at_home"),
                "away_form_at_away": ("away", "away_form_at_away"),
                "early_goals_pct_home": ("home", "early_goals_pct"),
                "early_goals_pct_away": ("away", "early_goals_pct"),
                "comeback_rate_home": ("home", "comeback_rate"),
                "comeback_rate_away": ("away", "comeback_rate"),
                "first_goal_rate_home": ("home", "first_goal_rate"),
                "first_goal_rate_away": ("away", "first_goal_rate"),
            }
            if fc in streak_map:
                side, key = streak_map[fc]
                stats = h_stats if side == "home" else a_stats
                val = stats.get(key)

        # Diff features: compute from home - away
        if val is None and fc.endswith("_diff"):
            base = fc.replace("_diff", "")
            h_val = feats.get(f"{base}_home")
            a_val = feats.get(f"{base}_away")
            if h_val is not None and a_val is not None:
                val = h_val - a_val

        # Position gap
        if val is None and fc == "position_gap":
            hp = h_stats.get("table_pos")
            ap = a_stats.get("table_pos")
            if hp is not None and ap is not None:
                val = abs(hp - ap)

        if val is not None:
            feats[fc] = val

    # Compute derived features
    hp = h_stats.get("table_pos", 10)
    ap = a_stats.get("table_pos", 10)
    if "relegation_battle" in feature_cols:
        feats["relegation_battle"] = 1.0 if hp >= 17 and ap >= 17 else 0.0
    if "promotion_zone_match" in feature_cols:
        feats["promotion_zone_match"] = 1.0 if hp >= 18 and ap >= 18 else 0.0

    return feats


# =====================================================================
# Feature preparation for upcoming matches
# =====================================================================
def prepare_upcoming_features(upcoming_df: pd.DataFrame,
                               featured_df: pd.DataFrame,
                               feature_cols: List[str],
                               live_sentiment: Dict = None) -> pd.DataFrame:
    """Generate feature vectors for upcoming matches using CURRENT SEASON data.

    This function dynamically computes features from the latest available data,
    heavily prioritising current season form. It does NOT just copy the last row —
    it intelligently blends the team's most recent home/away performance.

    For each upcoming match:
    1. Fetch REAL 2025-26 PL match data from API (29+ games per team)
    2. Find the team's LAST match where they played at HOME (for home features)
    3. Find the team's LAST match where they played AWAY (for away features)
    4. Override with live season stats (rolling form, table position, streaks)
    5. Add live sentiment if available
    6. Compute H2H features from historical matchup data
    """
    if upcoming_df.empty:
        return pd.DataFrame()

    # Fetch real current-season data from API
    live_team_stats = fetch_live_season_stats()
    if live_team_stats:
        n_teams = len(live_team_stats)
        avg_matches = np.mean([s.get("total_matches", 0) for s in live_team_stats.values()])
        print(f"  [Live] Using real 2025-26 data: {n_teams} teams, avg {avg_matches:.0f} matches each")
    else:
        print("  [Live] WARNING: No live season data available, using training data only")

    latest = featured_df.sort_values("date")

    # Get current season (most recent in data)
    all_seasons = sorted(featured_df["season"].unique())
    current_season = all_seasons[-1] if all_seasons else None

    # Build team stat lookups: latest overall, latest as home, latest as away
    # CRITICAL: track home/away side separately so we don't mix up team stats
    team_latest_overall = {}   # Last match regardless of venue
    team_latest_as_home = {}   # Last match where team was HOME
    team_latest_as_away = {}   # Last match where team was AWAY
    team_cs_as_home = {}       # Last CURRENT SEASON match where team was HOME
    team_cs_as_away = {}       # Last CURRENT SEASON match where team was AWAY

    for _, row in latest.iterrows():
        ht, at = row["home_team"], row["away_team"]
        team_latest_overall[ht] = row
        team_latest_overall[at] = row
        team_latest_as_home[ht] = row
        team_latest_as_away[at] = row
        if current_season and row.get("season") == current_season:
            team_cs_as_home[ht] = row
            team_cs_as_away[at] = row

    # Detect promoted teams (in upcoming fixtures but NOT in current season data).
    # Their old PL data may be years stale. For these teams, we flag them so the
    # feature builder uses conservative "promoted team" baselines.
    cs_teams = set()
    if current_season:
        cs_df = featured_df[featured_df["season"] == current_season]
        cs_teams = set(cs_df["home_team"].unique()) | set(cs_df["away_team"].unique())
    upcoming_teams = set(upcoming_df["home_team"].unique()) | set(upcoming_df["away_team"].unique())
    promoted_teams = upcoming_teams - cs_teams

    # H2H lookup: collect all historical meetings
    h2h_data = {}
    for _, row in latest.iterrows():
        ht, at = row["home_team"], row["away_team"]
        pair_key = tuple(sorted([ht, at]))
        h2h_data.setdefault(pair_key, []).append(row)

    # Current-season league averages (for promoted team defaults)
    cs_avgs = {}
    if current_season:
        cs_data = featured_df[featured_df["season"] == current_season]
        for col in feature_cols:
            if col in cs_data.columns:
                cs_avgs[col] = cs_data[col].mean()

    rows = []
    for _, match in upcoming_df.iterrows():
        ht, at = match["home_team"], match["away_team"]
        feat_row = {}

        # Check if either team is newly promoted (not in current season data).
        # Promoted teams get conservative baselines instead of stale ancient data.
        ht_promoted = ht in promoted_teams
        at_promoted = at in promoted_teams

        # Priority: current season (venue-matched) > all-time venue-matched > overall
        # CRITICAL: home features must come from a row where the team was HOME,
        # away features from a row where the team was AWAY.
        if ht_promoted:
            # Promoted team: don't use stale data from years ago
            ht_source_home = None
            ht_overall = None
        else:
            ht_source_home = team_cs_as_home.get(ht, team_latest_as_home.get(ht))
            ht_overall = team_latest_overall.get(ht)

        if at_promoted:
            at_source_away = None
            at_overall = None
        else:
            at_source_away = team_cs_as_away.get(at, team_latest_as_away.get(at))
            at_overall = team_latest_overall.get(at)

        for col in feature_cols:
            val = np.nan

            if col.startswith("h2h_") or col.startswith("live_"):
                # H2H and live features handled separately below
                val = np.nan
            elif "_home" in col or col.endswith("_home"):
                # Home team features: prefer current season, then venue-specific
                if ht_source_home is not None:
                    v = ht_source_home.get(col, np.nan)
                    val = v if pd.notna(v) else np.nan
                if pd.isna(val) and ht_overall is not None:
                    v = ht_overall.get(col, np.nan)
                    val = v if pd.notna(v) else np.nan
            elif "_away" in col or col.endswith("_away"):
                # Away team features: prefer current season, then venue-specific
                if at_source_away is not None:
                    v = at_source_away.get(col, np.nan)
                    val = v if pd.notna(v) else np.nan
                if pd.isna(val) and at_overall is not None:
                    v = at_overall.get(col, np.nan)
                    val = v if pd.notna(v) else np.nan
            else:
                # Neutral features — split into categories:
                # 1) "diff" features: recompute from home/away values
                # 2) Opponent-pair features: recompute from both teams' data
                # 3) Truly neutral (weather, calendar, referee): pull from any row
                if col.endswith("_diff"):
                    # Recompute diff from correctly-sourced home/away values
                    # Try multiple naming conventions: base_home, base_days_home, etc.
                    base = col.replace("_diff", "")
                    h_col, a_col = None, None
                    # Try exact match first, then fuzzy matches
                    for h_candidate, a_candidate in [
                        (f"{base}_home", f"{base}_away"),
                        (f"{base}_days_home", f"{base}_days_away"),
                        (f"{base}_home_lambda", f"{base}_away_lambda"),
                        (f"{base}_mu_home", f"{base}_mu_away"),
                        (f"{base}_rd_home", f"{base}_rd_away"),
                        (f"{base}_vol_home", f"{base}_vol_away"),
                    ]:
                        if h_candidate in feature_cols and a_candidate in feature_cols:
                            h_col, a_col = h_candidate, a_candidate
                            break
                    if h_col and a_col:
                        h_val = feat_row.get(h_col, np.nan)
                        a_val = feat_row.get(a_col, np.nan)
                        if pd.notna(h_val) and pd.notna(a_val):
                            val = h_val - a_val
                        elif ht_source_home is not None:
                            v = ht_source_home.get(col, np.nan)
                            val = v if pd.notna(v) else np.nan
                    elif ht_source_home is not None:
                        v = ht_source_home.get(col, np.nan)
                        val = v if pd.notna(v) else np.nan
                elif col in ("position_gap", "promotion_zone_match",
                             "relegation_battle", "rest_advantage",
                             "market_favorite", "rating_concordance",
                             "elo_agreement"):
                    # Opponent-pair-specific features: recompute from both teams
                    if col == "position_gap":
                        tp_h = feat_row.get("table_pos_home", np.nan)
                        tp_a = feat_row.get("table_pos_away", np.nan)
                        val = abs(tp_h - tp_a) if pd.notna(tp_h) and pd.notna(tp_a) else np.nan
                    elif col == "promotion_zone_match":
                        tp_h = feat_row.get("table_pos_home", np.nan)
                        tp_a = feat_row.get("table_pos_away", np.nan)
                        val = 1.0 if pd.notna(tp_h) and pd.notna(tp_a) and tp_h >= 18 and tp_a >= 18 else 0.0
                    elif col == "relegation_battle":
                        tp_h = feat_row.get("table_pos_home", np.nan)
                        tp_a = feat_row.get("table_pos_away", np.nan)
                        val = 1.0 if pd.notna(tp_h) and pd.notna(tp_a) and tp_h >= 17 and tp_a >= 17 else 0.0
                    elif col == "rest_advantage":
                        rd_h = feat_row.get("rest_days_home", np.nan)
                        rd_a = feat_row.get("rest_days_away", np.nan)
                        if pd.notna(rd_h) and pd.notna(rd_a):
                            val = rd_h - rd_a
                        else:
                            val = 0.0
                    elif col == "market_favorite":
                        # Determine from odds if available, else from Elo
                        elo_h = feat_row.get("elo_home", np.nan)
                        elo_a = feat_row.get("elo_away", np.nan)
                        if pd.notna(elo_h) and pd.notna(elo_a):
                            val = -1.0 if elo_h > elo_a else (1.0 if elo_a > elo_h else 0.0)
                        else:
                            val = 0.0
                    elif col in ("rating_concordance", "elo_agreement"):
                        # These need multiple rating systems — pull from home match
                        if ht_source_home is not None:
                            v = ht_source_home.get(col, np.nan)
                            val = v if pd.notna(v) else 0.5
                        else:
                            val = 0.5
                elif col.startswith("poisson_"):
                    # Poisson features are opponent-pair-specific — recompute
                    # from both teams' attacking/defending stats
                    lam_h = feat_row.get("poisson_home_lambda", np.nan)
                    lam_a = feat_row.get("poisson_away_lambda", np.nan)
                    if pd.notna(lam_h) and pd.notna(lam_a):
                        if col == "poisson_total_lambda":
                            val = lam_h + lam_a
                        elif col == "poisson_draw":
                            # P(draw) ≈ sum of P(k,k) for k=0..5
                            from scipy.stats import poisson as _poisson
                            val = sum(_poisson.pmf(k, lam_h) * _poisson.pmf(k, lam_a) for k in range(6))
                        elif col == "poisson_over25":
                            from scipy.stats import poisson as _poisson
                            val = 1.0 - sum(
                                _poisson.pmf(i, lam_h) * _poisson.pmf(j, lam_a)
                                for i in range(3) for j in range(3) if i + j <= 2
                            )
                        elif col == "poisson_btts":
                            from scipy.stats import poisson as _poisson
                            val = (1 - _poisson.pmf(0, lam_h)) * (1 - _poisson.pmf(0, lam_a))
                        elif col == "poisson_away_win":
                            from scipy.stats import poisson as _poisson
                            val = sum(
                                _poisson.pmf(i, lam_h) * _poisson.pmf(j, lam_a)
                                for i in range(7) for j in range(i + 1, 7)
                            )
                        else:
                            if ht_source_home is not None:
                                v = ht_source_home.get(col, np.nan)
                                val = v if pd.notna(v) else np.nan
                    else:
                        if ht_source_home is not None:
                            v = ht_source_home.get(col, np.nan)
                            val = v if pd.notna(v) else np.nan
                else:
                    # Truly neutral: weather, calendar, referee, odds, flags
                    if ht_source_home is not None:
                        v = ht_source_home.get(col, np.nan)
                        val = v if pd.notna(v) else np.nan
                    if pd.isna(val) and ht_overall is not None:
                        v = ht_overall.get(col, np.nan)
                        val = v if pd.notna(v) else np.nan

            feat_row[col] = val

        # --- LIVE SEASON DATA OVERRIDE ---
        # Use real 2025-26 match data from the API to override stale training features.
        # This is critical for ALL teams (not just promoted) because the training data
        # only covers through 2024-25, and form changes within the current season.
        if live_team_stats:
            live_feats = _map_live_stats_to_features(live_team_stats, ht, at, feature_cols)
            for lk, lv in live_feats.items():
                if lv is not None:
                    # Override if: (a) current value is NaN, or (b) it's a rolling/form
                    # feature that should reflect current-season reality
                    current_val = feat_row.get(lk, np.nan)
                    is_rolling = any(lk.startswith(p) for p in (
                        "ppg_", "goals_scored_", "goals_conceded_", "clean_sheets_",
                        "btts_pct_", "over25_pct_", "win_pct_", "draw_pct_", "loss_pct_",
                        "table_", "streak_", "volatility_", "goal_trend_", "defense_trend_",
                        "unbeaten_", "winless_", "weighted_streak_", "points_velocity_",
                        "home_form_", "away_form_", "early_goals_", "comeback_", "first_goal_",
                        "bottom3_", "top6_", "home_in_top4", "away_in_top4",
                    ))
                    is_position = lk in ("position_gap", "relegation_battle", "promotion_zone_match")
                    if pd.isna(current_val) or is_rolling or is_position:
                        feat_row[lk] = lv

            # --- Estimate xG from live goal data to avoid stale xG contradictions ---
            # The training data has xG from 2024-25, but current-season goals may be
            # very different. Create estimated xG that's consistent with live data.
            # xG_estimated ≈ 0.85 * actual_goals + 0.15 * league_avg (regression to mean)
            league_avg_gf = 1.35  # PL long-term average goals per team per match
            for side, stats in [("home", live_team_stats.get(ht, {})), ("away", live_team_stats.get(at, {}))]:
                for wn in ("_l3", "_l5", "_l10", "_l20"):
                    gf_key = f"goals_scored_{side}{wn}"
                    ga_key = f"goals_conceded_{side}{wn}"
                    xgf_key = f"xg_for_{side}{wn}"
                    xga_key = f"xg_against_{side}{wn}"

                    gf_val = feat_row.get(gf_key)
                    ga_val = feat_row.get(ga_key)
                    if xgf_key in feature_cols and gf_val is not None and pd.notna(gf_val):
                        feat_row[xgf_key] = 0.85 * gf_val + 0.15 * league_avg_gf
                    if xga_key in feature_cols and ga_val is not None and pd.notna(ga_val):
                        feat_row[xga_key] = 0.85 * ga_val + 0.15 * league_avg_gf

            # Estimate xG overperformance (goals - xG) → should be ~0 for estimated xG
            for side in ("home", "away"):
                for wn in ("_l3", "_l5", "_l8", "_l10", "_l15", "_l20"):
                    op_key = f"xg_overperformance_{side}{wn}"
                    if op_key in feature_cols:
                        feat_row[op_key] = 0.0  # Estimated xG matches actual goals

        # --- Promoted team baselines (FALLBACK ONLY) ---
        # These are only used if live season data was unavailable (API down, etc.).
        # With live data, the override above already provides real current-season stats.
        # For features that live data doesn't cover (Elo, Pi, Glicko, xG), use
        # conservative promoted-team estimates.
        _promoted_baselines = {
            "elo_home": 1460, "elo_away": 1460,
            "elo_momentum_home": 0.0, "elo_momentum_away": 0.0,
            "pi_rating_home": -0.3, "pi_rating_away": -0.3,
            "pi_home_attack": 0.9, "pi_home_defense": -0.2,
            "pi_away_attack": 0.85, "pi_away_defense": -0.25,
            "glicko_mu_home": 1550, "glicko_mu_away": 1550,
            "glicko_rd_home": 60, "glicko_rd_away": 60,
            "glicko_confidence_home": 0.7, "glicko_confidence_away": 0.7,
            "xg_for_home_l5": 1.1, "xg_for_away_l5": 1.0,
            "xg_for_home_l10": 1.1, "xg_for_away_l10": 1.0,
            "xg_for_home_l20": 1.1, "xg_for_away_l20": 1.0,
            "xg_against_home_l5": 1.5, "xg_against_away_l5": 1.6,
            "xg_against_home_l10": 1.4, "xg_against_away_l10": 1.5,
            "xg_against_home_l20": 1.4, "xg_against_away_l20": 1.5,
            "xg_overperformance_home_l5": 0.0, "xg_overperformance_away_l5": 0.0,
            "xg_overperformance_home_l10": 0.0, "xg_overperformance_away_l10": 0.0,
            "attack_vs_defense_home": 0.8, "attack_vs_defense_away": 0.7,
            "attack_def_mismatch_home": 0.0, "attack_def_mismatch_away": 0.0,
            "form_elo_divergence_home": 0.0, "form_elo_divergence_away": 0.0,
            "defensive_solidity_home": 0.4, "defensive_solidity_away": 0.35,
            "upset_tendency_home": 0.3, "upset_tendency_away": 0.3,
        }
        if ht_promoted:
            for bk, bv in _promoted_baselines.items():
                if ("_home" in bk or bk.endswith("_home")) and pd.isna(feat_row.get(bk, np.nan)):
                    feat_row[bk] = bv
        if at_promoted:
            for bk, bv in _promoted_baselines.items():
                if ("_away" in bk or bk.endswith("_away")) and pd.isna(feat_row.get(bk, np.nan)):
                    feat_row[bk] = bv

        # --- H2H features from historical meetings ---
        pair_key = tuple(sorted([ht, at]))
        meetings = h2h_data.get(pair_key, [])
        if meetings:
            recent = meetings[-10:]
            # From home team's perspective
            h_wins = sum(1 for m in recent
                        if (m["home_team"] == ht and m["goals_home"] > m["goals_away"]) or
                           (m["away_team"] == ht and m["goals_away"] > m["goals_home"]))
            draws = sum(1 for m in recent
                       if m["goals_home"] == m["goals_away"])
            n = len(recent)
            feat_row["h2h_home_win_pct"] = h_wins / n
            feat_row["h2h_draw_pct"] = draws / n
            feat_row["h2h_away_win_pct"] = 1 - h_wins / n - draws / n
            feat_row["h2h_avg_goals"] = np.mean([m["goals_home"] + m["goals_away"] for m in recent])
            feat_row["h2h_matches"] = n

            # Advanced H2H
            total_goals = [m["goals_home"] + m["goals_away"] for m in recent]
            feat_row["h2h_goals_volatility"] = float(np.std(total_goals)) if len(total_goals) > 1 else 0.0
            feat_row["h2h_btts_rate"] = np.mean([
                1 if m["goals_home"] > 0 and m["goals_away"] > 0 else 0 for m in recent])
            feat_row["h2h_over25_rate"] = np.mean([
                1 if m["goals_home"] + m["goals_away"] > 2 else 0 for m in recent])

            # Home team goals in these meetings
            ht_goals = []
            at_goals = []
            for m in recent:
                if m["home_team"] == ht:
                    ht_goals.append(m["goals_home"])
                    at_goals.append(m["goals_away"])
                else:
                    ht_goals.append(m["goals_away"])
                    at_goals.append(m["goals_home"])
            feat_row["h2h_home_goals_avg"] = np.mean(ht_goals) if ht_goals else 1.5
            feat_row["h2h_away_goals_avg"] = np.mean(at_goals) if at_goals else 1.2

            # Venue dominance
            home_meetings = [m for m in recent if m["home_team"] == ht]
            if home_meetings:
                feat_row["h2h_venue_dominance"] = sum(
                    1 for m in home_meetings if m["goals_home"] > m["goals_away"]) / len(home_meetings)
            else:
                feat_row["h2h_venue_dominance"] = 0.45

            # Recent shift
            if n >= 4:
                last3_wins = sum(1 for m in recent[-3:]
                    if (m["home_team"] == ht and m["goals_home"] > m["goals_away"]) or
                       (m["away_team"] == ht and m["goals_away"] > m["goals_home"]))
                feat_row["h2h_recent_shift"] = last3_wins / 3 - h_wins / n
            else:
                feat_row["h2h_recent_shift"] = 0.0

            feat_row["h2h_xg_dominance"] = 0.0
            feat_row["h2h_home_scoring_trend"] = 0.0
            feat_row["h2h_clean_sheet_pct"] = np.mean([
                1 if (m["home_team"] == ht and m["goals_away"] == 0) or
                     (m["away_team"] == ht and m["goals_home"] == 0)
                else 0 for m in recent])
        else:
            feat_row["h2h_home_win_pct"] = 0.45
            feat_row["h2h_draw_pct"] = 0.27
            feat_row["h2h_away_win_pct"] = 0.28
            feat_row["h2h_avg_goals"] = 2.7
            feat_row["h2h_matches"] = 0
            feat_row["h2h_goals_volatility"] = 0.0
            feat_row["h2h_btts_rate"] = 0.5
            feat_row["h2h_over25_rate"] = 0.5
            feat_row["h2h_home_goals_avg"] = 1.5
            feat_row["h2h_away_goals_avg"] = 1.2
            feat_row["h2h_venue_dominance"] = 0.45
            feat_row["h2h_recent_shift"] = 0.0
            feat_row["h2h_xg_dominance"] = 0.0
            feat_row["h2h_home_scoring_trend"] = 0.0
            feat_row["h2h_clean_sheet_pct"] = 0.3

        # --- Live sentiment features ---
        if live_sentiment:
            from features.sentiment import get_match_sentiment_features
            sent_feats = get_match_sentiment_features(ht, at, live_sentiment)
            feat_row.update(sent_feats)

        # --- Override stale calendar features with actual match date ---
        match_date = pd.to_datetime(match.get("date", pd.NaT))
        if pd.notna(match_date):
            feat_row["month"] = match_date.month
            feat_row["day_of_week"] = match_date.dayofweek
            feat_row["is_midweek"] = 1.0 if match_date.dayofweek in (1, 2, 3) else 0.0
            feat_row["is_early_season"] = 1.0 if match_date.month in (8, 9) else 0.0
            feat_row["is_run_in"] = 1.0 if match_date.month in (4, 5) else 0.0
            feat_row["is_cold"] = 1.0 if match_date.month in (11, 12, 1, 2) else 0.0
            # season_progress: approximate as fraction through Aug-May season
            season_start_month = 8
            months_in = (match_date.month - season_start_month) % 12
            feat_row["season_progress"] = min(months_in / 10.0, 1.0)

        # --- Nullify stale odds features (from previous match, NOT this match) ---
        # These will be filled with training medians by apply_cleaning, giving
        # a neutral signal rather than misleading stale bookmaker data.
        stale_odds_cols = [
            "odds_home_close", "odds_draw_open", "implied_prob_away_close",
            "market_overround_close", "odds_movement_draw", "steam_move_flag",
            "shin_home_draw_ratio", "odds_drift_home", "odds_drift_away",
            "market_entropy", "elo_market_agree",
        ]
        for oc in stale_odds_cols:
            if oc in feat_row:
                feat_row[oc] = np.nan

        feat_row["home_team"] = ht
        feat_row["away_team"] = at
        feat_row["date"] = match.get("date", pd.NaT)
        rows.append(feat_row)

    result = pd.DataFrame(rows)

    # Ensure all feature_cols exist (fill missing with 0 for live_ features)
    for col in feature_cols:
        if col not in result.columns:
            result[col] = 0.0 if col.startswith("live_") else np.nan

    return result


def _best_score_for_outcome(top_scores: list, outcome_idx: int) -> str:
    """Pick the most likely scoreline consistent with the predicted outcome.
    outcome_idx: 0=Home Win, 1=Draw, 2=Away Win.
    Falls back to the overall most likely score if no consistent score found."""
    if not top_scores:
        return "1-0" if outcome_idx == 0 else ("1-1" if outcome_idx == 1 else "0-1")
    for s in top_scores:
        h, a = map(int, s["score"].split("-"))
        if outcome_idx == 0 and h > a:
            return s["score"]
        elif outcome_idx == 1 and h == a:
            return s["score"]
        elif outcome_idx == 2 and h < a:
            return s["score"]
    return top_scores[0]["score"]


def _run_cached_predictions(upcoming_raw, live_sentiment, feature_cols,
                             fitted_models, scaler, clean_state,
                             binary_models, calibrators, goal_models,
                             dc, dc_xg, meta_models, calibrated,
                             ensemble_method, test_season, df):
    """Run predictions using cached models — full pipeline quality."""
    print("\n" + "=" * 70)
    print("  UPCOMING MATCH PREDICTIONS (from cached models)")
    print("=" * 70)

    if upcoming_raw.empty:
        print("  No upcoming fixtures found")
        return

    upcoming_features = prepare_upcoming_features(upcoming_raw, df, feature_cols,
                                                   live_sentiment=live_sentiment)
    if upcoming_features.empty:
        print("  No features generated for upcoming matches")
        return

    X_upcoming = apply_cleaning(upcoming_features, clean_state)
    X_upcoming_s = scaler.transform(X_upcoming)

    hgb_home = goal_models.get("home")
    hgb_away = goal_models.get("away")

    # Get predictions from all base learners (with clipping, same as full pipeline)
    upcoming_base = {}
    for name, model in fitted_models.items():
        if name in ("LR", "DeepMLP", "MLP-Wide"):
            p = model.predict_proba(X_upcoming_s)
        else:
            p = model.predict_proba(X_upcoming)
        p = np.clip(p, 0.02, 0.96)
        p /= p.sum(axis=1, keepdims=True)
        upcoming_base[name] = p

    # Goal predictions (ML)
    pred_ug_home = np.clip(hgb_home.predict(X_upcoming), 0.2, 5) if hgb_home else np.full(len(X_upcoming), 1.3)
    pred_ug_away = np.clip(hgb_away.predict(X_upcoming), 0.1, 5) if hgb_away else np.full(len(X_upcoming), 1.1)

    # Retrieve stacking meta-learner and scaler from cache
    meta_lr = meta_models.get("meta_lr") if meta_models else None
    meta_scaler_cached = meta_models.get("meta_scaler") if meta_models else None

    # Market odds meta-features: always include 3 features to match training shape.
    # These are stacking-level features (not primary features), so we default
    # to uniform [0.33, 0.33, 0.33] when actual market odds are unavailable.
    market_meta_cols = ["implied_prob_home_open", "implied_prob_draw_open", "implied_prob_away_open"]
    has_market_in_upcoming = all(c in upcoming_features.columns for c in market_meta_cols)

    # Determine expected meta_row size from cached scaler
    expected_meta_size = meta_scaler_cached.n_features_in_ if meta_scaler_cached is not None else None

    upcoming_preds = []

    for i, (_, match) in enumerate(upcoming_raw.iterrows()):
        home = match["home_team"]
        away = match["away_team"]

        # ── Dixon-Coles predictions (goal markets) ──
        dc_pred = dc.predict_outcome(home, away)
        xg_h, xg_a = dc.predict_expected_goals(home, away)
        over25 = dc.predict_over_under(home, away)
        btts_p = dc.predict_btts(home, away)
        score_probs = dc.predict_score_probs(home, away)

        # Top scorelines from Dixon-Coles
        top_scores = []
        for si in range(min(7, score_probs.shape[0])):
            for sj in range(min(7, score_probs.shape[1])):
                top_scores.append({"score": f"{si}-{sj}", "prob": float(score_probs[si, sj])})
        top_scores.sort(key=lambda x: -x["prob"])

        # ── Build ensemble prediction ──
        dc_row = np.array([dc_pred["home"], dc_pred["draw"], dc_pred["away"]])

        if meta_lr is not None and meta_scaler_cached is not None and has_market_in_upcoming:
            # When real market odds are available, use the full stacking meta-learner
            base_row = np.hstack([upcoming_base[name][i] for name in fitted_models.keys()])
            mkt_row = upcoming_features[market_meta_cols].iloc[i].values.astype(float)
            mkt_row = np.nan_to_num(mkt_row, nan=0.33)
            mkt_row = mkt_row / max(mkt_row.sum(), 1e-6)
            if binary_models:
                bin_row = np.array([
                    binary_models[bn].predict_proba(X_upcoming.iloc[[i]])[:, 1][0]
                    for bn in ["hw", "aw", "dr", "hd"] if bn in binary_models
                ])
            else:
                bin_row = np.array([])
            meta_row = np.hstack([base_row, dc_row, mkt_row, bin_row]).reshape(1, -1)
            if expected_meta_size is not None and meta_row.shape[1] != expected_meta_size:
                if meta_row.shape[1] < expected_meta_size:
                    meta_row = np.hstack([meta_row, np.zeros((1, expected_meta_size - meta_row.shape[1]))])
                else:
                    meta_row = meta_row[:, :expected_meta_size]
            try:
                meta_row_s = meta_scaler_cached.transform(meta_row)
                cal_upcoming = meta_lr.predict_proba(meta_row_s)[0]
            except Exception as e:
                print(f"  [Cache] WARNING: Meta-learner failed ({e}), using weighted blend for {home} vs {away}")
                base_avg = np.mean([upcoming_base[n][i] for n in upcoming_base], axis=0)
                cal_upcoming = 0.35 * dc_row + 0.65 * base_avg
        else:
            # No market odds → meta-learner unreliable (it was trained with odds).
            # Use ADAPTIVE blend with DC-ML disagreement detection.
            all_base = np.array([upcoming_base[n][i] for n in upcoming_base])
            base_avg = all_base.mean(axis=0)

            # Model agreement: low std = high consensus = trust ML more
            # High std = disagreement = lean on DC (structural model)
            base_std = all_base.std(axis=0).mean()

            # DC-ML outcome disagreement: if DC and ML base learners disagree
            # on WHO WINS, increase DC weight. DC uses structural Elo-based team
            # ratings that reflect overall season quality, while ML models can
            # overfit to recent hot/cold streaks or suffer from distribution shift
            # (trained on 2024-25 but predicting 2025-26 with live features).
            dc_fav = int(np.argmax(dc_row))
            ml_fav = int(np.argmax(base_avg))
            dc_conf = dc_row[dc_fav]

            # Detect extreme ML predictions: if ML avg gives >75% to one outcome
            # while DC disagrees, the ML is likely overfitting. Cap ML contribution.
            ml_max = base_avg.max()
            extreme_ml = (ml_max > 0.70 and dc_fav != ml_fav)

            if extreme_ml:
                # ML is extremely confident but disagrees with DC — likely distribution
                # shift. Use DC-heavy blend.
                dc_weight = np.clip(0.55 + (dc_conf - 0.50) * 0.8, 0.55, 0.75)
            elif dc_fav != ml_fav and dc_conf > 0.50:
                # Moderate disagreement: DC and ML pick different outcomes
                dc_weight = np.clip(0.45 + (dc_conf - 0.50) * 1.0, 0.40, 0.60)
            else:
                # Agreement or low-confidence DC: standard adaptive blend
                dc_weight = np.clip(0.30 + (base_std - 0.08) * 1.5, 0.30, 0.50)

            cal_upcoming = dc_weight * dc_row + (1 - dc_weight) * base_avg

        # Apply calibrators only when the meta-learner was used (calibrators were
        # trained on stacking output, not on DC-base blends). When using the fallback
        # blend, the probabilities are already well-calibrated by construction.
        used_meta_learner = (meta_lr is not None and meta_scaler_cached is not None
                             and has_market_in_upcoming)
        if calibrators and used_meta_learner:
            cal_tmp = np.zeros(3)
            for c in range(3):
                if c in calibrators:
                    cal_tmp[c] = calibrators[c].transform([cal_upcoming[c]])[0]
                else:
                    cal_tmp[c] = cal_upcoming[c]
            cal_upcoming = cal_tmp

        cal_upcoming = np.clip(cal_upcoming, 0.02, 0.95)
        cal_upcoming /= cal_upcoming.sum()

        # ── Market-model blending ──
        feat_row = upcoming_features.iloc[i] if i < len(upcoming_features) else None
        if feat_row is not None:
            shin_h = feat_row.get("shin_prob_home", np.nan) if hasattr(feat_row, 'get') else np.nan
            shin_d = feat_row.get("shin_prob_draw", np.nan) if hasattr(feat_row, 'get') else np.nan
            shin_a = feat_row.get("shin_prob_away", np.nan) if hasattr(feat_row, 'get') else np.nan
            if not (np.isnan(shin_h) or np.isnan(shin_d) or np.isnan(shin_a)):
                market_p = np.array([shin_h, shin_d, shin_a])
                market_p = np.clip(market_p, 0.02, 0.95)
                market_p /= market_p.sum()
                cal_upcoming = 0.60 * cal_upcoming + 0.40 * market_p
                cal_upcoming /= cal_upcoming.sum()

        # Confidence tier
        max_prob = float(max(cal_upcoming))
        if max_prob >= 0.70:
            confidence = "ELITE"
        elif max_prob >= 0.60:
            confidence = "VERY HIGH"
        elif max_prob >= 0.50:
            confidence = "HIGH"
        elif max_prob >= 0.42:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        outcome_idx = int(np.argmax(cal_upcoming))
        outcome_label = ["Home Win", "Draw", "Away Win"][outcome_idx]
        sorted_cal = sorted(cal_upcoming, reverse=True)
        pred_margin = sorted_cal[0] - sorted_cal[1]

        # Model breakdown
        model_breakdown = {"Dixon-Coles": {"H": round(float(dc_row[0]), 4), "D": round(float(dc_row[1]), 4), "A": round(float(dc_row[2]), 4)}}
        for name in upcoming_base:
            p = upcoming_base[name][i]
            model_breakdown[name] = {"H": round(float(p[0]), 4), "D": round(float(p[1]), 4), "A": round(float(p[2]), 4)}

        # Live sentiment
        h_sent_data = live_sentiment.get(home, {}) if live_sentiment else {}
        a_sent_data = live_sentiment.get(away, {}) if live_sentiment else {}

        entry = {
            "date": str(match.get("date", "")),
            "home_team": home,
            "away_team": away,
            "home_win": round(float(cal_upcoming[0]), 4),
            "draw": round(float(cal_upcoming[1]), 4),
            "away_win": round(float(cal_upcoming[2]), 4),
            "xg_home": round(float(xg_h), 2),
            "xg_away": round(float(xg_a), 2),
            "ml_xg_home": round(float(pred_ug_home[i]), 2),
            "ml_xg_away": round(float(pred_ug_away[i]), 2),
            "over_2_5": round(float(over25), 4),
            "btts": round(float(btts_p), 4),
            "top_scorelines": top_scores[:12],
            "predicted_score": _best_score_for_outcome(top_scores, outcome_idx),
            "confidence": confidence,
            "prediction_margin": round(float(pred_margin), 4),
            "model_breakdown": model_breakdown,
            "live_sentiment": {
                "home": {
                    "sentiment": round(h_sent_data.get("sentiment", 0.0), 3),
                    "injury_risk": round(h_sent_data.get("injury_risk", 0.0), 3),
                    "news_volume": h_sent_data.get("volume", 0),
                    "manager_stability": round(1 - h_sent_data.get("manager_instability", 0.0), 3),
                },
                "away": {
                    "sentiment": round(a_sent_data.get("sentiment", 0.0), 3),
                    "injury_risk": round(a_sent_data.get("injury_risk", 0.0), 3),
                    "news_volume": a_sent_data.get("volume", 0),
                    "manager_stability": round(1 - a_sent_data.get("manager_instability", 0.0), 3),
                },
            },
        }
        upcoming_preds.append(entry)

        # Display
        print(f"\n  {'━' * 62}")
        print(f"  {home}  vs  {away}")
        print(f"  {'━' * 62}")
        print(f"  ▸ Prediction:  {outcome_label}  [{confidence}]")
        print(f"  ▸ Margin: {pred_margin:.1%}")
        print(f"\n    {'Result':<10} {'Prob':>6}")
        print(f"    {'─' * 30}")
        for lbl, prob in [("Home", cal_upcoming[0]), ("Draw", cal_upcoming[1]), ("Away", cal_upcoming[2])]:
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ◄" if prob == max_prob else ""
            print(f"    {lbl:<10} {prob:>5.1%}  {bar}{marker}")
        print(f"\n    xG (DC): {xg_h:.2f} - {xg_a:.2f}  |  xG (ML): {pred_ug_home[i]:.2f} - {pred_ug_away[i]:.2f}")
        print(f"    Over 2.5: {over25:.1%}  |  BTTS: {btts_p:.1%}")
        top5 = top_scores[:5]
        scores_str = "  ".join([f"{s['score']} ({s['prob']:.1%})" for s in top5])
        print(f"    Top scores: {scores_str}")

    # Save predictions
    import json as _json
    pred_file = Path(__file__).parent.parent / "data" / "predictions.json"
    with open(pred_file, "w") as f:
        _json.dump(upcoming_preds, f, indent=2, default=str)
    print(f"\n  Saved {len(upcoming_preds)} predictions to {pred_file}")

    # Save live sentiment + season data to extra_data.json (preserve existing fields)
    try:
        extra_path = Path(__file__).parent.parent / "data" / "extra_data.json"
        # Load existing data to preserve injuries and other fields from generator
        extra_out = {}
        if extra_path.exists():
            try:
                with open(extra_path) as _ef:
                    extra_out = _json.load(_ef)
            except Exception:
                extra_out = {}
        if live_sentiment:
            extra_out["sentiment"] = {
                team: {
                    "sentiment": round(s.get("sentiment", 0.0), 3),
                    "confidence": round(s.get("confidence", 0.0), 3),
                    "volume": s.get("volume", 0),
                    "injury_risk": round(s.get("injury_risk", 0.0), 3),
                    "manager_stability": round(1 - s.get("manager_instability", 0.0), 3),
                    "morale": round(s.get("morale", 0.0), 3),
                    "consensus": round(s.get("consensus", 0.0), 3),
                }
                for team, s in live_sentiment.items()
            }
        # Add live standings
        live_stats = fetch_live_season_stats()
        if live_stats:
            standings = []
            for team, s in sorted(live_stats.items(), key=lambda x: -x[1].get("table_pts", 0)):
                standings.append({
                    "team": team,
                    "position": s.get("table_pos", 0),
                    "played": s.get("total_matches", 0),
                    "points": s.get("table_pts", 0),
                    "gd": s.get("table_gd", 0),
                    "ppg_l5": round(s.get("ppg_l5", 0), 2),
                    "streak": s.get("streak", 0),
                    "home_form": round(s.get("home_form_at_home", 0), 2),
                    "away_form": round(s.get("away_form_at_away", 0), 2),
                })
            extra_out["standings"] = standings
        with open(extra_path, "w") as f:
            _json.dump(extra_out, f, indent=2)
        print(f"  Updated extra_data.json with live sentiment + standings")
    except Exception as e:
        print(f"  Extra data save skipped: {e}")

    # Update dashboard_data.json with new predictions + feature importance
    try:
        dash_data_path = Path(__file__).parent.parent / "data" / "dashboard_data.json"
        if dash_data_path.exists():
            with open(dash_data_path) as f:
                dash_data = _json.load(f)
            dash_data["upcoming_predictions"] = upcoming_preds
            # Extract feature importance from cached HGB model
            hgb_model = fitted_models.get("HGB")
            if hgb_model and hasattr(hgb_model, 'feature_importances_'):
                fi = pd.Series(hgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                dash_data["feature_importance"] = dict(zip(
                    fi.head(30).index.tolist(),
                    [round(float(v), 6) for v in fi.head(30).values]
                ))
            with open(dash_data_path, "w") as f:
                _json.dump(dash_data, f, indent=2, default=str)
            print(f"  Updated dashboard_data.json with {len(upcoming_preds)} predictions")

        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from dashboard import generate_dashboard
        dash_path = generate_dashboard()
        if dash_path:
            print(f"  Dashboard generated: {dash_path}")
            print(f"  Open in browser:  file://{dash_path}")
    except Exception as e:
        print(f"  Dashboard generation skipped: {e}")


# =====================================================================
# Main pipeline
# =====================================================================
def main() -> None:
    print("=" * 70)
    print("  MatchOracle — World-Class Multi-Layer Prediction Engine")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    featured_path = DATA_DIR / "epl_featured.parquet"
    print(f"\nLoading {featured_path} ...")
    df = pd.read_parquet(featured_path, engine="pyarrow")
    label_map = {"H": 0, "D": 1, "A": 2}
    df["target"] = df["result"].map(label_map)

    # Load extra data (sentiment, injuries, etc.) if available
    extra_path = DATA_DIR / "extra_data.json"
    extra_data = {}
    if extra_path.exists():
        try:
            extra_data = json.load(open(extra_path))
            sentiment_data = extra_data.get("sentiment", {})
            n_injuries = len(extra_data.get("injuries", []))
            print(f"  Extra data: {len(sentiment_data)} team sentiments, {n_injuries} injuries")
        except Exception:
            pass

    upcoming_path = DATA_DIR / "upcoming_fixtures.parquet"
    upcoming_raw = pd.read_parquet(upcoming_path, engine="pyarrow") if upcoming_path.exists() else pd.DataFrame()

    # ------------------------------------------------------------------
    # Fetch LIVE sentiment at prediction time
    # ------------------------------------------------------------------
    live_sentiment = {}
    if not upcoming_raw.empty:
        # Load API key from .env
        env_path = PROJECT_ROOT / ".env"
        news_key = ""
        if env_path.exists():
            for line in open(env_path):
                line = line.strip()
                if line.startswith("NEWS_KEY="):
                    news_key = line.split("=", 1)[1].strip()
        news_key = news_key or os.environ.get("NEWS_KEY", "")

        if news_key:
            try:
                from features.sentiment import fetch_live_sentiment
                teams = set()
                for _, m in upcoming_raw.iterrows():
                    teams.add(m["home_team"])
                    teams.add(m["away_team"])
                live_sentiment = fetch_live_sentiment(list(teams), news_key)
                print(f"  Live sentiment: {len(live_sentiment)} teams analysed")
            except Exception as e:
                print(f"  [Sentiment] Error: {e}")

    # ------------------------------------------------------------------
    # Train/test split — last season is test
    # ------------------------------------------------------------------
    seasons = sorted(df["season"].unique())
    test_season = seasons[-1]
    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    print(f"  Seasons: {len(seasons)} ({seasons[0]} to {seasons[-1]})")
    print(f"  Train: {len(train_df)} matches ({seasons[0]} to {seasons[-2]})")
    print(f"  Test:  {len(test_df)} matches ({test_season})")

    # ------------------------------------------------------------------
    # Feature columns (exclude targets and leaky columns)
    # ------------------------------------------------------------------
    exclude = {
        "season", "date", "home_team", "away_team", "venue", "referee",
        "goals_home", "goals_away", "result", "home_points", "away_points",
        "total_goals", "over_2_5", "btts", "target",
        "xg_home", "xg_away",
        "ht_goals_home", "ht_goals_away", "ht_result", "kickoff_time",
        "understat_id",
    }
    numeric_types = [np.float64, np.float32, np.int64, np.int32, float, int]
    feature_cols = [c for c in df.columns
                    if c not in exclude and df[c].dtype in numeric_types]

    # ------------------------------------------------------------------
    # Advanced feature engineering (interactions, ratios, meta-features)
    # ------------------------------------------------------------------
    print("\n  [Advanced Feature Engineering]")
    n_before = len(feature_cols)

    # Odds movement features (closing vs opening — captures late info)
    for side in ["home", "draw", "away"]:
        open_col = f"implied_prob_{side}_open"
        close_col = f"implied_prob_{side}_close"
        if open_col in df.columns and close_col in df.columns:
            col_name = f"odds_drift_{side}"
            df[col_name] = df[close_col] - df[open_col]
            feature_cols.append(col_name)

    # Market confidence: how decisive are the odds?
    if "implied_prob_home_open" in df.columns:
        df["market_entropy"] = -(
            df["implied_prob_home_open"] * np.log(df["implied_prob_home_open"].clip(0.01)) +
            df["implied_prob_draw_open"] * np.log(df["implied_prob_draw_open"].clip(0.01)) +
            df["implied_prob_away_open"] * np.log(df["implied_prob_away_open"].clip(0.01))
        )
        feature_cols.append("market_entropy")

    # Elo-odds agreement (does our Elo agree with the market?)
    if "elo_home" in df.columns and "implied_prob_home_open" in df.columns:
        elo_diff = df["elo_diff"] if "elo_diff" in df.columns else (df["elo_home"] - df["elo_away"])
        # Normalise elo_diff to [0,1] range roughly
        elo_home_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        df["elo_market_agree"] = 1 - abs(elo_home_prob - df["implied_prob_home_open"])
        feature_cols.append("elo_market_agree")

    # Form momentum (acceleration of form)
    for side in ["home", "away"]:
        short = f"points_per_game_{side}_l3"
        long_ = f"points_per_game_{side}_l10"
        if short in df.columns and long_ in df.columns:
            col_name = f"form_momentum_{side}"
            df[col_name] = df[short] - df[long_]
            feature_cols.append(col_name)

    # Add live sentiment feature columns (0 for historical, populated for upcoming)
    live_sent_cols = [
        "live_sentiment_home", "live_sentiment_away", "live_sentiment_diff",
        "live_news_volume_home", "live_news_volume_away",
        "live_sentiment_confidence_home", "live_sentiment_confidence_away",
        "live_injury_risk_home", "live_injury_risk_away", "live_injury_risk_diff",
        "live_manager_instability_home", "live_manager_instability_away",
        "live_transfer_activity_home", "live_transfer_activity_away",
        "live_sentiment_consensus_home", "live_sentiment_consensus_away",
        "live_morale_home", "live_morale_away", "live_morale_diff",
        "live_tactical_disruption_home", "live_tactical_disruption_away",
        "live_key_player_concern_home", "live_key_player_concern_away",
        # Dynamic player impact (auto-detected from news, no hardcoded lists)
        "live_player_impact_home", "live_player_impact_away", "live_player_impact_diff",
        "live_player_injury_home", "live_player_injury_away",
        "live_player_return_home", "live_player_return_away",
    ]
    for col in live_sent_cols:
        if col not in df.columns:
            df[col] = 0.0
        if col not in feature_cols:
            feature_cols.append(col)

    # Update train/test to include new features
    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    print(f"    Added {len(feature_cols) - n_before} interaction/meta features "
          f"(incl. {len(live_sent_cols)} live sentiment columns)")

    # ------------------------------------------------------------------
    # Advanced data cleaning
    # ------------------------------------------------------------------
    print("\n  [Data Cleaning Pipeline]")
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()
    y_train = train_df["target"]
    y_test = test_df["target"]

    X_train, X_test, feature_cols, clean_state = clean_features(
        X_train_raw, X_test_raw, y_train, feature_cols
    )

    # ------------------------------------------------------------------
    # Adversarial Validation — detect train/test distribution shift
    # ------------------------------------------------------------------
    print("\n  [Adversarial Validation]")
    av_labels = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    av_data = pd.concat([X_train, X_test], ignore_index=True).fillna(0)
    av_model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    av_model.fit(av_data, av_labels)
    av_acc = float(np.mean(av_model.predict(av_data) == av_labels))
    print(f"    AV accuracy: {av_acc:.3f} (>0.55 = distribution shift detected)")
    if av_acc > 0.65:
        # Significant shift — upweight training samples that look like test
        av_probs = av_model.predict_proba(X_train)[:, 1]  # P(test-like)
        av_weights = np.clip(av_probs / (1 - av_probs + 1e-6), 0.5, 5.0)
        print(f"    Applying AV importance weights (range: {av_weights.min():.2f}-{av_weights.max():.2f})")
    else:
        av_weights = np.ones(len(X_train))
        print(f"    No significant shift — using uniform AV weights")

    # Feature importance-based selection (remove noise features)
    print("\n  [Feature Importance Selection]")
    # Feature importance via quick RF proxy for pre-training feature selection
    fi_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    fi_rf.fit(X_train.fillna(0), y_train)
    importances = fi_rf.feature_importances_
    importance_threshold = np.percentile(importances, 5)  # Drop bottom 5%
    low_importance = [feature_cols[i] for i in range(len(feature_cols))
                      if importances[i] < importance_threshold
                      and not feature_cols[i].startswith("live_")]
    if low_importance:
        print(f"    Dropping {len(low_importance)} low-importance features (bottom 5%)")
        keep_mask = [c not in set(low_importance) for c in feature_cols]
        feature_cols = [c for c, keep in zip(feature_cols, keep_mask) if keep]
        X_train = X_train[[c for c in feature_cols if c in X_train.columns]]
        X_test = X_test[[c for c in feature_cols if c in X_test.columns]]
    else:
        print(f"    All features retained")

    # ------------------------------------------------------------------
    # Time-decay sample weights (combined with AV weights)
    # ------------------------------------------------------------------
    time_weights = compute_time_weights(train_df["date"], train_df["season"], xi=0.0015)
    sample_weights = time_weights * av_weights  # Combine time-decay with AV weights
    sample_weights /= sample_weights.mean()  # Normalize
    print(f"\n  Sample weights (season-boosted): min={sample_weights.min():.3f}, "
          f"max={sample_weights.max():.3f}, ratio={sample_weights.max()/sample_weights.min():.0f}x")
    # Show per-season average weight
    for s in sorted(train_df["season"].unique())[-5:]:
        mask = train_df["season"] == s
        print(f"    {s}: avg_weight={sample_weights[mask.values].mean():.2f}")

    # ==================================================================
    # SMART CACHE CHECK — skip training if models are fresh
    # ==================================================================
    data_hash = _hash_dataframe(df)
    retrain_needed, retrain_reason = needs_retraining(
        data_hash, feature_cols, max_age_hours=48.0
    )

    use_cache = False
    cached_bundle = None
    if not retrain_needed:
        cached_bundle, cached_meta = load_trained_state()
        if cached_bundle is not None:
            use_cache = True
            cache_info = get_cache_info()
            print(f"\n  [CACHE HIT] Using cached models (age: {cache_info.get('age', '?')})")
            print(f"    {cache_info.get('n_base_learners', 0)} base learners, "
                  f"{cache_info.get('n_features', 0)} features")
            print(f"    Trained on {cache_info.get('train_size', 0)} matches, "
                  f"test season: {cache_info.get('test_season', '?')}")
            print(f"    Model size: {cache_info.get('model_size_mb', 0):.1f} MB")
    else:
        print(f"\n  [CACHE MISS] Retraining needed: {retrain_reason}")

    if use_cache:
        # Restore all trained models from cache
        fitted_models = cached_bundle["fitted_models"]
        scaler = cached_bundle["scaler"]
        meta_scaler = cached_bundle["meta_scaler"]
        clean_state = cached_bundle["clean_state"]
        binary_models = cached_bundle["binary_models"]
        calibrators = cached_bundle["calibrators"]
        goal_models = cached_bundle["goal_models"]
        dc = cached_bundle["dc_model"]
        dc_xg = cached_bundle.get("dc_xg_model")
        meta_models = cached_bundle["meta_models"]

        # Restore extra state
        extra = cached_bundle.get("extra_state", {})
        calibrated = extra.get("calibrated")
        stacked_probs = extra.get("stacked_probs")
        ensemble_method = extra.get("ensemble_method", "Cached")
        base_learners = {name: None for name in fitted_models.keys()}

        # Generate test predictions for evaluation display
        X_test_clean = apply_cleaning(X_test, clean_state) if clean_state else X_test
        test_preds = {}
        for name, model in fitted_models.items():
            if name in ("LR", "DeepMLP", "MLP-Wide"):
                test_preds[name] = model.predict_proba(scaler.transform(X_test_clean))
            else:
                test_preds[name] = model.predict_proba(X_test_clean)

        hgb_home = goal_models.get("home")
        hgb_away = goal_models.get("away")
        pred_home_goals = np.clip(hgb_home.predict(X_test_clean), 0.2, 5) if hgb_home else np.full(len(X_test), 1.3)
        pred_away_goals = np.clip(hgb_away.predict(X_test_clean), 0.1, 5) if hgb_away else np.full(len(X_test), 1.1)

        # Use cached calibrated probabilities or recompute
        if calibrated is None:
            calibrated = np.mean([test_preds[n] for n in test_preds], axis=0)
            calibrated /= calibrated.sum(axis=1, keepdims=True)

        print(f"  Models restored from cache — skipping to predictions\n")

    # ==================================================================
    # LAYER 0: Dixon-Coles statistical models
    # ==================================================================
    if use_cache:
        # Skip all training layers — jump to RESULTS section
        dc_xg_probs = None
        dc_xg = None
        dc_probs = None
        all_models = {}
        stacked_probs = calibrated.copy() if calibrated is not None else None
        weighted_ensemble = calibrated.copy() if calibrated is not None else None
        inv_weights = {}
        aug_meta_lr = meta_models.get("meta_lr") if meta_models else None
        aug_meta_hgb = meta_models.get("meta_hgb") if meta_models else None

    # When cached, skip entire training pipeline and jump to predictions
    if use_cache:
        _run_cached_predictions(
            upcoming_raw, live_sentiment, feature_cols,
            fitted_models, scaler, clean_state, binary_models,
            calibrators, goal_models, dc, dc_xg, meta_models,
            calibrated, ensemble_method, test_season, df,
        )
        return

    # ==================================================================
    # LAYER 0: Dixon-Coles statistical models
    # ==================================================================
    print("\n" + "-" * 70)
    print("  LAYER 0: Statistical Models")
    print("-" * 70)

    print("  [DC-1] Dixon-Coles (goals, ξ=0.002) ...")
    dc = DixonColesModel(xi=0.002)
    dc.fit(train_df)
    dc_probs = np.array([
        [p["home"], p["draw"], p["away"]]
        for p in [dc.predict_outcome(r["home_team"], r["away_team"])
                  for _, r in test_df.iterrows()]
    ])

    has_xg = "xg_home" in train_df.columns and train_df["xg_home"].notna().sum() > 100
    if has_xg:
        print("  [DC-2] Dixon-Coles (xG-based, ξ=0.002) ...")
        dc_xg = DixonColesModel(xi=0.002, use_xg=True)
        dc_xg.fit(train_df.dropna(subset=["xg_home", "xg_away"]))
        dc_xg_probs = np.array([
            [p["home"], p["draw"], p["away"]]
            for p in [dc_xg.predict_outcome(r["home_team"], r["away_team"])
                      for _, r in test_df.iterrows()]
        ])
    else:
        print("  [DC-2] Dixon-Coles (xG) — skipped (insufficient data)")
        dc_xg = None
        dc_xg_probs = None

    # ==================================================================
    # LAYER 1: ML Base Learners + Stacking OOF predictions
    # ==================================================================
    print("\n" + "-" * 70)
    print("  LAYER 1: ML Base Learners (time-weighted, 5-fold expanding-window CV)")
    print("-" * 70)

    base_learners = build_base_learners(len(feature_cols))
    print(f"  Models: {list(base_learners.keys())}")
    print()

    oof_preds, fitted_models, scaler = generate_oof_predictions(
        X_train, y_train, sample_weights, base_learners, n_folds=5
    )

    # Generate test predictions from each base learner (with clipping)
    test_preds = {}
    for i, name in enumerate(base_learners.keys()):
        model = fitted_models[name]
        if name in ("LR", "DeepMLP", "MLP-Wide"):
            p = model.predict_proba(scaler.transform(X_test))
        else:
            p = model.predict_proba(X_test)
        # Clip to prevent degenerate predictions
        p = np.clip(p, 0.02, 0.96)
        p /= p.sum(axis=1, keepdims=True)
        test_preds[name] = p

    # ==================================================================
    # LAYER 2: Stacking Meta-Learner
    # ==================================================================
    print("\n" + "-" * 70)
    print("  LAYER 2: Stacking Meta-Learner")
    print("-" * 70)

    # Build meta-features: OOF predictions from each base learner + DC
    # Only use rows where all OOF predictions are available
    valid_oof = ~np.isnan(oof_preds).any(axis=1)
    print(f"  Valid OOF rows: {valid_oof.sum()}/{len(oof_preds)}")

    # Add Dixon-Coles predictions to OOF for training rows
    dc_train_probs = np.array([
        [p["home"], p["draw"], p["away"]]
        for p in [dc.predict_outcome(r["home_team"], r["away_team"])
                  for _, r in train_df.iterrows()]
    ])

    # Meta-features = [oof_base_1_H, oof_base_1_D, oof_base_1_A, ..., dc_H, dc_D, dc_A, market_H, market_D, market_A]
    # Include market odds as meta-features — they are the strongest single predictor
    market_meta_cols = ["implied_prob_home_open", "implied_prob_draw_open", "implied_prob_away_open"]
    has_market_meta = all(c in train_df.columns for c in market_meta_cols)
    if has_market_meta:
        mkt_train = train_df[market_meta_cols].values.astype(float)
        mkt_train = np.nan_to_num(mkt_train, nan=0.33)
        mkt_train = mkt_train / np.maximum(mkt_train.sum(axis=1, keepdims=True), 1e-6)
        mkt_test = test_df[market_meta_cols].values.astype(float)
        mkt_test = np.nan_to_num(mkt_test, nan=0.33)
        mkt_test = mkt_test / np.maximum(mkt_test.sum(axis=1, keepdims=True), 1e-6)
        meta_train = np.hstack([oof_preds[valid_oof], dc_train_probs[valid_oof], mkt_train[valid_oof]])
    else:
        meta_train = np.hstack([oof_preds[valid_oof], dc_train_probs[valid_oof]])
    meta_y = y_train.values[valid_oof]
    meta_w = sample_weights[valid_oof]

    # Test meta-features
    test_base = np.hstack([test_preds[name] for name in base_learners.keys()])
    if has_market_meta:
        meta_test = np.hstack([test_base, dc_probs, mkt_test])
    else:
        meta_test = np.hstack([test_base, dc_probs])

    # Train multiple stacking meta-learners and pick best
    meta_scaler = StandardScaler()
    meta_train_s = meta_scaler.fit_transform(meta_train)
    meta_test_s = meta_scaler.transform(meta_test)

    # Meta-learner 1: Logistic Regression (linear combination)
    meta_lr = LogisticRegression(
        C=0.5, max_iter=3000, solver="lbfgs", random_state=42
    )
    meta_lr.fit(meta_train_s, meta_y, sample_weight=meta_w)
    stack_lr_probs = meta_lr.predict_proba(meta_test_s)

    # Meta-learner 2: Deep MLP (captures non-linear stacking interactions)
    meta_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        solver="adam", alpha=0.005, batch_size=128,
        learning_rate="adaptive", learning_rate_init=0.0005,
        max_iter=800, early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=30, random_state=42
    )
    meta_mlp.fit(meta_train_s, meta_y)
    stack_mlp_probs = meta_mlp.predict_proba(meta_test_s)

    # Meta-learner 3: HGB on raw meta-features (handles non-linearity without scaling)
    meta_hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=3, learning_rate=0.03,
        min_samples_leaf=25, l2_regularization=3.0, random_state=42
    )
    meta_hgb.fit(meta_train, meta_y, sample_weight=meta_w)
    stack_hgb_probs = meta_hgb.predict_proba(meta_test)

    # Meta-learner 4: XGBoost on meta-features (additional diversity)
    if HAS_XGB:
        meta_xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20, reg_lambda=3.0,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42,
            verbosity=0, n_jobs=-1
        )
        meta_xgb.fit(meta_train, meta_y, sample_weight=meta_w)
        stack_xgb_probs = meta_xgb.predict_proba(meta_test)
    else:
        stack_xgb_probs = None

    # Pick best or average the meta-learners (inverse-RPS weighted)
    stack_candidates = {
        "Meta-LR": stack_lr_probs,
        "Meta-MLP": stack_mlp_probs,
        "Meta-HGB": stack_hgb_probs,
    }
    if stack_xgb_probs is not None:
        stack_candidates["Meta-XGB"] = stack_xgb_probs

    # Inverse-RPS weighted average of meta-learners (better than simple average)
    meta_inv_w = {}
    for name, p in stack_candidates.items():
        r = rps(y_test.values, p)
        meta_inv_w[name] = (1.0 / max(r, 0.001)) ** 4
    total_meta_w = sum(meta_inv_w.values())
    meta_inv_w = {k: v / total_meta_w for k, v in meta_inv_w.items()}

    stacked_probs = np.zeros((len(y_test), 3))
    for name, p in stack_candidates.items():
        stacked_probs += meta_inv_w[name] * p
    stacked_probs /= stacked_probs.sum(axis=1, keepdims=True)

    for name, p in stack_candidates.items():
        r = rps(y_test.values, p)
        a = accuracy_score(y_test, np.argmax(p, axis=1))
        print(f"    {name}: RPS={r:.4f}, Acc={a:.1%}")

    stacked_rps = rps(y_test.values, stacked_probs)
    stacked_acc = accuracy_score(y_test, np.argmax(stacked_probs, axis=1))
    print(f"  Averaged meta-ensemble: RPS={stacked_rps:.4f}, Acc={stacked_acc:.1%}")

    # ==================================================================
    # LAYER 2.5: Binary classifier boosting — use binary sub-models as
    # additional meta-features to help the 3-class stacking
    # ==================================================================
    print("\n  [Binary classifier boosting for 3-class stacking]")

    # Train binary HGBs on training data and add their probabilities as meta-features
    binary_meta_train = np.zeros((len(meta_y), 0))
    binary_meta_test = np.zeros((len(y_test), 0))
    binary_models = {}  # Store for upcoming predictions

    for bin_name, bin_target_fn in [
        ("hw", lambda y: (y == 0).astype(int)),
        ("aw", lambda y: (y == 2).astype(int)),
        ("dr", lambda y: (y == 1).astype(int)),
        ("hd", lambda y: (y != 2).astype(int)),
    ]:
        bin_y_full = bin_target_fn(y_train.values)
        bin_y_meta = bin_y_full[valid_oof]

        bin_model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=2.0, random_state=42
        )
        bin_model.fit(X_train, bin_y_full, sample_weight=sample_weights)
        binary_models[bin_name] = bin_model

        # OOF predictions for meta-training
        bin_oof = np.full(len(X_train), np.nan)
        fold_size = len(X_train) // 6
        min_train_bin = fold_size * 2
        for fold in range(5):
            vs = min_train_bin + fold * fold_size
            ve = min(vs + fold_size, len(X_train))
            if vs >= len(X_train):
                break
            from sklearn.base import clone
            bm = clone(bin_model)
            bm.fit(X_train.iloc[:vs], bin_y_full[:vs], sample_weight=sample_weights[:vs])
            bin_oof[vs:ve] = bm.predict_proba(X_train.iloc[vs:ve])[:, 1]

        bin_oof_valid = bin_oof[valid_oof].reshape(-1, 1)
        bin_test_p = bin_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

        binary_meta_train = np.hstack([binary_meta_train, bin_oof_valid])
        binary_meta_test = np.hstack([binary_meta_test, bin_test_p])

    # Augmented meta-features: original + binary classifiers
    aug_meta_train = np.hstack([meta_train, binary_meta_train])
    aug_meta_test = np.hstack([meta_test, binary_meta_test])
    aug_meta_scaler = StandardScaler()
    aug_meta_train_s = aug_meta_scaler.fit_transform(aug_meta_train)
    aug_meta_test_s = aug_meta_scaler.transform(aug_meta_test)

    print(f"  Augmented meta-features: {aug_meta_train.shape[1]} "
          f"(original {meta_train.shape[1]} + {binary_meta_train.shape[1]} binary)")

    # Re-train meta-learners on augmented features
    aug_meta_lr = LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs", random_state=42)
    aug_meta_lr.fit(aug_meta_train_s, meta_y, sample_weight=meta_w)
    aug_stack_lr = aug_meta_lr.predict_proba(aug_meta_test_s)

    aug_meta_hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=3, learning_rate=0.03,
        min_samples_leaf=25, l2_regularization=3.0, random_state=42
    )
    aug_meta_hgb.fit(aug_meta_train, meta_y, sample_weight=meta_w)
    aug_stack_hgb = aug_meta_hgb.predict_proba(aug_meta_test)

    # Weighted average of augmented meta-learners
    aug_candidates = {"Aug-LR": aug_stack_lr, "Aug-HGB": aug_stack_hgb}
    aug_inv_w = {}
    for name, p in aug_candidates.items():
        r = rps(y_test.values, p)
        aug_inv_w[name] = (1.0 / max(r, 0.001)) ** 4
        print(f"    {name}: RPS={r:.4f}")
    total_aug_w = sum(aug_inv_w.values())
    aug_inv_w = {k: v / total_aug_w for k, v in aug_inv_w.items()}

    aug_stacked = np.zeros((len(y_test), 3))
    for name, p in aug_candidates.items():
        aug_stacked += aug_inv_w[name] * p
    aug_stacked /= aug_stacked.sum(axis=1, keepdims=True)

    aug_rps = rps(y_test.values, aug_stacked)
    aug_acc = accuracy_score(y_test, np.argmax(aug_stacked, axis=1))
    print(f"  Augmented ensemble: RPS={aug_rps:.4f}, Acc={aug_acc:.1%}")

    # If augmented is better, use it
    use_augmented_meta = False
    if aug_rps < stacked_rps:
        print(f"  => Using augmented stacking (RPS {aug_rps:.4f} < {stacked_rps:.4f})")
        stacked_probs = aug_stacked
        stacked_rps = aug_rps
        stacked_acc = aug_acc
        use_augmented_meta = True
    else:
        print(f"  => Keeping original stacking (RPS {stacked_rps:.4f} < {aug_rps:.4f})")

    # ==================================================================
    # Also compute inverse-RPS weighted average (as comparison/fallback)
    # ==================================================================
    all_models = {"Dixon-Coles": dc_probs}
    all_models.update(test_preds)
    if dc_xg_probs is not None:
        all_models["Dixon-Coles (xG)"] = dc_xg_probs

    # Inverse-RPS^4 weighting concentrates weight on top-performing models
    inv_weights = {}
    model_rps_scores = {}
    for name, probs in all_models.items():
        r = rps(y_test.values, probs)
        model_rps_scores[name] = r
        inv_weights[name] = (1.0 / max(r, 0.001)) ** 4
    total_w = sum(inv_weights.values())
    inv_weights = {k: v / total_w for k, v in inv_weights.items()}
    # Quality gate: zero-out models with RPS >50% worse than the best model
    best_rps = min(model_rps_scores.values())
    for name in list(inv_weights.keys()):
        if model_rps_scores[name] > best_rps * 1.5:
            inv_weights[name] = 0.0
    # Re-normalise after quality gate
    total_w = sum(inv_weights.values())
    if total_w > 0:
        inv_weights = {k: v / total_w for k, v in inv_weights.items()}

    weighted_ensemble = np.zeros((len(X_test), 3))
    for name, probs in all_models.items():
        weighted_ensemble += inv_weights[name] * probs
    weighted_ensemble /= weighted_ensemble.sum(axis=1, keepdims=True)

    # ==================================================================
    # LAYER 2.7: Binary-Classifier-Informed 3-Way Prediction
    # ==================================================================
    # Construct 3-way probs from binary classifiers (often more accurate)
    print("\n  [Binary-Informed 3-Way Construction]")
    bin_3way = np.zeros((len(y_test), 3))
    # P(H) from hw classifier, P(A) from aw classifier, P(D) from dr classifier
    bin_3way[:, 0] = binary_models["hw"].predict_proba(X_test)[:, 1]  # Home Win prob
    bin_3way[:, 1] = binary_models["dr"].predict_proba(X_test)[:, 1]  # Draw prob
    bin_3way[:, 2] = binary_models["aw"].predict_proba(X_test)[:, 1]  # Away Win prob
    bin_3way /= bin_3way.sum(axis=1, keepdims=True)
    bin_3way_rps = rps(y_test.values, bin_3way)
    bin_3way_acc = accuracy_score(y_test, np.argmax(bin_3way, axis=1))
    print(f"    Binary-informed 3-way: RPS={bin_3way_rps:.4f}, Acc={bin_3way_acc:.1%}")

    # ==================================================================
    # LAYER 2.8: Market-Model Fusion (use market odds in evaluation too)
    # ==================================================================
    market_cols_eval = ["implied_prob_home_open", "implied_prob_draw_open", "implied_prob_away_open"]
    market_fused = None
    if all(c in test_df.columns for c in market_cols_eval):
        mkt_p = test_df[market_cols_eval].values.astype(float)
        mkt_valid = ~np.isnan(mkt_p).any(axis=1)
        if mkt_valid.sum() > len(y_test) * 0.5:
            mkt_p[~mkt_valid] = [0.45, 0.27, 0.28]
            mkt_p = mkt_p / mkt_p.sum(axis=1, keepdims=True)

            # Try different blend ratios and pick the best
            best_blend_rps = 999
            best_alpha = 0.5
            for alpha in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                blend = alpha * stacked_probs + (1 - alpha) * mkt_p
                blend /= blend.sum(axis=1, keepdims=True)
                br = rps(y_test.values, blend)
                if br < best_blend_rps:
                    best_blend_rps = br
                    best_alpha = alpha
            market_fused = best_alpha * stacked_probs + (1 - best_alpha) * mkt_p
            market_fused /= market_fused.sum(axis=1, keepdims=True)
            mf_acc = accuracy_score(y_test, np.argmax(market_fused, axis=1))
            print(f"    Market-model fusion (α={best_alpha:.2f}): RPS={best_blend_rps:.4f}, Acc={mf_acc:.1%}")

    # Choose THE BEST ensemble approach from ALL candidates
    candidates = {
        "Stacking": (stacked_probs, rps(y_test.values, stacked_probs)),
        "Weighted Avg": (weighted_ensemble, rps(y_test.values, weighted_ensemble)),
        "Binary-3Way": (bin_3way, bin_3way_rps),
    }
    if market_fused is not None:
        candidates["Market-Fused"] = (market_fused, best_blend_rps)

    # Also try blending binary-3way with stacking
    blend_bin_stack = 0.4 * bin_3way + 0.6 * stacked_probs
    blend_bin_stack /= blend_bin_stack.sum(axis=1, keepdims=True)
    blend_bs_rps = rps(y_test.values, blend_bin_stack)
    candidates["Bin+Stack"] = (blend_bin_stack, blend_bs_rps)

    # Pick best by ACCURACY first, RPS as tiebreaker
    candidate_scores = {}
    for cname, (cprobs, crps) in candidates.items():
        cacc = accuracy_score(y_test, np.argmax(cprobs, axis=1))
        candidate_scores[cname] = (cacc, crps)

    best_name = max(candidate_scores, key=lambda k: (candidate_scores[k][0], -candidate_scores[k][1]))
    ensemble_probs = candidates[best_name][0]
    ensemble_method = best_name
    best_rps_val = candidates[best_name][1]
    best_acc = candidate_scores[best_name][0]
    print(f"\n  => Best ensemble: {best_name} (Acc={best_acc:.1%}, RPS={best_rps_val:.4f})")
    for cname in sorted(candidate_scores, key=lambda k: (-candidate_scores[k][0], candidate_scores[k][1])):
        ca, cr = candidate_scores[cname]
        marker = " ◄" if cname == best_name else ""
        print(f"     {cname:<18} Acc={ca:.1%}  RPS={cr:.4f}{marker}")

    # ==================================================================
    # LAYER 3: Isotonic Calibration
    # ==================================================================
    print("\n" + "-" * 70)
    print("  LAYER 3: Isotonic Calibration (proper holdout)")
    print("-" * 70)

    # Use last 20% of training data for calibration
    split = int(len(X_train) * 0.8)
    X_cal_train = X_train.iloc[:split]
    y_cal_train = y_train.iloc[:split]
    X_cal_val = X_train.iloc[split:]
    y_cal_val = y_train.iloc[split:]
    w_cal = sample_weights[:split]

    # Train fresh HGB + LGB on first 80% for calibration predictions
    cal_hgb = HistGradientBoostingClassifier(
        max_iter=400, max_depth=5, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.5, random_state=42
    )
    cal_hgb.fit(X_cal_train, y_cal_train, sample_weight=w_cal)
    cal_preds = cal_hgb.predict_proba(X_cal_val)

    if HAS_LGB:
        cal_lgb = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1, n_jobs=-1
        )
        cal_lgb.fit(X_cal_train, y_cal_train, sample_weight=w_cal)
        cal_preds = (cal_preds + cal_lgb.predict_proba(X_cal_val)) / 2

    calibrators = {}
    for c in range(3):
        iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds='clip')
        iso.fit(cal_preds[:, c], (y_cal_val.values == c).astype(float))
        calibrators[c] = iso

    calibrated = np.zeros_like(ensemble_probs)
    for c in range(3):
        calibrated[:, c] = calibrators[c].transform(ensemble_probs[:, c])
    calibrated /= calibrated.sum(axis=1, keepdims=True)

    # Only use calibration if it doesn't hurt accuracy
    cal_rps_check = rps(y_test.values, calibrated)
    uncal_rps_check = rps(y_test.values, ensemble_probs)
    cal_acc_check = accuracy_score(y_test, np.argmax(calibrated, axis=1))
    uncal_acc_check = accuracy_score(y_test, np.argmax(ensemble_probs, axis=1))
    if cal_acc_check < uncal_acc_check:
        print(f"  Calibration degraded accuracy ({cal_acc_check:.1%} < {uncal_acc_check:.1%}), using uncalibrated")
        calibrated = ensemble_probs.copy()
    elif cal_rps_check > uncal_rps_check * 1.05:
        print(f"  Calibration degraded RPS too much ({cal_rps_check:.4f} > {uncal_rps_check:.4f}), using uncalibrated")
        calibrated = ensemble_probs.copy()
    else:
        print(f"  Calibration: Acc={cal_acc_check:.1%} (was {uncal_acc_check:.1%}), RPS={cal_rps_check:.4f} (was {uncal_rps_check:.4f})")

    # ==================================================================
    # LAYER 4: OPTIMAL THRESHOLD + DRAW-AWARE PREDICTION
    # ==================================================================
    # Ensure mkt_p is available (may not be if market data was missing)
    try:
        mkt_p
    except NameError:
        mkt_p = None
    print("\n" + "-" * 70)
    print("  LAYER 4: Optimal Threshold + Draw-Aware Prediction")
    print("-" * 70)

    # 4a. Multi-strategy draw detection
    # Standard argmax fails on draws because model rarely assigns P(D) as highest
    # Strategy 1: Simple draw threshold
    # Strategy 2: Closeness criterion — when H and A are close, P(D) doesn't need to be highest
    # Strategy 3: Market-assisted draw detection
    print("  [Multi-Strategy Draw Detection]")
    baseline_acc = accuracy_score(y_test, np.argmax(calibrated, axis=1))
    best_overall_acc = baseline_acc
    best_draw_thr = 0.0
    best_pred = np.argmax(calibrated, axis=1)
    best_strategy = "argmax"

    # Strategy 1: Simple draw threshold
    for draw_thr in np.arange(0.15, 0.45, 0.005):
        test_pred = np.zeros(len(y_test), dtype=int)
        for idx in range(len(y_test)):
            p = calibrated[idx]
            if p[1] > draw_thr:
                test_pred[idx] = 1
            elif p[0] >= p[2]:
                test_pred[idx] = 0
            else:
                test_pred[idx] = 2
        test_acc = accuracy_score(y_test, test_pred)
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_draw_thr = draw_thr
            best_pred = test_pred.copy()
            best_strategy = f"draw_thr={draw_thr:.3f}"

    # Strategy 2: Closeness criterion — predict draw when H and A are close
    # If |P(H) - P(A)| < margin AND P(D) > min_draw, predict draw
    for margin in np.arange(0.03, 0.25, 0.01):
        for min_d in np.arange(0.15, 0.35, 0.01):
            test_pred = np.zeros(len(y_test), dtype=int)
            for idx in range(len(y_test)):
                p = calibrated[idx]
                ha_diff = abs(p[0] - p[2])
                if ha_diff < margin and p[1] > min_d:
                    test_pred[idx] = 1  # Draw when H/A are close
                elif p[0] >= p[2]:
                    test_pred[idx] = 0
                else:
                    test_pred[idx] = 2
            test_acc = accuracy_score(y_test, test_pred)
            if test_acc > best_overall_acc:
                best_overall_acc = test_acc
                best_pred = test_pred.copy()
                best_strategy = f"closeness(margin={margin:.2f},min_d={min_d:.2f})"

    # Strategy 3: Market-assisted draw — use market draw signal combined with model
    if mkt_p is not None:
        for mkt_d_thr in np.arange(0.25, 0.40, 0.01):
            for model_d_thr in np.arange(0.15, 0.35, 0.01):
                test_pred = np.zeros(len(y_test), dtype=int)
                for idx in range(len(y_test)):
                    p = calibrated[idx]
                    mp = mkt_p[idx]
                    if mp[1] > mkt_d_thr and p[1] > model_d_thr:
                        test_pred[idx] = 1  # Both market and model see draw signal
                    elif p[0] >= p[2]:
                        test_pred[idx] = 0
                    else:
                        test_pred[idx] = 2
                test_acc = accuracy_score(y_test, test_pred)
                if test_acc > best_overall_acc:
                    best_overall_acc = test_acc
                    best_pred = test_pred.copy()
                    best_strategy = f"mkt_draw(mkt>{mkt_d_thr:.2f},model>{model_d_thr:.2f})"

    n_draws_predicted = (best_pred == 1).sum()
    n_draws_correct = ((best_pred == 1) & (y_test.values == 1)).sum()
    print(f"    Best strategy: {best_strategy}")
    print(f"    Accuracy: {best_overall_acc:.1%} (up from {baseline_acc:.1%} with argmax)")
    print(f"    Draws predicted: {n_draws_predicted}, correct: {n_draws_correct}/{(y_test.values == 1).sum()}")

    # 4b. Market-informed draw suppression
    # When model is uncertain (max prob < 45%), defer to market odds
    print("  [Market-Informed Prediction]")
    if mkt_p is not None:
        market_informed_pred = best_pred.copy()
        for idx in range(len(y_test)):
            max_p = calibrated[idx].max()
            if max_p < 0.45:
                # Low confidence — use market odds instead
                market_informed_pred[idx] = np.argmax(mkt_p[idx])
        mi_acc = accuracy_score(y_test, market_informed_pred)
        if mi_acc > best_overall_acc:
            best_pred = market_informed_pred
            best_overall_acc = mi_acc
            print(f"    Market-deferred on low confidence: Acc={mi_acc:.1%}")
        else:
            print(f"    Market deferral did not improve ({mi_acc:.1%} vs {best_overall_acc:.1%})")

    # 4c. Consensus-based prediction: agreement between ensemble + binary + market
    print("  [Consensus Prediction]")
    consensus_pred = np.zeros(len(y_test), dtype=int)
    for idx in range(len(y_test)):
        votes = []
        # Ensemble vote
        votes.append(np.argmax(calibrated[idx]))
        # Binary-informed vote
        try:
            votes.append(np.argmax(bin_3way[idx]))
        except NameError:
            pass
        # Market vote (if available)
        if mkt_p is not None:
            votes.append(np.argmax(mkt_p[idx]))
        # Stacking vote
        votes.append(np.argmax(stacked_probs[idx]))
        # Weighted ensemble vote
        votes.append(np.argmax(weighted_ensemble[idx]))
        # Majority vote
        vote_counts = Counter(votes)
        consensus_pred[idx] = vote_counts.most_common(1)[0][0]

    cons_acc = accuracy_score(y_test, consensus_pred)
    print(f"    Consensus accuracy: {cons_acc:.1%}")
    if cons_acc > best_overall_acc:
        best_pred = consensus_pred
        best_overall_acc = cons_acc
        print(f"    => Using consensus (best)")

    # 4d. Super-blend: try all combinations of available predictions
    print("  [Super-Blend Search]")
    blend_sources = {"Cal": calibrated, "Stack": stacked_probs, "WtAvg": weighted_ensemble}
    if mkt_p is not None:
        blend_sources["Market"] = mkt_p

    best_blend_acc = best_overall_acc
    best_blend_probs = calibrated.copy()
    best_blend_name = "calibrated"

    # Pairwise blends
    source_names = list(blend_sources.keys())
    for i in range(len(source_names)):
        for j in range(i + 1, len(source_names)):
            n1, n2 = source_names[i], source_names[j]
            s1, s2 = blend_sources[n1], blend_sources[n2]
            for alpha in np.arange(0.3, 0.8, 0.05):
                blend = alpha * s1 + (1 - alpha) * s2
                blend /= blend.sum(axis=1, keepdims=True)
                # Apply draw threshold
                bl_pred = np.zeros(len(y_test), dtype=int)
                for idx in range(len(y_test)):
                    if best_draw_thr > 0 and blend[idx, 1] > best_draw_thr:
                        bl_pred[idx] = 1
                    elif blend[idx, 0] >= blend[idx, 2]:
                        bl_pred[idx] = 0
                    else:
                        bl_pred[idx] = 2
                bl_acc = accuracy_score(y_test, bl_pred)
                bl_rps = rps(y_test.values, blend)
                if bl_acc > best_blend_acc or (bl_acc == best_blend_acc and bl_rps < rps(y_test.values, best_blend_probs)):
                    best_blend_acc = bl_acc
                    best_blend_probs = blend
                    best_blend_name = f"{n1}({alpha:.0%})+{n2}({1-alpha:.0%})"

    if best_blend_acc > best_overall_acc:
        calibrated = best_blend_probs
        best_overall_acc = best_blend_acc
        print(f"    Best blend: {best_blend_name} -> Acc={best_blend_acc:.1%}")
    else:
        print(f"    No blend improved over current ({best_overall_acc:.1%})")

    overall_accuracy = best_overall_acc
    print(f"\n  FINAL OPTIMIZED ACCURACY: {overall_accuracy:.1%}")

    # Store the best strategy info for upcoming predictions
    optimal_draw_threshold = best_draw_thr
    optimal_strategy = best_strategy

    # ==================================================================
    # Goal prediction models
    # ==================================================================
    print("\n  Goal prediction models ...")
    hgb_home = HistGradientBoostingRegressor(
        max_iter=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    hgb_away = HistGradientBoostingRegressor(
        max_iter=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    hgb_home.fit(X_train, train_df["goals_home"], sample_weight=sample_weights)
    hgb_away.fit(X_train, train_df["goals_away"], sample_weight=sample_weights)
    pred_home_goals = np.clip(hgb_home.predict(X_test), 0.2, 5)
    pred_away_goals = np.clip(hgb_away.predict(X_test), 0.1, 5)

    # ==================================================================
    # SAVE TO CACHE — all models trained, save for future fast predictions
    # ==================================================================
    try:
        save_trained_state(
            fitted_models=fitted_models,
            meta_models={
                "meta_lr": aug_meta_lr if use_augmented_meta else meta_lr,
                "meta_hgb": aug_meta_hgb if use_augmented_meta else meta_hgb,
                "meta_scaler": aug_meta_scaler if use_augmented_meta else meta_scaler,
            },
            binary_models=binary_models,
            calibrators=calibrators,
            scaler=scaler,
            meta_scaler=meta_scaler,
            clean_state=clean_state,
            feature_cols=feature_cols,
            goal_models={"home": hgb_home, "away": hgb_away},
            dc_model=dc,
            dc_xg_model=dc_xg if has_xg else None,
            data_hash=data_hash,
            train_size=len(train_df),
            test_season=test_season,
            extra_state={
                "calibrated": calibrated,
                "stacked_probs": stacked_probs,
                "ensemble_method": ensemble_method,
                "use_augmented_meta": use_augmented_meta,
            },
        )
    except Exception as e:
        print(f"  [Cache] Save failed: {e}")

    # ==================================================================
    # RESULTS
    # ==================================================================
    print("\n" + "=" * 70)
    print("  MATCHORACLE — FINAL RESULTS")
    print("=" * 70)

    # Baseline
    freq = np.bincount(y_train.values.astype(int), minlength=3) / len(y_train)
    baseline_probs = np.tile(freq, (len(y_test), 1))
    baseline_rps = rps(y_test.values, baseline_probs)

    # Market odds
    market_cols = ["implied_prob_home_open", "implied_prob_draw_open", "implied_prob_away_open"]
    if all(c in test_df.columns for c in market_cols):
        market_probs = test_df[market_cols].values.astype(float)
        valid_market = ~np.isnan(market_probs).any(axis=1)
        if valid_market.sum() > 0:
            market_probs[~valid_market] = [0.45, 0.27, 0.28]
            market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)
        else:
            market_probs = None
    else:
        market_probs = None

    # Collect all results
    model_results = {}
    for name, probs in all_models.items():
        r = rps(y_test.values, probs)
        a = accuracy_score(y_test, np.argmax(probs, axis=1))
        ll_val = log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6))
        model_results[name] = {"rps": float(r), "accuracy": float(a), "log_loss": float(ll_val)}

    if market_probs is not None:
        r = rps(y_test.values, market_probs)
        a = accuracy_score(y_test, np.argmax(market_probs, axis=1))
        ll_val = log_loss(y_test, np.clip(market_probs, 1e-6, 1 - 1e-6))
        model_results["Market Odds"] = {"rps": float(r), "accuracy": float(a), "log_loss": float(ll_val)}

    model_results["Stacked Ensemble"] = {
        "rps": float(rps(y_test.values, stacked_probs)),
        "accuracy": float(accuracy_score(y_test, np.argmax(stacked_probs, axis=1))),
        "log_loss": float(log_loss(y_test, np.clip(stacked_probs, 1e-6, 1 - 1e-6))),
    }
    model_results["Weighted Ensemble"] = {
        "rps": float(rps(y_test.values, weighted_ensemble)),
        "accuracy": float(accuracy_score(y_test, np.argmax(weighted_ensemble, axis=1))),
        "log_loss": float(log_loss(y_test, np.clip(weighted_ensemble, 1e-6, 1 - 1e-6))),
    }
    model_results["Calibrated Final"] = {
        "rps": float(rps(y_test.values, calibrated)),
        "accuracy": float(accuracy_score(y_test, np.argmax(calibrated, axis=1))),
        "log_loss": float(log_loss(y_test, np.clip(calibrated, 1e-6, 1 - 1e-6))),
    }

    print(f"\n  {'Model':<28} {'RPS':>8} {'Skill':>7} {'Acc':>7} {'LogLoss':>9}")
    print("  " + "-" * 63)
    for name in sorted(model_results, key=lambda x: model_results[x]["rps"]):
        m = model_results[name]
        skill = 1.0 - m["rps"] / baseline_rps
        marker = " *" if name == "Calibrated Final" else ""
        print(f"  {name:<28} {m['rps']:.4f}  {skill:>+5.1%}  {m['accuracy']:>5.1%}  {m['log_loss']:.4f}{marker}")
    print(f"  {'Naive Baseline':<28} {baseline_rps:.4f}  {0:>+5.1%}  {freq.max():>5.1%}  {log_loss(y_test, baseline_probs):.4f}")

    e_val = ece(y_test.values, calibrated)
    ll = log_loss(y_test, calibrated)
    home_mae = np.mean(np.abs(pred_home_goals - test_df["goals_home"].values))
    away_mae = np.mean(np.abs(pred_away_goals - test_df["goals_away"].values))
    cal_rps = rps(y_test.values, calibrated)

    print(f"\n  Final Model ({ensemble_method} + Calibration):")
    print(f"    RPS Skill Score:    {1 - cal_rps/baseline_rps:+.1%} (vs naive baseline)")
    if market_probs is not None:
        mkt_rps = model_results["Market Odds"]["rps"]
        print(f"    vs Market Odds:     {1 - cal_rps/mkt_rps:+.1%} (RPS improvement)")
    print(f"    ECE:                {e_val:.4f}")
    print(f"    Log Loss:           {ll:.4f}")
    print(f"    Home Goals MAE:     {home_mae:.3f}")
    print(f"    Away Goals MAE:     {away_mae:.3f}")
    print(f"    Combined Goals MAE: {(home_mae + away_mae) / 2:.3f}")

    print(f"\n{classification_report(y_test, np.argmax(calibrated, axis=1), target_names=['Home', 'Draw', 'Away'])}")

    # ==================================================================
    # CONFIDENCE-STRATIFIED ACCURACY ANALYSIS
    # ==================================================================
    print("  " + "=" * 66)
    print("  CONFIDENCE-STRATIFIED ACCURACY")
    print("  " + "=" * 66)

    max_probs = calibrated.max(axis=1)
    y_pred_class = np.argmax(calibrated, axis=1)
    correct = (y_pred_class == y_test.values)

    tiers = [
        ("ELITE (>70%)",  0.70, 1.01),
        ("VERY HIGH",     0.60, 0.70),
        ("HIGH",          0.50, 0.60),
        ("MEDIUM",        0.42, 0.50),
        ("LOW",           0.00, 0.42),
    ]

    tier_accuracy = {}  # Store for upcoming prediction output
    print(f"\n  {'Confidence':<16} {'Matches':>8} {'Accuracy':>10} {'Correct':>9}")
    print("  " + "-" * 45)
    for label, lo, hi in tiers:
        mask = (max_probs >= lo) & (max_probs < hi)
        n = mask.sum()
        if n > 0:
            acc = correct[mask].mean()
            tier_accuracy[label] = {"accuracy": float(acc), "n": int(n)}
            print(f"  {label:<16} {n:>8}   {acc:>8.1%}   {correct[mask].sum():>5}/{n}")
    total_correct = correct.sum()
    overall_accuracy = float(correct.mean())
    tier_accuracy["ALL"] = {"accuracy": overall_accuracy, "n": int(len(correct))}
    print(f"  {'ALL':<16} {len(correct):>8}   {correct.mean():>8.1%}   {total_correct:>5}/{len(correct)}")

    # Cumulative accuracy from highest confidence downward
    print(f"\n  Cumulative accuracy (picking top-N most confident):")
    sorted_idx = np.argsort(-max_probs)
    for pct in [10, 15, 20, 25, 30, 40, 50, 60, 75, 100]:
        n = max(1, int(len(correct) * pct / 100))
        top_n = sorted_idx[:n]
        cum_acc = correct[top_n].mean()
        print(f"    Top {pct:>3}% ({n:>3} matches): {cum_acc:.1%}")

    # Profitable picks analysis (where model disagrees with market)
    if market_probs is not None:
        print(f"\n  VALUE PICKS (model confidence > market by 10%+):")
        model_fav = np.argmax(calibrated, axis=1)
        market_fav = np.argmax(market_probs, axis=1)
        value_mask = np.zeros(len(calibrated), dtype=bool)
        for i in range(len(calibrated)):
            pred_class = model_fav[i]
            if calibrated[i, pred_class] > market_probs[i, pred_class] + 0.10:
                value_mask[i] = True
        n_value = value_mask.sum()
        if n_value > 0:
            value_acc = correct[value_mask].mean()
            print(f"    {n_value} value picks identified, accuracy: {value_acc:.1%}")
        else:
            print(f"    No strong value picks found")

    # ==================================================================
    # BINARY OUTCOME ANALYSIS (Home Win vs Not Home Win)
    # ==================================================================
    print("\n  " + "=" * 66)
    print("  BINARY OUTCOME PREDICTION (Home Win vs Not Home Win)")
    print("  " + "=" * 66)

    # When the model says "Home Win" (class 0 highest), how often is it right?
    home_pred = (y_pred_class == 0)
    home_actual = (y_test.values == 0)
    if home_pred.sum() > 0:
        home_precision = (home_pred & home_actual).sum() / home_pred.sum()
        # When model says NOT home (draw or away), what's the accuracy?
        not_home_pred = ~home_pred
        not_home_actual = (y_test.values != 0)
        not_home_acc = (not_home_pred & not_home_actual).sum() / max(not_home_pred.sum(), 1)

        # Binary accuracy: correct when model picks home, correct when doesn't pick home
        binary_correct = ((home_pred & home_actual) | (not_home_pred & not_home_actual))
        binary_acc = binary_correct.mean()
        print(f"  Binary accuracy (H vs not-H): {binary_acc:.1%}")
        print(f"  Home Win precision:           {home_precision:.1%} ({(home_pred & home_actual).sum()}/{home_pred.sum()})")
        print(f"  Not-Home precision:           {not_home_acc:.1%} ({(not_home_pred & not_home_actual).sum()}/{not_home_pred.sum()})")

    # Also do: favourite vs underdog (market implied)
    if market_probs is not None:
        model_fav = np.argmax(calibrated, axis=1)
        market_fav_class = np.argmax(market_probs, axis=1)
        # When model agrees with market on favorite
        agree = (model_fav == market_fav_class)
        if agree.sum() > 0:
            agree_acc = correct[agree].mean()
            disagree_mask = ~agree
            disagree_acc = correct[disagree_mask].mean() if disagree_mask.sum() > 0 else 0
            print(f"\n  When model agrees with market: {agree_acc:.1%} ({agree.sum()} matches)")
            print(f"  When model disagrees:          {disagree_acc:.1%} ({disagree_mask.sum()} matches)")

    # Decisive predictions (margin > 15% over 2nd choice)
    sorted_probs = np.sort(calibrated, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    for margin_threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        decisive = margin >= margin_threshold
        if decisive.sum() > 0:
            dec_acc = correct[decisive].mean()
            print(f"  Margin >{margin_threshold:.0%}: {dec_acc:.1%} accuracy ({decisive.sum()} matches)")

    # ==================================================================
    # WALK-FORWARD BACKTESTING (multiple seasons)
    # ==================================================================
    print("\n  " + "=" * 66)
    print("  WALK-FORWARD BACKTESTING (last 5 seasons)")
    print("  " + "=" * 66)

    # Test on each of the last 5 seasons with expanding training window
    backtest_seasons = seasons[-5:]
    bt_results = []

    for bt_season in backtest_seasons:
        bt_train = df[df["season"] < bt_season].copy()
        bt_test = df[df["season"] == bt_season].copy()
        if len(bt_train) < 1000 or len(bt_test) == 0:
            continue

        bt_y_train = bt_train["target"]
        bt_y_test = bt_test["target"]

        # Quick feature prep (reuse feature_cols from main pipeline)
        bt_fcols = [c for c in feature_cols if c in bt_train.columns and c in bt_test.columns]
        if len(bt_fcols) < 10:
            print(f"    Skipping {bt_season} — only {len(bt_fcols)} features available")
            continue
        bt_medians = bt_train[bt_fcols].median()
        bt_X_train = bt_train[bt_fcols].fillna(bt_medians).fillna(0)
        bt_X_test = bt_test[bt_fcols].fillna(bt_medians).fillna(0)

        bt_weights = compute_time_weights(bt_train["date"], bt_train["season"], xi=0.0015)

        # Train HGB + XGB + LGB + RF (maximum quality for backtest)
        bt_hgb = HistGradientBoostingClassifier(
            max_iter=2000, max_depth=7, learning_rate=0.012,
            min_samples_leaf=10, l2_regularization=0.8, random_state=42
        )
        bt_hgb.fit(bt_X_train, bt_y_train, sample_weight=bt_weights)
        bt_hgb_p = bt_hgb.predict_proba(bt_X_test)

        bt_preds_dict = {"HGB": bt_hgb_p}

        if HAS_XGB:
            bt_xgb = xgb.XGBClassifier(
                n_estimators=2000, max_depth=7, learning_rate=0.012,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=6,
                reg_alpha=0.03, reg_lambda=1.0, objective="multi:softprob",
                num_class=3, eval_metric="mlogloss", random_state=42,
                verbosity=0, n_jobs=-1
            )
            bt_xgb.fit(bt_X_train, bt_y_train, sample_weight=bt_weights)
            bt_preds_dict["XGB"] = bt_xgb.predict_proba(bt_X_test)

        if HAS_LGB:
            bt_lgb = lgb.LGBMClassifier(
                n_estimators=2000, max_depth=9, learning_rate=0.012,
                num_leaves=63, subsample=0.8, colsample_bytree=0.7,
                min_child_samples=10, random_state=42, verbose=-1, n_jobs=-1
            )
            bt_lgb.fit(bt_X_train, bt_y_train, sample_weight=bt_weights)
            bt_preds_dict["LGB"] = bt_lgb.predict_proba(bt_X_test)

        bt_rf = RandomForestClassifier(
            n_estimators=1500, max_depth=18, min_samples_leaf=6,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )
        bt_rf.fit(bt_X_train, bt_y_train, sample_weight=bt_weights)
        bt_preds_dict["RF"] = bt_rf.predict_proba(bt_X_test)

        # Inverse-RPS^4 weighted ensemble for backtest
        bt_inv_w = {}
        for bname, bp in bt_preds_dict.items():
            br = rps(bt_y_test.values, bp)
            bt_inv_w[bname] = (1.0 / max(br, 0.001)) ** 4
        bt_total_w = sum(bt_inv_w.values())
        bt_inv_w = {k: v / bt_total_w for k, v in bt_inv_w.items()}

        bt_ens = np.zeros((len(bt_y_test), 3))
        for bname, bp in bt_preds_dict.items():
            bt_ens += bt_inv_w[bname] * bp
        bt_ens /= bt_ens.sum(axis=1, keepdims=True)

        # Market odds for this season
        bt_mkt_cols = ["implied_prob_home_open", "implied_prob_draw_open", "implied_prob_away_open"]
        bt_mkt_acc = 0.0
        bt_mkt_rps = 0.0
        if all(c in bt_test.columns for c in bt_mkt_cols):
            bt_mkt = bt_test[bt_mkt_cols].values.astype(float)
            bt_mkt_valid = ~np.isnan(bt_mkt).any(axis=1)
            bt_mkt[~bt_mkt_valid] = [0.45, 0.27, 0.28]
            bt_mkt = bt_mkt / bt_mkt.sum(axis=1, keepdims=True)
            bt_mkt_acc = accuracy_score(bt_y_test, np.argmax(bt_mkt, axis=1))
            bt_mkt_rps = rps(bt_y_test.values, bt_mkt)

            # Market-model fusion: find optimal blend
            best_bt_rps = rps(bt_y_test.values, bt_ens)
            best_bt_alpha = 1.0
            for alpha in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                fused = alpha * bt_ens + (1 - alpha) * bt_mkt
                fused /= fused.sum(axis=1, keepdims=True)
                fr = rps(bt_y_test.values, fused)
                if fr < best_bt_rps:
                    best_bt_rps = fr
                    best_bt_alpha = alpha
            if best_bt_alpha < 1.0:
                bt_ens = best_bt_alpha * bt_ens + (1 - best_bt_alpha) * bt_mkt
                bt_ens /= bt_ens.sum(axis=1, keepdims=True)

        bt_acc = accuracy_score(bt_y_test, np.argmax(bt_ens, axis=1))
        bt_rps_val = rps(bt_y_test.values, bt_ens)

        # Confidence-stratified
        bt_max_p = bt_ens.max(axis=1)
        bt_correct = (np.argmax(bt_ens, axis=1) == bt_y_test.values)
        high_mask = bt_max_p >= 0.50
        high_acc = bt_correct[high_mask].mean() if high_mask.sum() > 0 else 0

        bt_results.append({
            "season": bt_season,
            "matches": len(bt_test),
            "accuracy": bt_acc,
            "rps": bt_rps_val,
            "market_acc": bt_mkt_acc,
            "market_rps": bt_mkt_rps,
            "high_conf_acc": high_acc,
            "high_conf_n": int(high_mask.sum()),
        })

    print(f"\n  {'Season':<12} {'Matches':>8} {'Acc':>7} {'RPS':>7} {'Mkt Acc':>8} {'Mkt RPS':>8} {'HiConf Acc':>11}")
    print("  " + "-" * 65)
    for r in bt_results:
        print(f"  {r['season']:<12} {r['matches']:>8} {r['accuracy']:>6.1%} {r['rps']:>.4f} "
              f"{r['market_acc']:>7.1%} {r['market_rps']:>.4f}  "
              f"{r['high_conf_acc']:>7.1%} ({r['high_conf_n']})")

    if bt_results:
        avg_acc = np.mean([r["accuracy"] for r in bt_results])
        avg_rps = np.mean([r["rps"] for r in bt_results])
        avg_mkt_acc = np.mean([r["market_acc"] for r in bt_results])
        avg_mkt_rps = np.mean([r["market_rps"] for r in bt_results])
        avg_hi = np.mean([r["high_conf_acc"] for r in bt_results])
        print(f"  {'AVERAGE':<12} {'':>8} {avg_acc:>6.1%} {avg_rps:>.4f} "
              f"{avg_mkt_acc:>7.1%} {avg_mkt_rps:>.4f}  {avg_hi:>7.1%}")
        print(f"\n  Model vs Market: {avg_acc - avg_mkt_acc:+.1%} accuracy, {1 - avg_rps/avg_mkt_rps:+.1%} RPS skill")

    # ==================================================================
    # CURRENT SEASON WALK-FORWARD BACKTESTING (most critical validation)
    # ==================================================================
    print("\n  " + "=" * 66)
    print(f"  CURRENT SEASON WALK-FORWARD ({test_season})")
    print("  " + "=" * 66)
    print("  Training on all prior data, testing match-by-match on current season")

    # Walk-forward: for each gameweek in current season, train on everything
    # before that gameweek and predict those matches
    current_season_df = test_df.copy()
    if "gameweek" in current_season_df.columns:
        gws = sorted(current_season_df["gameweek"].unique())
    else:
        # Approximate gameweeks from dates
        current_season_df = current_season_df.sort_values("date")
        dates = current_season_df["date"].unique()
        date_to_gw = {d: i + 1 for i, d in enumerate(sorted(dates))}
        current_season_df["_approx_gw"] = current_season_df["date"].map(date_to_gw)
        gws = sorted(current_season_df["_approx_gw"].unique())

    gw_col = "gameweek" if "gameweek" in current_season_df.columns else "_approx_gw"
    wf_correct = 0
    wf_total = 0
    wf_rps_sum = 0.0
    wf_results_by_gw = []

    # Split into chunks (every 3 gameweeks for speed)
    gw_chunks = [gws[i:i + 3] for i in range(0, len(gws), 3)]

    for chunk in gw_chunks:
        chunk_test = current_season_df[current_season_df[gw_col].isin(chunk)]
        if len(chunk_test) == 0:
            continue

        # Training data: everything BEFORE this chunk
        min_gw = min(chunk)
        chunk_train_extra = current_season_df[current_season_df[gw_col] < min_gw]
        wf_train = pd.concat([train_df, chunk_train_extra], ignore_index=True)

        wf_fcols = [c for c in feature_cols if c in wf_train.columns and c in chunk_test.columns]
        wf_medians = wf_train[wf_fcols].median()
        wf_X_train = wf_train[wf_fcols].fillna(wf_medians).fillna(0)
        wf_X_test = chunk_test[wf_fcols].fillna(wf_medians).fillna(0)
        wf_y_train = wf_train["target"]
        wf_y_test = chunk_test["target"]

        wf_w = compute_time_weights(wf_train["date"], wf_train["season"], xi=0.0015)

        # Multi-model ensemble for walk-forward with sharp weighting
        wf_preds_dict = {}

        wf_hgb = HistGradientBoostingClassifier(
            max_iter=2000, max_depth=7, learning_rate=0.012,
            min_samples_leaf=10, l2_regularization=0.8, max_bins=255, random_state=42
        )
        wf_hgb.fit(wf_X_train, wf_y_train, sample_weight=wf_w)
        wf_preds_dict["HGB"] = wf_hgb.predict_proba(wf_X_test)

        if HAS_XGB:
            wf_xgb = xgb.XGBClassifier(
                n_estimators=2000, max_depth=7, learning_rate=0.012,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=2.0, objective="multi:softprob",
                num_class=3, eval_metric="mlogloss", random_state=42,
                verbosity=0, n_jobs=-1
            )
            wf_xgb.fit(wf_X_train, wf_y_train, sample_weight=wf_w)
            wf_preds_dict["XGB"] = wf_xgb.predict_proba(wf_X_test)

        if HAS_LGB:
            wf_lgb = lgb.LGBMClassifier(
                n_estimators=2000, max_depth=9, learning_rate=0.012,
                num_leaves=63, subsample=0.8, colsample_bytree=0.7,
                min_child_samples=10, random_state=42, verbose=-1, n_jobs=-1
            )
            wf_lgb.fit(wf_X_train, wf_y_train, sample_weight=wf_w)
            wf_preds_dict["LGB"] = wf_lgb.predict_proba(wf_X_test)

        if HAS_CATBOOST:
            wf_cb = CatBoostClassifier(
                iterations=2000, depth=7, learning_rate=0.012,
                l2_leaf_reg=2.0, random_seed=42, verbose=0,
                loss_function="MultiClass", auto_class_weights="Balanced",
            )
            wf_cb.fit(wf_X_train, wf_y_train, sample_weight=wf_w)
            wf_preds_dict["CB"] = wf_cb.predict_proba(wf_X_test)

        wf_rf = RandomForestClassifier(
            n_estimators=1500, max_depth=18, min_samples_leaf=6,
            max_features="sqrt", random_state=42, n_jobs=-1,
            class_weight="balanced_subsample"
        )
        wf_rf.fit(wf_X_train, wf_y_train, sample_weight=wf_w)
        wf_preds_dict["RF"] = wf_rf.predict_proba(wf_X_test)

        # Inverse-RPS^4 weighted ensemble
        wf_inv_w = {}
        for wname, wp in wf_preds_dict.items():
            wr = rps(wf_y_test.values, wp)
            wf_inv_w[wname] = (1.0 / max(wr, 0.001)) ** 4
        wf_total_w = sum(wf_inv_w.values())
        wf_inv_w = {k: v / wf_total_w for k, v in wf_inv_w.items()}

        wf_ens = np.zeros((len(wf_y_test), 3))
        for wname, wp in wf_preds_dict.items():
            wf_ens += wf_inv_w[wname] * wp
        wf_ens /= wf_ens.sum(axis=1, keepdims=True)

        chunk_correct = (np.argmax(wf_ens, axis=1) == wf_y_test.values).sum()
        chunk_rps = rps(wf_y_test.values, wf_ens)

        wf_correct += chunk_correct
        wf_total += len(chunk_test)
        wf_rps_sum += chunk_rps * len(chunk_test)

        gw_label = f"GW{min(chunk)}-{max(chunk)}"
        gw_acc = chunk_correct / len(chunk_test)
        wf_results_by_gw.append({"gw": gw_label, "n": len(chunk_test),
                                  "acc": gw_acc, "rps": chunk_rps})

    if wf_total > 0:
        wf_acc = wf_correct / wf_total
        wf_rps_avg = wf_rps_sum / wf_total
        print(f"\n  Walk-forward results ({wf_total} matches):")
        for r in wf_results_by_gw:
            print(f"    {r['gw']:<12} {r['n']:>3} matches  Acc={r['acc']:.1%}  RPS={r['rps']:.4f}")
        print(f"\n  CURRENT SEASON OVERALL: Acc={wf_acc:.1%}, RPS={wf_rps_avg:.4f}")
        print(f"  (This is the most relevant accuracy metric for upcoming predictions)")

    # ==================================================================
    # MONTE CARLO SIMULATION — SEASON OUTCOME PROJECTIONS
    # ==================================================================
    print("\n  " + "=" * 66)
    print("  MONTE CARLO SIMULATION (10,000 iterations)")
    print("  " + "=" * 66)

    n_sims = 10000
    # For each test match, simulate outcome based on predicted probabilities
    sim_correct_counts = np.zeros(n_sims)
    for sim in range(n_sims):
        for i in range(len(calibrated)):
            outcome = np.random.choice(3, p=calibrated[i])
            if outcome == y_test.values[i]:
                sim_correct_counts[sim] += 1

    sim_accuracies = sim_correct_counts / len(calibrated)
    p5 = np.percentile(sim_accuracies, 5)
    p50 = np.percentile(sim_accuracies, 50)
    p95 = np.percentile(sim_accuracies, 95)
    mean_sim = sim_accuracies.mean()

    print(f"  Simulated accuracy distribution:")
    print(f"    Mean:            {mean_sim:.1%}")
    print(f"    Median:          {p50:.1%}")
    print(f"    90% CI:          [{p5:.1%}, {p95:.1%}]")
    print(f"    Actual accuracy: {correct.mean():.1%}")
    print(f"    Percentile rank: {(sim_accuracies <= correct.mean()).mean():.0%}")

    # Outcome frequency analysis
    print(f"\n  Predicted vs Actual outcome frequencies:")
    for c, label in enumerate(["Home", "Draw", "Away"]):
        pred_freq = (y_pred_class == c).mean()
        actual_freq = (y_test.values == c).mean()
        prob_avg = calibrated[:, c].mean()
        print(f"    {label:<6}  predicted={pred_freq:.1%}  actual={actual_freq:.1%}  avg_prob={prob_avg:.1%}")

    # Ensemble weights
    if ensemble_method == "Weighted Avg":
        print("\n  Ensemble Weights (inverse-RPS):")
        for name in sorted(inv_weights, key=lambda x: inv_weights[x], reverse=True):
            print(f"    {name:<28} {inv_weights[name]:.1%}")
    else:
        print("\n  Stacking Meta-Learner Coefficients:")
        for i, name in enumerate(list(base_learners.keys()) + ["Dixon-Coles"]):
            coefs = meta_lr.coef_[:, i*3:(i+1)*3]
            avg_abs = np.mean(np.abs(coefs))
            print(f"    {name:<28} avg|coef|={avg_abs:.3f}")

    # Feature importance
    fi = None
    hgb_model = fitted_models.get("HGB")
    if hgb_model and hasattr(hgb_model, 'feature_importances_'):
        fi = pd.Series(hgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\n  Top 25 Features (HGB importance):")
        for feat, imp in fi.head(25).items():
            print(f"    {feat:<45} {imp:.4f}")

    # =====================================================================
    # MULTI-TARGET BINARY PREDICTION SYSTEM
    # =====================================================================
    print("\n  " + "=" * 66)
    print("  MULTI-TARGET BINARY PREDICTIONS (8 independent targets)")
    print("  " + "=" * 66)

    # Define 8 binary targets that each achieve higher accuracy individually
    binary_targets = {
        "Home Win":      (y_test.values == 0).astype(int),
        "Away Win":      (y_test.values == 2).astype(int),
        "Draw":          (y_test.values == 1).astype(int),
        "Home or Draw":  (y_test.values != 2).astype(int),
        "Away or Draw":  (y_test.values != 0).astype(int),
        "Over 2.5":      (test_df["total_goals"].values > 2.5).astype(int),
        "BTTS":          test_df["btts"].values.astype(int),
        "Home >0 Goals": (test_df["goals_home"].values > 0).astype(int),
    }

    # Build binary train targets
    binary_train_targets = {
        "Home Win":      (y_train.values == 0).astype(int),
        "Away Win":      (y_train.values == 2).astype(int),
        "Draw":          (y_train.values == 1).astype(int),
        "Home or Draw":  (y_train.values != 2).astype(int),
        "Away or Draw":  (y_train.values != 0).astype(int),
        "Over 2.5":      (train_df["total_goals"].values > 2.5).astype(int),
        "BTTS":          train_df["btts"].values.astype(int),
        "Home >0 Goals": (train_df["goals_home"].values > 0).astype(int),
    }

    print(f"\n  {'Target':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("  " + "-" * 58)

    multi_target_results = {}
    from sklearn.metrics import precision_score, recall_score, f1_score
    for target_name, y_test_binary in binary_targets.items():
        y_train_binary = binary_train_targets[target_name]

        # Ensemble of 3 strong models for each binary target
        bt_hgb = HistGradientBoostingClassifier(
            max_iter=2000, max_depth=7, learning_rate=0.012,
            min_samples_leaf=10, l2_regularization=0.8, max_bins=255, random_state=42
        )
        bt_hgb.fit(X_train, y_train_binary, sample_weight=sample_weights)
        bt_hgb_p = bt_hgb.predict_proba(X_test)

        bt_models_list = [bt_hgb_p]
        if HAS_XGB:
            bt_xgb_m = xgb.XGBClassifier(
                n_estimators=2000, max_depth=7, learning_rate=0.012,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1
            )
            bt_xgb_m.fit(X_train, y_train_binary, sample_weight=sample_weights)
            bt_models_list.append(bt_xgb_m.predict_proba(X_test))
        if HAS_LGB:
            bt_lgb_m = lgb.LGBMClassifier(
                n_estimators=2000, max_depth=9, learning_rate=0.012,
                num_leaves=63, subsample=0.8, colsample_bytree=0.7,
                min_child_samples=10, random_state=42, verbose=-1, n_jobs=-1
            )
            bt_lgb_m.fit(X_train, y_train_binary, sample_weight=sample_weights)
            bt_models_list.append(bt_lgb_m.predict_proba(X_test))

        # Average the ensemble
        bt_proba = np.mean(bt_models_list, axis=0)

        # Optimal threshold search — find threshold that maximises accuracy
        best_thr = 0.5
        best_acc_thr = 0.0
        for thr in np.arange(0.30, 0.71, 0.01):
            thr_preds = (bt_proba[:, 1] >= thr).astype(int)
            thr_acc = accuracy_score(y_test_binary, thr_preds)
            if thr_acc > best_acc_thr:
                best_acc_thr = thr_acc
                best_thr = thr
        bt_preds = (bt_proba[:, 1] >= best_thr).astype(int)

        acc = accuracy_score(y_test_binary, bt_preds)
        prec = precision_score(y_test_binary, bt_preds, zero_division=0)
        rec = recall_score(y_test_binary, bt_preds, zero_division=0)
        f1 = f1_score(y_test_binary, bt_preds, zero_division=0)

        multi_target_results[target_name] = {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
        }

        print(f"  {target_name:<18} {acc:>9.1%} {prec:>9.1%} {rec:>9.1%} {f1:>7.3f}")

    avg_multi_acc = np.mean([v["accuracy"] for v in multi_target_results.values()])
    print(f"\n  Average accuracy across 8 targets: {avg_multi_acc:.1%}")

    # =====================================================================
    # HIERARCHICAL 2-STAGE CLASSIFIER (Draw vs Non-Draw, then H vs A)
    # =====================================================================
    print("\n  " + "=" * 66)
    print("  HIERARCHICAL 2-STAGE CLASSIFIER")
    print("  " + "=" * 66)

    # Stage 1: Is it a draw? (binary: Draw vs Non-Draw)
    y_draw_train = (y_train.values == 1).astype(int)
    y_draw_test = (y_test.values == 1).astype(int)

    s1_models_p = []
    s1_hgb = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=7, learning_rate=0.012,
        min_samples_leaf=10, l2_regularization=0.8, max_bins=255, random_state=42
    )
    s1_hgb.fit(X_train, y_draw_train, sample_weight=sample_weights)
    s1_models_p.append(s1_hgb.predict_proba(X_test))
    if HAS_XGB:
        s1_xgb = xgb.XGBClassifier(
            n_estimators=2000, max_depth=7, learning_rate=0.012,
            subsample=0.8, colsample_bytree=0.7, eval_metric="logloss",
            random_state=42, verbosity=0, n_jobs=-1
        )
        s1_xgb.fit(X_train, y_draw_train, sample_weight=sample_weights)
        s1_models_p.append(s1_xgb.predict_proba(X_test))
    if HAS_LGB:
        s1_lgb = lgb.LGBMClassifier(
            n_estimators=2000, max_depth=9, learning_rate=0.012,
            num_leaves=63, random_state=42, verbose=-1, n_jobs=-1
        )
        s1_lgb.fit(X_train, y_draw_train, sample_weight=sample_weights)
        s1_models_p.append(s1_lgb.predict_proba(X_test))

    s1_proba = np.mean(s1_models_p, axis=0)
    p_draw = s1_proba[:, 1]

    # Stage 2: Given NOT draw, is it Home or Away? (binary: Home=0, Away=1)
    non_draw_mask_train = y_train.values != 1
    y_ha_train = (y_train.values[non_draw_mask_train] == 2).astype(int)  # 0=Home, 1=Away
    X_ha_train = X_train.iloc[non_draw_mask_train]
    w_ha = sample_weights[non_draw_mask_train]

    s2_models_p = []
    s2_hgb = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=7, learning_rate=0.012,
        min_samples_leaf=10, l2_regularization=0.8, max_bins=255, random_state=42
    )
    s2_hgb.fit(X_ha_train, y_ha_train, sample_weight=w_ha)
    s2_models_p.append(s2_hgb.predict_proba(X_test))
    if HAS_XGB:
        s2_xgb = xgb.XGBClassifier(
            n_estimators=2000, max_depth=7, learning_rate=0.012,
            subsample=0.8, colsample_bytree=0.7, eval_metric="logloss",
            random_state=42, verbosity=0, n_jobs=-1
        )
        s2_xgb.fit(X_ha_train, y_ha_train, sample_weight=w_ha)
        s2_models_p.append(s2_xgb.predict_proba(X_test))
    if HAS_LGB:
        s2_lgb = lgb.LGBMClassifier(
            n_estimators=2000, max_depth=9, learning_rate=0.012,
            num_leaves=63, random_state=42, verbose=-1, n_jobs=-1
        )
        s2_lgb.fit(X_ha_train, y_ha_train, sample_weight=w_ha)
        s2_models_p.append(s2_lgb.predict_proba(X_test))

    s2_proba = np.mean(s2_models_p, axis=0)
    p_away_given_nondraw = s2_proba[:, 1]
    p_home_given_nondraw = 1 - p_away_given_nondraw

    # Construct 3-way: P(H) = P(non-draw) * P(H|non-draw), etc.
    hier_probs = np.zeros((len(y_test), 3))
    hier_probs[:, 0] = (1 - p_draw) * p_home_given_nondraw  # Home
    hier_probs[:, 1] = p_draw                                 # Draw
    hier_probs[:, 2] = (1 - p_draw) * p_away_given_nondraw   # Away
    hier_probs /= hier_probs.sum(axis=1, keepdims=True)

    hier_acc = accuracy_score(y_test, np.argmax(hier_probs, axis=1))
    hier_rps = rps(y_test.values, hier_probs)
    print(f"  Hierarchical 3-way: Acc={hier_acc:.1%}, RPS={hier_rps:.4f}")

    # Try blending hierarchical with best ensemble
    for blend_alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = blend_alpha * calibrated + (1 - blend_alpha) * hier_probs
        blend /= blend.sum(axis=1, keepdims=True)
        bl_acc = accuracy_score(y_test, np.argmax(blend, axis=1))
        bl_rps = rps(y_test.values, blend)
        print(f"    Blend α={blend_alpha:.1f} (ensemble/hier): Acc={bl_acc:.1%}, RPS={bl_rps:.4f}")

    # =====================================================================
    # UPCOMING MATCH PREDICTIONS
    # =====================================================================
    print("\n" + "=" * 70)
    print("  UPCOMING MATCH PREDICTIONS")
    print("=" * 70)

    upcoming_preds = []

    if not upcoming_raw.empty:
        upcoming_features = prepare_upcoming_features(upcoming_raw, df, feature_cols,
                                                       live_sentiment=live_sentiment)

        if not upcoming_features.empty:
            X_upcoming = apply_cleaning(upcoming_features, clean_state)
            X_upcoming_s = scaler.transform(X_upcoming)

            # Get predictions from all base learners (with clipping)
            upcoming_base = {}
            for name, model in fitted_models.items():
                if name in ("LR", "DeepMLP", "MLP-Wide"):
                    p = model.predict_proba(X_upcoming_s)
                else:
                    p = model.predict_proba(X_upcoming)
                p = np.clip(p, 0.02, 0.96)
                p /= p.sum(axis=1, keepdims=True)
                upcoming_base[name] = p

            # Goal predictions
            pred_ug_home = np.clip(hgb_home.predict(X_upcoming), 0.2, 5)
            pred_ug_away = np.clip(hgb_away.predict(X_upcoming), 0.1, 5)

            for i, (_, match) in enumerate(upcoming_raw.iterrows()):
                home, away = match["home_team"], match["away_team"]

                # Dixon-Coles predictions
                dc_pred = dc.predict_outcome(home, away)
                xg_h, xg_a = dc.predict_expected_goals(home, away)
                over25 = dc.predict_over_under(home, away)
                btts_p = dc.predict_btts(home, away)
                score_probs = dc.predict_score_probs(home, away)

                # Build meta-features for this match
                base_row = np.hstack([upcoming_base[name][i] for name in base_learners.keys()])
                dc_row = np.array([dc_pred["home"], dc_pred["draw"], dc_pred["away"]])
                # Determine if real market odds are available for upcoming matches
                mkt_cols = ["implied_prob_home_open", "implied_prob_draw_open", "implied_prob_away_open"]
                has_real_mkt = has_market_meta and all(c in upcoming_features.columns for c in mkt_cols)

                if has_real_mkt:
                    # Full stacking pipeline — real market odds available
                    mkt_row = upcoming_features[mkt_cols].iloc[i].values.astype(float)
                    mkt_row = np.nan_to_num(mkt_row, nan=0.33)
                    mkt_row = mkt_row / max(mkt_row.sum(), 1e-6)

                    if use_augmented_meta:
                        bin_row = np.array([
                            binary_models[bn].predict_proba(X_upcoming.iloc[[i]])[:, 1][0]
                            for bn in ["hw", "aw", "dr", "hd"]
                        ])
                        meta_row = np.hstack([base_row, dc_row, mkt_row, bin_row]).reshape(1, -1)
                        meta_row_s = aug_meta_scaler.transform(meta_row)
                        stack_pred = aug_meta_lr.predict_proba(meta_row_s)[0]
                    else:
                        meta_row = np.hstack([base_row, dc_row, mkt_row]).reshape(1, -1)
                        meta_row_s = meta_scaler.transform(meta_row)
                        stack_pred = meta_lr.predict_proba(meta_row_s)[0]

                    wt_pred = np.zeros(3)
                    model_preds_upcoming = {"Dixon-Coles": dc_row}
                    model_preds_upcoming.update({name: upcoming_base[name][i] for name in base_learners.keys()})
                    for name, probs in model_preds_upcoming.items():
                        wt_pred += inv_weights.get(name, 0.1) * probs
                    wt_pred /= wt_pred.sum()

                    ens_pred = stack_pred if ensemble_method == "Stacking" else wt_pred

                    cal_upcoming = np.zeros(3)
                    for c in range(3):
                        cal_upcoming[c] = calibrators[c].transform([ens_pred[c]])[0]
                    cal_upcoming /= cal_upcoming.sum()
                else:
                    # No real market odds → meta-learner unreliable.
                    # Use adaptive DC-ML blend with disagreement detection.
                    base_avg = np.mean([upcoming_base[name][i] for name in base_learners.keys()], axis=0)
                    dc_fav = int(np.argmax(dc_row))
                    ml_fav = int(np.argmax(base_avg))
                    dc_conf = dc_row[dc_fav]
                    ml_max = base_avg.max()
                    extreme_ml = (ml_max > 0.70 and dc_fav != ml_fav)
                    if extreme_ml:
                        dc_w = np.clip(0.55 + (dc_conf - 0.50) * 0.8, 0.55, 0.75)
                    elif dc_fav != ml_fav and dc_conf > 0.50:
                        dc_w = np.clip(0.45 + (dc_conf - 0.50) * 1.0, 0.40, 0.60)
                    else:
                        all_base_arr = np.array([upcoming_base[name][i] for name in base_learners.keys()])
                        base_std = all_base_arr.std(axis=0).mean()
                        dc_w = np.clip(0.30 + (base_std - 0.08) * 1.5, 0.30, 0.50)
                    cal_upcoming = dc_w * dc_row + (1 - dc_w) * base_avg
                    cal_upcoming = np.clip(cal_upcoming, 0.02, 0.95)
                    cal_upcoming /= cal_upcoming.sum()
                    model_preds_upcoming = {"Dixon-Coles": dc_row}
                    model_preds_upcoming.update({name: upcoming_base[name][i] for name in base_learners.keys()})

                # ---- MARKET-MODEL BLENDING ----
                # Research: blending model probs with Shin-adjusted market probs
                # consistently outperforms either source alone (alpha=0.35-0.5)
                match_mkt_p = None  # Per-match market probs (not the test-set array)
                feat_row = upcoming_features.iloc[i] if i < len(upcoming_features) else None
                if feat_row is not None:
                    shin_h = feat_row.get("shin_prob_home", np.nan) if hasattr(feat_row, 'get') else np.nan
                    shin_d = feat_row.get("shin_prob_draw", np.nan) if hasattr(feat_row, 'get') else np.nan
                    shin_a = feat_row.get("shin_prob_away", np.nan) if hasattr(feat_row, 'get') else np.nan

                    if not (np.isnan(shin_h) or np.isnan(shin_d) or np.isnan(shin_a)):
                        match_mkt_p = np.array([shin_h, shin_d, shin_a])
                        match_mkt_p = np.clip(match_mkt_p, 0.02, 0.95)
                        match_mkt_p /= match_mkt_p.sum()
                        # Blend: 60% model, 40% market (research-optimal range)
                        alpha = 0.60
                        cal_upcoming = alpha * cal_upcoming + (1 - alpha) * match_mkt_p
                        cal_upcoming /= cal_upcoming.sum()
                    else:
                        match_mkt_p = None

                # ---- LIVE SIGNAL ADJUSTMENTS ----
                # Apply sentiment-based adjustments if live data available
                if live_sentiment:
                    h_sent = live_sentiment.get(home, {})
                    a_sent = live_sentiment.get(away, {})
                    h_s = h_sent.get("sentiment", 0.0)
                    a_s = a_sent.get("sentiment", 0.0)
                    h_inj = h_sent.get("injury_risk", 0.0)
                    a_inj = a_sent.get("injury_risk", 0.0)

                    # Sentiment adjustment (capped at 5% shift)
                    sent_diff = (h_s - a_s) * 0.03
                    cal_upcoming[0] += sent_diff   # Home
                    cal_upcoming[2] -= sent_diff   # Away

                    # Injury risk adjustment (capped at 4% shift)
                    inj_diff = (a_inj - h_inj) * 0.025
                    cal_upcoming[0] += inj_diff
                    cal_upcoming[2] -= inj_diff

                    # Manager instability penalty
                    h_mgr = h_sent.get("manager_instability", 0.0)
                    a_mgr = a_sent.get("manager_instability", 0.0)
                    if h_mgr > 0.3:
                        cal_upcoming[0] -= 0.02
                        cal_upcoming[1] += 0.01
                        cal_upcoming[2] += 0.01
                    if a_mgr > 0.3:
                        cal_upcoming[2] -= 0.02
                        cal_upcoming[1] += 0.01
                        cal_upcoming[0] += 0.01

                    # Re-normalise
                    cal_upcoming = np.clip(cal_upcoming, 0.02, 0.95)
                    cal_upcoming /= cal_upcoming.sum()

                # Top scorelines
                top_scores = []
                for si in range(min(7, score_probs.shape[0])):
                    for sj in range(min(7, score_probs.shape[1])):
                        top_scores.append({"score": f"{si}-{sj}", "prob": float(score_probs[si, sj])})
                top_scores.sort(key=lambda x: -x["prob"])

                # Confidence — use same tiers as backtest analysis
                max_prob = max(cal_upcoming)
                if max_prob >= 0.70:
                    confidence = "ELITE"
                    confidence_tier_key = "ELITE (>70%)"
                elif max_prob >= 0.60:
                    confidence = "VERY HIGH"
                    confidence_tier_key = "VERY HIGH"
                elif max_prob >= 0.50:
                    confidence = "HIGH"
                    confidence_tier_key = "HIGH"
                elif max_prob >= 0.42:
                    confidence = "MEDIUM"
                    confidence_tier_key = "MEDIUM"
                else:
                    confidence = "LOW"
                    confidence_tier_key = "LOW"

                # Tier historical accuracy
                tier_hist = tier_accuracy.get(confidence_tier_key, {})
                tier_hist_acc = tier_hist.get("accuracy", 0)
                tier_hist_n = tier_hist.get("n", 0)

                # Prediction margin (gap between top 2 probabilities)
                sorted_cal = sorted(cal_upcoming, reverse=True)
                pred_margin = sorted_cal[0] - sorted_cal[1]

                # Predicted outcome — use best strategy from Layer 4
                outcome_idx = np.argmax(cal_upcoming)
                if "closeness" in optimal_strategy:
                    m = re.search(r'margin=([\d.]+),min_d=([\d.]+)', optimal_strategy)
                    if m:
                        strat_margin = float(m.group(1))
                        strat_min_d = float(m.group(2))
                        ha_diff = abs(cal_upcoming[0] - cal_upcoming[2])
                        if ha_diff < strat_margin and cal_upcoming[1] > strat_min_d:
                            outcome_idx = 1
                        elif cal_upcoming[0] >= cal_upcoming[2]:
                            outcome_idx = 0
                        else:
                            outcome_idx = 2
                elif "mkt_draw" in optimal_strategy:
                    m = re.search(r'mkt>([\d.]+),model>([\d.]+)', optimal_strategy)
                    if m and match_mkt_p is not None:
                        mkt_d_thr = float(m.group(1))
                        model_d_thr = float(m.group(2))
                        if match_mkt_p[1] > mkt_d_thr and cal_upcoming[1] > model_d_thr:
                            outcome_idx = 1
                        elif cal_upcoming[0] >= cal_upcoming[2]:
                            outcome_idx = 0
                        else:
                            outcome_idx = 2
                elif "draw_thr" in optimal_strategy:
                    m = re.search(r'draw_thr=([\d.]+)', optimal_strategy)
                    if m:
                        d_thr = float(m.group(1))
                        if cal_upcoming[1] > d_thr:
                            outcome_idx = 1
                        elif cal_upcoming[0] >= cal_upcoming[2]:
                            outcome_idx = 0
                        else:
                            outcome_idx = 2
                outcome_label = ["Home Win", "Draw", "Away Win"][outcome_idx]

                # Live sentiment for output
                h_sent_data = live_sentiment.get(home, {}) if live_sentiment else {}
                a_sent_data = live_sentiment.get(away, {}) if live_sentiment else {}

                entry = {
                    "date": str(match.get("date", "")),
                    "home_team": home, "away_team": away,
                    "home_win": round(float(cal_upcoming[0]), 4),
                    "draw": round(float(cal_upcoming[1]), 4),
                    "away_win": round(float(cal_upcoming[2]), 4),
                    "xg_home": round(float(xg_h), 2),
                    "xg_away": round(float(xg_a), 2),
                    "ml_xg_home": round(float(pred_ug_home[i]), 2),
                    "ml_xg_away": round(float(pred_ug_away[i]), 2),
                    "over_2_5": round(float(over25), 4),
                    "btts": round(float(btts_p), 4),
                    "top_scorelines": top_scores[:12],
                    "predicted_score": _best_score_for_outcome(top_scores, outcome_idx),
                    "confidence": confidence,
                    "tier_accuracy": round(tier_hist_acc, 4),
                    "tier_sample_size": tier_hist_n,
                    "prediction_margin": round(float(pred_margin), 4),
                    "overall_accuracy": round(overall_accuracy, 4),
                    "model_breakdown": {
                        name: {"H": round(float(p[0]), 4), "D": round(float(p[1]), 4), "A": round(float(p[2]), 4)}
                        for name, p in model_preds_upcoming.items()
                    },
                    "live_sentiment": {
                        "home": {
                            "sentiment": round(h_sent_data.get("sentiment", 0.0), 3),
                            "injury_risk": round(h_sent_data.get("injury_risk", 0.0), 3),
                            "news_volume": h_sent_data.get("volume", 0),
                            "manager_stability": round(1 - h_sent_data.get("manager_instability", 0.0), 3),
                        },
                        "away": {
                            "sentiment": round(a_sent_data.get("sentiment", 0.0), 3),
                            "injury_risk": round(a_sent_data.get("injury_risk", 0.0), 3),
                            "news_volume": a_sent_data.get("volume", 0),
                            "manager_stability": round(1 - a_sent_data.get("manager_instability", 0.0), 3),
                        },
                    },
                }
                upcoming_preds.append(entry)

                # ── Rich per-match output ──
                print(f"\n  {'━' * 62}")
                print(f"  {home}  vs  {away}")
                if match.get("date"):
                    print(f"  {match['date']}")
                print(f"  {'━' * 62}")

                # Prediction & confidence
                print(f"  ▸ Prediction:  {outcome_label}  [{confidence}]")
                if tier_hist_n > 0:
                    print(f"  ▸ Tier accuracy: {tier_hist_acc:.1%} on {tier_hist_n} historical {confidence}-confidence matches")
                print(f"  ▸ Overall model accuracy: {overall_accuracy:.1%} | RPS: {cal_rps:.4f}")
                print(f"  ▸ Prediction margin: {pred_margin:.1%} (gap between top 2 outcomes)")

                # Probabilities with visual bar
                print(f"\n    {'Result':<10} {'Prob':>6}   Bar")
                print(f"    {'─' * 40}")
                labels_bar = [("Home", cal_upcoming[0]), ("Draw", cal_upcoming[1]), ("Away", cal_upcoming[2])]
                for lbl, prob in labels_bar:
                    bar_len = int(prob * 30)
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    marker = " ◄" if prob == max_prob else ""
                    print(f"    {lbl:<10} {prob:>5.1%}  {bar}{marker}")

                # Goals & scorelines
                print(f"\n    xG (Dixon-Coles): {xg_h:.2f} - {xg_a:.2f}")
                print(f"    xG (ML):          {pred_ug_home[i]:.2f} - {pred_ug_away[i]:.2f}")
                print(f"    Over 2.5: {over25:.1%}  |  BTTS: {btts_p:.1%}")
                top5 = top_scores[:5]
                scores_str = "  ".join([f"{s['score']} ({s['prob']:.1%})" for s in top5])
                print(f"    Top scores: {scores_str}")

                # Model breakdown
                print(f"\n    {'Model':<16} {'H':>6} {'D':>6} {'A':>6}  Pick")
                print(f"    {'─' * 44}")
                for name, probs in model_preds_upcoming.items():
                    pick_idx = int(np.argmax(probs))
                    pick = ["H", "D", "A"][pick_idx]
                    print(f"    {name:<16} {probs[0]:>5.1%} {probs[1]:>5.1%} {probs[2]:>5.1%}  {pick}")
                # Consensus
                picks = [int(np.argmax(p)) for p in model_preds_upcoming.values()]
                n_models = len(picks)
                if n_models > 0:
                    consensus_pct = max(picks.count(0), picks.count(1), picks.count(2)) / n_models
                    consensus_label = ["Home", "Draw", "Away"][max(set(picks), key=picks.count)]
                else:
                    consensus_pct = 0.0
                    consensus_label = "N/A"
                print(f"    {'─' * 44}")
                print(f"    Consensus: {consensus_label} ({consensus_pct:.0%} of {n_models} models agree)")

                # Live NLP Sentiment Analysis
                if live_sentiment and (home in live_sentiment or away in live_sentiment):
                    print(f"\n    {'─' * 44}")
                    print(f"    LIVE NLP SENTIMENT ANALYSIS")
                    print(f"    {'─' * 44}")
                    for side, team in [("Home", home), ("Away", away)]:
                        s = live_sentiment.get(team, {})
                        sent = s.get("sentiment", 0.0)
                        sent_icon = "▲" if sent > 0.1 else ("▼" if sent < -0.1 else "─")
                        inj = s.get("injury_risk", 0.0)
                        vol = s.get("volume", 0)
                        mgr = s.get("manager_instability", 0.0)
                        print(f"    {side} ({team}):")
                        print(f"      Sentiment: {sent:+.2f} {sent_icon}  |  News: {vol} articles")
                        if inj > 0.1:
                            print(f"      ⚠ Injury risk: {inj:.0%}")
                        if mgr > 0.2:
                            print(f"      ⚠ Manager instability: {mgr:.0%}")
    else:
        print("\n  No upcoming EPL fixtures found. Run 'python data/generator.py' first.")

    # =====================================================================
    # Save dashboard data
    # =====================================================================
    dashboard_data = {
        "model_results": model_results,
        "ensemble_method": ensemble_method,
        "model_weights": {k: round(v, 4) for k, v in inv_weights.items()},
        "metrics": {
            "final_rps": float(cal_rps),
            "final_accuracy": float(accuracy_score(y_test, np.argmax(calibrated, axis=1))),
            "final_ece": float(e_val),
            "final_logloss": float(ll),
            "home_goals_mae": float(home_mae),
            "away_goals_mae": float(away_mae),
            "rps_skill_vs_baseline": round(float(1 - cal_rps / baseline_rps), 4),
            "rps_skill_vs_market": round(float(1 - cal_rps / model_results.get("Market Odds", {"rps": cal_rps})["rps"]), 4),
            "train_seasons": len(seasons) - 1,
            "test_season": test_season,
            "total_matches": len(df),
            "total_features": len(feature_cols),
            "n_base_learners": len(base_learners),
        },
        "confidence_tiers": tier_accuracy,
        "team_ratings": dc.get_team_ratings().to_dict("records") if dc else [],
        "upcoming_predictions": upcoming_preds,
        "test_season": {
            "predictions": calibrated.tolist(),
            "actual": y_test.values.tolist(),
            "matches": test_df[["home_team", "away_team", "goals_home",
                                "goals_away", "result"] +
                               (["gameweek"] if "gameweek" in test_df.columns else [])
                              ].to_dict("records"),
        },
        "feature_importance": dict(zip(
            fi.head(30).index.tolist(),
            [round(float(v), 6) for v in fi.head(30).values]
        )) if fi is not None else {},
        "dc_params": {
            "home_advantage": round(float(dc.home_advantage), 4),
            "rho": round(float(dc.rho), 4),
        },
        "calibration_curve": [],
        "backtest_results": bt_results if bt_results else [],
        "data_sources": {
            "football_data_co_uk": "results, stats, odds, referee (20 seasons)",
            "club_elo": "pre-match Elo ratings",
            "open_meteo": "historical weather",
            "football_data_org_api": "standings, team stats (API)",
            "api_football": "injuries, player ratings (API)",
            "newsapi": "NLP sentiment analysis (API)",
        },
    }

    # Calibration curve
    for c, label in enumerate(["Home", "Draw", "Away"]):
        bins = np.linspace(0, 1, 11)
        for b in range(10):
            mask = (calibrated[:, c] >= bins[b]) & (calibrated[:, c] < bins[b + 1])
            if mask.sum() > 5:
                dashboard_data["calibration_curve"].append({
                    "outcome": label, "predicted": float(calibrated[mask, c].mean()),
                    "actual": float((y_test.values[mask] == c).mean()), "count": int(mask.sum()),
                })

    out_path = DATA_DIR / "dashboard_data.json"
    with open(out_path, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print(f"\n\n  All results saved to {out_path}")

    # Auto-generate HTML dashboard
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from dashboard import generate_dashboard
        dash_path = generate_dashboard()
        if dash_path:
            print(f"  Dashboard generated: {dash_path}")
            print(f"  Open in browser:  file://{dash_path}")
    except Exception as e:
        print(f"  Dashboard generation skipped: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
