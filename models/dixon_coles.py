"""
MatchOracle: Dixon-Coles Model (1997)
The gold-standard statistical model for football score prediction.
Includes the original model plus Bivariate Poisson and xG-based extensions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class DixonColesModel:
    """
    Full Dixon-Coles model with:
    - Team attack/defense parameters
    - Home advantage
    - Rho correction for low-scoring outcomes  
    - Time-decay weighting
    - xG-based variant
    """
    
    def __init__(self, xi: float = 0.002, use_xg: bool = False):
        self.xi = xi  # Time decay rate
        self.use_xg = use_xg
        self.params = None
        self.teams = None
        self.team_to_idx = None
        
    def _dc_tau(self, x: int, y: int, lambda_h: float, lambda_a: float, rho: float) -> float:
        """Dixon-Coles correction factor for low-scoring outcomes."""
        if x == 0 and y == 0:
            return 1 - lambda_h * lambda_a * rho
        elif x == 0 and y == 1:
            return 1 + lambda_h * rho
        elif x == 1 and y == 0:
            return 1 + lambda_a * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _neg_log_likelihood(self, params: np.ndarray, 
                            home_idx: np.ndarray, away_idx: np.ndarray,
                            home_goals: np.ndarray, away_goals: np.ndarray,
                            weights: np.ndarray) -> float:
        """Negative log-likelihood for Dixon-Coles model (vectorized)."""
        n_teams = len(self.teams)
        
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]
        mu = params[2*n_teams + 2]
        
        lambda_h = np.exp(mu + attack[home_idx] - defense[away_idx] + home_adv)
        lambda_a = np.exp(mu + attack[away_idx] - defense[home_idx])
        
        # Clip for numerical stability
        lambda_h = np.clip(lambda_h, 0.01, 15)
        lambda_a = np.clip(lambda_a, 0.01, 15)
        
        if self.use_xg:
            # For xG, round to nearest integer for Poisson
            gh = np.round(home_goals).astype(int)
            ga = np.round(away_goals).astype(int)
        else:
            gh = home_goals.astype(int)
            ga = away_goals.astype(int)
        
        # Poisson log-probs (vectorized)
        log_p_home = poisson.logpmf(gh, lambda_h)
        log_p_away = poisson.logpmf(ga, lambda_a)
        
        # Dixon-Coles tau correction (vectorized)
        tau = np.ones(len(gh))
        mask_00 = (gh == 0) & (ga == 0)
        mask_01 = (gh == 0) & (ga == 1)
        mask_10 = (gh == 1) & (ga == 0)
        mask_11 = (gh == 1) & (ga == 1)
        
        tau[mask_00] = 1 - lambda_h[mask_00] * lambda_a[mask_00] * rho
        tau[mask_01] = 1 + lambda_h[mask_01] * rho
        tau[mask_10] = 1 + lambda_a[mask_10] * rho
        tau[mask_11] = 1 - rho
        
        tau = np.clip(tau, 1e-10, None)
        log_tau = np.log(tau)
        
        ll = np.sum(weights * (log_p_home + log_p_away + log_tau))
        
        penalty = 100 * (np.sum(attack) ** 2 + np.sum(defense) ** 2)
        
        return -ll + penalty
    
    def fit(self, df: pd.DataFrame, 
            home_col: str = "home_team", away_col: str = "away_team",
            date_col: str = "date") -> "DixonColesModel":
        """Fit the Dixon-Coles model."""
        
        self.teams = sorted(set(df[home_col].unique()) | set(df[away_col].unique()))
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)
        
        # Prepare data
        home_idx = df[home_col].map(self.team_to_idx).values
        away_idx = df[away_col].map(self.team_to_idx).values
        
        if self.use_xg and "xg_home" in df.columns:
            home_goals = df["xg_home"].values
            away_goals = df["xg_away"].values
        else:
            home_goals = df["goals_home"].values
            away_goals = df["goals_away"].values
        
        # Time-decay weights
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
            max_date = dates.max()
            days_ago = (max_date - dates).dt.days.values
            weights = np.exp(-self.xi * days_ago)
        else:
            weights = np.ones(len(df))
        
        # Initial parameters
        x0 = np.zeros(2 * n_teams + 3)
        x0[2 * n_teams] = 0.25   # home advantage
        x0[2 * n_teams + 1] = -0.05  # rho
        x0[2 * n_teams + 2] = 0.2   # mu (base rate)
        
        # Bounds
        bounds = (
            [(-2, 2)] * n_teams +     # attack
            [(-2, 2)] * n_teams +     # defense
            [(0, 1)] +                 # home advantage
            [(-0.5, 0.5)] +           # rho
            [(-1, 1)]                  # mu
        )
        
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-6}
        )
        
        self.params = result.x
        self.attack = {t: self.params[i] for i, t in enumerate(self.teams)}
        self.defense = {t: self.params[n_teams + i] for i, t in enumerate(self.teams)}
        self.home_advantage = self.params[2 * n_teams]
        self.rho = self.params[2 * n_teams + 1]
        self.mu = self.params[2 * n_teams + 2]
        
        return self
    
    def predict_score_probs(self, home_team: str, away_team: str,
                            max_goals: int = 8) -> np.ndarray:
        """Predict scoreline probability distribution.
        Handles unknown teams by using league-average parameters."""
        avg_attack = np.mean(list(self.attack.values()))
        avg_defense = np.mean(list(self.defense.values()))

        h_att = self.attack.get(home_team, avg_attack)
        h_def = self.defense.get(home_team, avg_defense)
        a_att = self.attack.get(away_team, avg_attack)
        a_def = self.defense.get(away_team, avg_defense)

        lambda_h = np.exp(self.mu + h_att - a_def + self.home_advantage)
        lambda_a = np.exp(self.mu + a_att - h_def)
        
        probs = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
                tau = self._dc_tau(i, j, lambda_h, lambda_a, self.rho)
                probs[i, j] = p * tau
        
        # Normalize
        probs /= probs.sum()
        return probs
    
    def predict_outcome(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Predict match outcome probabilities (H/D/A)."""
        probs = self.predict_score_probs(home_team, away_team)
        
        p_home = sum(probs[i, j] for i in range(8) for j in range(8) if i > j)
        p_draw = sum(probs[i, i] for i in range(8))
        p_away = sum(probs[i, j] for i in range(8) for j in range(8) if i < j)
        
        total = p_home + p_draw + p_away
        return {
            "home": p_home / total,
            "draw": p_draw / total,
            "away": p_away / total,
        }
    
    def predict_expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Predict expected goals for each team."""
        avg_attack = np.mean(list(self.attack.values()))
        avg_defense = np.mean(list(self.defense.values()))
        lambda_h = np.exp(
            self.mu + self.attack.get(home_team, avg_attack)
            - self.defense.get(away_team, avg_defense) + self.home_advantage
        )
        lambda_a = np.exp(
            self.mu + self.attack.get(away_team, avg_attack)
            - self.defense.get(home_team, avg_defense)
        )
        return lambda_h, lambda_a
    
    def predict_over_under(self, home_team: str, away_team: str, 
                           line: float = 2.5) -> float:
        """Predict probability of over `line` total goals."""
        probs = self.predict_score_probs(home_team, away_team)
        over = sum(probs[i, j] for i in range(8) for j in range(8) if i + j > line)
        return over
    
    def predict_btts(self, home_team: str, away_team: str) -> float:
        """Predict probability of both teams to score."""
        probs = self.predict_score_probs(home_team, away_team)
        btts = sum(probs[i, j] for i in range(1, 8) for j in range(1, 8))
        return btts
    
    def get_team_ratings(self) -> pd.DataFrame:
        """Return team ratings as a DataFrame."""
        data = []
        for team in self.teams:
            lambda_h = np.exp(self.mu + self.attack[team] + self.home_advantage)
            lambda_a = np.exp(self.mu + self.attack[team])
            data.append({
                "team": team,
                "attack": round(self.attack[team], 4),
                "defense": round(self.defense[team], 4),
                "attack_rank": 0,
                "defense_rank": 0,
                "expected_home_goals": round(lambda_h, 2),
                "expected_away_goals": round(lambda_a, 2),
            })
        
        df = pd.DataFrame(data)
        df["attack_rank"] = df["attack"].rank(ascending=False).astype(int)
        df["defense_rank"] = df["defense"].rank(ascending=True).astype(int)  # Lower = better
        return df.sort_values("attack", ascending=False)


class BivariatePoissonModel:
    """Bivariate Poisson model (Karlis & Ntzoufras 2003) with direct goal correlation."""
    
    def __init__(self, xi: float = 0.002):
        self.xi = xi
        self.params = None
        
    def fit(self, df: pd.DataFrame) -> "BivariatePoissonModel":
        """Fit bivariate Poisson model."""
        teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        self.teams = teams
        self.team_to_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        
        # Simplified: use per-team average goals as attack/defense proxy
        team_stats = {}
        for team in teams:
            home_games = df[df["home_team"] == team]
            away_games = df[df["away_team"] == team]
            
            goals_for = (home_games["goals_home"].sum() + away_games["goals_away"].sum())
            goals_against = (home_games["goals_away"].sum() + away_games["goals_home"].sum())
            games = len(home_games) + len(away_games)
            
            team_stats[team] = {
                "attack": goals_for / max(games, 1),
                "defense": goals_against / max(games, 1),
            }
        
        self.team_stats = team_stats
        league_avg = df["total_goals"].mean() / 2
        self.league_avg = league_avg
        
        # Estimate correlation parameter
        corr = df["goals_home"].corr(df["goals_away"])
        self.lambda3 = max(0, corr * 0.3)  # Correlation component
        
        return self
    
    def predict_outcome(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Predict using bivariate Poisson."""
        ha = self.team_stats[home_team]
        aa = self.team_stats[away_team]
        
        lambda1 = ha["attack"] * (self.league_avg / aa["defense"]) * 1.1 - self.lambda3
        lambda2 = aa["attack"] * (self.league_avg / ha["defense"]) * 0.9 - self.lambda3
        lambda1 = max(0.3, lambda1)
        lambda2 = max(0.2, lambda2)
        
        p_home, p_draw, p_away = 0, 0, 0
        for i in range(8):
            for j in range(8):
                # Approximate bivariate Poisson
                p = 0
                for k in range(min(i, j) + 1):
                    p += (poisson.pmf(i - k, lambda1) * 
                          poisson.pmf(j - k, lambda2) * 
                          poisson.pmf(k, self.lambda3))
                
                if i > j: p_home += p
                elif i == j: p_draw += p
                else: p_away += p
        
        total = p_home + p_draw + p_away
        return {"home": p_home/total, "draw": p_draw/total, "away": p_away/total}


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "epl_matches.parquet", engine="pyarrow")

    # Fit on all but last season, test on last
    seasons = sorted(df["season"].unique())
    train = df[df["season"] != seasons[-1]]
    test = df[df["season"] == seasons[-1]]
    
    # Standard Dixon-Coles
    dc = DixonColesModel(xi=0.002)
    dc.fit(train)
    
    print("=== DIXON-COLES MODEL ===")
    print(f"\nHome advantage: {dc.home_advantage:.4f}")
    print(f"Rho (correlation): {dc.rho:.4f}")
    print(f"\nTeam Ratings:")
    print(dc.get_team_ratings().to_string(index=False))
    
    # Test predictions
    print("\n\n=== SAMPLE PREDICTIONS ===")
    test_matches = [
        ("Arsenal", "Manchester City"),
        ("Liverpool", "Chelsea"),
        ("Manchester United", "Tottenham"),
        ("Southampton", "Manchester City"),
        ("Arsenal", "Southampton"),
    ]
    
    for home, away in test_matches:
        pred = dc.predict_outcome(home, away)
        xg_h, xg_a = dc.predict_expected_goals(home, away)
        over25 = dc.predict_over_under(home, away)
        btts = dc.predict_btts(home, away)
        
        print(f"\n{home} vs {away}")
        print(f"  H: {pred['home']:.1%} | D: {pred['draw']:.1%} | A: {pred['away']:.1%}")
        print(f"  xG: {xg_h:.2f} - {xg_a:.2f}")
        print(f"  Over 2.5: {over25:.1%} | BTTS: {btts:.1%}")
    
    # xG-based Dixon-Coles (only if xG data exists)
    if "xg_home" in train.columns and train["xg_home"].notna().any():
        dc_xg = DixonColesModel(xi=0.002, use_xg=True)
        dc_xg.fit(train)
        print("\n\n=== XG-BASED DIXON-COLES ===")
        for home, away in test_matches[:3]:
            pred = dc_xg.predict_outcome(home, away)
            print(f"{home} vs {away}: H={pred['home']:.1%} D={pred['draw']:.1%} A={pred['away']:.1%}")
