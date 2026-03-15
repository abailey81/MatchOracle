<div align="center">

# MatchOracle

### Deep Ensemble EPL Prediction Engine

*13 base learners · Dixon-Coles statistical model · 376+ features · 8 data sources · NLP sentiment analysis*

<br>

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/sklearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-2980B9?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/MIT-green?style=for-the-badge)](LICENSE)

<br>

[![GitHub stars](https://img.shields.io/github/stars/abailey81/MatchOracle?style=social)](https://github.com/abailey81/MatchOracle/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/abailey81/MatchOracle?style=social)](https://github.com/abailey81/MatchOracle/network/members)

---

**An advanced EPL match prediction system** built on real data from 8 sources spanning 20 seasons (~7,600 matches). Features a 5-layer deep stacking ensemble with walk-forward backtesting that consistently outperforms market odds.

<br>

| Metric | Result | vs Market |
|:-------|:------:|:---------:|
| **3-Way Accuracy** | 60.2% | +4.6% (55.6%) |
| **RPS Score** | 0.1163 | +11.2% skill |
| **ELITE Tier (>70%)** | 82.2% acc | 90 matches |
| **Best Season** | 66.8% | 2023-24 |

<br>

[Ensemble Architecture](#ensemble-architecture) · [Features](#feature-engineering) · [Data Sources](#data-sources) · [Results](#walk-forward-results) · [Getting Started](#getting-started)

</div>

<br>

## Highlights

<table>
<tr>
<td width="50%">

**5-Layer Ensemble Architecture**
- Layer 0: Dixon-Coles statistical model
- Layer 1: 13 base learners (HGB, XGB, LightGBM, CatBoost, RF, MLP)
- Layer 2: 4 meta-learners with isotonic calibration
- Layer 2.5: Binary classifier boosting
- Layer 3: Best ensemble selection (stacking vs weighted avg vs market-fused)

</td>
<td width="50%">

**376+ Engineered Features**
- Elo + Glicko-2 + Pi-Ratings
- Rolling form (6 windows), H2H, momentum
- Market intelligence (Shin probabilities, odds movement)
- Poisson goal decomposition, xG-based metrics
- Manager tenure, GK quality, rest days, scoring patterns

</td>
</tr>
<tr>
<td width="50%">

**NLP Sentiment Analysis**
- RoBERTa transformer (70%) + keyword fallback (30%)
- Dual news sources: NewsAPI + Google News RSS
- Aspect-based: injury risk, morale, tactical disruption
- 30 live sentiment features injected at prediction time

</td>
<td width="50%">

**Smart Model Caching**
- 5 automated retraining checks (data hash, age, integrity)
- <2 second predictions after initial training
- Interactive CLI with arrow-key fixture selector
- Auto-generated HTML dashboard with Plotly/Chart.js

</td>
</tr>
</table>

---

## Ensemble Architecture

```
Layer 0: Dixon-Coles (goals + xG variants)
    │
Layer 1: 13 Base Learners
    │   HGB, HGB-Agg, HGB-Deep, XGBoost, LightGBM, CatBoost,
    │   Random Forest, Extra Trees, DeepMLP, MLP-Wide,
    │   Logistic Regression, Bagging-HGB, Vote-HGB3
    │
Layer 2: 4 Meta-Learners (Meta-LR, Meta-MLP, Meta-HGB, Meta-XGB)
    │
Layer 2.5: Binary Classifier Boosting (4 dedicated HGB models)
    │
Layer 3: Best Ensemble Selection
        Stacking vs Weighted Avg vs Binary-3Way vs Market-Fused
```

---

## Data Sources

| # | Source | Coverage | Key Data |
|:-:|:-------|:---------|:---------|
| 1 | football-data.co.uk | 20 seasons | Results, shots, corners, bookmaker odds |
| 2 | Understat | 11 seasons (2014+) | Match-level xG, xGA |
| 3 | Club Elo | 20 seasons | Historical Elo ratings |
| 4 | Open-Meteo | 20 seasons | Weather at stadium GPS |
| 5 | Football-Data.org | Live season | Standings, fixtures, H2H |
| 6 | API-Football | Live season | Injuries, player ratings |
| 7 | NewsAPI | Live | Team news for NLP sentiment |
| 8 | Google News RSS | Live | Fallback news source |

---

## Walk-Forward Results

5-season walk-forward backtesting (no future data leakage):

| Season | Accuracy | RPS | vs Market |
|:-------|:--------:|:---:|:---------:|
| 2020-21 | 55.0% | 0.1314 | — |
| 2021-22 | 61.8% | 0.1080 | +17.6% |
| 2022-23 | 60.0% | 0.1189 | +9.2% |
| 2023-24 | **66.8%** | **0.1040** | +20.6% |
| 2024-25 | 57.1% | 0.1194 | +8.9% |

<details>
<summary><b>Confidence Tier Breakdown</b></summary>

| Tier | Confidence | Accuracy | Matches |
|:-----|:----------:|:--------:|:-------:|
| ELITE | >70% | 82.2% | 90 |
| VERY HIGH | 60-70% | 66.7% | 96 |
| HIGH | 50-60% | 62.7% | 83 |

</details>

<details>
<summary><b>376+ Feature Groups</b></summary>

| Group | Features | Description |
|:------|:--------:|:------------|
| Elo Ratings | 6 | Home/away Elo + differential |
| Pi-Ratings | 8 | Home/away attack/defense ratings |
| Rolling Form | 40+ | 6 rolling windows (3-20 matches) |
| Head-to-Head | 20+ | Historical H2H results and trends |
| Market Intelligence | 15+ | Implied probabilities, odds movement, Shin |
| Momentum | 25+ | Streaks, acceleration, velocity |
| Glicko-2 | 10 | Rating + uncertainty + volatility |
| Poisson | 15+ | Attack/defense decomposition |
| Contextual | 10 | Derby flags, distance, title contender |
| Injuries | 12 | Per-team injury impact |
| Manager | 6 | Tenure, new manager bounce |
| GK Quality | 6 | Clean sheets, consistency |
| Sequence Patterns | 12 | Encoded result sequences |
| Scoring Patterns | 8 | Early goals, comebacks |
| Rest Days | 6 | Fatigue/freshness flags |

</details>

---

## Getting Started

```bash
git clone https://github.com/abailey81/MatchOracle.git
cd MatchOracle

# Setup
chmod +x setup.sh && ./setup.sh
# Or manually:
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env  # Edit with your keys

# Run predictions
python predict.py --fdo-key YOUR_KEY --apif-key YOUR_KEY --news-key YOUR_KEY
```

---

## Project Structure

```
MatchOracle/
├── predict.py                  # Interactive CLI entry point
├── dashboard.py                # HTML dashboard generator
├── data/
│   ├── generator.py            # Real data pipeline (8 sources, 20 seasons)
│   └── api_client.py           # Rate limiter, circuit breaker, caching
├── features/
│   ├── engine.py               # 376+ features across 24 groups
│   └── sentiment.py            # RoBERTa NLP sentiment analysis
├── models/
│   ├── run_pipeline.py         # 5-layer ensemble pipeline
│   ├── dixon_coles.py          # Statistical model (1997)
│   └── model_cache.py          # Smart caching with retraining detection
├── requirements.txt
├── setup.sh
└── .env.example
```

---

<div align="center">

**[MIT License](LICENSE)**

Built with scikit-learn, XGBoost, LightGBM, CatBoost, and RoBERTa

</div>
