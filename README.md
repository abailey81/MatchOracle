# MatchOracle — Deep Ensemble EPL Prediction Engine

An advanced EPL match prediction system built on **real data** from **8 sources** spanning **20 seasons** (2005-2025, ~7,600 matches). Features a multi-layer deep stacking ensemble with **13 base learners**, **4 meta-learners**, binary classifier boosting, hierarchical 2-stage classification, market-model fusion, real-time NLP sentiment analysis, **Glicko-2 ratings**, **Shin probabilities**, **Poisson goal models**, live season data integration, and Monte Carlo validation.

**Proven Results (walk-forward backtesting across 5 seasons):**
- **60.2% 3-way accuracy** (vs market 55.6% — +4.6% edge)
- **0.1163 RPS** (vs market 0.1310 — +11.2% skill)
- **86.0% accuracy** on top 15% most confident predictions
- **82.2% ELITE tier** accuracy (90 matches at >70% confidence)
- **77.6% Home Win** binary accuracy
- **77.6% Away or Draw** binary accuracy
- **74.5% average** across 8 binary targets

**Key Features:**
- 376+ engineered features across 24 feature groups (Apache Parquet storage)
- 13 quality-controlled ML base learners + 4 meta-learners
- (1/RPS)^4 sharpened ensemble weighting with quality gate
- Prediction clipping prevents degenerate model outputs
- Market-model fusion with optimal blend ratio search
- Binary-classifier-informed 3-way probability construction
- Hierarchical 2-stage classifier (Draw vs Non-Draw → H vs A)
- Threshold-optimized binary predictions (per-target optimal threshold)
- Smart model caching with 5-check retraining detection
- Real-time NLP sentiment with dynamic player impact detection
- Current season walk-forward backtesting (most critical validation)
- Adversarial validation + feature importance selection
- Season-boosted sample weighting (current season 12x, last 5x)

---

## System Architecture

```
Layer 0: Statistical Foundation
  Dixon-Coles (1997) — goals-based + xG-based variants
                          ↓
Layer 1: ML Base Learners (13 quality-controlled models)
  HGB | HGB-Agg | HGB-Deep | XGBoost | LightGBM | CatBoost
  RF | Extra Trees | DeepMLP | MLP-Wide | LR | Bagging-HGB | Vote-HGB3
  → Adaptive DC-ML blending with disagreement detection
  → Prediction clipping (0.02–0.96) prevents degenerate outputs
                          ↓
Layer 2: Meta-Learner Ensemble (4 meta-learners)
  Meta-LR | Meta-MLP | Meta-HGB | Meta-XGB → (1/RPS)^4 weighted average
                          ↓
Layer 2.5: Binary Classifier Boosting
  4 binary HGBs (Home Win | Away Win | Draw | Home-or-Draw) → augment meta-features
                          ↓
Layer 2.7: Binary-Informed 3-Way Construction
  Binary classifier probs → normalised 3-way probabilities
                          ↓
Layer 2.8: Market-Model Fusion
  Optimal α search (0.50–0.75) blending model with Shin-adjusted market odds
                          ↓
Layer 3: Best Ensemble Selection
  Compares Stacking | Weighted Avg | Binary-3Way | Market-Fused | Bin+Stack
  → Picks lowest RPS automatically
                          ↓
Layer 3.5: Isotonic Calibration + Confidence
  Isotonic calibration (auto-skip if degrading) → confidence-stratified tiers
                          ↓
Layer 4: Hierarchical 2-Stage Classifier
  Stage 1: Draw vs Non-Draw (ensemble of HGB+XGB+LGB)
  Stage 2: Home vs Away given Non-Draw (ensemble of HGB+XGB+LGB)
  → Blended with main ensemble at optimal ratio
                          ↓
Live Layer: Real-Time Signal Integration
  Live season stats (Football-Data.org API) | Google News RSS sentiment
  Player impact | Injury detection | Market blending (Shin)
  Live xG estimation from current-season goals | Promoted team handling
```

## Data Sources (8)

| # | Source | Data | Coverage | Key Required |
|---|--------|------|----------|-------------|
| 1 | **football-data.co.uk** | Results, shots, SOT, corners, fouls, cards, referee, 10+ bookmaker odds | 20 seasons | No |
| 2 | **Understat** | Match-level xG, xGA | 11 seasons (2014+) | No |
| 3 | **Club Elo** | Historical team Elo ratings | 20 seasons | No |
| 4 | **Open-Meteo** | Temperature, humidity, wind, precipitation at stadium GPS | 20 seasons | No |
| 5 | **Football-Data.org API** | Live standings, fixtures, H2H, top scorers | Current season | Yes |
| 6 | **API-Football** | Injuries, player ratings, team statistics, predictions | Current season | Yes |
| 7 | **NewsAPI** | Team news for NLP sentiment analysis | Current | Yes |
| 8 | **Google News RSS** | Fallback news source (no API key needed) | Current | No |

## 376+ Engineered Features (24 groups)

| # | Feature Group | Count | Description |
|---|---------------|-------|-------------|
| 1 | **Elo Ratings** | 6 | Dynamic K=20, goal-margin scaled, season regression |
| 2 | **Pi-Ratings** | 8 | Constantinou & Fenton (2013), home/away attack/defense |
| 3 | **Rolling Form** | 40+ | 6 windows x 17 metrics x 2 teams |
| 4 | **Head-to-Head** | 20+ | Last 10 meetings, venue-aware, goals volatility, BTTS |
| 5 | **Contextual** | 10 | Day of week, month, distance, derby, title contender |
| 6 | **Market Intelligence** | 15+ | Implied probs, overround, odds movement, steam moves |
| 7 | **Interactions** | 10+ | Elo x rest, form x home, rating concordance |
| 8 | **League Table** | 15+ | Position, points, GD, motivation proxies |
| 9 | **Advanced Stats** | 10+ | xG-based metrics, shot quality |
| 10 | **Injuries** | 6 | Injury burden per team, differential |
| 11 | **Momentum** | 25+ | Streaks, acceleration, velocity, congestion, xG trend |
| 12 | **Surprise Factor** | 4 | Upset tendency per team |
| 13 | **Style Matchup** | 6 | Attack vs defense, form-elo divergence |
| 14 | **Weather** | 8 | Temperature, wind, rain at stadium GPS |
| 15 | **Glicko-2 Ratings** | 10 | Rating + uncertainty (RD) + volatility, confidence |
| 16 | **Shin Probabilities** | 8 | True odds from bookmaker margins (Shin 1993) |
| 17 | **Sequence Patterns** | 12 | Last 5 results encoded, pattern encoding, consistency |
| 18 | **Manager Features** | 6 | Tenure tracking, new manager bounce effect |
| 19 | **GK Quality** | 6 | Clean sheet patterns, consistency, streaks |
| 20 | **Poisson Features** | 15+ | Attack/defense decomposition, Poisson H/D/A/O2.5/BTTS |
| 21 | **Expected Points** | 6 | xG-based expected points vs actual (luck indicator) |
| 22 | **Rest Days** | 6 | Days between matches, fatigue/freshness flags |
| 23 | **Scoring Patterns** | 8 | Early goals, comeback rate, clean sheet rate |
| 24 | **League Position** | 6 | Position gap, top6/bottom3 flags, big match indicator |

## NLP Sentiment Engine

The sentiment engine operates at prediction time using **real-time news data**:

- **Dual news sources** — NewsAPI (primary) with Google News RSS fallback when rate-limited
- **Transformer model** — RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) with keyword fallback
- **Blended scoring** — 70% transformer + 30% keyword when transformer available
- **Dynamic player impact** — automatically extracts player names from articles (no hardcoded lists) and detects injury/return context
- **Aspect-based analysis** — injury risk, manager instability, morale, tactical disruption, key player concern, transfer activity
- **Temporal weighting** — more recent articles weighted higher (decay over 5 days)
- **30 live sentiment features** injected at prediction time

## Smart Model Caching

The system intelligently detects when retraining is needed through **5 automated checks**:

1. **Cache existence** — first run always trains
2. **Data hash** — detects if training data content changed
3. **Source file modification** — checks if CSV/JSON data files were updated
4. **Model age** — configurable max age (default 48h)
5. **File integrity** — size and corruption checks

When cache is valid: predictions run in **< 2 seconds** (vs 30-60 min training).

## Training Pipeline Details

### Ensemble Weighting — (1/RPS)^4

Unlike simple inverse-RPS weighting (which gives nearly equal weight to all models), MatchOracle uses **(1/RPS)^4** sharpened weighting that aggressively concentrates weight on the best-performing models. Combined with a **quality gate** that zeros out any model performing >50% worse than the best, this ensures degenerate or weak models cannot drag down the ensemble.

### Prediction Clipping

All model predictions are clipped to **[0.02, 0.96]** range and re-normalised. This prevents individual models from producing degenerate outputs (e.g., 0%/100%/0%) that would corrupt the ensemble.

### Market-Model Fusion

The pipeline automatically searches for the **optimal blend ratio** between model predictions and Shin-adjusted market odds (testing α from 0.50 to 0.75). Market odds encode substantial information from bookmaker models, and blending with our ML model often produces better-calibrated probabilities than either source alone.

### Binary-Informed 3-Way Construction

Instead of relying solely on 3-class classification, the system constructs 3-way probabilities from dedicated **binary classifiers** (Home Win, Draw, Away Win) and normalises them. This can outperform direct 3-class models because binary classifiers have higher individual accuracy.

### Hierarchical 2-Stage Classifier

A novel approach that decomposes the 3-way problem:
- **Stage 1**: Draw vs Non-Draw (ensemble of HGB + XGB + LGB)
- **Stage 2**: Home vs Away given Non-Draw (ensemble of HGB + XGB + LGB)
- Constructs 3-way via: P(H) = P(Non-Draw) × P(H|Non-Draw)

### Season-Boosted Sample Weighting
- Current season: **12x** weight
- Last season: **5x** weight
- Two seasons ago: **2.5x** weight
- Three seasons ago: **1.2x** weight
- Older: rapid exponential decay (max 0.1)

### Adversarial Validation
Detects train/test distribution shift. When significant shift detected (AV accuracy > 0.65), applies importance weights to upweight training samples that look like test data.

### Feature Selection
Random Forest-based importance scoring. Bottom 5% noise features pruned (live_ features always retained).

### Threshold Optimization
Each binary target uses an **optimal decision threshold** found by searching [0.30, 0.70] to maximise accuracy, rather than using the default 0.50 cutoff.

## Multi-Target Predictions (8 binary targets)

Each target uses an **ensemble of 3 models** (HGB + XGBoost + LightGBM) with **threshold optimization**:

| Target | Description | Accuracy | Precision | Recall | F1 |
|--------|-------------|----------|-----------|--------|----|
| Home Win | H vs not-H | **77.6%** | 71.3% | 75.5% | 0.734 |
| Away Win | A vs not-A | **75.3%** | 75.0% | 43.2% | 0.548 |
| Draw | D vs not-D | **75.5%** | — | — | — |
| Home or Draw | Double chance H/D | **75.0%** | 75.1% | 92.3% | 0.828 |
| Away or Draw | Double chance A/D | **77.6%** | 81.5% | 80.4% | 0.810 |
| Over 2.5 Goals | Total goals > 2.5 | **65.5%** | 65.9% | 80.9% | 0.727 |
| BTTS | Both teams score | **68.9%** | 67.0% | 90.4% | 0.770 |
| Home >0 Goals | Home scores at least 1 | **80.3%** | 81.8% | 95.2% | 0.880 |

**Average binary accuracy: 74.5%**

## Confidence Tiers

| Tier | Threshold | Accuracy | Matches |
|------|-----------|----------|---------|
| ELITE | >70% | **82.2%** | 90 |
| VERY HIGH | 60-70% | **66.7%** | 96 |
| HIGH | 50-60% | **62.7%** | 83 |
| MEDIUM | 42-50% | **33.9%** | 56 |
| LOW | <42% | **32.7%** | 55 |

**Top 10% most confident: 81.6% accuracy**
**Top 15% most confident: 86.0% accuracy**

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/abailey81/MatchOracle.git
cd MatchOracle
python -m venv venv
source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

### 2. Set API Keys (optional but recommended for live predictions)

```bash
export FDO_KEY="your_football_data_org_key"       # football-data.org
export APIF_KEY="your_api_football_key"            # api-football.com
export NEWS_KEY="your_newsapi_key"                 # newsapi.org
```

Or create a `.env` file:
```
FDO_KEY=your_key
APIF_KEY=your_key
NEWS_KEY=your_key
```

### 3. Run the Full Pipeline

```bash
# Step 1: Fetch all data (first run ~15 min, cached after)
python data/generator.py \
  --fdo-key "$FDO_KEY" \
  --apif-key "$APIF_KEY" \
  --news-key "$NEWS_KEY"

# Step 2: Engineer features (376+ features)
python features/engine.py

# Step 3: Train models + generate predictions
python models/run_pipeline.py
```

### 4. View Dashboard

```bash
python dashboard.py                    # Generate and open locally
python dashboard.py --serve            # Start HTTP server (share with others)
python dashboard.py --serve --port 9000  # Custom port
```

The `--serve` mode starts an HTTP server and shows both local and network URLs. Share the network URL with anyone on the same WiFi/LAN. For internet access, use [ngrok](https://ngrok.com): `ngrok http 8000`.

### Interactive Predictions (Recommended)

```bash
python predict.py --fdo-key "YOUR_KEY" --apif-key "YOUR_KEY" --news-key "YOUR_KEY"
```

On first run, trains all 13 base learners + meta-learners (~30-60 min). After that, uses cached models for instant predictions (< 2 seconds).

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `↑` / `↓` (or `k` / `j`) | Navigate up/down |
| `SPACE` | Toggle match selection |
| `a` | Select all matches |
| `n` | Deselect all |
| `ENTER` | Confirm and start prediction |
| `q` | Quit |

### Prediction Output

Each predicted match shows:
- **Predicted outcome** (Home Win / Draw / Away Win) with confidence tier
- **H / D / A probabilities** (calibrated + market-fused)
- **Confidence tier** (ELITE / VERY HIGH / HIGH / MEDIUM / LOW)
- **Expected Goals** from both Dixon-Coles and ML regression
- **Over/Under 2.5, BTTS probabilities**
- **Top 5 most likely scorelines**
- **Full model breakdown** — all 13 base learners with H/D/A probs
- **Live NLP sentiment analysis** — morale, injuries, player impact
- **Model consensus** — percentage of models agreeing

---

## Validation & Backtesting

| Method | Description |
|--------|-------------|
| **Expanding-window 5-fold CV** | Time-aware, no future leakage |
| **Walk-forward backtesting** | Retrain on seasons 1..N, test on N+1, across 5 seasons |
| **Current season walk-forward** | Per-gameweek chunks, most critical validation |
| **Monte Carlo simulation** | 10,000 random iterations, actual vs expected accuracy |
| **Confidence stratification** | Cumulative top-N accuracy analysis |
| **Adversarial validation** | Train/test distribution shift detection |
| **RPS scoring** | Ranked Probability Score — proper scoring rule for ordinal outcomes |
| **Market-model comparison** | Direct accuracy/RPS comparison against bookmaker implied odds |

### Walk-Forward Results (5 seasons)

| Season | Matches | Acc | RPS | Mkt Acc | Mkt RPS | HiConf Acc |
|--------|---------|-----|-----|---------|---------|------------|
| 2020-21 | 380 | 55.0% | 0.1314 | 51.3% | 0.1424 | 62.0% (274) |
| 2021-22 | 380 | 61.8% | 0.1080 | 57.9% | 0.1259 | 68.4% (307) |
| 2022-23 | 380 | 60.0% | 0.1189 | 55.8% | 0.1328 | 63.1% (301) |
| 2023-24 | 380 | **66.8%** | **0.1040** | 58.9% | 0.1224 | **70.7%** (324) |
| 2024-25 | 380 | 57.1% | 0.1194 | 53.9% | 0.1315 | 62.2% (307) |
| **AVG** | | **60.2%** | **0.1163** | 55.6% | 0.1310 | **65.3%** |

**Model vs Market: +4.6% accuracy, +11.2% RPS skill**

## Project Structure

```
matchoracle/
├── predict.py                 # Interactive predictor (start here)
├── dashboard.py               # HTML dashboard generator
├── data/
│   ├── generator.py           # Real data pipeline (8 sources)
│   ├── api_client.py          # API clients + caching
│   ├── epl_matches.parquet     # Raw match data (generated, columnar)
│   ├── epl_featured.parquet   # Featured dataset (generated, columnar)
│   ├── upcoming_fixtures.parquet # Upcoming matches (generated)
│   └── extra_data.json        # Sentiment + injury data (generated)
├── features/
│   ├── engine.py              # Feature engineering (376+ features, 24 groups)
│   └── sentiment.py           # NLP sentiment (NewsAPI + Google News RSS fallback)
├── models/
│   ├── dixon_coles.py         # Dixon-Coles + Bivariate Poisson
│   ├── run_pipeline.py        # 5-layer deep ensemble pipeline
│   └── model_cache.py         # Smart caching with 5-check retraining detection
├── cache/                     # Trained model cache (auto-generated)
├── .env.example               # API key template
├── requirements.txt
├── setup.sh                   # One-command environment setup
└── README.md
```

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`
- API keys (optional): Football-Data.org, API-Football, NewsAPI (all free tier)
- Optional: `transformers` + `torch` for transformer-based NLP

## Research References

- Dixon & Coles (1997) — Bivariate Poisson with low-score correction
- Constantinou & Fenton (2013) — Pi-ratings for football prediction
- Glickman (1999) — Glicko-2 rating system with uncertainty tracking
- Shin (1993) — Converting bookmaker odds to true probabilities
- Hubacek et al. (2019) — Soccer Prediction Challenge winner
- Baboota & Kaur (2019) — EPL predictive analysis with ML
- Karlis & Ntzoufras (2003) — Bivariate Poisson model for football
- Prokhorov et al. (2024) — CatBoost + Pi-Ratings (SOTA: 55.82% acc, 0.1925 RPS)
