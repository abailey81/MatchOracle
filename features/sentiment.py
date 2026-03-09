"""
MatchOracle — Real-Time NLP Sentiment Engine
==============================================
Fetches latest news for EPL teams and performs keyword-weighted
sentiment analysis. Designed to run at prediction time for
maximum freshness.

Analyses:
- Overall sentiment polarity (-1 to +1)
- News volume (buzz indicator)
- Injury/suspension mentions (negative signal)
- Managerial change mentions (volatility signal)
- Transfer activity mentions
- Fan/media confidence indicators
- Key player availability signals
"""

import re
import time
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# EPL team keywords for news search
TEAM_SEARCH_TERMS = {
    "Arsenal": ["Arsenal FC", "Arsenal"],
    "Aston Villa": ["Aston Villa", "Villa Park"],
    "Bournemouth": ["AFC Bournemouth", "Bournemouth FC"],
    "Brentford": ["Brentford FC", "Brentford"],
    "Brighton": ["Brighton Hove Albion", "Brighton FC"],
    "Chelsea": ["Chelsea FC", "Chelsea"],
    "Crystal Palace": ["Crystal Palace FC", "Crystal Palace"],
    "Everton": ["Everton FC", "Everton"],
    "Fulham": ["Fulham FC", "Fulham"],
    "Ipswich": ["Ipswich Town", "Ipswich FC"],
    "Leeds": ["Leeds United", "Leeds FC"],
    "Leicester": ["Leicester City", "Leicester FC"],
    "Liverpool": ["Liverpool FC", "Liverpool"],
    "Manchester City": ["Manchester City", "Man City"],
    "Manchester United": ["Manchester United", "Man United", "Man Utd"],
    "Newcastle": ["Newcastle United", "Newcastle FC"],
    "Nottm Forest": ["Nottingham Forest", "Nott Forest"],
    "Southampton": ["Southampton FC", "Southampton"],
    "Sunderland": ["Sunderland AFC", "Sunderland FC", "Sunderland"],
    "Burnley": ["Burnley FC", "Burnley"],
    "Tottenham": ["Tottenham Hotspur", "Tottenham Spurs"],
    "West Ham": ["West Ham United", "West Ham"],
    "Wolves": ["Wolverhampton Wanderers", "Wolves FC"],
}

# Sentiment keywords with weights
POSITIVE_KEYWORDS = {
    # Strong positive
    "victory": 0.8, "wins": 0.7, "win": 0.6, "triumph": 0.8,
    "dominant": 0.7, "brilliant": 0.8, "excellent": 0.7,
    "impressive": 0.6, "outstanding": 0.8, "superb": 0.7,
    # Moderate positive
    "confident": 0.5, "optimistic": 0.5, "boost": 0.5,
    "returns": 0.4, "fit": 0.4, "recovered": 0.5,
    "unbeaten": 0.6, "streak": 0.3, "form": 0.3,
    "signing": 0.4, "deal": 0.3, "strengthen": 0.5,
    "clean sheet": 0.5, "shutout": 0.5,
    # Light positive
    "positive": 0.3, "good": 0.2, "strong": 0.3,
    "improved": 0.3, "progress": 0.3, "momentum": 0.4,
}

NEGATIVE_KEYWORDS = {
    # Strong negative
    "injury": -0.7, "injured": -0.7, "sidelined": -0.8,
    "ruled out": -0.9, "surgery": -0.9, "torn": -0.8,
    "sacked": -0.8, "fired": -0.8, "dismissed": -0.7,
    "defeat": -0.6, "loss": -0.5, "lost": -0.5,
    "crisis": -0.8, "disaster": -0.8,
    # Moderate negative
    "suspended": -0.6, "ban": -0.6, "red card": -0.6,
    "doubt": -0.4, "doubtful": -0.5, "concern": -0.4,
    "struggling": -0.5, "slump": -0.6, "poor": -0.4,
    "relegation": -0.5, "drop": -0.3,
    "hamstring": -0.6, "knee": -0.5, "ankle": -0.5,
    "muscle": -0.5, "fracture": -0.8, "concussion": -0.7,
    # Light negative
    "miss": -0.3, "missing": -0.4, "absence": -0.4,
    "fatigue": -0.3, "tired": -0.3, "rotation": -0.2,
    "criticism": -0.3, "pressure": -0.2,
}

# Context-aware modifiers
INJURY_PATTERNS = [
    r"injur(?:y|ed|ies)", r"sidelined", r"ruled out", r"hamstring",
    r"knee", r"ankle", r"muscle", r"torn", r"fracture", r"surgery",
    r"concussion", r"fitness (?:doubt|concern|test)",
]

MANAGER_PATTERNS = [
    r"sacked", r"fired", r"dismissed", r"new manager",
    r"managerial change", r"interim", r"appointment",
    r"steps down", r"resigned", r"leaves",
]

TRANSFER_PATTERNS = [
    r"sign(?:s|ed|ing)", r"transfer", r"loan", r"deal",
    r"bid", r"offer", r"target", r"acquire",
]

# Formation and tactical system keywords
FORMATION_PATTERNS = [
    r"4-3-3", r"4-2-3-1", r"3-5-2", r"3-4-3", r"4-4-2", r"5-3-2",
    r"5-4-1", r"4-1-4-1", r"diamond", r"false nine", r"inverted",
]


# Transformer-based sentiment model (optional, loaded lazily)
_TRANSFORMER_PIPELINE = None
_TRANSFORMER_LOADED = False

def _load_transformer():
    """Try to load a pre-trained transformer for sentiment analysis."""
    global _TRANSFORMER_PIPELINE, _TRANSFORMER_LOADED
    if _TRANSFORMER_LOADED:
        return _TRANSFORMER_PIPELINE
    _TRANSFORMER_LOADED = True
    try:
        from transformers import pipeline
        _TRANSFORMER_PIPELINE = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        print("  [Sentiment] Loaded RoBERTa transformer model")
    except Exception:
        _TRANSFORMER_PIPELINE = None
    return _TRANSFORMER_PIPELINE


def _transformer_sentiment(text: str) -> float:
    """Get sentiment score from transformer model."""
    pipe = _load_transformer()
    if pipe is None:
        return None
    try:
        results = pipe(text[:512])[0]
        # Map labels to scores
        score = 0.0
        for r in results:
            label = r["label"].lower()
            if "positive" in label:
                score += r["score"]
            elif "negative" in label:
                score -= r["score"]
            # neutral contributes 0
        return np.clip(score, -1.0, 1.0)
    except Exception:
        return None


# Aspect-based sentiment patterns
MORALE_PATTERNS = [
    r"confident", r"belief", r"togeth", r"spirit", r"determination",
    r"delight", r"thrilled", r"motivated", r"focused", r"ready",
    r"under pressure", r"tensions?", r"conflict", r"unhappy",
    r"frustrated", r"worried", r"concern",
]

TACTICAL_PATTERNS = [
    r"new formation", r"tactical", r"system change", r"switch",
    r"playing style", r"press(?:ing)?", r"counter.attack",
    r"defensive", r"attacking",
]

KEY_PLAYER_PATTERNS = [
    r"captain", r"star player", r"key player", r"top scorer",
    r"playmaker", r"goalkeeper", r"centre.back",
]


def _extract_player_names(text: str) -> list:
    """Dynamically extract likely player names from text using NER-style heuristics.

    Detects capitalized multi-word names (e.g. 'Marcus Rashford') and single
    capitalized surnames that appear near football context words.
    No hardcoded player lists — fully dynamic.
    """
    # Pattern 1: Full names (FirstName LastName) — most reliable
    full_names = re.findall(r"\b([A-Z][a-z]{1,15}\s[A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20})?)\b", text)

    # Pattern 2: Capitalized single words near football context
    # (avoids matching city names, team names by checking surrounding context)
    football_context = r"(?:scored|goal|assist|injur|sidelined|ruled out|return|fit|doubt|miss|absent|surgery|captain|striker|midfielder|defender|goalkeeper|winger|forward|playmaker|star)"
    single_names = []
    for m in re.finditer(r"\b([A-Z][a-z]{2,20})\b", text):
        name = m.group(1)
        # Check if football context exists within 100 chars
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 100)
        context = text[start:end].lower()
        if re.search(football_context, context):
            # Filter out common non-player words
            skip_words = {"Premier", "League", "United", "City", "Palace",
                          "Forest", "Villa", "Town", "Wednesday", "Saturday",
                          "Sunday", "Monday", "Tuesday", "January", "February",
                          "March", "April", "May", "June", "July", "August",
                          "September", "October", "November", "December",
                          "Arsenal", "Chelsea", "Liverpool", "Tottenham",
                          "Newcastle", "Brighton", "Everton", "Fulham",
                          "Brentford", "Wolves", "Bournemouth", "Ipswich",
                          "Leicester", "Southampton", "England", "Spain",
                          "France", "Germany", "Brazil", "Argentina",
                          "Manager", "Coach", "Report", "Update", "Breaking"}
            if name not in skip_words:
                single_names.append(name)

    return list(set(full_names + single_names))


def _detect_player_impact(text: str) -> dict:
    """Dynamically detect player-related impact from article text.

    No hardcoded player lists. Extracts player names from the text itself
    and assesses whether the context is positive or negative.
    """
    players = _extract_player_names(text)
    if not players:
        return {"player_mentioned": False, "player_injury": 0.0,
                "player_absence": 0.0, "player_return": 0.0, "impact_score": 0.0}

    text_lower = text.lower()
    injury_score = 0.0
    absence_score = 0.0
    return_score = 0.0

    neg_patterns = [r"injur", r"ruled out", r"sidelined", r"doubt",
                    r"miss(?:es|ing)?", r"absent", r"out of",
                    r"surgery", r"torn", r"fracture", r"hamstring",
                    r"knee", r"ankle", r"muscle", r"ban", r"suspend"]
    pos_patterns = [r"return", r"\bfit\b", r"available", r"recover",
                    r"back in", r"training", r"ready", r"passed.*test"]

    for player in players:
        player_lower = player.lower()
        idx = text_lower.find(player_lower)
        if idx < 0:
            continue
        context = text_lower[max(0, idx - 80):idx + len(player_lower) + 80]

        for p in neg_patterns:
            if re.search(p, context):
                injury_score += 0.4
                absence_score += 0.3
                break

        for p in pos_patterns:
            if re.search(p, context):
                return_score += 0.3
                break

    impact = return_score - injury_score - absence_score
    return {
        "player_mentioned": True,
        "player_injury": min(1.0, injury_score),
        "player_absence": min(1.0, absence_score),
        "player_return": min(1.0, return_score),
        "impact_score": np.clip(impact, -1.0, 1.0),
    }


def _compute_article_sentiment(title: str, description: str,
                                published_at: str = None) -> dict:
    """Analyse a single article's sentiment with multi-method approach.

    Uses transformer model if available, falls back to keyword matching.
    Also performs aspect-based analysis for injuries, morale, tactics.
    Dynamically detects player mentions for impact assessment.
    """
    text = f"{title} {description}".lower()
    full_text = f"{title} {description}"

    # Method 1: Transformer sentiment (if available)
    transformer_score = _transformer_sentiment(full_text[:512])

    # Method 2: Keyword-based sentiment (always computed)
    keyword_score = 0.0
    n_hits = 0

    for word, weight in POSITIVE_KEYWORDS.items():
        count = text.count(word)
        if count > 0:
            keyword_score += weight * count
            n_hits += count

    for word, weight in NEGATIVE_KEYWORDS.items():
        count = text.count(word)
        if count > 0:
            keyword_score += weight * count
            n_hits += count

    if n_hits > 0:
        keyword_score = np.clip(keyword_score / max(n_hits, 1), -1.0, 1.0)

    # Blend: transformer (70%) + keyword (30%) if both available
    if transformer_score is not None:
        score = 0.7 * transformer_score + 0.3 * keyword_score
    else:
        score = keyword_score

    # Temporal weighting: more recent articles get higher weight
    time_weight = 1.0
    if published_at:
        try:
            from datetime import datetime
            pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            hours_ago = (datetime.now(pub_date.tzinfo) - pub_date).total_seconds() / 3600
            time_weight = max(0.3, 1.0 - hours_ago / (5 * 24))  # Decay over 5 days
        except Exception:
            pass

    # Aspect-based analysis
    injury_mentions = sum(1 for p in INJURY_PATTERNS if re.search(p, text))
    manager_mentions = sum(1 for p in MANAGER_PATTERNS if re.search(p, text))
    transfer_mentions = sum(1 for p in TRANSFER_PATTERNS if re.search(p, text))
    morale_mentions = sum(1 for p in MORALE_PATTERNS if re.search(p, text))
    tactical_mentions = sum(1 for p in TACTICAL_PATTERNS if re.search(p, text))
    key_player_mentions = sum(1 for p in KEY_PLAYER_PATTERNS if re.search(p, text))

    # Morale sub-sentiment (positive or negative morale signals)
    morale_positive = sum(1 for p in MORALE_PATTERNS[:10] if re.search(p, text))
    morale_negative = sum(1 for p in MORALE_PATTERNS[10:] if re.search(p, text))
    morale_score = (morale_positive - morale_negative) / max(morale_mentions, 1)

    # Dynamic player impact detection (no hardcoded player lists)
    player_impact = _detect_player_impact(full_text)

    # If a player is injured, amplify the negative signal
    if player_impact["player_injury"] > 0:
        score = score - 0.12 * player_impact["player_injury"]
        injury_mentions += 1

    return {
        "sentiment": score * time_weight,
        "raw_sentiment": score,
        "time_weight": time_weight,
        "transformer_used": transformer_score is not None,
        "injury_signal": injury_mentions,
        "manager_signal": manager_mentions,
        "transfer_signal": transfer_mentions,
        "morale_signal": morale_score,
        "tactical_change": tactical_mentions,
        "key_player_news": key_player_mentions,
        "has_content": n_hits > 0 or transformer_score is not None,
        "player_impact": player_impact["impact_score"],
        "player_injury": player_impact["player_injury"],
        "player_return": player_impact["player_return"],
    }


def _fetch_google_news_rss(team: str) -> List[dict]:
    """Fallback: fetch news from Google News RSS when NewsAPI is rate-limited."""
    import re as _re
    search_terms = TEAM_SEARCH_TERMS.get(team, [team])
    query = "+".join(search_terms[0].split()) + "+Premier+League"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-GB&gl=GB&ceid=GB:en"

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            content = resp.text
            titles = _re.findall(r"<title>(.*?)</title>", content)
            descriptions = _re.findall(r"<description>(.*?)</description>", content)
            articles = []
            for i, title in enumerate(titles[2:32]):  # Skip header items
                desc = descriptions[i + 1] if i + 1 < len(descriptions) else ""
                # Strip HTML from description
                desc = _re.sub(r"<[^>]+>", "", desc)
                articles.append({
                    "title": title,
                    "description": desc,
                    "publishedAt": "",
                })
            return articles
    except Exception:
        pass
    return []


def fetch_team_news(team: str, api_key: str, days: int = 5) -> List[dict]:
    """Fetch recent news articles for a team from NewsAPI, with Google News fallback."""
    search_terms = TEAM_SEARCH_TERMS.get(team, [team])
    query = " OR ".join(f'"{t}"' for t in search_terms)

    from datetime import datetime, timedelta
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_date,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 30,
                "apiKey": api_key,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("articles", [])
            if articles:
                return articles
            # NewsAPI returned 0 results — try Google News
            return _fetch_google_news_rss(team)
        elif resp.status_code == 429:
            # Rate limited — fall back to Google News RSS
            return _fetch_google_news_rss(team)
    except Exception:
        pass
    return _fetch_google_news_rss(team)


def analyse_team_sentiment(articles: List[dict]) -> dict:
    """Aggregate sentiment analysis across multiple articles for a team."""
    if not articles:
        return {
            "sentiment": 0.0,
            "confidence": 0.0,
            "volume": 0,
            "injury_risk": 0.0,
            "manager_instability": 0.0,
            "transfer_activity": 0.0,
            "consensus": 0.0,
        }

    sentiments = []
    injury_total = 0
    manager_total = 0
    transfer_total = 0
    morale_scores = []
    tactical_changes = 0
    key_player_news = 0
    content_articles = 0
    transformer_count = 0

    player_injury_total = 0.0
    player_return_total = 0.0
    player_impact_scores = []

    for article in articles:
        title = article.get("title", "") or ""
        desc = article.get("description", "") or ""
        published = article.get("publishedAt", "")
        team = article.get("_team", None)  # Set by fetch_and_analyse
        if not title and not desc:
            continue

        result = _compute_article_sentiment(title, desc, published)
        if result["has_content"]:
            sentiments.append(result["sentiment"])
            content_articles += 1
        if result.get("transformer_used"):
            transformer_count += 1
        injury_total += result["injury_signal"]
        manager_total += result["manager_signal"]
        transfer_total += result["transfer_signal"]
        morale_scores.append(result.get("morale_signal", 0.0))
        tactical_changes += result.get("tactical_change", 0)
        key_player_news += result.get("key_player_news", 0)
        player_injury_total += result.get("player_injury", 0.0)
        player_return_total += result.get("player_return", 0.0)
        if result.get("player_impact", 0.0) != 0.0:
            player_impact_scores.append(result["player_impact"])

    if not sentiments:
        return {
            "sentiment": 0.0, "confidence": 0.0, "volume": len(articles),
            "injury_risk": 0.0, "manager_instability": 0.0,
            "transfer_activity": 0.0, "consensus": 0.0,
            "morale": 0.0, "tactical_disruption": 0.0,
            "key_player_concern": 0.0, "transformer_pct": 0.0,
        }

    avg_sentiment = float(np.mean(sentiments))
    consensus = 1.0 - float(np.std(sentiments)) if len(sentiments) > 1 else 0.5
    avg_morale = float(np.mean(morale_scores)) if morale_scores else 0.0

    avg_player_impact = float(np.mean(player_impact_scores)) if player_impact_scores else 0.0

    return {
        "sentiment": avg_sentiment,
        "confidence": min(1.0, content_articles / 10),
        "volume": len(articles),
        "injury_risk": min(1.0, injury_total / max(len(articles), 1)),
        "manager_instability": min(1.0, manager_total / max(len(articles), 1)),
        "transfer_activity": min(1.0, transfer_total / max(len(articles), 1)),
        "consensus": consensus,
        "morale": np.clip(avg_morale, -1.0, 1.0),
        "tactical_disruption": min(1.0, tactical_changes / max(len(articles), 1)),
        "key_player_concern": min(1.0, key_player_news / max(len(articles), 1)),
        "transformer_pct": transformer_count / max(content_articles, 1),
        "player_impact": np.clip(avg_player_impact, -1.0, 1.0),
        "player_injury_risk": min(1.0, player_injury_total),
        "player_return_boost": min(1.0, player_return_total),
    }


def fetch_live_sentiment(teams: List[str], api_key: str,
                          max_workers: int = 5) -> Dict[str, dict]:
    """Fetch and analyse sentiment for multiple teams in parallel.
    Returns a dict of team -> sentiment analysis results.
    """
    if not api_key:
        print("  [Sentiment] No NEWS_KEY — skipping live sentiment")
        return {}

    print(f"  [Sentiment] Fetching live news for {len(teams)} teams ...")
    results = {}

    def _fetch_and_analyse(team):
        articles = fetch_team_news(team, api_key, days=5)
        analysis = analyse_team_sentiment(articles)
        return team, analysis

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_and_analyse, t): t for t in teams}
        for future in as_completed(futures):
            try:
                team, analysis = future.result()
                results[team] = analysis
            except Exception as e:
                team = futures[future]
                results[team] = {
                    "sentiment": 0.0, "confidence": 0.0, "volume": 0,
                    "injury_risk": 0.0, "manager_instability": 0.0,
                    "transfer_activity": 0.0, "consensus": 0.0,
                }

    # Report
    positive = sum(1 for v in results.values() if v["sentiment"] > 0.1)
    negative = sum(1 for v in results.values() if v["sentiment"] < -0.1)
    neutral = len(results) - positive - negative
    total_articles = sum(v["volume"] for v in results.values())
    print(f"  [Sentiment] {total_articles} articles analysed: "
          f"{positive} positive, {negative} negative, {neutral} neutral teams")

    return results


def get_match_sentiment_features(home_team: str, away_team: str,
                                  sentiment_data: Dict[str, dict]) -> dict:
    """Extract sentiment features for a specific match from pre-fetched data."""
    h = sentiment_data.get(home_team, {})
    a = sentiment_data.get(away_team, {})

    h_sent = h.get("sentiment", 0.0)
    a_sent = a.get("sentiment", 0.0)

    return {
        "live_sentiment_home": h_sent,
        "live_sentiment_away": a_sent,
        "live_sentiment_diff": h_sent - a_sent,
        "live_news_volume_home": h.get("volume", 0),
        "live_news_volume_away": a.get("volume", 0),
        "live_sentiment_confidence_home": h.get("confidence", 0.0),
        "live_sentiment_confidence_away": a.get("confidence", 0.0),
        "live_injury_risk_home": h.get("injury_risk", 0.0),
        "live_injury_risk_away": a.get("injury_risk", 0.0),
        "live_injury_risk_diff": h.get("injury_risk", 0.0) - a.get("injury_risk", 0.0),
        "live_manager_instability_home": h.get("manager_instability", 0.0),
        "live_manager_instability_away": a.get("manager_instability", 0.0),
        "live_transfer_activity_home": h.get("transfer_activity", 0.0),
        "live_transfer_activity_away": a.get("transfer_activity", 0.0),
        "live_sentiment_consensus_home": h.get("consensus", 0.0),
        "live_sentiment_consensus_away": a.get("consensus", 0.0),
        # Advanced aspect-based features
        "live_morale_home": h.get("morale", 0.0),
        "live_morale_away": a.get("morale", 0.0),
        "live_morale_diff": h.get("morale", 0.0) - a.get("morale", 0.0),
        "live_tactical_disruption_home": h.get("tactical_disruption", 0.0),
        "live_tactical_disruption_away": a.get("tactical_disruption", 0.0),
        "live_key_player_concern_home": h.get("key_player_concern", 0.0),
        "live_key_player_concern_away": a.get("key_player_concern", 0.0),
        # Dynamic player impact features (auto-detected from news)
        "live_player_impact_home": h.get("player_impact", 0.0),
        "live_player_impact_away": a.get("player_impact", 0.0),
        "live_player_impact_diff": h.get("player_impact", 0.0) - a.get("player_impact", 0.0),
        "live_player_injury_home": h.get("player_injury_risk", 0.0),
        "live_player_injury_away": a.get("player_injury_risk", 0.0),
        "live_player_return_home": h.get("player_return_boost", 0.0),
        "live_player_return_away": a.get("player_return_boost", 0.0),
    }
