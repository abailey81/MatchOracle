"""
MatchOracle — Advanced API Client Infrastructure
==================================================
Production-grade API client with:
  - Disk-based caching with TTL (time-to-live)
  - Token bucket rate limiter (per-API)
  - Exponential backoff retry handler with jitter
  - ThreadPoolExecutor parallelization
  - Request deduplication and connection pooling
  - Circuit breaker pattern for failing APIs

Supports:
  1. Football-Data.org   — standings, team stats, H2H
  2. API-Football        — player ratings, injuries, detailed match stats
  3. NewsAPI             — football news for NLP sentiment analysis
"""

import hashlib
import json
import os
import pickle
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent / ".api_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Rate Limiter — Token Bucket algorithm
# ---------------------------------------------------------------------------
class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter.

    Args:
        rate: Maximum requests per second
        burst: Maximum burst capacity (tokens stored)
    """

    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            if time.monotonic() > deadline:
                return False
            wait = max(0.01, (1.0 - self._tokens) / self.rate)
            time.sleep(min(wait, 0.5))


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """Prevents repeated calls to a failing API.

    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._state = "CLOSED"
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._state == "OPEN":
                if time.monotonic() - self._opened_at > self.reset_timeout:
                    self._state = "HALF_OPEN"
                    return False
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = time.monotonic()


# ---------------------------------------------------------------------------
# Disk Cache with TTL
# ---------------------------------------------------------------------------
class DiskCache:
    """JSON/pickle disk cache with time-to-live support."""

    def __init__(self, cache_dir: Path, default_ttl: int = 86400):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _key_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key[:60])
        return self.cache_dir / f"{safe_key}_{h}.cache"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            with self._lock:
                data = pickle.loads(path.read_bytes())
            if data["expires_at"] < time.time():
                path.unlink(missing_ok=True)
                return None
            return data["value"]
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.default_ttl
        data = {"value": value, "expires_at": time.time() + ttl, "stored_at": time.time()}
        path = self._key_path(key)
        with self._lock:
            path.write_bytes(pickle.dumps(data))

    def clear(self):
        for f in self.cache_dir.glob("*.cache"):
            f.unlink(missing_ok=True)

    def stats(self) -> Dict[str, int]:
        files = list(self.cache_dir.glob("*.cache"))
        valid = 0
        expired = 0
        for f in files:
            try:
                data = pickle.loads(f.read_bytes())
                if data["expires_at"] >= time.time():
                    valid += 1
                else:
                    expired += 1
            except Exception:
                expired += 1
        return {"total": len(files), "valid": valid, "expired": expired}


# ---------------------------------------------------------------------------
# Base API Client
# ---------------------------------------------------------------------------
class BaseAPIClient:
    """Base HTTP client with retry, rate limiting, caching, and circuit breaker."""

    def __init__(self, name: str, base_url: str, rate_limit: float,
                 burst: int = 1, cache_ttl: int = 86400,
                 headers: Optional[Dict] = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = TokenBucketRateLimiter(rate=rate_limit, burst=burst)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=300)
        self.cache = DiskCache(CACHE_DIR / name, default_ttl=cache_ttl)
        self._stats = {"requests": 0, "cache_hits": 0, "errors": 0}
        self._lock = threading.Lock()

        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if headers:
            self.session.headers.update(headers)

    def get(self, endpoint: str, params: Optional[Dict] = None,
            cache_ttl: Optional[int] = None, cache_key: Optional[str] = None) -> Optional[Dict]:
        """Make a GET request with caching, rate limiting, and circuit breaker."""
        key = cache_key or f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"

        cached = self.cache.get(key)
        if cached is not None:
            with self._lock:
                self._stats["cache_hits"] += 1
            return cached

        if self.circuit_breaker.is_open:
            return None

        if not self.rate_limiter.acquire(timeout=30):
            return None

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        with self._lock:
            self._stats["requests"] += 1

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self.cache.set(key, data, ttl=cache_ttl)
            self.circuit_breaker.record_success()
            return data
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                time.sleep(min(retry_after, 120))
                return self.get(endpoint, params, cache_ttl, cache_key)
            self.circuit_breaker.record_failure()
            with self._lock:
                self._stats["errors"] += 1
            return None
        except Exception:
            self.circuit_breaker.record_failure()
            with self._lock:
                self._stats["errors"] += 1
            return None

    def get_parallel(self, requests_list: List[Dict], max_workers: int = 4) -> List[Optional[Dict]]:
        """Execute multiple requests in parallel with rate limiting."""
        results = [None] * len(requests_list)

        def _fetch(idx: int, req: Dict):
            result = self.get(**req)
            results[idx] = result

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch, i, req): i for i, req in enumerate(requests_list)}
            for future in as_completed(futures):
                future.result()

        return results

    @property
    def stats(self) -> Dict:
        with self._lock:
            return {**self._stats, "cache": self.cache.stats()}


# ---------------------------------------------------------------------------
# Football-Data.org Client
# ---------------------------------------------------------------------------
class FootballDataOrgClient(BaseAPIClient):
    """Football-Data.org API v4 client.

    Free tier: 10 requests/minute
    Provides: standings, matches, team stats, head-to-head
    """

    def __init__(self, api_key: str):
        super().__init__(
            name="football_data_org",
            base_url="https://api.football-data.org/v4",
            rate_limit=0.15,  # ~9 req/min (conservative under 10/min limit)
            burst=2,
            cache_ttl=3600,  # 1 hour cache
            headers={"X-Auth-Token": api_key},
        )
        self.competition_id = "PL"  # Premier League

    def get_standings(self, season: Optional[int] = None) -> Optional[Dict]:
        params = {}
        if season:
            params["season"] = season
        return self.get(f"competitions/{self.competition_id}/standings", params,
                       cache_ttl=3600, cache_key=f"standings_{season or 'current'}")

    def get_matches(self, season: Optional[int] = None,
                    matchday: Optional[int] = None,
                    status: Optional[str] = None) -> Optional[Dict]:
        params = {}
        if season:
            params["season"] = season
        if matchday:
            params["matchday"] = matchday
        if status:
            params["status"] = status
        key = f"matches_{season}_{matchday}_{status}"
        return self.get(f"competitions/{self.competition_id}/matches", params,
                       cache_ttl=1800, cache_key=key)

    def get_team(self, team_id: int) -> Optional[Dict]:
        return self.get(f"teams/{team_id}", cache_ttl=86400,
                       cache_key=f"team_{team_id}")

    def get_head_to_head(self, match_id: int) -> Optional[Dict]:
        return self.get(f"matches/{match_id}/head2head",
                       params={"limit": 20},
                       cache_ttl=86400,
                       cache_key=f"h2h_{match_id}")

    def get_scorers(self, season: Optional[int] = None) -> Optional[Dict]:
        params = {"limit": 50}
        if season:
            params["season"] = season
        return self.get(f"competitions/{self.competition_id}/scorers", params,
                       cache_ttl=3600, cache_key=f"scorers_{season or 'current'}")

    def fetch_multi_season_standings(self, seasons: List[int]) -> Dict[int, Any]:
        """Fetch standings for multiple seasons in parallel."""
        requests_list = [
            {"endpoint": f"competitions/{self.competition_id}/standings",
             "params": {"season": s},
             "cache_ttl": 86400 * 7,  # 7 day cache for historical
             "cache_key": f"standings_{s}"}
            for s in seasons
        ]
        results = self.get_parallel(requests_list, max_workers=2)
        return {s: r for s, r in zip(seasons, results) if r is not None}


# ---------------------------------------------------------------------------
# API-Football Client (via RapidAPI)
# ---------------------------------------------------------------------------
class APIFootballClient(BaseAPIClient):
    """API-Football v3 client (via RapidAPI).

    Free tier: 100 requests/day, 10 req/min
    Provides: player ratings, injuries, lineups, detailed match stats
    """

    def __init__(self, api_key: str):
        super().__init__(
            name="api_football",
            base_url="https://v3.football.api-sports.io",
            rate_limit=0.15,  # ~9 req/min
            burst=2,
            cache_ttl=3600,
            headers={
                "x-apisports-key": api_key,
            },
        )
        self.league_id = 39  # EPL
        self._daily_requests = 0
        self._daily_limit = 95  # conservative under 100

    def _check_daily_limit(self) -> bool:
        if self._daily_requests >= self._daily_limit:
            print(f"    API-Football: Daily limit reached ({self._daily_requests}/{self._daily_limit})")
            return False
        return True

    def get(self, endpoint: str, params: Optional[Dict] = None,
            cache_ttl: Optional[int] = None, cache_key: Optional[str] = None) -> Optional[Dict]:
        if not self._check_daily_limit():
            cached = self.cache.get(cache_key or f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}")
            return cached

        result = super().get(endpoint, params, cache_ttl, cache_key)
        if result is not None:
            self._daily_requests += 1
        return result

    def get_injuries(self, season: int, fixture_id: Optional[int] = None) -> Optional[Dict]:
        params = {"league": self.league_id, "season": season}
        if fixture_id:
            params["fixture"] = fixture_id
        return self.get("injuries", params, cache_ttl=3600,
                       cache_key=f"injuries_{season}_{fixture_id}")

    def get_predictions(self, fixture_id: int) -> Optional[Dict]:
        return self.get("predictions", {"fixture": fixture_id},
                       cache_ttl=3600, cache_key=f"predictions_{fixture_id}")

    def get_team_statistics(self, team_id: int, season: int) -> Optional[Dict]:
        params = {"team": team_id, "season": season, "league": self.league_id}
        return self.get("teams/statistics", params, cache_ttl=86400,
                       cache_key=f"team_stats_{team_id}_{season}")

    def get_fixtures(self, season: int, round_: Optional[str] = None) -> Optional[Dict]:
        params = {"league": self.league_id, "season": season}
        if round_:
            params["round"] = round_
        return self.get("fixtures", params, cache_ttl=3600,
                       cache_key=f"fixtures_{season}_{round_}")

    def get_fixture_stats(self, fixture_id: int) -> Optional[Dict]:
        return self.get("fixtures/statistics", {"fixture": fixture_id},
                       cache_ttl=86400 * 30,  # 30-day cache for historical
                       cache_key=f"fixture_stats_{fixture_id}")

    def get_player_ratings(self, fixture_id: int) -> Optional[Dict]:
        return self.get("fixtures/players", {"fixture": fixture_id},
                       cache_ttl=86400 * 30,
                       cache_key=f"player_ratings_{fixture_id}")

    def get_standings(self, season: int) -> Optional[Dict]:
        params = {"league": self.league_id, "season": season}
        return self.get("standings", params, cache_ttl=3600,
                       cache_key=f"standings_{season}")


# ---------------------------------------------------------------------------
# NewsAPI Client
# ---------------------------------------------------------------------------
class NewsAPIClient(BaseAPIClient):
    """NewsAPI client for football news sentiment analysis.

    Free tier: 100 requests/day, no per-minute limit specified
    Provides: news articles about EPL teams for NLP sentiment features
    """

    def __init__(self, api_key: str):
        super().__init__(
            name="newsapi",
            base_url="https://newsapi.org/v2",
            rate_limit=0.5,  # 30 req/min (conservative)
            burst=5,
            cache_ttl=7200,  # 2 hour cache for news
            headers={"X-Api-Key": api_key},
        )
        self._daily_requests = 0
        self._daily_limit = 90

    def get(self, endpoint: str, params: Optional[Dict] = None,
            cache_ttl: Optional[int] = None, cache_key: Optional[str] = None) -> Optional[Dict]:
        if self._daily_requests >= self._daily_limit:
            cached = self.cache.get(cache_key or f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}")
            return cached
        result = super().get(endpoint, params, cache_ttl, cache_key)
        if result is not None:
            self._daily_requests += 1
        return result

    def search_team_news(self, team_name: str, days_back: int = 7,
                         page_size: int = 10) -> Optional[Dict]:
        """Search for recent news about a specific team."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": f'"{team_name}" AND (Premier League OR EPL OR football)',
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": page_size,
        }
        return self.get("everything", params, cache_ttl=7200,
                       cache_key=f"team_news_{team_name}_{from_date}")

    def search_match_news(self, home_team: str, away_team: str,
                          days_back: int = 3) -> Optional[Dict]:
        """Search for pre-match news and previews."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": f'("{home_team}" OR "{away_team}") AND (Premier League OR match OR preview)',
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 15,
        }
        return self.get("everything", params, cache_ttl=3600,
                       cache_key=f"match_news_{home_team}_{away_team}_{from_date}")

    def get_epl_headlines(self, page_size: int = 20) -> Optional[Dict]:
        """Get latest EPL headlines from top sources."""
        params = {
            "q": "Premier League",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
        }
        return self.get("everything", params, cache_ttl=3600,
                       cache_key=f"epl_headlines_{datetime.now().strftime('%Y%m%d')}")


# ---------------------------------------------------------------------------
# NLP Sentiment Analyzer
# ---------------------------------------------------------------------------
class SentimentAnalyzer:
    """Lightweight NLP sentiment analyzer for football news.

    Uses a lexicon-based approach (no external ML dependencies) combined
    with football-specific sentiment signals.
    """

    # Football-specific positive/negative terms
    POSITIVE_TERMS = {
        "win", "victory", "dominant", "impressive", "excellent", "superb",
        "brilliant", "outstanding", "confident", "strong", "unbeaten",
        "streak", "form", "scoring", "clean sheet", "top", "title",
        "champions", "promoted", "boost", "return", "fit", "signing",
        "transfer", "new signing", "momentum", "crucial", "comeback",
        "clinical", "world-class", "lethal", "unstoppable", "perfect",
    }

    NEGATIVE_TERMS = {
        "loss", "defeat", "poor", "terrible", "dreadful", "struggling",
        "crisis", "sacked", "fired", "injured", "injury", "suspended",
        "banned", "red card", "relegated", "bottom", "worst", "collapse",
        "nightmare", "dismal", "woeful", "dire", "slump", "concern",
        "doubt", "uncertain", "setback", "blow", "miss", "absent",
        "hamstring", "knee", "ligament", "fracture", "ruled out",
    }

    AMPLIFIERS = {"very", "extremely", "highly", "absolutely", "completely"}
    NEGATORS = {"not", "no", "never", "hardly", "barely", "without"}

    def __init__(self):
        self._positive = {t.lower() for t in self.POSITIVE_TERMS}
        self._negative = {t.lower() for t in self.NEGATIVE_TERMS}

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a text. Returns scores in [-1, 1]."""
        if not text:
            return {"sentiment": 0.0, "positive": 0.0, "negative": 0.0, "subjectivity": 0.0}

        words = text.lower().split()
        n = len(words)
        if n == 0:
            return {"sentiment": 0.0, "positive": 0.0, "negative": 0.0, "subjectivity": 0.0}

        pos_count = 0
        neg_count = 0
        for i, word in enumerate(words):
            is_negated = i > 0 and words[i - 1] in self.NEGATORS
            multiplier = 1.5 if (i > 0 and words[i - 1] in self.AMPLIFIERS) else 1.0

            if word in self._positive:
                if is_negated:
                    neg_count += multiplier
                else:
                    pos_count += multiplier
            elif word in self._negative:
                if is_negated:
                    pos_count += multiplier * 0.5  # negated negative is weakly positive
                else:
                    neg_count += multiplier

            # Check 2-word phrases
            if i < n - 1:
                phrase = f"{word} {words[i + 1]}"
                if phrase in self._positive:
                    pos_count += 1.5 if not is_negated else 0
                    neg_count += 1.5 if is_negated else 0
                elif phrase in self._negative:
                    neg_count += 1.5 if not is_negated else 0
                    pos_count += 0.5 if is_negated else 0

        total = pos_count + neg_count
        if total == 0:
            return {"sentiment": 0.0, "positive": 0.0, "negative": 0.0, "subjectivity": 0.0}

        sentiment = (pos_count - neg_count) / max(total, 1)
        subjectivity = min(total / (n * 0.1), 1.0)

        return {
            "sentiment": round(max(-1, min(1, sentiment)), 4),
            "positive": round(pos_count / n, 4),
            "negative": round(neg_count / n, 4),
            "subjectivity": round(subjectivity, 4),
        }

    def analyze_articles(self, articles: List[Dict]) -> Dict[str, float]:
        """Analyze sentiment across multiple articles. Returns aggregated scores."""
        if not articles:
            return {"sentiment": 0.0, "positive": 0.0, "negative": 0.0,
                    "subjectivity": 0.0, "volume": 0, "consensus": 0.0}

        sentiments = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
            s = self.analyze_text(text)
            sentiments.append(s)

        n = len(sentiments)
        avg_sentiment = sum(s["sentiment"] for s in sentiments) / n
        avg_positive = sum(s["positive"] for s in sentiments) / n
        avg_negative = sum(s["negative"] for s in sentiments) / n
        avg_subjectivity = sum(s["subjectivity"] for s in sentiments) / n

        # Consensus: how much do articles agree? (low std = high consensus)
        sent_values = [s["sentiment"] for s in sentiments]
        if n > 1:
            import statistics
            std = statistics.stdev(sent_values)
            consensus = max(0, 1 - std)
        else:
            consensus = 0.5

        return {
            "sentiment": round(avg_sentiment, 4),
            "positive": round(avg_positive, 4),
            "negative": round(avg_negative, 4),
            "subjectivity": round(avg_subjectivity, 4),
            "volume": n,
            "consensus": round(consensus, 4),
        }


# ---------------------------------------------------------------------------
# Unified Data Fetcher — orchestrates all APIs
# ---------------------------------------------------------------------------
class UnifiedDataFetcher:
    """Orchestrates data fetching from all APIs with intelligent parallelization."""

    # Team name mapping from standard names to API-specific names
    TEAM_ID_MAP_FDO = {
        "Arsenal": 57, "Aston Villa": 58, "Bournemouth": 1044,
        "Brentford": 402, "Brighton": 397, "Chelsea": 61,
        "Crystal Palace": 354, "Everton": 62, "Fulham": 63,
        "Ipswich": 349, "Leicester": 338, "Liverpool": 64,
        "Manchester City": 65, "Manchester United": 66, "Newcastle": 67,
        "Nottm Forest": 351, "Southampton": 340, "Tottenham": 73,
        "West Ham": 563, "Wolves": 76,
    }

    def __init__(self, fdo_key: Optional[str] = None,
                 apif_key: Optional[str] = None,
                 news_key: Optional[str] = None):
        self.fdo = FootballDataOrgClient(fdo_key) if fdo_key else None
        self.apif = APIFootballClient(apif_key) if apif_key else None
        self.news = NewsAPIClient(news_key) if news_key else None
        self.sentiment = SentimentAnalyzer()

    def fetch_team_sentiment(self, teams: List[str], days_back: int = 7) -> Dict[str, Dict]:
        """Fetch and analyze news sentiment for multiple teams."""
        if not self.news:
            return {}

        print(f"  Fetching news sentiment for {len(teams)} teams ...")
        results = {}

        for team in teams:
            news_data = self.news.search_team_news(team, days_back=days_back, page_size=10)
            if news_data and "articles" in news_data:
                articles = news_data["articles"]
                sentiment = self.sentiment.analyze_articles(articles)
                results[team] = sentiment
            else:
                results[team] = {
                    "sentiment": 0.0, "positive": 0.0, "negative": 0.0,
                    "subjectivity": 0.0, "volume": 0, "consensus": 0.0,
                }

        n_with_data = sum(1 for v in results.values() if v["volume"] > 0)
        print(f"  => Sentiment data for {n_with_data}/{len(teams)} teams")
        return results

    def fetch_fdo_standings(self, seasons: List[int]) -> Dict[int, List[Dict]]:
        """Fetch league standings from Football-Data.org for multiple seasons."""
        if not self.fdo:
            return {}

        print(f"  Fetching FDO standings for {len(seasons)} seasons ...")
        all_standings = {}

        for season in seasons:
            data = self.fdo.get_standings(season)
            if data and "standings" in data:
                table = data["standings"]
                if table and len(table) > 0:
                    entries = []
                    for entry in table[0].get("table", []):
                        entries.append({
                            "position": entry.get("position"),
                            "team": entry.get("team", {}).get("name"),
                            "team_id": entry.get("team", {}).get("id"),
                            "points": entry.get("points"),
                            "played": entry.get("playedGames"),
                            "won": entry.get("won"),
                            "draw": entry.get("draw"),
                            "lost": entry.get("lost"),
                            "goals_for": entry.get("goalsFor"),
                            "goals_against": entry.get("goalsAgainst"),
                            "goal_diff": entry.get("goalDifference"),
                            "form": entry.get("form"),
                        })
                    all_standings[season] = entries

        print(f"  => Standings for {len(all_standings)} seasons")
        return all_standings

    def fetch_apif_team_stats(self, team_ids: List[int], season: int) -> Dict[int, Dict]:
        """Fetch detailed team statistics from API-Football."""
        if not self.apif:
            return {}

        print(f"  Fetching API-Football team stats for {len(team_ids)} teams ...")
        results = {}

        for team_id in team_ids:
            data = self.apif.get_team_statistics(team_id, season)
            if data and "response" in data:
                resp = data["response"]
                stats = {}
                if isinstance(resp, dict):
                    stats = {
                        "form": resp.get("form", ""),
                        "clean_sheets_total": _safe_nested(resp, "clean_sheet", "total"),
                        "failed_to_score_total": _safe_nested(resp, "failed_to_score", "total"),
                        "avg_goals_for": _safe_nested(resp, "goals", "for", "average", "total"),
                        "avg_goals_against": _safe_nested(resp, "goals", "against", "average", "total"),
                        "penalty_scored": _safe_nested(resp, "penalty", "scored", "total"),
                        "penalty_missed": _safe_nested(resp, "penalty", "missed", "total"),
                        "biggest_win_streak": _safe_nested(resp, "biggest", "streak", "wins"),
                        "biggest_loss_streak": _safe_nested(resp, "biggest", "streak", "loses"),
                    }
                results[team_id] = stats

        print(f"  => Team stats for {len(results)} teams")
        return results

    def fetch_injuries(self, season: int) -> List[Dict]:
        """Fetch current injuries from API-Football."""
        if not self.apif:
            return []

        data = self.apif.get_injuries(season)
        if not data or "response" not in data:
            return []

        injuries = []
        for entry in data["response"]:
            player = entry.get("player", {})
            team = entry.get("team", {})
            injuries.append({
                "player": player.get("name"),
                "team": team.get("name"),
                "team_id": team.get("id"),
                "type": player.get("type"),
                "reason": player.get("reason"),
            })

        print(f"  => {len(injuries)} current injuries")
        return injuries

    def print_stats(self):
        """Print API usage statistics."""
        print("\n  API Usage Statistics:")
        if self.fdo:
            s = self.fdo.stats
            print(f"    Football-Data.org: {s['requests']} requests, "
                  f"{s['cache_hits']} cache hits, {s['errors']} errors")
        if self.apif:
            s = self.apif.stats
            print(f"    API-Football:      {s['requests']} requests, "
                  f"{s['cache_hits']} cache hits, {s['errors']} errors")
        if self.news:
            s = self.news.stats
            print(f"    NewsAPI:           {s['requests']} requests, "
                  f"{s['cache_hits']} cache hits, {s['errors']} errors")


def _safe_nested(d: dict, *keys):
    """Safely navigate nested dict."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d
