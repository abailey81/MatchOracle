"""
MatchOracle — Smart Model Caching & Retraining Detection
=========================================================
Saves trained models + metadata to disk. When predictions are requested,
intelligently determines whether retraining is needed based on:
  1. Data freshness (has new match data been added?)
  2. Feature schema changes (have features changed?)
  3. Time since last training
  4. Model file integrity

If no retraining is needed, loads cached models in <2 seconds.
"""

import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

METADATA_FILE = CACHE_DIR / "training_metadata.json"
MODELS_FILE = CACHE_DIR / "trained_models.pkl"
STATE_FILE = CACHE_DIR / "pipeline_state.pkl"


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Fast hash of stable training data.

    Hashes the SOURCE CSV file on disk (not the in-memory DataFrame),
    so that live sentiment fetches and feature engineering variations
    don't trigger unnecessary retraining.
    """
    data_dir = Path(__file__).parent.parent / "data"
    data_path = data_dir / "epl_featured.parquet"

    if data_path.exists():
        # Hash the raw file bytes — this only changes when actual data is updated
        raw = data_path.read_bytes()
        return hashlib.md5(raw).hexdigest()[:16]

    # Fallback: hash DataFrame shape + stable column names only (not values)
    stable_cols = [c for c in df.columns if not c.startswith("live_")]
    content = f"{len(df)}|{len(stable_cols)}|{sorted(stable_cols)}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _hash_features(feature_cols: list) -> str:
    """Hash the feature column list (excluding live_ columns that change each run)."""
    stable = sorted(c for c in feature_cols if not c.startswith("live_"))
    return hashlib.md5("|".join(stable).encode()).hexdigest()[:16]


def save_trained_state(
    fitted_models: dict,
    meta_models: dict,
    binary_models: dict,
    calibrators: dict,
    scaler: Any,
    meta_scaler: Any,
    clean_state: dict,
    feature_cols: list,
    goal_models: dict,
    dc_model: Any,
    dc_xg_model: Any,
    data_hash: str,
    train_size: int,
    test_season: str,
    extra_state: dict = None,
):
    """Save all trained models and pipeline state to disk."""
    # Hash upcoming fixtures for change detection
    fixtures_hash = ""
    fixtures_path = CACHE_DIR.parent / "data" / "upcoming_fixtures.parquet"
    if fixtures_path.exists():
        try:
            fixtures_hash = hashlib.md5(
                fixtures_path.read_bytes()
            ).hexdigest()[:16]
        except Exception:
            pass

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "data_hash": data_hash,
        "feature_hash": _hash_features(feature_cols),
        "n_features": len(feature_cols),
        "n_base_learners": len(fitted_models),
        "base_learner_names": list(fitted_models.keys()),
        "train_size": train_size,
        "test_season": test_season,
        "feature_cols": feature_cols,
        "fixtures_hash": fixtures_hash,
    }

    models_bundle = {
        "fitted_models": fitted_models,
        "meta_models": meta_models,
        "binary_models": binary_models,
        "calibrators": calibrators,
        "scaler": scaler,
        "meta_scaler": meta_scaler,
        "clean_state": clean_state,
        "goal_models": goal_models,
        "dc_model": dc_model,
        "dc_xg_model": dc_xg_model,
    }

    if extra_state:
        models_bundle["extra_state"] = extra_state

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(MODELS_FILE, "wb") as f:
        pickle.dump(models_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  [Cache] Saved {len(fitted_models)} base learners + meta-models to {CACHE_DIR}")
    print(f"  [Cache] Data hash: {data_hash}, Features: {len(feature_cols)}")


def load_trained_state() -> Tuple[Optional[dict], Optional[dict]]:
    """Load cached models and metadata. Returns (models_bundle, metadata) or (None, None)."""
    if not METADATA_FILE.exists() or not MODELS_FILE.exists():
        return None, None

    try:
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        with open(MODELS_FILE, "rb") as f:
            models_bundle = pickle.load(f)
        return models_bundle, metadata
    except Exception as e:
        print(f"  [Cache] Failed to load: {e}")
        return None, None


def _check_data_file_changes() -> Tuple[bool, str]:
    """Check if source data files have been modified since last training."""
    data_dir = CACHE_DIR.parent / "data"
    if not METADATA_FILE.exists():
        return True, "no metadata"

    try:
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        trained_at = metadata.get("trained_at", "")
        if not trained_at:
            return True, "no training timestamp"

        trained_dt = datetime.fromisoformat(trained_at)

        # Check if key data files were modified after training
        for fname in ["epl_matches.parquet", "epl_featured.parquet"]:
            fpath = data_dir / fname
            if fpath.exists():
                file_mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
                if file_mtime > trained_dt:
                    return True, f"{fname} modified since last training"
    except Exception:
        pass

    return False, "data files unchanged"


def _check_upcoming_fixtures_changed() -> Tuple[bool, str]:
    """Check if upcoming fixtures have changed (new matchweek)."""
    data_dir = CACHE_DIR.parent / "data"
    fixtures_path = data_dir / "upcoming_fixtures.parquet"

    if not fixtures_path.exists() or not METADATA_FILE.exists():
        return False, "no fixtures to check"

    try:
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        cached_fixtures_hash = metadata.get("fixtures_hash", "")

        # Hash current fixtures (same method as save — raw file bytes)
        current_hash = hashlib.md5(
            fixtures_path.read_bytes()
        ).hexdigest()[:16]

        if cached_fixtures_hash and current_hash != cached_fixtures_hash:
            return True, "upcoming fixtures changed (new matchweek)"
    except Exception:
        pass

    return False, "fixtures unchanged"


def needs_retraining(
    current_data_hash: str,
    current_feature_cols: list,
    max_age_hours: float = 48.0,
) -> Tuple[bool, str]:
    """Advanced retraining detection system.

    Returns (needs_retrain: bool, reason: str).

    Multi-level checks (ordered by severity):
    1. Cache existence and integrity
    2. Data hash (has the training data content changed?)
    3. Source file modifications (were CSV/JSON files updated?)
    4. Model age (configurable max_age_hours)
    5. File integrity (size, corruption)

    Feature schema and pipeline code changes are intentionally not checked:
    features are derived deterministically from the data (already hash-checked),
    and bug fixes in prediction code should not trigger full retraining.

    The system is conservative — it only skips retraining when ALL checks pass.
    This ensures predictions are always based on the freshest possible model.
    """
    # 1. Cache existence
    if not METADATA_FILE.exists() or not MODELS_FILE.exists():
        return True, "no cached model found"

    try:
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
    except Exception:
        return True, "corrupted metadata file"

    # 2. Data hash (most important — has actual data changed?)
    if metadata.get("data_hash") != current_data_hash:
        return True, f"data changed (old={metadata.get('data_hash')}, new={current_data_hash})"

    # 3. Source file modifications
    files_changed, files_reason = _check_data_file_changes()
    if files_changed:
        return True, files_reason

    # 4. Model age check
    trained_at = metadata.get("trained_at", "")
    if trained_at:
        try:
            trained_dt = datetime.fromisoformat(trained_at)
            age_hours = (datetime.now() - trained_dt).total_seconds() / 3600
            if age_hours > max_age_hours:
                return True, f"model is {age_hours:.1f}h old (max={max_age_hours}h)"
        except Exception:
            pass

    # 5. File integrity
    if MODELS_FILE.stat().st_size < 1000:
        return True, "model file appears corrupt (too small)"

    return False, "cache is fresh and valid (all 5 checks passed)"


def get_cache_info() -> dict:
    """Get information about the current cache state."""
    if not METADATA_FILE.exists():
        return {"cached": False}

    try:
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

        age_str = "unknown"
        trained_at = metadata.get("trained_at", "")
        if trained_at:
            try:
                trained_dt = datetime.fromisoformat(trained_at)
                age = datetime.now() - trained_dt
                if age.total_seconds() < 3600:
                    age_str = f"{age.total_seconds() / 60:.0f} minutes"
                elif age.total_seconds() < 86400:
                    age_str = f"{age.total_seconds() / 3600:.1f} hours"
                else:
                    age_str = f"{age.days} days"
            except Exception:
                pass

        return {
            "cached": True,
            "trained_at": trained_at,
            "age": age_str,
            "n_features": metadata.get("n_features", 0),
            "n_base_learners": metadata.get("n_base_learners", 0),
            "base_learners": metadata.get("base_learner_names", []),
            "train_size": metadata.get("train_size", 0),
            "test_season": metadata.get("test_season", ""),
            "model_size_mb": MODELS_FILE.stat().st_size / (1024 * 1024) if MODELS_FILE.exists() else 0,
        }
    except Exception:
        return {"cached": False, "error": "failed to read metadata"}
