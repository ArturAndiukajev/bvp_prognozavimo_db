"""
data_preparation.py
===================
Two-mode preprocessing pipeline for macroeconomic nowcasting.

Modes
-----
common_frequency  — aggregate all series to one target frequency (default: monthly)
                    and run the full stationarity pipeline (KPSS/ADF + differencing).
mixed_frequency   — preserve native frequencies (D, W, M, Q).
                    Apply lighter stationarity transforms suitable for MIDAS /
                    Bridge Equation / mixed-frequency DFM models.

Usage examples
--------------
# Common-frequency (monthly) pipeline, all providers
python scripts/data_preparation.py --mode common_frequency

# Mixed-frequency pipeline, limit to 100 series for a quick test
python scripts/data_preparation.py --mode mixed_frequency --limit-series 100

# Common-frequency with custom settings
python scripts/data_preparation.py --mode common_frequency --min-obs 48 --kpss-alpha 0.10
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import warnings
import signal
import traceback
import gc
import shutil
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
import time
import concurrent.futures
from sqlalchemy.exc import OperationalError

# Project root = parent of this script's directory (scripts/ -> project root)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUT_DIR = str(_PROJECT_ROOT / "data" / "processed")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from statsmodels.tsa.stattools import kpss, adfuller, acf
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*interpolate.*")
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_prep")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)

engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=10,
    connect_args={"connect_timeout": 10},
)

# ---------------------------------------------------------------------------
# Signals & Checkpointing
# ---------------------------------------------------------------------------
_INTERRUPT_REQUESTED = False

def _signal_handler(signum, frame):
    global _INTERRUPT_REQUESTED
    if _INTERRUPT_REQUESTED:
        logger.warning("\n[INTERRUPT] Forced immediate exit!")
        sys.exit(1)
    logger.warning("\n[INTERRUPT] Signal received. Will finish current chunk, save smoothly, and exit. Press Ctrl+C again to force quit.")
    _INTERRUPT_REQUESTED = True

class CheckpointManager:
    def __init__(self, out_dir: Path, run_suffix: str, chunk_size: int, mode: str):
        self.chk_dir = out_dir / f".checkpoints_{run_suffix}"
        self.manifest_file = self.chk_dir / "manifest.json"
        self.chunk_size = chunk_size
        self.mode = mode
        self.completed_chunks = set()

    def setup(self, resume: bool):
        if resume and self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r") as f:
                    data = json.load(f)
                    self.completed_chunks = set(data.get("completed_chunks", []))
                    if self.completed_chunks:
                        logger.info(f"Resuming from checkpoint: skipped {len(self.completed_chunks)} completed chunks.")
            except Exception as e:
                logger.warning(f"Failed to read manifest: {e}. Starting fresh.")
                self.completed_chunks = set()
        
        if not resume and self.chk_dir.exists():
            logger.info("Starting fresh: clearing old partial checkpoints.")
            shutil.rmtree(self.chk_dir, ignore_errors=True)

        self.chk_dir.mkdir(parents=True, exist_ok=True)

    def mark_chunk_completed(self, chunk_idx: int):
        self.completed_chunks.add(chunk_idx)
        with open(self.manifest_file, "w") as f:
            json.dump({
                "mode": self.mode,
                "chunk_size": self.chunk_size,
                "completed_chunks": sorted(list(self.completed_chunks)),
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, f, indent=4)

    def is_completed(self, chunk_idx: int) -> bool:
        return chunk_idx in self.completed_chunks

# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class PrepMode(str, Enum):
    COMMON_FREQUENCY = "common_frequency"
    MIXED_FREQUENCY = "mixed_frequency"


@dataclass
class PrepConfig:
    # General
    mode: PrepMode = PrepMode.COMMON_FREQUENCY
    min_obs: int = 8
    fill_internal_limit: int = 3
    impute_ragged_edges: bool = False
    ragged_edges_limit: int = 4

    # Fast Mode & Scalability
    fast_mode: bool = False
    allowed_frequencies: str = "D,W,M,Q,A"
    seasonality_mode: str = "full"      # 'full', 'fast', 'metadata_only'
    report_level: str = "full"          # 'full', 'compact'
    prefilter_max_series: Optional[int] = None

    # Common-frequency options
    target_freq: str = "ME"           # pandas offset alias for the target frequency
    daily_weekly_rule: str = "mean"   # how to aggregate D/W to target

    # Stationarity (common-frequency)
    use_yoy: bool = True
    kpss_alpha: float = 0.05
    max_diffs: int = 2                # max extra diffs applied after base transform

    # Stationarity (mixed-frequency stationarity (lighter)
    mf_max_diffs: int = 1             # only 1 extra diff in mixed-frequency mode
    mf_use_log_diff: bool = True      # use log-diff(1) instead of YoY for MF

    # New Requirements
    min_span_years: float = 14.0      # Minimum span between first and last valid observation

    # Execution
    use_multiprocess: bool = False    # True = ProcessPoolExecutor (Linux/macOS); False = Thread
    n_workers: int = os.cpu_count() or 4
    
    # Checkpointing & Scalability
    chunk_size: int = 1000
    resume_from_checkpoint: bool = False
    debug: bool = False


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def kpss_pvalue(x: pd.Series, regression: str = "c") -> Optional[float]:
    """KPSS p-value. Returns None if series is too short or test fails."""
    x = x.dropna()
    if len(x) < 10:
        return None
    try:
        _, pval, _, _ = kpss(x.values, regression=regression, nlags="auto")
        return float(pval)
    except Exception:
        return None


def adf_pvalue(x: pd.Series, regression: str = "c") -> Optional[float]:
    """ADF p-value. Returns None if series is too short or test fails."""
    x = x.dropna()
    if len(x) < 15:
        return None
    try:
        res = adfuller(x.values, regression=regression, autolag="AIC")
        return float(res[1])
    except Exception:
        return None


def safe_log(x: pd.Series) -> pd.Series:
    """Log transform only when strictly positive."""
    if (x.dropna() <= 0).any():
        return x
    return np.log(x)


def winsorize_series(s: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """Clips the extreme 1% tails of the distribution to dampen severe outliers."""
    s_clean = s.dropna()
    if s_clean.empty:
        return s
    lower = s_clean.quantile(limits[0])
    upper = s_clean.quantile(1 - limits[1])
    return s.clip(lower=lower, upper=upper)


def yoy_transform(x: pd.Series, periods: int) -> pd.Series:
    """Log-diff YoY if strictly positive, else simple diff YoY."""
    if (x.dropna() <= 0).any():
        return x.diff(periods)
    return np.log(x).diff(periods)


def is_almost_constant(x: pd.Series, tol: float = 1e-12) -> bool:
    x = x.dropna()
    if len(x) < 10:
        return True
    return float(np.nanstd(x.values)) < tol


def calculate_span_years(s: pd.Series) -> float:
    """Calculate effective data span in calendar years (first to last valid observation)."""
    valid_dates = s.dropna().index
    if len(valid_dates) < 2:
        return 0.0
    return (valid_dates.max() - valid_dates.min()).days / 365.25


def normalize_dropped_reason(value):
    return None if pd.isna(value) else value

def derive_status(dropped_reason):
    return "success" if normalize_dropped_reason(dropped_reason) is None else "dropped"

def create_report_row(
    sid: Any, 
    freq: str, 
    mode: str, 
    dropped_reason: Optional[str] = None,
    span_years: float = 0.0,
    initial_obs: int = 0,
    final_obs: int = 0,
    transform: Optional[str] = None
) -> Dict[str, Any]:
    """Standardized diagnostic row for all stages (pre-filter, worker, success, fail)."""
    dropped_reason = normalize_dropped_reason(dropped_reason)
    return {
        "series_id": sid,
        "frequency": freq,
        "mode": mode,
        "dropped_reason": dropped_reason,
        "transform_applied": transform,
        "span_years": round(span_years, 2),
        "initial_non_nan": initial_obs,
        "final_non_nan": final_obs,
        "is_seasonal": False,
        "is_sa": False,
        "d0_kpss_p": None,
        "diffs_used": 0,
        "status": derive_status(dropped_reason)
    }


def detect_seasonality(s: pd.Series, series_meta: pd.Series, cfg: PrepConfig) -> Dict[str, Any]:
    """
    Detect if a series is seasonal or seasonally adjusted.
    Uses metadata keywords first. Falls back to statistical ACF check ONLY if unknown.
    """
    # 1. Metadata check
    meta_str = " ".join([str(v) for v in series_meta.values]).lower()

    sa_keywords = ["seasonal adjustment", "seasonally adjusted", " sa", "_sa", " -sa"]
    nsa_keywords = ["not seasonally adjusted", "unadjusted", " nsa", "_nsa", " -nsa"]

    is_sa_meta = any(k in meta_str for k in sa_keywords)
    is_nsa_meta = any(k in meta_str for k in nsa_keywords)

    mode = cfg.seasonality_mode if not cfg.fast_mode else "metadata_only"

    # 2. Statistical check (ACF at seasonal lags)
    is_seasonal_stat = False
    acf_val = 0.0

    # --> YOUR GENIUS LOGIC: Only run math if metadata is unknown! <--
    if mode != "metadata_only" and not is_sa_meta and not is_nsa_meta:
        freq = str(series_meta.get("frequency", "")).upper()
        lag = 12 if "M" in freq else (4 if "Q" in freq else None)
        
        if lag is not None:
            min_len = (lag * 3) if mode == "fast" else (lag * 2)

            if len(s.dropna()) > min_len:
                try:
                    # Calculate ACF on first difference to remove trend
                    diffed = s.dropna().diff(1).dropna()
                    if len(diffed) > lag:
                        vals = acf(diffed, nlags=lag, fft=True)
                        acf_val = vals[lag]
                        threshold = 0.3 if mode == "fast" else 0.2
                        if acf_val > threshold:
                            is_seasonal_stat = True
                except Exception:
                    pass

    # 3. Decision Logic
    if is_nsa_meta or is_seasonal_stat:
        status = "seasonal"
        reason = "nsa_metadata" if is_nsa_meta else f"acf_lag_{lag if 'lag' in locals() and lag else 'N/A'}_stat"
    elif is_sa_meta:
        status = "seasonally_adjusted"
        reason = "sa_metadata"
    else:
        status = "unknown"
        reason = "no_strong_evidence"

    return {
        "status": status,
        "reason": reason,
        "is_sa": status == "seasonally_adjusted",
        "is_seasonal": status == "seasonal",
        "acf_seasonal": acf_val
    }


def load_signature_cache(out_dir: Path) -> dict:
    cache_path = out_dir / "stationarity_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
    return {}

def save_signature_cache(out_dir: Path, current_cache: dict, new_reports: list):
    cache_path = out_dir / "stationarity_cache.json"
    
    # Update the dictionary with new successful runs
    for diag in new_reports:
        if diag.get("dropped_reason") is None: # Only cache successful series
            sid_str = str(diag["series_id"])
            current_cache[sid_str] = {
                "obs_count": diag["initial_non_nan"],
                "diffs_used": diag["diffs_used"],
                "transform_applied": diag["transform_applied"],
                "is_seasonal": diag["is_seasonal"],
                "is_sa": diag["is_sa"]
            }
            
    with open(cache_path, "w") as f:
        json.dump(current_cache, f, indent=4)


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------

def handle_missing_values(s: pd.Series, limit: int = 3) -> pd.Series:
    """
    Interpolate small INTERNAL gaps (≤ limit) in a DatetimeIndex series.
    Does not extrapolate beyond the last valid observation.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        # Fall back to linear interpolation for non-datetime indices
        return s.interpolate(method="linear", limit=limit, limit_area="inside")
    return s.interpolate(method="time", limit=limit, limit_area="inside")


def handle_ragged_edges(s: pd.Series, impute: bool = False, limit: int = 2) -> pd.Series:
    """Optionally forward-fill trailing NaNs (ragged right edge)."""
    if impute:
        return s.ffill(limit=limit)
    return s


def custom_impute(s: pd.Series) -> Tuple[pd.Series, int, int]:
    """
    Custom imputation rules:
    - 1-gap: average of neighbors
    - 2-3 gaps: linear interpolation
    - >3 gaps: left empty
    Returns (imputed_series, count_avg_1, count_linear_23)
    """
    if s.notna().all():
        return s, 0, 0

    s_clean = s.copy()
    mask = s.isna()
    # Find blocks of NaNs
    # Using shift-difference to identify starts/ends of NaN blocks
    nan_blocks = (mask != mask.shift()).astype(int).cumsum()
    nan_groups = mask.groupby(nan_blocks)

    avg_count = 0
    linear_count = 0

    for name, group in nan_groups:
        if not group.all(): # Not a NaN block
            continue

        gap_len = len(group)
        start_idx = group.index[0]
        end_idx = group.index[-1]

        # Get position in original series to check neighbors
        loc_start = s.index.get_loc(start_idx)
        loc_end = s.index.get_loc(end_idx)

        # Avoid extrapolating at edges
        if loc_start == 0 or loc_end == len(s) - 1:
            continue

        if gap_len == 1:
            # fill with mean of neighbors
            prev_val = s.iloc[loc_start - 1]
            next_val = s.iloc[loc_end + 1]
            if pd.notna(prev_val) and pd.notna(next_val):
                s_clean.iloc[loc_start] = (prev_val + next_val) / 2
                avg_count += 1
        elif 2 <= gap_len <= 3:
            # linear interpolation
            prev_val = s.iloc[loc_start - 1]
            next_val = s.iloc[loc_end + 1]
            if pd.notna(prev_val) and pd.notna(next_val):
                # Simple linear interpolation for 2 or 3 values
                # pandas interpolate with limit=3 and limit_area='inside' handles this
                # but we only want it for blocks of EXACTLY 2 or 3.
                # Use a temporary slice for interpolation
                subset = s.iloc[loc_start-1 : loc_end+2].interpolate(method='linear')
                s_clean.iloc[loc_start : loc_end+1] = subset.iloc[1:-1]
                linear_count += gap_len

    return s_clean, avg_count, linear_count


# ---------------------------------------------------------------------------
# Frequency Alignment — Common-Frequency Mode
# ---------------------------------------------------------------------------

_FREQ_MAP = {
    "D": "daily",
    "W": "weekly",
    "M": "monthly",
    "ME": "monthly",
    "MS": "monthly",
    "Q": "quarterly",
    "QE": "quarterly",
    "QS": "quarterly",
    "A": "annual",
    "Y": "annual",
}

def monthly_resample(s: pd.Series, freq: str, daily_weekly_rule: str = "mean") -> pd.Series:
    """Resample a series to standard month-end frequency ('ME')."""
    if s.empty:
        return s

    if freq in ("D", "W"):
        if daily_weekly_rule == "last":
            return s.resample("ME").last()
        elif daily_weekly_rule == "sum":
            return s.resample("ME").sum(min_count=1)
        else:
            return s.resample("ME").mean()
    elif freq in ("M", "ME", "MS"):
        return s.resample("ME").last()
    elif freq in ("Q", "QE", "QS"):
        # Quarterly → keep quarter-end month only (NaN for in-between months)
        return s.resample("Q").last().resample("ME").asfreq()
    else:
        return s.resample("ME").mean()


def resample_to_target(s: pd.Series, freq: str, target_freq: str, rule: str = "mean") -> pd.Series:
    """Generic resampler: handles D/W/M → target_freq (e.g. ME, QE)."""
    if s.empty:
        return s
    if target_freq in ("ME", "M", "MS"):
        return monthly_resample(s, freq, rule)
    elif target_freq in ("QE", "Q", "QS"):
        if freq in ("D", "W", "M", "ME", "MS"):
            if rule == "last":
                return s.resample("QE").last()
            elif rule == "sum":
                return s.resample("QE").sum(min_count=1)
            else:
                return s.resample("QE").mean()
        elif freq in ("Q", "QE", "QS"):
            return s.resample("QE").last()
        else:
            return s.resample("QE").mean()
    else:
        return s.resample(target_freq).mean()


# ---------------------------------------------------------------------------
# Data Extraction
# ---------------------------------------------------------------------------

def safe_read_sql(sql_query, engine, params=None, max_retries=3, backoff_factor=2.0) -> pd.DataFrame:
    """Executes a pd.read_sql call with exponential backoff and jitter for connection drops."""
    import random
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                return pd.read_sql(sql_query, conn, params=params)
        except OperationalError as e:
            if attempt == max_retries - 1:
                logger.error(f"Max retries exceeded reading from DB. Fatal Error: {e}")
                raise
            
            jitter = random.uniform(0, 1.0)
            sleep_time = (backoff_factor ** attempt) + jitter
            logger.warning(f"DB Read Error (Attempt {attempt+1}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}")
            time.sleep(sleep_time)
            # Dispose engine pool to violently clear broken socket connections
            engine.dispose()

def fetch_metadata(
    providers: Optional[List[str]] = None,
    dataset_key: Optional[str] = None,
    limit_datasets: Optional[int] = None,
    limit_series: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely retrieves strictly the metadata for the target series using bound parameters.
    """
    where_clauses = []
    params = {}

    if dataset_key:
        where_clauses.append("d.key = :dataset_key")
        params["dataset_key"] = dataset_key

    if providers:
        where_clauses.append("p.name = ANY(:providers)")
        params["providers"] = providers  

    if limit_datasets:
        # Use a subquery to robustly limit datasets within the *already filtered* provider scope
        prov_filter = "p_sub.name = ANY(:providers)" if providers else "1=1"
        where_clauses.append(f"""
            d.id IN (
                SELECT d_sub.id FROM datasets d_sub
                JOIN providers p_sub ON d_sub.provider_id = p_sub.id
                WHERE {prov_filter}
                ORDER BY d_sub.id
                LIMIT :limit_datasets
            )
        """)
        params["limit_datasets"] = limit_datasets

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    # We apply limit_series at the very end to guarantee it limits within the correctly selected scope
    limit_sql = "LIMIT :limit_series" if limit_series else ""
    if limit_series:
        params["limit_series"] = limit_series

    meta_sql = text(f"""
        SELECT 
            s.id AS series_id, s.key AS series_key, s.frequency, 
            s.country, s.transform, s.unit, 
            d.key AS dataset_key, p.name AS provider
        FROM series s
        JOIN datasets d ON d.id = s.dataset_id
        JOIN providers p ON p.id = d.provider_id
        {where_sql}
        ORDER BY s.id
        {limit_sql}
    """)

    return safe_read_sql(meta_sql, engine, params=params)


def fetch_obs_chunk(chunk_sids: List[int], min_date: str) -> pd.DataFrame:
    """
    Fetches exactly one chunk of observations from DB.
    """
    if not chunk_sids:
        return pd.DataFrame()
        
    obs_sql = text("""
        SELECT series_id, period_date, value
        FROM observations
        WHERE series_id = ANY(:chunk_ids)
          AND period_date >= :min_date
    """)
    chunk_params = {"chunk_ids": chunk_sids, "min_date": min_date}
    return safe_read_sql(obs_sql, engine, params=chunk_params)





# (Legacy intermediate panel builders removed in favor of streaming)


# ---------------------------------------------------------------------------
# Stationarity Pipeline — Common-Frequency Mode
# ---------------------------------------------------------------------------

def apply_transform_and_diff(s: pd.Series, transform_applied: str, diffs_used: int) -> pd.Series:
    """Applies a specific base transformation and differencing to a series."""
    if "yoy_diff" in transform_applied:
        periods = int(transform_applied.split("_")[-1])
        z = yoy_transform(s, periods=periods)
    else:
        z = safe_log(s).diff(1)
        
    if diffs_used > 0:
        z = z.diff(diffs_used)
        
    return z


def _process_cf_series(args: Tuple) -> Tuple[int, pd.Series, Dict]:
    """Worker: resamples + transforms + stationarity filter for one series (CF mode)."""
    sid, vals, dates, series_meta, cfg, cache_dict = args
    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    freq = str(series_meta.get("frequency", ""))
    x = resample_to_target(s, freq, cfg.target_freq, cfg.daily_weekly_rule)

    diag: Dict[str, Any] = {
        "series_id": sid,
        "frequency": series_meta.get("frequency", ""),
        "mode": "common_frequency",
        "dropped_reason": None,
        "transform_applied": None,
        "span_years": calculate_span_years(x),
        "initial_non_nan": int(x.notna().sum()),
        "gaps_avg_1": 0,
        "gaps_linear_23": 0,
        "is_seasonal": False,
        "is_sa": False,
        "detect_reason": None,
        "d0_kpss_p": None,
        "diffs_used": 0,
        "final_non_nan": 0,
    }

    # 1. Span Check
    if diag["span_years"] < cfg.min_span_years:
        diag["dropped_reason"] = f"too_short_span_{diag['span_years']:.1f}<{cfg.min_span_years}"
        return sid, x, diag

    if diag["initial_non_nan"] < cfg.min_obs:
        diag["dropped_reason"] = f"too_few_obs_initial_<{cfg.min_obs}"
        return sid, x, diag

    # 2. Custom Imputation
    x_imputed, c1, c23 = custom_impute(x)
    diag["gaps_avg_1"] = c1
    diag["gaps_linear_23"] = c23

    sid_str = str(sid)
    current_count = diag["initial_non_nan"]

    # Check if we have a valid cache hit (data hasn't grown by more than 10%)
    if sid_str in cache_dict and current_count <= cache_dict[sid_str].get("obs_count", 0) * 1.1:
        # 🟢 CACHE HIT: Skip heavy stats and use known signature
        cached = cache_dict[sid_str]
        diag["is_seasonal"] = cached["is_seasonal"]
        diag["is_sa"] = cached["is_sa"]
        diag["detect_reason"] = "loaded_from_cache"
        diag["transform_applied"] = cached["transform_applied"]
        diag["diffs_used"] = cached["diffs_used"]

    else:
        # 🔴 CACHE MISS: Run expensive tests
        # 3. Seasonality Detection
        s_diag = detect_seasonality(x_imputed, series_meta, cfg)
        diag["is_seasonal"] = s_diag["is_seasonal"]
        diag["is_sa"] = s_diag["is_sa"]
        diag["detect_reason"] = s_diag["reason"]

        # 4. Adaptive Base Transformation
        if diag["is_seasonal"]:
            periods = 4 if str(diag["frequency"]).upper() in ("Q", "QE", "QS") else 12
            diag["transform_applied"] = f"yoy_diff_{periods}"
        else:
            diag["transform_applied"] = "log_diff_1"

        # Temporary transform to determine diffs_used via KPSS
        if "yoy_diff" in diag["transform_applied"]:
            temp_z = yoy_transform(x_imputed, periods=int(diag["transform_applied"].split("_")[-1]))
        else:
            temp_z = safe_log(x_imputed).diff(1)

        diag["d0_kpss_p"] = kpss_pvalue(temp_z)

        diffs_applied = 0
        for d in range(1, cfg.max_diffs + 1):
            p = kpss_pvalue(temp_z)
            if p is None or p >= cfg.kpss_alpha:   # stationary → stop
                break
            temp_z = temp_z.diff(1)
            diffs_applied += 1
            diag[f"d{d}_kpss_p"] = kpss_pvalue(temp_z)

        diag["diffs_used"] = diffs_applied

    # Unified Transformation & Differencing Application
    final = apply_transform_and_diff(x_imputed, diag["transform_applied"], diag["diffs_used"])

    

    # --> ADD THIS LINE: Protect models from extreme shocks (like COVID)
    final = winsorize_series(final, limits=(0.01, 0.01))

    # 6. Ragged edges
    final = handle_ragged_edges(final, impute=cfg.impute_ragged_edges, limit=cfg.ragged_edges_limit)

    # 7. Quality filters
    diag["final_non_nan"] = int(final.notna().sum())

    if is_almost_constant(final):
        diag["dropped_reason"] = "almost_constant"
        return sid, final, diag

    if diag["final_non_nan"] < cfg.min_obs:
        diag["dropped_reason"] = f"too_few_obs_final_<{cfg.min_obs}"
        return sid, final, diag

    p_final = kpss_pvalue(final)
    if p_final is not None and p_final < cfg.kpss_alpha:
        diag["dropped_reason"] = "kpss_nonstationary_after_diffs"

    if cfg.report_level == "compact" or cfg.fast_mode:
        compact_diag = {
            "series_id": diag["series_id"],
            "frequency": diag["frequency"],
            "dropped_reason": diag["dropped_reason"],
            "transform_applied": diag["transform_applied"],
            "final_non_nan": diag["final_non_nan"],
            "span_years": round(diag["span_years"], 2)
        }
        return sid, final, compact_diag

    return sid, final, diag


# ---------------------------------------------------------------------------
# Stationarity Pipeline — Mixed-Frequency Mode
# ---------------------------------------------------------------------------
# Stationarity Pipeline — Mixed-Frequency Mode
# ---------------------------------------------------------------------------

def _process_mf_series(args: Tuple) -> Tuple[int, pd.Series, Dict]:
    """
    Worker: lighter transforms for one series (MF mode).
    """
    sid, vals, dates, series_meta, cfg, canon_freq, cache_dict = args
    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()

    # Resample to regularize the Native frequency grid
    _resample_rule = {"D": "D", "W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}
    _agg_func = {"D": "last", "W": "mean", "M": "last", "Q": "last", "A": "last"}

    rule = _resample_rule.get(canon_freq, "ME")
    agg = _agg_func.get(canon_freq, "last")

    if agg == "last":
        x = s.resample(rule).last()
    elif agg == "mean":
        x = s.resample(rule).mean()
    else:
        x = s.resample(rule).last()

    diag: Dict[str, Any] = {
        "series_id": sid,
        "frequency": series_meta.get("frequency", ""),
        "mode": "mixed_frequency",
        "dropped_reason": None,
        "transform_applied": None,
        "span_years": calculate_span_years(x),
        "initial_non_nan": int(x.notna().sum()),
        "gaps_avg_1": 0,
        "gaps_linear_23": 0,
        "is_seasonal": False,
        "is_sa": False,
        "detect_reason": None,
        "d0_kpss_p": None,
        "diffs_used": 0,
        "final_non_nan": 0,
    }

    # 1. Span Check
    if diag["span_years"] < cfg.min_span_years:
        diag["dropped_reason"] = f"too_short_span_{diag['span_years']:.1f}<{cfg.min_span_years}"
        return sid, x, diag

    if diag["initial_non_nan"] < cfg.min_obs:
        diag["dropped_reason"] = f"too_few_obs_initial_<{cfg.min_obs}"
        return sid, x, diag

    # 2. Custom Imputation
    x_imputed, c1, c23 = custom_impute(x)
    diag["gaps_avg_1"] = c1
    diag["gaps_linear_23"] = c23

    sid_str = str(sid)
    current_count = diag["initial_non_nan"]

    if sid_str in cache_dict and current_count <= cache_dict[sid_str].get("obs_count", 0) * 1.1:
        # 🟢 CACHE HIT
        cached = cache_dict[sid_str]
        diag["is_seasonal"] = cached["is_seasonal"]
        diag["is_sa"] = cached["is_sa"]
        diag["detect_reason"] = "loaded_from_cache"
        diag["transform_applied"] = cached["transform_applied"]
        diag["diffs_used"] = cached["diffs_used"]

    else:
        # 🔴 CACHE MISS
        # 3. Seasonality Detection
        s_diag = detect_seasonality(x_imputed, series_meta, cfg)
        diag["is_seasonal"] = s_diag["is_seasonal"]
        diag["is_sa"] = s_diag["is_sa"]
        diag["detect_reason"] = s_diag["reason"]

        # 4. Adaptive Transformation (MF mode)
        if diag["is_seasonal"]:
            freq_str = str(diag["frequency"]).upper()
            if "M" in freq_str: periods = 12
            elif "Q" in freq_str: periods = 4
            elif "W" in freq_str: periods = 52
            elif "D" in freq_str: periods = 365
            else: periods = 1

            diag["transform_applied"] = f"yoy_diff_{periods}"
        else:
            diag["transform_applied"] = "log_diff_1"

        if "yoy_diff" in diag["transform_applied"]:
            temp_z = yoy_transform(x_imputed, periods=int(diag["transform_applied"].split("_")[-1]))
        else:
            temp_z = safe_log(x_imputed).diff(1)

        diag["d0_kpss_p"] = kpss_pvalue(temp_z)

        # 5. At most 1 extra diff for MF
        diffs_applied = 0
        p = kpss_pvalue(temp_z)
        if p is not None and p < cfg.kpss_alpha and cfg.mf_max_diffs >= 1:
            temp_z = temp_z.diff(1)
            diffs_applied = 1
            diag["d1_kpss_p"] = kpss_pvalue(temp_z)

        diag["diffs_used"] = diffs_applied

    # Unified Transformation & Differencing Application
    final = apply_transform_and_diff(x_imputed, diag["transform_applied"], diag["diffs_used"])

    # 6. Ragged edges
    final = handle_ragged_edges(final, impute=cfg.impute_ragged_edges, limit=cfg.ragged_edges_limit)

    # 7. Quality filters
    diag["final_non_nan"] = int(final.notna().sum())

    if is_almost_constant(final):
        diag["dropped_reason"] = "almost_constant"
        return sid, final, diag

    if diag["final_non_nan"] < cfg.min_obs:
        diag["dropped_reason"] = f"too_few_obs_final_<{cfg.min_obs}"
        return sid, final, diag

    p_final = kpss_pvalue(final)
    if p_final is not None and p_final < cfg.kpss_alpha:
        diag["kpss_warning"] = f"still_nonstationary_p={p_final:.3f}"

    if cfg.report_level == "compact" or cfg.fast_mode:
        compact_diag = {
            "series_id": diag["series_id"],
            "frequency": diag["frequency"],
            "dropped_reason": diag["dropped_reason"],
            "transform_applied": diag["transform_applied"],
            "final_non_nan": diag["final_non_nan"],
            "span_years": round(diag["span_years"], 2)
        }
        return sid, final, compact_diag

    return sid, final, diag


# ---------------------------------------------------------------------------
# Parallel Panel Preparation
# ---------------------------------------------------------------------------

def _run_parallel(
    args_list: List[Tuple],
    worker_fn,
    cfg: PrepConfig,
    label: str,
    chunk_idx: int = 1,
    total_chunks: int = 1
) -> Tuple[Dict[int, pd.Series], List[Dict]]:
    """
    Run worker_fn over args_list using ThreadPoolExecutor (Windows-safe default)
    or ProcessPoolExecutor when cfg.use_multiprocess=True.
    """
    total = len(args_list)
    out_valid: Dict[int, pd.Series] = {}
    report_rows: List[Dict] = []
    t0 = time.perf_counter()

    Executor = (
        concurrent.futures.ProcessPoolExecutor
        if cfg.use_multiprocess
        else concurrent.futures.ThreadPoolExecutor
    )

    with Executor(max_workers=cfg.n_workers) as executor:
        futures = {executor.submit(worker_fn, arg): arg[0] for arg in args_list}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if _INTERRUPT_REQUESTED:
                # Let pool finish naturally but don't log normal progress to not spam
                pass

            try:
                sid, final_series, diag = future.result()
            except Exception as exc:
                sid = futures[future]
                if cfg.debug:
                    logger.error(f"  [{label}] series {sid} worker exception:\n{traceback.format_exc()}")
                else:
                    logger.error(f"  [{label}] series {sid} raised: {exc}")
                diag = {"series_id": sid, "dropped_reason": f"worker_exception: {exc}"}
                report_rows.append(diag)
                continue

            report_rows.append(diag)
            if normalize_dropped_reason(diag.get("dropped_reason")) is None:
                out_valid[sid] = final_series

            if i % 100 == 0 or i == total:
                dt = time.perf_counter() - t0
                rate = i / dt if dt > 0 else 0
                logger.info(
                    f"  [{label} Chunk {chunk_idx}/{total_chunks}] {i}/{total} ({i/total:.1%}) "
                    f"| kept={len(out_valid)} | speed={rate:.1f} it/s"
                )

    return out_valid, report_rows


# def prepare_panel_cf_chunk(
#     df_chunk: pd.DataFrame, meta: pd.DataFrame, cfg: PrepConfig,
#     chk: CheckpointManager, chunk_idx: int, total_chunks: int
# ) -> List[Dict]:
#     report_rows = []
#     if df_chunk.empty:
#         chk.mark_chunk_completed(chunk_idx)
#         return report_rows

#     groups = list(df_chunk.groupby("series_id"))
#     args_list = []
#     for sid, group in groups:
#         vals = group["value"].values
#         dates = group["period_date"].values 
#         series_meta = meta.loc[sid] if sid in meta.index else pd.Series()
#         args_list.append((sid, vals, dates, series_meta, cfg))

#     out_valid, chunk_report_rows = _run_parallel(args_list, _process_cf_series, cfg, "CF", chunk_idx, total_chunks)


def prepare_panel_cf_chunk(
    df_chunk: pd.DataFrame, meta: pd.DataFrame, cfg: PrepConfig,
    chk: CheckpointManager, chunk_idx: int, total_chunks: int,
    stationarity_cache: dict
) -> List[Dict]:
    report_rows = []
    if df_chunk.empty:
        chk.mark_chunk_completed(chunk_idx)
        return report_rows

    groups = list(df_chunk.groupby("series_id", observed=True))
    args_list = []
    for sid, group in groups:
        vals = group["value"].values
        dates = group["period_date"].values 
        series_meta = meta.loc[sid] if sid in meta.index else pd.Series()
        args_list.append((sid, vals, dates, series_meta, cfg, stationarity_cache))

    out_valid, chunk_report_rows = _run_parallel(args_list, _process_cf_series, cfg, "CF", chunk_idx, total_chunks)
    report_rows.extend(chunk_report_rows)

    prepared = pd.DataFrame(out_valid).sort_index() if out_valid else pd.DataFrame()
    if not prepared.empty:
        prepared.index.name = "period_date"
        out_path = chk.chk_dir / f"panel_chunk_{chunk_idx}.parquet"
        prepared.reset_index().to_parquet(out_path, index=False)
        
    del out_valid
    del prepared
    del args_list
    return report_rows


# Groups of frequency codes that should be treated together
_MF_FREQ_GROUPS: Dict[str, List[str]] = {
    "D":  ["D", "d", "BD"],
    "W":  ["W", "w", "WE", "WS"],
    "M":  ["M", "ME", "MS", "m"],
    "Q":  ["Q", "QE", "QS", "q"],
    "A":  ["A", "Y", "AE", "AS", "YS"],
}

def _canonical_mf_freq(freq: str) -> str:
    """Map a raw DB frequency string to a canonical MF group key."""
    freq_upper = (freq or "").strip().upper()
    for canon, aliases in _MF_FREQ_GROUPS.items():
        if freq_upper in [a.upper() for a in aliases]:
            return canon
    return "M"  # default to monthly if unknown


# def prepare_panel_mf_chunk(
#     df_chunk: pd.DataFrame, meta: pd.DataFrame, cfg: PrepConfig,
#     chk: CheckpointManager, chunk_idx: int, total_chunks: int
# ) -> List[Dict]:
#     report_rows = []
#     if df_chunk.empty:
#         chk.mark_chunk_completed(chunk_idx)
#         return report_rows

#     groups = list(df_chunk.groupby("series_id"))
#     args_list = []
#     for sid, group in groups:
#         vals = group["value"].values
#         dates = group["period_date"].values
#         series_meta = meta.loc[sid] if sid in meta.index else pd.Series()
#         raw_freq = str(series_meta.get("frequency", ""))
#         canon = _canonical_mf_freq(raw_freq)
#         args_list.append((sid, vals, dates, series_meta, cfg, canon))

def prepare_panel_mf_chunk(
    df_chunk: pd.DataFrame, meta: pd.DataFrame, cfg: PrepConfig,
    chk: CheckpointManager, chunk_idx: int, total_chunks: int,
    stationarity_cache: dict
) -> List[Dict]:
    report_rows = []
    if df_chunk.empty:
        chk.mark_chunk_completed(chunk_idx)
        return report_rows

    groups = list(df_chunk.groupby("series_id", observed=True))
    args_list = []
    for sid, group in groups:
        vals = group["value"].values
        dates = group["period_date"].values
        series_meta = meta.loc[sid] if sid in meta.index else pd.Series()
        raw_freq = str(series_meta.get("frequency", ""))
        canon = _canonical_mf_freq(raw_freq)
        args_list.append((sid, vals, dates, series_meta, cfg, canon, stationarity_cache))

    # Run workers

    out_valid, chunk_report_rows = _run_parallel(args_list, _process_mf_series, cfg, "MF", chunk_idx, total_chunks)
    report_rows.extend(chunk_report_rows)
    
    canon_mapping = {arg[0]: arg[5] for arg in args_list}
    freq_data = {canon: {} for canon in set(canon_mapping.values())}
    
    if out_valid:
        for sid, s_clean in out_valid.items():
            canon = canon_mapping[sid]
            freq_data[canon][sid] = s_clean
            
    for freq, s_dict in freq_data.items():
        if s_dict:
            panel = pd.DataFrame(s_dict).sort_index()
            panel.index.name = "period_date"
            out_path = chk.chk_dir / f"mf_panel_{freq}_chunk_{chunk_idx}.parquet"
            panel.reset_index().to_parquet(out_path, index=False)


    del out_valid
    del freq_data
    del args_list
    return report_rows


def _log_prep_summary(
    label: str,
    total_series_metadata: int,
    total_series_reported: int,
    kept_series: int,
    dropped_series: int,
    report_rows: list,
    min_date_str: Optional[str] = None,
    max_date_str: Optional[str] = None
) -> dict:
    drop_counts = {}
    kept_freqs = {}
    dropped_freqs = {}

    if report_rows:
        drop_reasons = pd.Series([normalize_dropped_reason(r.get("dropped_reason")) for r in report_rows]).value_counts()
        drop_counts = drop_reasons.to_dict()
        reasons_str = ", ".join(f"{r}={c}" for r, c in drop_reasons.items() if r is not None)

        kept_freqs = pd.Series([str(r.get("frequency")) for r in report_rows if normalize_dropped_reason(r.get("dropped_reason")) is None]).value_counts().to_dict()
        dropped_freqs = pd.Series([str(r.get("frequency")) for r in report_rows if normalize_dropped_reason(r.get("dropped_reason")) is not None]).value_counts().to_dict()
    else:
        reasons_str = "n/a"

    transforms = pd.Series([r.get("transform_applied") for r in report_rows if r.get("transform_applied")]).value_counts()
    tx_counts = transforms.to_dict()
    tx_str = ", ".join(f"{t}={c}" for t, c in transforms.items())

    logger.info(
        f"{label} Summary: input_metadata={total_series_metadata} | reported={total_series_reported} | kept={kept_series} | dropped={dropped_series}\n"
        f"  Kept Freqs:   {kept_freqs}\n"
        f"  Drop Freqs:   {dropped_freqs}\n"
        f"  Date Bounds:  [{min_date_str}, {max_date_str}]\n"
        f"  Drop reasons: {reasons_str or 'none'}\n"
        f"  Transforms:   {tx_str or 'none'}"
    )

    return {
        "total_series_metadata": total_series_metadata,
        "total_series_reported": total_series_reported,
        "kept_series": kept_series,
        "dropped_series": dropped_series,
        "kept_frequencies": kept_freqs,
        "dropped_frequencies": dropped_freqs,
        "panel_min_date": min_date_str,
        "panel_max_date": max_date_str,
        "drop_reasons_breakdown": drop_counts,
        "transformations_applied": tx_counts
    }


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def validate_panel(panel: pd.DataFrame, label: str = "panel") -> None:
    """Log shape, NaN share, index type, and warn on issues."""
    if panel.empty:
        logger.warning(f"[validate] {label} is EMPTY")
        return

    nan_share = panel.isna().mean().mean()
    logger.info(
        f"[validate] {label}: shape={panel.shape} | "
        f"nan_share={nan_share:.1%} | "
        f"index_type={type(panel.index).__name__} | "
        f"index_range=[{panel.index.min()}..{panel.index.max()}]"
    )

    if not isinstance(panel.index, pd.DatetimeIndex):
        logger.warning(f"[validate] {label} index is NOT DatetimeIndex -- alignment issues likely!")

    if nan_share > 0.8:
        logger.warning(f"[validate] {label} has >80% NaN -- models may receive near-empty inputs")


# ---------------------------------------------------------------------------
# Output filename helpers
# ---------------------------------------------------------------------------

def sanitize_token(value: str) -> str:
    """
    Make a string safe for use in a filename.
    """
    import re
    s = str(value).strip().lower()
    s = re.sub(r"[\s,./\\=:*?\"<>|]+", "_", s)
    s = re.sub(r"_+", "_", s)           # collapse runs
    s = s.strip("_")
    return s


def build_run_suffix(args, cfg: "PrepConfig", providers_list) -> str:
    """
    Build a descriptive, deterministic, filename-safe suffix from the current
    run configuration.
    """
    parts = []
    parts.append(cfg.mode.value)
    
    if cfg.mode.value == "common_frequency":
        parts.append(str(cfg.target_freq).upper())
        
    prov_str = "-".join(sorted(sanitize_token(p) for p in providers_list)) if providers_list else "all-providers"
    if getattr(args, "dataset_key", None):
        prov_str += f"_{sanitize_token(args.dataset_key)}"
    parts.append(prov_str)

    if getattr(args, "min_date", None):
        parts.append(f"from{args.min_date.replace('-', '')}")
        
    if getattr(args, "max_date", None):
        parts.append(f"max{args.max_date.replace('-', '')}")
    elif getattr(args, "cutoff_date", None):
        parts.append(f"publag{args.cutoff_date.replace('-', '')}")
        
    if hasattr(args, "seasonality_mode") and args.seasonality_mode:
        parts.append(args.seasonality_mode)
    else:
        parts.append(cfg.seasonality_mode)
    
    if cfg.mode.value == "common_frequency":
        tx_mode = "yoy" if cfg.use_yoy else "noyoy"
        parts.append(tx_mode)
        kpss_str = str(cfg.kpss_alpha).replace(".", "")
        parts.append(f"kpss{kpss_str}")
        parts.append(f"diff{cfg.max_diffs}")
    else:  # mixed_frequency
        tx_mode = "logdiff" if cfg.mf_use_log_diff else "diff"
        parts.append(tx_mode)
        kpss_str = str(cfg.kpss_alpha).replace(".", "")
        parts.append(f"kpss{kpss_str}")
        parts.append(f"mfdiff{cfg.mf_max_diffs}")

    if getattr(cfg, "impute_ragged_edges", False):
        parts.append(f"ragged{getattr(cfg, 'ragged_edges_limit', 2)}")

    if getattr(cfg, "fill_internal_limit", 3) != 3:
        parts.append(f"fill{cfg.fill_internal_limit}")

    suffix = "_".join(t for t in parts if t)

    # Safety: cap at 180 chars to stay within OS filename limits
    if len(suffix) > 180:
        important = parts[:2] + parts[-4:]
        suffix = "_".join(t for t in important if t) + "_trunc"

    return suffix


def safe_consolidate_panels(checkpoint_dir: Path, pattern: str) -> pd.DataFrame:
    """
    Safely consolidates many wide-format parquet chunks into a single DataFrame.
    Bypasses pandas outer-join cascades by building a dictionary of single-column Series,
    allowing pd.DataFrame to allocate the entire memory block once optimally.
    """
    import gc
    files = sorted(list(checkpoint_dir.glob(pattern)))
    if not files:
        return pd.DataFrame()
        
    logger.info(f"  Safely bridging {len(files)} parquet chunks via block allocator pass...")
    
    series_dict = {}
    for idx, f in enumerate(files):
        df_chunk = pd.read_parquet(f)
        if "period_date" in df_chunk.columns:
            df_chunk.set_index("period_date", inplace=True)
            
        for col in df_chunk.columns:
            series_dict[col] = df_chunk[col].astype(np.float32)
            
        del df_chunk
        if idx % 10 == 0:
            gc.collect()
            
    final_df = pd.DataFrame(series_dict).sort_index()
    final_df.index.name = "period_date"
    
    del series_dict
    gc.collect()
    
    return final_df


def aggregate_chunk_reports(checkpoint_dir: Path) -> List[Dict]:
    """Reads all report chunks from disk to avoid retaining all_report_rows in memory."""
    files = sorted(list(checkpoint_dir.glob("report_chunk_*.csv")))
    all_rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "dropped_reason" in df.columns:
                df["dropped_reason"] = df["dropped_reason"].apply(normalize_dropped_reason)
            if "status" in df.columns:
                df["status"] = df["dropped_reason"].apply(derive_status)
            all_rows.extend(df.to_dict("records"))
        except Exception as e:
            logger.warning(f"Could not read chunk report {f}: {e}")
    return all_rows





# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Nowcasting Data Preparation Pipeline")

    # Mode
    ap.add_argument(
        "--mode",
        type=str,
        default="common_frequency",
        choices=["common_frequency", "mixed_frequency"],
        help="Preprocessing mode: common_frequency or mixed_frequency",
    )
    ap.add_argument(
        "--target-freq",
        type=str,
        default="ME",
        help="Target frequency for common_frequency mode (pandas offset alias, e.g. ME, QE)",
    )

    # Data selection
    ap.add_argument("--out-dir", type=str, default=_DEFAULT_OUT_DIR,
                    help="Output directory (default: <project_root>/data/processed, CWD-independent)")
    ap.add_argument("--min-date", type=str, default="2000-01-01")
    ap.add_argument("--max-date", type=str, default=None, help="Maximum date to include")
    ap.add_argument("--providers", type=str, default="eurostat,alfred,yahoo_finance",
                    help="Comma-separated provider names")
    ap.add_argument("--dataset-key", type=str, default=None)
    ap.add_argument("--limit-datasets", type=int, default=None)
    ap.add_argument("--limit-series", type=int, default=None)

    # Aggregation rule for D/W series
    ap.add_argument("--daily-weekly-rule", type=str, default="mean", choices=["mean", "last", "sum"])

    # Stationarity (CF)
    ap.add_argument("--min-obs", type=int, default=36)
    ap.add_argument("--no-yoy", action="store_true", help="Use log-diff(1) instead of YoY for CF mode")
    ap.add_argument("--kpss-alpha", type=float, default=0.05)
    ap.add_argument("--max-diffs", type=int, default=2)

    # Stationarity (MF)
    ap.add_argument("--mf-max-diffs", type=int, default=1)
    ap.add_argument("--mf-no-log-diff", action="store_true", help="Use diff(1) instead of log-diff for MF mode")

    # Missing data
    ap.add_argument("--fill-limit", type=int, default=3)
    ap.add_argument("--impute-ragged-edges", action="store_true")
    ap.add_argument("--ragged-edges-limit", type=int, default=2)

    # Execution
    ap.add_argument("--use-multiprocess", action="store_true",
                    help="Use ProcessPoolExecutor instead of ThreadPoolExecutor (better on Linux/macOS)")

    # --- NEW: Scalability & Checkpointing ---
    ap.add_argument("--chunk-size", type=int, default=500, help="Number of series per processing chunk")
    ap.add_argument("--resume-from-checkpoint", action="store_true", help="Resume from last interrupted run checkpoints")
    ap.add_argument("--debug", action="store_true", help="Print verbose worker tracebacks")

    # --- NEW: Pseudo Real-Time Time Machine ---
    ap.add_argument("--cutoff-date", type=str, default=None,
                    help="Simulate a specific date in history to prevent data leakage (YYYY-MM-DD)")

    # --- Fast mode & Scalability ---
    ap.add_argument("--fast-mode", action="store_true", help="Enable fast mode with lighter checks/reports")
    ap.add_argument("--allowed-frequencies", type=str, default=None, help="Comma-separated allowed frequencies (default: D,W,M,Q for CF, D,W,M,Q,A for MF)")
    ap.add_argument("--seasonality-mode", type=str, default="full", choices=["full", "fast", "metadata_only"])
    ap.add_argument("--report-level", type=str, default="full", choices=["full", "compact"])
    ap.add_argument("--min-span-years", type=float, default=14.0)
    ap.add_argument("--prefilter-max-series", type=int, default=None)

    args = ap.parse_args()

    providers_list = [p.strip() for p in args.providers.split(",")] if args.providers else None

    allowed_freqs = args.allowed_frequencies
    if not allowed_freqs:
        if args.mode == "common_frequency":
            allowed_freqs = "D,W,M,Q"
        else:
            allowed_freqs = "D,W,M,Q,A"

    cfg = PrepConfig(
        mode=PrepMode(args.mode),
        fast_mode=args.fast_mode,
        allowed_frequencies=allowed_freqs,
        seasonality_mode=args.seasonality_mode,
        report_level=args.report_level,
        prefilter_max_series=args.prefilter_max_series,
        min_span_years=args.min_span_years,
        target_freq=args.target_freq,
        min_obs=args.min_obs,
        fill_internal_limit=args.fill_limit,
        use_yoy=not args.no_yoy,
        kpss_alpha=args.kpss_alpha,
        max_diffs=args.max_diffs,
        mf_max_diffs=args.mf_max_diffs,
        mf_use_log_diff=not args.mf_no_log_diff,
        daily_weekly_rule=args.daily_weekly_rule,
        impute_ragged_edges=args.impute_ragged_edges,
        ragged_edges_limit=args.ragged_edges_limit,
        use_multiprocess=args.use_multiprocess,
        chunk_size=args.chunk_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug=args.debug,
    )

    logger.info("=== Data Preparation Pipeline ===")
    
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)
    logger.info(f"Mode: {cfg.mode.value} | Providers: {providers_list} | Min Date: {args.min_date}")

    # 1. Extract Metadata
    df_meta = fetch_metadata(
        providers=providers_list,
        dataset_key=args.dataset_key,
        limit_datasets=args.limit_datasets,
        limit_series=args.limit_series,
    )

    if df_meta.empty:
        logger.warning("No metadata retrieved from DB. Pipeline stopped.")
        return

    series_ids = df_meta["series_id"].tolist()
    total_series = len(series_ids)
    total_chunks = math.ceil(total_series / cfg.chunk_size)
    logger.info(f"Extracted {total_series:,} series metadata records. Processing in {total_chunks} chunks.")

    meta_cols = ["series_id", "provider", "dataset_key", "series_key", "frequency", "country", "transform", "unit"]
    meta = df_meta[meta_cols].drop_duplicates("series_id").set_index("series_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_suffix = build_run_suffix(args, cfg, providers_list)
    logger.info(f"Output filename suffix: {run_suffix}")

    chk = CheckpointManager(out_dir, run_suffix, cfg.chunk_size, str(cfg.mode.value))
    chk.setup(cfg.resume_from_checkpoint)

    # 2. Load stationarity cache
    stationarity_cache = load_signature_cache(out_dir)
    logger.info(f"Loaded {len(stationarity_cache)} stationarity signatures from cache.")

    logger.info(f"[{cfg.mode.value}] Pipeline starting...")

    # =========================================================================
    # MAIN LOOP PHASE: Chunk Processing
    # =========================================================================
    for chunk_idx in range(1, total_chunks + 1):
        if _INTERRUPT_REQUESTED:
            logger.warning(f"Interrupt requested. Halting chunk loop before chunk {chunk_idx}.")
            break
            
        if chk.is_completed(chunk_idx):
            logger.info(f"Skipping chunk {chunk_idx}/{total_chunks} (already completed).")
            continue
            
        start_idx = (chunk_idx - 1) * cfg.chunk_size
        end_idx = start_idx + cfg.chunk_size
        chunk_sids = series_ids[start_idx:end_idx]
        
        # 2a. Initialize diagnostics for EVERY series in the metadata chunk (Full Accounting)
        chunk_diagnostics: Dict[int, Dict[str, Any]] = {}
        for sid in chunk_sids:
            s_meta = meta.loc[sid] if sid in meta.index else pd.Series()
            chunk_diagnostics[sid] = create_report_row(sid, str(s_meta.get("frequency", "N/A")), cfg.mode.value, dropped_reason="no_observations_found")

        # 2b. Fetch Observations
        df_chunk_raw = fetch_obs_chunk(chunk_sids, args.min_date)
        if df_chunk_raw.empty:
            report_df = pd.DataFrame(list(chunk_diagnostics.values())).sort_values("series_id")
            report_df.to_csv(chk.chk_dir / f"report_chunk_{chunk_idx}.csv", index=False)
            chk.mark_chunk_completed(chunk_idx)
            continue
        
        # 2c. Merge and Optimize
        df_chunk = df_chunk_raw.merge(df_meta, on="series_id", how="inner")
        df_chunk["value"] = df_chunk["value"].astype(np.float32) 
        df_chunk["series_id"] = df_chunk["series_id"].astype("category") 
        df_chunk["period_date"] = pd.to_datetime(df_chunk["period_date"], errors='coerce')
        df_chunk = df_chunk.dropna(subset=["period_date"])

        # Mark as 'data_found' for metadata IDs that returned rows
        found_ids = df_chunk["series_id"].unique()
        for sid in found_ids:
            chunk_diagnostics[int(sid)]["dropped_reason"] = None

        if df_chunk.empty:
            report_df = pd.DataFrame(list(chunk_diagnostics.values())).sort_values("series_id")
            report_df.to_csv(chk.chk_dir / f"report_chunk_{chunk_idx}.csv", index=False)
            chk.mark_chunk_completed(chunk_idx)
            continue

        # 2d. Cutoff / Time Machine Filter
        if args.cutoff_date:
            cutoff = pd.to_datetime(args.cutoff_date)
            is_q = df_chunk['frequency'].str.contains('Q', case=False, na=False)
            is_m = df_chunk['frequency'].str.contains('M', case=False, na=False)
            q_limit = cutoff - pd.Timedelta(days=60)
            m_limit = cutoff - pd.Timedelta(days=30)
            
            mask = (is_q & (df_chunk['period_date'] <= q_limit)) | \
                   (is_m & (df_chunk['period_date'] <= m_limit)) | \
                   (~(is_q | is_m) & (df_chunk['period_date'] <= cutoff))
            
            dropped_sids = df_chunk.loc[~mask, "series_id"].unique()
            df_chunk = df_chunk[mask]
            for sid in dropped_sids:
                if int(sid) not in df_chunk["series_id"].values: # Only if it was totally removed
                    chunk_diagnostics[int(sid)]["dropped_reason"] = "prefilter_cutoff_date"

        # 2e. Frequency Filter
        if cfg.allowed_frequencies:
            allowed = [x.strip().upper() for x in cfg.allowed_frequencies.split(",")]
            bad_mask = ~df_chunk['frequency'].str.upper().isin(allowed)
            dropped_sids = df_chunk.loc[bad_mask, "series_id"].unique()
            df_chunk = df_chunk[~bad_mask]
            for sid in dropped_sids:
                freq_val = meta.loc[int(sid)]["frequency"] if int(sid) in meta.index else "?"
                chunk_diagnostics[int(sid)]["dropped_reason"] = f"prefilter_disallowed_freq_{freq_val}"

        # 2f. Heuristic Filtering
        if not df_chunk.empty:
            obs_stats = df_chunk.dropna(subset=["value"]).groupby("series_id", observed=True)
            v_counts = obs_stats.size()
            v_bounds = obs_stats["period_date"].agg(["min", "max"])
            v_spans = (v_bounds["max"] - v_bounds["min"]).dt.days / 365.25
            
            too_few = v_counts[v_counts < cfg.min_obs].index
            too_short = v_spans[v_spans < cfg.min_span_years].index
            
            for sid in too_few:
                chunk_diagnostics[int(sid)].update({
                    "dropped_reason": f"prefilter_too_few_obs_{int(v_counts[sid])}<{cfg.min_obs}",
                    "initial_non_nan": int(v_counts[sid]),
                    "span_years": float(v_spans[sid])
                })
            for sid in too_short:
                if chunk_diagnostics[int(sid)]["dropped_reason"] is None:
                    chunk_diagnostics[int(sid)].update({
                        "dropped_reason": f"prefilter_too_short_span_{float(v_spans[sid]):.2f}<{cfg.min_span_years}",
                        "initial_non_nan": int(v_counts[sid]),
                        "span_years": float(v_spans[sid])
                    })

            valid_mask = ~df_chunk['series_id'].isin(too_few) & ~df_chunk['series_id'].isin(too_short)
            df_chunk = df_chunk[valid_mask]

        if df_chunk.empty:
            report_df = pd.DataFrame(list(chunk_diagnostics.values())).sort_values("series_id")
            if not report_df.empty and "dropped_reason" in report_df.columns:
                report_df["dropped_reason"] = report_df["dropped_reason"].apply(normalize_dropped_reason)
                report_df["status"] = report_df["dropped_reason"].apply(derive_status)
            report_df.to_csv(chk.chk_dir / f"report_chunk_{chunk_idx}.csv", index=False)
            chk.mark_chunk_completed(chunk_idx)
            continue

        # 2g. Worker Phase
        if cfg.mode == PrepMode.COMMON_FREQUENCY:
            chunk_results = prepare_panel_cf_chunk(df_chunk, meta, cfg, chk, chunk_idx, total_chunks, stationarity_cache)
        else:
            chunk_results = prepare_panel_mf_chunk(df_chunk, meta, cfg, chk, chunk_idx, total_chunks, stationarity_cache)
        
        for res in chunk_results:
            sid = int(res["series_id"])
            if sid in chunk_diagnostics:
                chunk_diagnostics[sid].update(res)

        # 2h. Save chunk report
        report_df = pd.DataFrame(list(chunk_diagnostics.values())).sort_values("series_id")
        if not report_df.empty and "dropped_reason" in report_df.columns:
            report_df["dropped_reason"] = report_df["dropped_reason"].apply(normalize_dropped_reason)
            report_df["status"] = report_df["dropped_reason"].apply(derive_status)
        report_df.to_csv(chk.chk_dir / f"report_chunk_{chunk_idx}.csv", index=False)
        chk.mark_chunk_completed(chunk_idx)
            
        del df_chunk
        gc.collect()

    # =========================================================================
    # POST-LOOP PHASE: Consolidation & Saving
    # =========================================================================

    out_panel  = out_dir / f"panel_{run_suffix}.parquet"
    out_report = out_dir / f"report_{run_suffix}.csv"
    out_meta   = out_dir / f"meta_{run_suffix}.csv"
    out_summary = out_dir / f"pipeline_summary_{run_suffix}.json"

    if not _INTERRUPT_REQUESTED or chk.completed_chunks:
        logger.info("Reading pipeline diagnostic logs from chunk disk...")
        all_report_rows = aggregate_chunk_reports(chk.chk_dir)
        total_s = len(all_report_rows)

        if total_s != total_series:
            logger.error(f"REPORTING INVARIANT FAILED: Expected {total_series} rows but assembled {total_s} from chunks.")

        # Reconstruct DataFrame immediately to derive consistent status & compute lengths
        report_df = pd.DataFrame(all_report_rows)
        if not report_df.empty and "dropped_reason" in report_df.columns:
            report_df["dropped_reason"] = report_df["dropped_reason"].apply(normalize_dropped_reason)
            report_df["status"] = report_df["dropped_reason"].apply(derive_status)
            
            # Persist the normalized status strings back to list mapping for summary calculations
            all_report_rows = report_df.to_dict("records")
            kept_s = int(report_df["dropped_reason"].isna().sum())
        else:
            kept_s = 0
            
        dropped_s = total_series - kept_s

        save_signature_cache(out_dir, stationarity_cache, report_df.to_dict("records"))
        logger.info("Stationarity cache updated and saved to JSON.")
       
        if cfg.mode == PrepMode.COMMON_FREQUENCY:
            logger.info("Consolidating via Partitioned Parquet (Parallel I/O)...")
            partitioned_dir = out_dir / f"panel_{run_suffix}.dataset"
            partitioned_dir.mkdir(parents=True, exist_ok=True)
            
            min_dt, max_dt = None, None
        
            for f in chk.chk_dir.glob("panel_chunk_*.parquet"):
                chunk_df = pd.read_parquet(f)
                if chunk_df.empty: continue
                    
                chunk_long = chunk_df.melt(id_vars="period_date", var_name="series_id")
                chunk_long["series_id"] = chunk_long["series_id"].astype(int)
                chunk_long = chunk_long.merge(meta[['provider']], left_on="series_id", right_index=True)
                
                current_min = chunk_df["period_date"].min()
                current_max = chunk_df["period_date"].max()
                min_dt = current_min if min_dt is None else min(min_dt, current_min)
                max_dt = current_max if max_dt is None else max(max_dt, current_max)
        
                chunk_long.to_parquet(partitioned_dir, engine='pyarrow', index=False, partition_cols=['provider'])
            
            min_dt_str = min_dt.strftime("%Y-%m-%d") if min_dt else None
            max_dt_str = max_dt.strftime("%Y-%m-%d") if max_dt else None
            
            summary_dict = _log_prep_summary(
                label="[CF]", 
                total_series_metadata=total_series, 
                total_series_reported=total_s, 
                kept_series=kept_s, 
                dropped_series=dropped_s,  
                report_rows=all_report_rows, 
                min_date_str=min_dt_str, 
                max_date_str=max_dt_str
            )
            with open(out_summary, "w") as f:
                json.dump(summary_dict, f, indent=4)
            logger.info(f"Saved: {out_summary}")

        else:
            logger.info("Consolidating MF datasets...")
            freqs_found = {f.stem.split("_")[2] for f in chk.chk_dir.glob("mf_panel_*_chunk_*.parquet")}
            summary_dicts = {}
            
            for freq in freqs_found:
                partitioned_dir = out_dir / f"mf_panel_{freq}_{run_suffix}.dataset"
                partitioned_dir.mkdir(parents=True, exist_ok=True)
                kept_series_count = 0

                for f in chk.chk_dir.glob(f"mf_panel_{freq}_chunk_*.parquet"):
                    chunk_df = pd.read_parquet(f)
                    if chunk_df.empty: continue
                    kept_series_count += (len(chunk_df.columns) - 1)
                    chunk_long = chunk_df.melt(id_vars="period_date", var_name="series_id")
                    chunk_long["series_id"] = chunk_long["series_id"].astype(int)
                    chunk_long = chunk_long.merge(meta[['provider']], left_on="series_id", right_index=True)
                    chunk_long.to_parquet(partitioned_dir, engine='pyarrow', index=False, partition_cols=['provider'])
                
                summary_dicts[f"freq_{freq}"] = {"kept_series": kept_series_count}

            global_summary = _log_prep_summary(
                label="[MF_ALL]", 
                total_series_metadata=total_series, 
                total_series_reported=total_s, 
                kept_series=kept_s, 
                dropped_series=dropped_s,  
                report_rows=all_report_rows
            )
            summary_dicts["global"] = global_summary
            with open(out_summary, "w") as f:
                json.dump(summary_dicts, f, indent=4)
            logger.info(f"Saved: {out_summary}")

        # Final reports
        report_df = pd.DataFrame(all_report_rows)
        if not report_df.empty and "series_id" in report_df.columns:
            dups = report_df["series_id"].duplicated().sum()
            if dups > 0:
                logger.error(f"REPORTING INVARIANT FAILED: Found {dups} duplicate series_ids in final report.")
                report_df = report_df.drop_duplicates("series_id")
            report_df = report_df.set_index("series_id").sort_index()
            
        if not report_df.empty:
            report_df.to_csv(out_report)
            logger.info(f"Saved: {out_report}")

        meta.to_csv(out_meta)
        logger.info(f"Saved: {out_meta}")
        
        if not _INTERRUPT_REQUESTED:
            shutil.rmtree(chk.chk_dir, ignore_errors=True)
            logger.info("Removed checkpoints directory.")

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
