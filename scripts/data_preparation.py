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
import logging
import argparse
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
import time
import concurrent.futures

# Project root = parent of this script's directory (scripts/ -> project root)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUT_DIR = str(_PROJECT_ROOT / "data" / "processed")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from statsmodels.tsa.stattools import kpss, adfuller
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
    n_workers: int = min(os.cpu_count() or 4, 16)


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


def detect_seasonality(s: pd.Series, series_meta: pd.Series) -> Dict[str, Any]:
    """
    Detect if a series is seasonal or seasonally adjusted.
    Uses metadata keywords and statistical ACF check.
    """
    # 1. Metadata check
    # Check fields like 'series_key', 'dataset_key', 'unit' for clues
    meta_str = " ".join([str(v) for v in series_meta.values]).lower()
    
    sa_keywords = ["seasonal adjustment", "seasonally adjusted", " sa", "_sa", " -sa"]
    nsa_keywords = ["not seasonally adjusted", "unadjusted", " nsa", "_nsa", " -nsa"]
    
    is_sa_meta = any(k in meta_str for k in sa_keywords)
    is_nsa_meta = any(k in meta_str for k in nsa_keywords)
    
    # 2. Statistical check (ACF at seasonal lags)
    # Only if enough data
    is_seasonal_stat = False
    acf_val = 0.0
    
    freq = str(series_meta.get("frequency", "")).upper()
    lag = 12 if "M" in freq else (4 if "Q" in freq else None)
    
    if lag and len(s.dropna()) > lag * 2:
        try:
            from statsmodels.tsa.stattools import acf
            # Calculate ACF on first difference to remove trend
            diffed = s.dropna().diff(1).dropna()
            if len(diffed) > lag:
                vals = acf(diffed, nlags=lag, fft=True)
                acf_val = vals[lag]
                # Heuristic: ACF > 0.2 at seasonal lag is a strong indicator of seasonality
                if acf_val > 0.2:
                    is_seasonal_stat = True
        except Exception:
            pass

    # Decision Logic
    if is_nsa_meta or is_seasonal_stat:
        status = "seasonal"
        reason = "nsa_metadata" if is_nsa_meta else f"acf_lag_{lag}_stat"
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

def fetch_observations(
    providers: Optional[List[str]] = None,
    dataset_key: Optional[str] = None,
    limit_datasets: Optional[int] = None,
    limit_series: Optional[int] = None,
    min_date: str = "2000-01-01",
) -> pd.DataFrame:
    """
    Pull observations from the DB with optional filtering.
    Returns a long-format DataFrame with columns:
        period_date, value, series_id, series_key, frequency,
        country, transform, unit, dataset_key, provider
    """
    where: List[str] = [f"o.period_date >= '{min_date}'"]
    dataset_filter_subquery = ""
    series_filter_subquery = ""

    if dataset_key:
        where.append(f"d.key = '{dataset_key}'")

    if providers:
        p_str = "', '".join(providers)
        where.append(f"p.name IN ('{p_str}')")

    if limit_datasets:
        prov_clause = f"WHERE p_sub.name IN ('{p_str}')" if providers else ""
        dataset_filter_subquery = f"""
            INNER JOIN (
                SELECT d_sub.id
                FROM datasets d_sub
                JOIN providers p_sub ON d_sub.provider_id = p_sub.id
                {prov_clause}
                ORDER BY d_sub.id LIMIT {limit_datasets}
            ) d_lim ON d.id = d_lim.id
        """

    if limit_series:
        series_filter_subquery = f"""
            INNER JOIN (
                SELECT s_sub.id
                FROM series s_sub
                ORDER BY s_sub.id LIMIT {limit_series}
            ) s_lim ON s.id = s_lim.id
        """

    sql = f"""
    SELECT
      o.period_date,
      o.value,
      s.id AS series_id,
      s.key AS series_key,
      s.frequency,
      s.country,
      s.transform,
      s.unit,
      d.key AS dataset_key,
      p.name AS provider
    FROM observations o
    JOIN series s ON s.id = o.series_id
    {series_filter_subquery}
    JOIN datasets d ON d.id = s.dataset_id
    {dataset_filter_subquery}
    JOIN providers p ON p.id = d.provider_id
    WHERE {" AND ".join(where)}
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    if df.empty:
        return df

    df["period_date"] = pd.to_datetime(df["period_date"])
    df = df.sort_values(["series_id", "period_date"])
    df = df.drop_duplicates(subset=["series_id", "period_date"], keep="last")
    return df


# ---------------------------------------------------------------------------
# Panel Building — Common-Frequency Mode
# ---------------------------------------------------------------------------

def build_common_freq_panel(
    df_long: pd.DataFrame,
    cfg: PrepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate all series to a single common frequency (default: monthly 'ME').
    Returns (panel, meta) where panel.index is a DatetimeIndex.
    """
    if df_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta_cols = ["series_id", "provider", "dataset_key", "series_key",
                 "frequency", "country", "transform", "unit"]
    meta = df_long[meta_cols].drop_duplicates("series_id").set_index("series_id")

    groups = list(df_long.groupby("series_id"))
    logger.info(f"[CF] Aggregating {len(groups)} series → {cfg.target_freq}")

    out_series: Dict[int, pd.Series] = {}
    for sid, group in groups:
        s = pd.Series(
            group["value"].values,
            index=pd.DatetimeIndex(group["period_date"])
        ).sort_index()
        freq = str(meta.loc[sid, "frequency"]) if sid in meta.index else ""
        out_series[sid] = resample_to_target(s, freq, cfg.target_freq, cfg.daily_weekly_rule)

    panel = pd.DataFrame(out_series)
    panel.index.name = "period_date"
    panel.index = pd.to_datetime(panel.index)
    return panel, meta


# ---------------------------------------------------------------------------
# Panel Building — Mixed-Frequency Mode
# ---------------------------------------------------------------------------

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


def build_mixed_freq_panel(
    df_long: pd.DataFrame,
    cfg: PrepConfig,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Preserve native frequencies; do NOT force a common cadence.
    Returns:
        freq_panels — dict keyed by canonical freq ('D','W','M','Q','A'),
                      values are DataFrames with DatetimeIndex at native cadence.
        meta        — series metadata DataFrame indexed by series_id.
    """
    if df_long.empty:
        return {}, pd.DataFrame()

    meta_cols = ["series_id", "provider", "dataset_key", "series_key",
                 "frequency", "country", "transform", "unit"]
    meta = df_long[meta_cols].drop_duplicates("series_id").set_index("series_id")

    groups = list(df_long.groupby("series_id"))
    logger.info(f"[MF] Preserving native frequencies for {len(groups)} series")

    # Resample rules per canonical frequency (just to ensure a clean, regular index)
    _resample_rule = {"D": "D", "W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}
    _agg_func = {"D": "last", "W": "mean", "M": "last", "Q": "last", "A": "last"}

    freq_data: Dict[str, Dict[int, pd.Series]] = {k: {} for k in _resample_rule}

    for sid, group in groups:
        raw_freq = str(meta.loc[sid, "frequency"]) if sid in meta.index else ""
        canon = _canonical_mf_freq(raw_freq)
        rule = _resample_rule.get(canon, "ME")
        agg = _agg_func.get(canon, "last")

        s = pd.Series(
            group["value"].values,
            index=pd.DatetimeIndex(group["period_date"])
        ).sort_index()

        if agg == "last":
            resampled = s.resample(rule).last()
        elif agg == "mean":
            resampled = s.resample(rule).mean()
        else:
            resampled = s.resample(rule).last()

        freq_data[canon][sid] = resampled

    freq_panels: Dict[str, pd.DataFrame] = {}
    for canon, series_dict in freq_data.items():
        if series_dict:
            panel = pd.DataFrame(series_dict)
            panel.index.name = "period_date"
            panel.index = pd.to_datetime(panel.index)
            freq_panels[canon] = panel
            logger.info(f"  [MF] freq={canon}: {panel.shape[1]} series × {panel.shape[0]} periods")

    return freq_panels, meta


# ---------------------------------------------------------------------------
# Stationarity Pipeline — Common-Frequency Mode
# ---------------------------------------------------------------------------

def _process_cf_series(args: Tuple) -> Tuple[int, pd.Series, Dict]:
    """Worker: transforms + stationarity filter for one series (CF mode)."""
    sid, x, series_meta, cfg = args

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

    # 3. Seasonality Detection
    s_diag = detect_seasonality(x_imputed, series_meta)
    diag["is_seasonal"] = s_diag["is_seasonal"]
    diag["is_sa"] = s_diag["is_sa"]
    diag["detect_reason"] = s_diag["reason"]

    # 4. Adaptive Base Transformation
    # In CF mode, we are strict. If seasonal/NSA, use YoY. If SA, use simple log-diff.
    if diag["is_seasonal"]:
        periods = 4 if str(diag["frequency"]).upper() in ("Q", "QE", "QS") else 12
        # YoY transformation (log-diff if positive)
        z = yoy_transform(x_imputed, periods=periods)
        diag["transform_applied"] = f"yoy_diff_{periods}"
    else:
        # Seasonally adjusted or unknown -> prefer log-diff(1)
        z = safe_log(x_imputed).diff(1)
        diag["transform_applied"] = "log_diff_1"

    # 5. Stationarity: KPSS-driven extra differencing
    diag["d0_kpss_p"] = kpss_pvalue(z)

    current = z
    diffs_applied = 0
    for d in range(1, cfg.max_diffs + 1):
        p = kpss_pvalue(current)
        if p is None or p >= cfg.kpss_alpha:   # stationary → stop
            break
        current = current.diff(1)
        diffs_applied += 1
        diag[f"d{d}_kpss_p"] = kpss_pvalue(current)

    diag["diffs_used"] = diffs_applied
    final = current

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
        return sid, final, diag

    return sid, final, diag


# ---------------------------------------------------------------------------
# Stationarity Pipeline — Mixed-Frequency Mode
# ---------------------------------------------------------------------------

def _process_mf_series(args: Tuple) -> Tuple[int, pd.Series, Dict]:
    """
    Worker: lighter transforms for one series (MF mode).
    - Adaptive transform based on seasonality
    - 14-year span check
    - Custom imputation
    """
    sid, x, series_meta, cfg = args

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

    # 3. Seasonality Detection
    s_diag = detect_seasonality(x_imputed, series_meta)
    diag["is_seasonal"] = s_diag["is_seasonal"]
    diag["is_sa"] = s_diag["is_sa"]
    diag["detect_reason"] = s_diag["reason"]

    # 4. Adaptive Transformation (MF mode)
    # We prefer to keep high-frequency info. 
    # Even if seasonal, we might prefer log-diff(1) if it's for a model that handles it,
    # but the user requested: "If NSA/seasonal, prefer YoY".
    if diag["is_seasonal"]:
        freq_str = str(diag["frequency"]).upper()
        if "M" in freq_str: periods = 12
        elif "Q" in freq_str: periods = 4
        elif "W" in freq_str: periods = 52
        elif "D" in freq_str: periods = 365 # approximate
        else: periods = 1
        
        z = yoy_transform(x_imputed, periods=periods)
        diag["transform_applied"] = f"yoy_diff_{periods}"
    else:
        # Seasonally adjusted or weekly/daily with no clear seasonality -> simple log diff
        z = safe_log(x_imputed).diff(1)
        diag["transform_applied"] = "log_diff_1"

    diag["d0_kpss_p"] = kpss_pvalue(z)

    # 5. At most 1 extra diff for MF
    current = z
    diffs_applied = 0
    p = kpss_pvalue(current)
    if p is not None and p < cfg.kpss_alpha and cfg.mf_max_diffs >= 1:
        current = current.diff(1)
        diffs_applied = 1
        diag["d1_kpss_p"] = kpss_pvalue(current)

    diag["diffs_used"] = diffs_applied
    final = current

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

    return sid, final, diag


# ---------------------------------------------------------------------------
# Parallel Panel Preparation
# ---------------------------------------------------------------------------

def _run_parallel(
    args_list: List[Tuple],
    worker_fn,
    cfg: PrepConfig,
    label: str,
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
            try:
                sid, final_series, diag = future.result()
            except Exception as exc:
                sid = futures[future]
                logger.error(f"  {label} series {sid} raised: {exc}")
                diag = {"series_id": sid, "dropped_reason": f"worker_exception: {exc}"}
                report_rows.append(diag)
                continue

            report_rows.append(diag)
            if diag.get("dropped_reason") is None:
                out_valid[sid] = final_series

            if i % 100 == 0 or i == total:
                dt = time.perf_counter() - t0
                rate = i / dt if dt > 0 else 0
                logger.info(
                    f"  {label}: {i}/{total} ({i/total:.1%}) "
                    f"| kept={len(out_valid)} | {rate:.1f} it/s"
                )

    return out_valid, report_rows


def prepare_panel_cf(
    panel: pd.DataFrame, meta: pd.DataFrame, cfg: PrepConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Transform + stationarity filter for common-frequency panel."""
    if panel.empty:
        return panel, pd.DataFrame(), {}

    logger.info(f"[CF] Preparing {panel.shape[1]} series (stationarity pipeline)...")

    args_list = []
    for sid in panel.columns:
        series_meta = meta.loc[sid] if sid in meta.index else pd.Series()
        args_list.append((sid, panel[sid], series_meta, cfg))

    out_valid, report_rows = _run_parallel(args_list, _process_cf_series, cfg, "CF")

    prepared = pd.DataFrame(out_valid, index=panel.index).sort_index() if out_valid else pd.DataFrame(index=panel.index)
    report = pd.DataFrame(report_rows).set_index("series_id").sort_index() if report_rows else pd.DataFrame()

    summary_dict = _log_prep_summary("[CF]", panel.shape[1], out_valid, report_rows)
    return prepared, report, summary_dict


def prepare_panel_mf(
    freq_panels: Dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    cfg: PrepConfig,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict]:
    """Transform + stationarity filter for each native-frequency panel."""
    if not freq_panels:
        return {}, pd.DataFrame(), {}

    all_report_rows: List[Dict] = []
    prepared_panels: Dict[str, pd.DataFrame] = {}
    all_summary_dicts: Dict[str, Dict] = {}

    for freq, panel in freq_panels.items():
        if panel.empty:
            continue
        logger.info(f"[MF] Preparing freq={freq}: {panel.shape[1]} series...")

        args_list = []
        for sid in panel.columns:
            series_meta = meta.loc[sid] if sid in meta.index else pd.Series({"frequency": freq})
            args_list.append((sid, panel[sid], series_meta, cfg))

        out_valid, report_rows = _run_parallel(args_list, _process_mf_series, cfg, f"MF-{freq}")
        all_report_rows.extend(report_rows)

        if out_valid:
            prepared = pd.DataFrame(out_valid, index=panel.index).sort_index()
            prepared_panels[freq] = prepared

        summary_dict = _log_prep_summary(f"[MF-{freq}]", panel.shape[1], out_valid, report_rows)
        all_summary_dicts[freq] = summary_dict

    report = (
        pd.DataFrame(all_report_rows).set_index("series_id").sort_index()
        if all_report_rows
        else pd.DataFrame()
    )
    return prepared_panels, report, all_summary_dicts


def _log_prep_summary(label: str, total: int, out_valid: dict, report_rows: list) -> dict:
    kept = len(out_valid)
    dropped = total - kept
    
    drop_counts = {}
    kept_freqs = {}
    dropped_freqs = {}
    min_date_str = None
    max_date_str = None

    if report_rows:
        drop_reasons = pd.Series([r.get("dropped_reason") for r in report_rows]).value_counts()
        drop_counts = drop_reasons.to_dict()
        reasons_str = ", ".join(f"{r}={c}" for r, c in drop_reasons.items() if r is not None)
        
        kept_freqs = pd.Series([str(r.get("frequency")) for r in report_rows if r.get("dropped_reason") is None]).value_counts().to_dict()
        dropped_freqs = pd.Series([str(r.get("frequency")) for r in report_rows if r.get("dropped_reason") is not None]).value_counts().to_dict()
    else:
        reasons_str = "n/a"

    if out_valid:
        try:
            global_min = min(s.dropna().index.min() for s in out_valid.values() if not s.dropna().empty)
            global_max = max(s.dropna().index.max() for s in out_valid.values() if not s.dropna().empty)
            min_date_str = global_min.strftime("%Y-%m-%d")
            max_date_str = global_max.strftime("%Y-%m-%d")
        except Exception:
            pass

    transforms = pd.Series([r.get("transform_applied") for r in report_rows if r.get("transform_applied")]).value_counts()
    tx_counts = transforms.to_dict()
    tx_str = ", ".join(f"{t}={c}" for t, c in transforms.items())

    logger.info(
        f"{label} Summary: total={total} | kept={kept} | dropped={dropped}\n"
        f"  Kept Freqs:   {kept_freqs}\n"
        f"  Drop Freqs:   {dropped_freqs}\n"
        f"  Date Bounds:  [{min_date_str}, {max_date_str}]\n"
        f"  Drop reasons: {reasons_str or 'none'}\n"
        f"  Transforms:   {tx_str or 'none'}"
    )
    
    return {
        "total_series_input": total,
        "kept_series": kept,
        "dropped_series": dropped,
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
    Strips leading/trailing whitespace, lowercases, and replaces
    spaces, commas, slashes, backslashes and other unsafe chars with underscores.
    Collapses consecutive underscores.
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

    Example result:
        common_frequency_ME_eurostat-alfred_from2000-01-01_mean_yoy_kpss005_diff2
    """
    parts = []

    # 1. Mode
    parts.append(cfg.mode.value)                        # e.g. common_frequency

    # 2. Target frequency (CF only)
    if cfg.mode.value == "common_frequency":
        parts.append(sanitize_token(cfg.target_freq))  # e.g. me

    # 3. Providers / dataset key
    if args.dataset_key:
        parts.append("dataset")
        parts.append(sanitize_token(args.dataset_key))
    elif providers_list:
        joined = "-".join(sanitize_token(p) for p in providers_list)
        parts.append(joined)

    # 4. Limit-datasets / limit-series
    if getattr(args, "limit_datasets", None):
        parts.append(f"ld{args.limit_datasets}")
    if getattr(args, "limit_series", None):
        parts.append(f"ls{args.limit_series}")

    # 5. Min date
    if getattr(args, "min_date", None):
        parts.append(f"from{args.min_date}")

    # 6. Stationarity / transform settings
    if cfg.mode.value == "common_frequency":
        # daily-weekly aggregation rule
        parts.append(sanitize_token(cfg.daily_weekly_rule))    # mean | last | sum
        # YoY
        parts.append("yoy" if cfg.use_yoy else "noyoy")
        # KPSS alpha  (0.05 -> kpss005, 0.01 -> kpss001)
        kpss_str = f"kpss{str(cfg.kpss_alpha).replace('.', '')}"
        parts.append(kpss_str)
        # max diffs
        parts.append(f"diff{cfg.max_diffs}")
    else:  # mixed_frequency
        parts.append("logdiff" if cfg.mf_use_log_diff else "diff")
        parts.append(f"mfdiff{cfg.mf_max_diffs}")

    # 7. Ragged edge imputation
    if cfg.impute_ragged_edges:
        parts.append(f"ragged{cfg.ragged_edges_limit}")

    # 8. Non-default fill limit
    if cfg.fill_internal_limit != 3:
        parts.append(f"fill{cfg.fill_internal_limit}")

    suffix = "_".join(t for t in parts if t)

    # Safety: cap at 180 chars to stay within OS filename limits (255 byte limit).
    # If too long, keep mode + providers/dataset + last parts.
    if len(suffix) > 180:
        # Keep first 2 parts (mode + freq/providers) and last 4 (stationarity)
        important = parts[:2] + parts[-4:]
        suffix = "_".join(t for t in important if t) + "_trunc"

    return suffix





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

    args = ap.parse_args()

    providers_list = [p.strip() for p in args.providers.split(",")] if args.providers else None

    cfg = PrepConfig(
        mode=PrepMode(args.mode),
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
    )

    logger.info("=== Data Preparation Pipeline ===")
    logger.info(f"Mode: {cfg.mode.value} | Providers: {providers_list} | Min Date: {args.min_date}")

    # 1. Extract
    df_long = fetch_observations(
        providers=providers_list,
        dataset_key=args.dataset_key,
        limit_datasets=args.limit_datasets,
        limit_series=args.limit_series,
        min_date=args.min_date,
    )

    if df_long.empty:
        logger.warning("No data retrieved from DB. Pipeline stopped.")
        return

    logger.info(f"Extracted {len(df_long):,} observations across {df_long['series_id'].nunique()} series.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the shared filename suffix once for this run
    run_suffix = build_run_suffix(args, cfg, providers_list)
    logger.info(f"Output filename suffix: {run_suffix}")

    if cfg.mode == PrepMode.COMMON_FREQUENCY:
        # ---- Common-Frequency Path ----
        logger.info("[CF] Building monthly panel...")
        panel, meta = build_common_freq_panel(df_long, cfg)
        validate_panel(panel, "raw_cf_panel")

        logger.info("[CF] Applying stationarity pipeline...")
        prepared, report, summary_dict = prepare_panel_cf(panel, meta, cfg)
        validate_panel(prepared, "prepared_cf_panel")

        out_panel  = out_dir / f"panel_{run_suffix}.parquet"
        out_report = out_dir / f"report_{run_suffix}.csv"
        out_meta   = out_dir / f"meta_{run_suffix}.csv"
        out_summary = out_dir / f"pipeline_summary_{run_suffix}.json"

        if not prepared.empty:
            # Reset index so the date column is named 'period_date' inside the file
            prepared.reset_index().rename(columns={"index": "period_date"}).to_parquet(
                out_panel, index=False
            )
            logger.info(f"Saved: {out_panel} (shape={prepared.shape})")

        if not report.empty:
            report.to_csv(out_report)
            logger.info(f"Saved: {out_report}")
            
        import json
        with open(out_summary, "w") as f:
            json.dump(summary_dict, f, indent=4)
        logger.info(f"Saved: {out_summary}")

        meta.to_csv(out_meta)
        logger.info(f"Saved: {out_meta}")

    else:
        # ---- Mixed-Frequency Path ----
        logger.info("[MF] Building native-frequency panels...")
        freq_panels, meta = build_mixed_freq_panel(df_long, cfg)

        logger.info("[MF] Applying mixed-frequency stationarity pipeline...")
        prepared_panels, report, summary_dicts = prepare_panel_mf(freq_panels, meta, cfg)

        for freq, panel in prepared_panels.items():
            validate_panel(panel, f"mf_panel_{freq}")
            out_path = out_dir / f"mf_panel_{freq}_{run_suffix}.parquet"
            panel.reset_index().rename(columns={"index": "period_date"}).to_parquet(
                out_path, index=False
            )
            logger.info(f"Saved: {out_path} (shape={panel.shape})")

        out_report = out_dir / f"report_{run_suffix}.csv"
        out_meta   = out_dir / f"meta_{run_suffix}.csv"
        out_summary = out_dir / f"pipeline_summary_{run_suffix}.json"
        
        if not report.empty:
            report.to_csv(out_report)
            logger.info(f"Saved: {out_report}")
            
        import json
        with open(out_summary, "w") as f:
            json.dump(summary_dicts, f, indent=4)
        logger.info(f"Saved: {out_summary}")

        if not report.empty:
            report.to_csv(out_report)
            logger.info(f"Saved: {out_report}")

        meta.to_csv(out_meta)
        logger.info(f"Saved: {out_meta}")

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()