import os
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
import time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from statsmodels.tsa.stattools import kpss, adfuller
import concurrent.futures
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_prep")

# ----------------------------
# DB
# ----------------------------
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

# ============================================================
# Core Mathematical / Statistical Transforms
# ============================================================

def kpss_pvalue(x: pd.Series, regression: str = "c") -> Optional[float]:
    """KPSS p-value for non-NaN series. Returns None if too short or fails."""
    x = x.dropna()
    if len(x) < 10:
        return None
    try:
        # Use a simpler autolag or fixed lags if needed. auto sometimes fails on very short series
        stat, pval, lags, crit = kpss(x.values, regression=regression, nlags="auto")
        return float(pval)
    except Exception:
        return None

def adf_pvalue(x: pd.Series, regression: str = "c") -> Optional[float]:
    """ADF p-value for non-NaN series."""
    x = x.dropna()
    if len(x) < 15:
        return None
    try:
        res = adfuller(x.values, regression=regression, autolag="AIC")
        return float(res[1]) # p-value
    except Exception:
        return None

def safe_log(x: pd.Series) -> pd.Series:
    """Log transform only if strictly positive."""
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

# ============================================================
# Missing Data Handling
# ============================================================

def handle_missing_values(s: pd.Series, limit: int = 3) -> pd.Series:
    """
    Interpolates ONLY small internal gaps up to `limit` periods.
    Does NOT extrapolate/forward-fill to the end (ragged edges).
    """
    # limit_direction="inside" ensures it only fills gaps surrounded by valid data
    # Unfortunately pandas interpolate(limit_direction="inside") requires >= 1.0
    # In pandas 2.x limit_area="inside" is preferred.
    return s.interpolate(method="time", limit=limit, limit_area="inside")

def handle_ragged_edges(s: pd.Series, impute: bool = False, limit: int = 2) -> pd.Series:
    """
    By default, keeps ragged edges as NaN.
    If impute=True, uses a simple forward fill up to `limit` periods.
    """
    if impute:
        return s.ffill(limit=limit)
    return s

# ============================================================
# Frequency Alignment
# ============================================================

def monthly_resample(s: pd.Series, freq: str, daily_weekly_rule: str = "mean") -> pd.Series:
    """
    Resample a series to standard month-end frequency ('ME').
    freq: metadata frequency hint (D, W, M, Q, A)
    """
    if s.empty:
        return s

    if freq in ("D", "W"):
        # daily/weekly to monthly
        if daily_weekly_rule == "last":
            return s.resample("ME").last()
        elif daily_weekly_rule == "sum":
            return s.resample("ME").sum()
        else:
            return s.resample("ME").mean()
    elif freq in ("M", "ME", "MS"):
        # preserve month-end
        return s.resample("ME").last()
    elif freq in ("Q", "QE", "QS"):
        # For Quarterly: keep on the exact quarter-ending month. Do NOT spread to in-between months.
        # Resampling to 'Q' ensures it's bucketed by quarter, taking the last (the actual Q data).
        # We then resample to 'ME' keeping only the quarter-end entries, leaving NaNs for other months.
        return s.resample("Q").last().resample("ME").asfreq()
    else:
        # fallback
        return s.resample("ME").mean()

# ============================================================
# Data Extraction
# ============================================================

def fetch_observations(
    providers: Optional[List[str]] = None,
    dataset_key: Optional[str] = None,
    limit_datasets: Optional[int] = None,
    limit_series: Optional[int] = None,
    min_date: str = "2000-01-01",
) -> pd.DataFrame:
    """
    Extracts observations joined with meta. Flexible filtering.
    """
    where: List[str] = [f"o.period_date >= '{min_date}'"]
    params: Dict[str, Any] = {}

    dataset_filter_subquery = ""
    series_filter_subquery = ""

    if dataset_key:
        where.append(f"d.key = '{dataset_key}'")

    if providers:
        # Ensure parameterized lists are handled correctly
        # Using string interpolation here for simplicity due to SQLAlchemy constraints on IN clauses with lists in plain text
        p_str = "', '".join(providers)
        where.append(f"p.name IN ('{p_str}')")

    # If limiting datasets or series, we use subqueries
    if limit_datasets:
        dataset_filter_subquery = f"""
            INNER JOIN (
                SELECT d_sub.id 
                FROM datasets d_sub 
                JOIN providers p_sub ON d_sub.provider_id = p_sub.id
                {"WHERE p_sub.name IN ('" + p_str + "')" if providers else ""}
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
    
    # Ensuring uniqueness in case of multiple releases returning multiple rows per date
    df = df.sort_values(["series_id", "period_date"])
    # If the user's DB holds multiple vintages, this guarantees only the latest is selected
    # This assumes 'observed_at' logic was handled. If not, dropping duplicates on date is safe.
    df = df.drop_duplicates(subset=["series_id", "period_date"], keep="last")
    
    return df

# ============================================================
# Panel Building
# ============================================================

def build_monthly_panel(df_long: pd.DataFrame, daily_weekly_rule: str = "mean") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta_cols = ["series_id", "provider", "dataset_key", "series_key", "frequency", "country", "transform", "unit"]
    meta = df_long[meta_cols].drop_duplicates("series_id").set_index("series_id")
    
    groups = list(df_long.groupby("series_id"))
    total = len(groups)
    logger.info(f"Monthly panel build: {total} series...")

    out_series = {}
    
    # Perform grouping inline. Minimal compute, no strict need for parallelism
    for sid, group in groups:
        s = pd.Series(group["value"].values, index=group["period_date"]).sort_index()
        freq = str(meta.loc[sid, "frequency"]) if sid in meta.index else ""
        out_series[sid] = monthly_resample(s, freq, daily_weekly_rule)

    panel = pd.DataFrame(out_series)
    panel.index.name = "month_end"
    return panel, meta

# ============================================================
# Transform & Stationarity Pipeline
# ============================================================

@dataclass
class PrepConfig:
    min_obs: int = 24
    kpss_alpha: float = 0.05
    max_diffs: int = 2
    fill_internal_limit: int = 3
    use_yoy: bool = True
    daily_weekly_rule: str = "mean"
    impute_ragged_edges: bool = False
    ragged_edges_limit: int = 2

def process_single_series(args: Tuple[int, pd.Series, str, PrepConfig]) -> Tuple[int, pd.Series, Dict]:
    """Worker function for parallel processing."""
    sid, x, freq_hint, cfg = args
    
    diag = {
        "series_id": sid,
        "frequency": freq_hint,
        "dropped_reason": None,
        "transform_applied": None,
        "initial_non_nan": int(x.notna().sum()),
        "d0_kpss_p": None,
        "d1_kpss_p": None,
        "d2_kpss_p": None,
        "diffs_used": 0,
        "final_non_nan": 0
    }

    if diag["initial_non_nan"] < cfg.min_obs:
        diag["dropped_reason"] = f"too_few_obs_initial_<{cfg.min_obs}"
        return sid, x, diag

    # 1. Fill internal gaps
    x_f = handle_missing_values(x, limit=cfg.fill_internal_limit)

    # 2. Base transformation
    if cfg.use_yoy:
        if freq_hint == "Q":
            z = yoy_transform(x_f, periods=4)
            diag["transform_applied"] = "logdiff_4"
        else:
            z = yoy_transform(x_f, periods=12)
            diag["transform_applied"] = "logdiff_12"
    else:
        z = safe_log(x_f).diff(1)
        diag["transform_applied"] = "logdiff_1"

    # 3. Stationarity check and differencing
    diag["d0_kpss_p"] = kpss_pvalue(z)
    
    current = z
    for d in range(1, cfg.max_diffs + 1):
        p = kpss_pvalue(current)
        # If stationary (fail to reject H0), stop differencing
        if p is not None and p >= cfg.kpss_alpha:
            diag["diffs_used"] = d - 1
            break
        # Still non-stationary, apply first difference
        current = current.diff(1)
        diag[f"d{d}_kpss_p"] = kpss_pvalue(current)

    final = current

    # 4. Handle Ragged Edges
    final = handle_ragged_edges(final, impute=cfg.impute_ragged_edges, limit=cfg.ragged_edges_limit)

    # 5. Quality Filter
    diag["final_non_nan"] = int(final.notna().sum())
    
    if is_almost_constant(final):
        diag["dropped_reason"] = "almost_constant"
        return sid, final, diag

    if diag["final_non_nan"] < cfg.min_obs:
        diag["dropped_reason"] = f"too_few_obs_final_<{cfg.min_obs}"
        return sid, final, diag

    # Final stationarity strict check
    p_final = kpss_pvalue(final)
    if p_final is not None and p_final < cfg.kpss_alpha:
        diag["dropped_reason"] = "kpss_nonstationary_after_diffs"
        return sid, final, diag

    return sid, final, diag

def prepare_panel(panel: pd.DataFrame, meta: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parallelized preparation of the panel."""
    if panel.empty:
        return panel, pd.DataFrame()

    total = panel.shape[1]
    logger.info(f"Transforming and testing {total} series...")

    # Prepare arguments for multiprocessing
    args_list = []
    for sid in panel.columns:
        freq_hint = str(meta.loc[sid, "frequency"]) if sid in meta.index else ""
        args_list.append((sid, panel[sid], freq_hint, cfg))

    report_rows = []
    out_valid = {}

    import multiprocessing
    max_workers = min(os.cpu_count() or 4, 16)
    
    t0 = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, (sid, final_series, diag) in enumerate(executor.map(process_single_series, args_list), 1):
            report_rows.append(diag)
            
            if diag.get("dropped_reason") is None:
                out_valid[sid] = final_series
                
            if i % 50 == 0 or i == total:
                dt = time.perf_counter() - t0
                rate = i / dt if dt > 0 else 0
                logger.info(f"Processed: {i}/{total} ({i/total:.1%}) | kept={len(out_valid)} | {rate:.2f} it/s")

    prepared_panel = pd.DataFrame(out_valid, index=panel.index).sort_index()
    report_df = pd.DataFrame(report_rows).set_index("series_id").sort_index()

    logger.info(f"Prepared panel DONE: kept={len(out_valid)}/{total} series")
    return prepared_panel, report_df

# ============================================================
# Main Execution
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Data Preparation Pipeline for Nowcasting")
    ap.add_argument("--out-dir", type=str, default="data/processed", help="Directory to save outputs")
    ap.add_argument("--min-date", type=str, default="2000-01-01")
    ap.add_argument("--providers", type=str, default="eurostat,alfred,yahoo_finance", help="Comma-separated providers")
    ap.add_argument("--dataset-key", type=str, default=None, help="Filter by a specific dataset key")
    ap.add_argument("--limit-datasets", type=int, default=None, help="Process only the first N datasets")
    ap.add_argument("--limit-series", type=int, default=None, help="Process only the first N series overall")
    ap.add_argument("--daily-weekly-rule", type=str, default="mean", choices=["mean", "last", "sum"])
    ap.add_argument("--min-obs", type=int, default=36, help="Minimum observations required after transform")
    ap.add_argument("--no-yoy", action="store_true", help="Disable year-over-year transforms in favor of standard diffs")
    ap.add_argument("--kpss-alpha", type=float, default=0.05)
    ap.add_argument("--max-diffs", type=int, default=2)
    ap.add_argument("--fill-limit", type=int, default=3, help="Max internal nan limit to interpolate")
    ap.add_argument("--impute-ragged-edges", action="store_true", help="Forward fill ragged edges? Default: false")
    ap.add_argument("--ragged-edges-limit", type=int, default=2, help="Limit for ragged edge forward fill")
    
    args = ap.parse_args()

    providers_list = [p.strip() for p in args.providers.split(",")] if args.providers else None

    # Load config
    cfg = PrepConfig(
        min_obs=args.min_obs,
        kpss_alpha=args.kpss_alpha,
        max_diffs=args.max_diffs,
        fill_internal_limit=args.fill_limit,
        use_yoy=not args.no_yoy,
        daily_weekly_rule=args.daily_weekly_rule,
        impute_ragged_edges=args.impute_ragged_edges,
        ragged_edges_limit=args.ragged_edges_limit
    )

    logger.info("=== Data Preparation Pipeline ===")
    logger.info(f"Providers: {providers_list} | Min Date: {args.min_date}")
    
    # 1. Extraction
    df_long = fetch_observations(
        providers=providers_list,
        dataset_key=args.dataset_key,
        limit_datasets=args.limit_datasets,
        limit_series=args.limit_series,
        min_date=args.min_date
    )
    
    if df_long.empty:
        logger.warning("No data retrieved from Database. Pipeline stopped.")
        return

    logger.info(f"Extracted {len(df_long)} observations.")

    # 2. Alignment
    panel, meta = build_monthly_panel(df_long, daily_weekly_rule=cfg.daily_weekly_rule)

    # 3. Stationarity & Transforms
    prepared, report = prepare_panel(panel, meta, cfg)

    # 4. Save Outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_panel_path = out_dir / "panel_monthly.parquet"
    out_report_path = out_dir / "panel_report.csv"
    out_meta_path = out_dir / "panel_meta.csv"

    if not prepared.empty:
        prepared.to_parquet(out_panel_path)
        logger.info(f"Saved modeling dataset: {out_panel_path} (shape={prepared.shape})")
    
    report.to_csv(out_report_path)
    logger.info(f"Saved diagnostics report: {out_report_path}")

    meta.to_csv(out_meta_path)
    logger.info(f"Saved metadata: {out_meta_path}")

if __name__ == "__main__":
    main()