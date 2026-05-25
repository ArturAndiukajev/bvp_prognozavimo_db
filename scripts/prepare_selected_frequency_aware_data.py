import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import kpss, acf
from typing import Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("frequency_prep")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kpss_pvalue(x: pd.Series, regression: str = "c") -> float:
    """KPSS p-value. Returns 1.0 if series is too short or test fails."""
    x = x.dropna()
    if len(x) < 10:
        return 1.0
    try:
        _, pval, _, _ = kpss(x.values, regression=regression, nlags="auto")
        return float(pval)
    except Exception:
        return 1.0

def detect_seasonality(s: pd.Series, series_meta: pd.Series) -> bool:
    """
    Detect if a series is seasonal.
    Uses metadata keywords first, falls back to statistical ACF check.
    """
    meta_str = " ".join([str(v) for v in series_meta.values]).lower()
    
    # Metadata keywords
    nsa_keywords = ["not seasonally adjusted", "unadjusted", " nsa", "_nsa", " -nsa", "s_adj=nsa"]
    sa_keywords = ["seasonal adjustment", "seasonally adjusted", " sa", "_sa", " -sa", "s_adj=sa", "s_adj=sca"]
    
    if any(k in meta_str for k in nsa_keywords):
        return True
    if any(k in meta_str for k in sa_keywords):
        return False
    
    # Statistical check
    freq = str(series_meta.get("frequency", "")).upper()
    lag = 12 if "M" in freq else (4 if "Q" in freq else None)
    
    if lag is not None and len(s.dropna()) > (lag * 2):
        try:
            diffed = s.dropna().diff(1).dropna()
            if len(diffed) > lag:
                vals = acf(diffed, nlags=lag, fft=True)
                if vals[lag] > 0.2:
                    return True
        except Exception:
            pass
            
    return False

def custom_impute(s: pd.Series) -> pd.Series:
    """
    Custom imputation rules:
    - 1-gap: average of neighbors
    - 2-3 gaps: linear interpolation
    - >3 gaps: left empty
    """
    if s.notna().all():
        return s

    s_clean = s.copy()
    mask = s.isna()
    # Find blocks of NaNs
    nan_blocks = (mask != mask.shift()).astype(int).cumsum()
    nan_groups = mask.groupby(nan_blocks)

    for name, group in nan_groups:
        if not group.all(): continue # Not a NaN block
        
        gap_len = len(group)
        start_idx = group.index[0]
        end_idx = group.index[-1]

        # Avoid extrapolating at edges
        try:
            loc_start = s.index.get_loc(start_idx)
            loc_end = s.index.get_loc(end_idx)
        except Exception:
            continue

        if loc_start == 0 or loc_end == len(s) - 1:
            continue

        if gap_len == 1:
            prev_val = s.iloc[loc_start - 1]
            next_val = s.iloc[loc_end + 1]
            if pd.notna(prev_val) and pd.notna(next_val):
                s_clean.iloc[loc_start] = (prev_val + next_val) / 2
        elif 2 <= gap_len <= 3:
            prev_val = s.iloc[loc_start - 1]
            next_val = s.iloc[loc_end + 1]
            if pd.notna(prev_val) and pd.notna(next_val):
                subset = s.iloc[loc_start-1 : loc_end+2].interpolate(method='linear')
                s_clean.iloc[loc_start : loc_end+1] = subset.iloc[1:-1]
                
    return s_clean

def winsorize_series(s: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    s_clean = s.dropna()
    if s_clean.empty: return s
    lower = s_clean.quantile(limits[0])
    upper = s_clean.quantile(1 - limits[1])
    return s.clip(lower=lower, upper=upper)

def safe_log_diff(s: pd.Series, periods: int = 1) -> pd.Series:
    if (s.dropna() <= 0).any():
        return s.diff(periods)
    return np.log(s).diff(periods)

def apply_stationarity_pipeline(s: pd.Series, freq: str, is_seasonal: bool, kpss_alpha: float = 0.05) -> Tuple[pd.Series, str, int]:
    """Applies transformation and differencing to achieve stationarity."""
    p_lag = 12 if "M" in freq else (4 if "Q" in freq else 1)
    
    if is_seasonal:
        transform = f"yoy_diff_{p_lag}"
        z = safe_log_diff(s, periods=p_lag)
    else:
        transform = "log_diff_1"
        z = safe_log_diff(s, periods=1)
        
    # Extra differencing if needed
    diffs = 0
    temp_z = z.copy()
    while kpss_pvalue(temp_z) < kpss_alpha and diffs < 2:
        temp_z = temp_z.diff(1)
        diffs += 1
        
    return temp_z, transform, diffs

# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Frequency-aware data preparation for selected Eurostat and GT data.")
    parser.add_argument("--gt-suffix", type=str, default="", help="Suffix for Google Trends input files (e.g. v1 or lt)")
    parser.add_argument("--gt-transform", type=str, default="yoy", choices=["level", "diff", "yoy"], help="Google Trends transformation")
    parser.add_argument("--kpss-alpha", type=float, default=0.05, help="KPSS alpha for stationarity test")
    parser.add_argument("--min-obs-monthly", type=int, default=36, help="Min observations for monthly series")
    parser.add_argument("--min-obs-quarterly", type=int, default=36, help="Min observations for quarterly series")
    parser.add_argument("--min-span-years", type=float, default=14.0, help="Min data span in years")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "selected_raw"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Suffix handling
    gt_sfx = f"_{args.gt_suffix}" if args.gt_suffix else ""

    # Load Inputs
    logger.info(f"Loading inputs (GT suffix: '{args.gt_suffix}')...")
    try:
        euro_meta = pd.read_csv(raw_dir / "selected_eurostat_metadata_for_prep.csv")
        euro_obs = pd.read_csv(raw_dir / "selected_eurostat_observations.csv")
        
        gt_obs_file = raw_dir / f"selected_google_trends_observations_monthend{gt_sfx}.csv"
        logger.info(f"Using GT input: {gt_obs_file.name}")
        gt_obs = pd.read_csv(gt_obs_file)
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        return

    # Convert dates
    euro_obs["period_date"] = pd.to_datetime(euro_obs["period_date"])
    gt_obs["period_date"] = pd.to_datetime(gt_obs["period_date"])

    summary = {
        "eurostat_monthly": {"input": 0, "kept": 0, "dropped": 0},
        "eurostat_quarterly": {"input": 0, "kept": 0, "dropped": 0},
        "google_trends": {"input": 0, "kept": 0, "dropped": 0, "suffix": args.gt_suffix},
        "transformations": {}
    }
    
    # Track column frequencies for VintageBuilder
    col_freq_map = {}

    # ---------------------------------------------------------
    # 1. Eurostat Split
    # ---------------------------------------------------------
    euro_meta["freq_clean"] = euro_meta["frequency"].str.upper().str[0] # M or Q
    m_sids = euro_meta[euro_meta["freq_clean"] == "M"]["series_id"].unique()
    q_sids = euro_meta[euro_meta["freq_clean"] == "Q"]["series_id"].unique()
    
    summary["eurostat_monthly"]["input"] = len(m_sids)
    summary["eurostat_quarterly"]["input"] = len(q_sids)

    # ---------------------------------------------------------
    # 2. Identify GDP Target (Case-insensitive)
    # ---------------------------------------------------------
    logger.info("Identifying GDP Target...")
    # target: namq_10_gdp, CLV_I10, SCA, B1GQ, LT
    gdp_mask = (
        (euro_meta["dataset_key"].str.lower() == "namq_10_gdp") &
        (euro_meta["unit"].str.lower() == "clv_i10") &
        (euro_meta["series_key"].str.lower().str.contains("s_adj=sca")) &
        (euro_meta["series_key"].str.lower().str.contains("na_item=b1gq")) &
        (euro_meta["series_key"].str.lower().str.contains("geo\\\\time_period=lt"))
    )
    gdp_row = euro_meta[gdp_mask]
    
    if gdp_row.empty:
        # Fallback to na_item=B1G
        gdp_mask = (
            (euro_meta["dataset_key"].str.lower() == "namq_10_gdp") &
            (euro_meta["unit"].str.lower() == "clv_i10") &
            (euro_meta["series_key"].str.lower().str.contains("s_adj=sca")) &
            (euro_meta["series_key"].str.lower().str.contains("na_item=b1g")) &
            (euro_meta["series_key"].str.lower().str.contains("geo\\\\time_period=lt"))
        )
        gdp_row = euro_meta[gdp_mask]

    gdp_sid = None
    if not gdp_row.empty:
        gdp_sid = int(gdp_row.iloc[0]["series_id"])
        logger.info(f"GDP Target identified: series_id={gdp_sid}")
    else:
        logger.warning("Could not identify GDP target series in metadata.")

    # ---------------------------------------------------------
    # 3. Eurostat Monthly
    # ---------------------------------------------------------
    logger.info("Processing Eurostat Monthly...")
    m_obs = euro_obs[euro_obs["series_id"].isin(m_sids)].copy()
    m_obs["period_date"] = m_obs["period_date"] + pd.offsets.MonthEnd(0)
    m_obs = m_obs.groupby(["period_date", "series_id"])["value"].mean().reset_index()
    
    m_wide = m_obs.pivot(index="period_date", columns="series_id", values="value").sort_index()
    
    m_prepared = pd.DataFrame(index=m_wide.index)
    m_reports = []

    for sid in m_wide.columns:
        s = m_wide[sid]
        meta = euro_meta[euro_meta["series_id"] == sid].iloc[0]
        
        initial_non_nan = int(s.notna().sum())
        span_years = (s.dropna().index.max() - s.dropna().index.min()).days / 365.25 if initial_non_nan > 1 else 0
        
        dropped_reason = None
        if initial_non_nan < args.min_obs_monthly:
            dropped_reason = f"too_few_obs_{initial_non_nan}<{args.min_obs_monthly}"
        elif span_years < args.min_span_years:
            dropped_reason = f"too_short_span_{span_years:.1f}<{args.min_span_years}"
            
        if dropped_reason:
            summary["eurostat_monthly"]["dropped"] += 1
            m_reports.append({
                "series_id": sid, "frequency": "M", "status": "dropped", "dropped_reason": dropped_reason,
                "initial_non_nan": initial_non_nan, "span_years": round(span_years, 2)
            })
            continue

        s_imputed = custom_impute(s)
        is_seasonal = detect_seasonality(s_imputed, meta)
        z, trans_name, d_extra = apply_stationarity_pipeline(s_imputed, "M", is_seasonal, args.kpss_alpha)
        z = winsorize_series(z)
        
        final_non_nan = int(z.notna().sum())
        if final_non_nan < args.min_obs_monthly:
            summary["eurostat_monthly"]["dropped"] += 1
            m_reports.append({
                "series_id": sid, "frequency": "M", "status": "dropped", "dropped_reason": "too_few_obs_after_transform",
                "initial_non_nan": initial_non_nan, "final_non_nan": final_non_nan
            })
            continue

        m_prepared[sid] = z
        col_freq_map[str(sid)] = "M"
        m_reports.append({
            "series_id": sid, "frequency": "M", "status": "kept",
            "is_seasonal": is_seasonal, "transform": trans_name, "extra_diffs": d_extra,
            "initial_non_nan": initial_non_nan, "final_non_nan": final_non_nan,
            "span_years": round(span_years, 2), "kpss_final_p": round(kpss_pvalue(z), 4),
            "start_date": str(z.dropna().index.min().date()), "end_date": str(z.dropna().index.max().date())
        })
        summary["transformations"][f"M_{trans_name}_d{d_extra}"] = summary["transformations"].get(f"M_{trans_name}_d{d_extra}", 0) + 1

    summary["eurostat_monthly"]["kept"] = len(m_prepared.columns)
    
    # Save Monthly
    m_prepared.to_parquet(out_dir / "selected_eurostat_monthly_prepared.parquet")
    m_prepared.to_csv(out_dir / "selected_eurostat_monthly_prepared.csv")
    pd.DataFrame(m_reports).to_csv(out_dir / "selected_eurostat_monthly_report.csv", index=False)

    # ---------------------------------------------------------
    # 4. Eurostat Quarterly
    # ---------------------------------------------------------
    logger.info("Processing Eurostat Quarterly...")
    q_obs = euro_obs[euro_obs["series_id"].isin(q_sids)].copy()
    q_obs["period_date"] = q_obs["period_date"] + pd.offsets.QuarterEnd(0)
    q_obs = q_obs.groupby(["period_date", "series_id"])["value"].mean().reset_index()
    
    q_wide = q_obs.pivot(index="period_date", columns="series_id", values="value").sort_index()
    
    q_prepared = pd.DataFrame(index=q_wide.index)
    q_reports = []

    for sid in q_wide.columns:
        if gdp_sid and sid == gdp_sid:
            logger.info(f"Excluding GDP target {sid} from quarterly predictors.")
            q_reports.append({"series_id": sid, "frequency": "Q", "status": "target_excluded"})
            continue

        s = q_wide[sid]
        meta = euro_meta[euro_meta["series_id"] == sid].iloc[0]
        
        initial_non_nan = int(s.notna().sum())
        span_years = (s.dropna().index.max() - s.dropna().index.min()).days / 365.25 if initial_non_nan > 1 else 0
        
        dropped_reason = None
        if initial_non_nan < args.min_obs_quarterly:
            dropped_reason = f"too_few_obs_{initial_non_nan}<{args.min_obs_quarterly}"
        elif span_years < args.min_span_years:
            dropped_reason = f"too_short_span_{span_years:.1f}<{args.min_span_years}"
            
        if dropped_reason:
            summary["eurostat_quarterly"]["dropped"] += 1
            q_reports.append({
                "series_id": sid, "frequency": "Q", "status": "dropped", "dropped_reason": dropped_reason,
                "initial_non_nan": initial_non_nan, "span_years": round(span_years, 2)
            })
            continue

        s_imputed = custom_impute(s)
        is_seasonal = detect_seasonality(s_imputed, meta)
        z, trans_name, d_extra = apply_stationarity_pipeline(s_imputed, "Q", is_seasonal, args.kpss_alpha)
        z = winsorize_series(z)
        
        final_non_nan = int(z.notna().sum())
        if final_non_nan < args.min_obs_quarterly:
            summary["eurostat_quarterly"]["dropped"] += 1
            q_reports.append({
                "series_id": sid, "frequency": "Q", "status": "dropped", "dropped_reason": "too_few_obs_after_transform",
                "initial_non_nan": initial_non_nan, "final_non_nan": final_non_nan
            })
            continue

        q_prepared[sid] = z
        col_freq_map[str(sid)] = "Q"
        q_reports.append({
            "series_id": sid, "frequency": "Q", "status": "kept",
            "is_seasonal": is_seasonal, "transform": trans_name, "extra_diffs": d_extra,
            "initial_non_nan": initial_non_nan, "final_non_nan": final_non_nan,
            "span_years": round(span_years, 2), "kpss_final_p": round(kpss_pvalue(z), 4),
            "start_date": str(z.dropna().index.min().date()), "end_date": str(z.dropna().index.max().date())
        })
        summary["transformations"][f"Q_{trans_name}_d{d_extra}"] = summary["transformations"].get(f"Q_{trans_name}_d{d_extra}", 0) + 1

    summary["eurostat_quarterly"]["kept"] = len(q_prepared.columns)
    
    # Save Quarterly
    q_prepared.to_parquet(out_dir / "selected_eurostat_quarterly_prepared.parquet")
    q_prepared.to_csv(out_dir / "selected_eurostat_quarterly_prepared.csv")
    pd.DataFrame(q_reports).to_csv(out_dir / "selected_eurostat_quarterly_report.csv", index=False)

    # ---------------------------------------------------------
    # 5. Google Trends
    # ---------------------------------------------------------
    logger.info("Processing Google Trends...")
    summary["google_trends"]["input"] = len(gt_obs["series_id"].unique())
    
    gt_obs["period_date"] = gt_obs["period_date"] + pd.offsets.MonthEnd(0)
    gt_agg = gt_obs.groupby(["period_date", "series_id"])["value"].mean().reset_index()
    
    gt_wide = gt_agg.pivot(index="period_date", columns="series_id", values="value").sort_index()
    gt_wide.columns = [f"gt_{c}" for c in gt_wide.columns]
    
    gt_prepared = pd.DataFrame(index=gt_wide.index)
    gt_reports = []

    for col in gt_wide.columns:
        s = gt_wide[col]
        initial_non_nan = int(s.notna().sum())
        span_years = (s.dropna().index.max() - s.dropna().index.min()).days / 365.25 if initial_non_nan > 1 else 0
        
        if initial_non_nan < args.min_obs_monthly or span_years < args.min_span_years:
            summary["google_trends"]["dropped"] += 1
            gt_reports.append({"series_id": col, "status": "dropped", "initial_non_nan": initial_non_nan, "span_years": round(span_years, 2)})
            continue

        s_imputed = custom_impute(s)
        
        if args.gt_transform == "diff":
            z = s_imputed.diff(1)
        elif args.gt_transform == "yoy":
            z = s_imputed.diff(12)
        else: # level
            z = s_imputed
            
        z = winsorize_series(z)
        final_non_nan = int(z.notna().sum())
        
        if final_non_nan < args.min_obs_monthly:
            summary["google_trends"]["dropped"] += 1
            gt_reports.append({"series_id": col, "status": "dropped", "initial_non_nan": initial_non_nan, "final_non_nan": final_non_nan})
            continue

        gt_prepared[col] = z
        col_freq_map[col] = "GT"
        gt_reports.append({
            "series_id": col, "status": "kept", "transform_applied": args.gt_transform,
            "initial_non_nan": initial_non_nan, "final_non_nan": final_non_nan,
            "span_years": round(span_years, 2), "kpss_final_p": round(kpss_pvalue(z), 4),
            "start_date": str(z.dropna().index.min().date()), "end_date": str(z.dropna().index.max().date())
        })

    summary["google_trends"]["kept"] = len(gt_prepared.columns)
    
    # Save GT with suffix
    gt_out_file = f"selected_google_trends_monthly_prepared{gt_sfx}.parquet"
    logger.info(f"Saving GT prepared data to: {gt_out_file}")
    gt_prepared.to_parquet(out_dir / gt_out_file)
    gt_prepared.to_csv(out_dir / gt_out_file.replace(".parquet", ".csv"))
    pd.DataFrame(gt_reports).to_csv(out_dir / f"selected_google_trends_monthly_report{gt_sfx}.csv", index=False)

    # ---------------------------------------------------------
    # 6. GDP Target
    # ---------------------------------------------------------
    if gdp_sid:
        gdp_s = euro_obs[euro_obs["series_id"] == gdp_sid].copy()
        gdp_s["period_date"] = gdp_s["period_date"] + pd.offsets.QuarterEnd(0)
        gdp_s = gdp_s.groupby("period_date")["value"].mean().sort_index()
        
        gdp_s.to_frame("gdp_index").to_parquet(out_dir / "gdp_target_quarterly_raw.parquet")
        
        gdp_target = np.log(gdp_s).diff(4)
        gdp_target.name = "gdp_target"
        gdp_target.to_frame().to_parquet(out_dir / "gdp_target_quarterly.parquet")
        
        summary["gdp_target"] = {
            "series_id": int(gdp_sid),
            "initial_non_nan": int(gdp_s.notna().sum()),
            "final_non_nan": int(gdp_target.notna().sum()),
            "start": str(gdp_target.dropna().index.min().date()),
            "end": str(gdp_target.dropna().index.max().date()),
            "transform": "yoy_log_diff"
        }
        pd.DataFrame([summary["gdp_target"]]).to_csv(out_dir / "gdp_target_quarterly_report.csv", index=False)

    # ---------------------------------------------------------
    # 7. Metadata / Frequency Map with suffix
    # ---------------------------------------------------------
    map_out_file = out_dir / f"selected_column_frequency_map{gt_sfx}.json"
    with open(map_out_file, "w") as f:
        json.dump(col_freq_map, f, indent=4)
    logger.info(f"Saved frequency map to: {map_out_file.name}")

    # ---------------------------------------------------------
    # 8. Final Summary
    # ---------------------------------------------------------
    summary["date_ranges"] = {
        "eurostat_monthly": [str(m_prepared.index.min().date()), str(m_prepared.index.max().date())] if not m_prepared.empty else [],
        "eurostat_quarterly": [str(q_prepared.index.min().date()), str(q_prepared.index.max().date())] if not q_prepared.empty else [],
        "google_trends": [str(gt_prepared.index.min().date()), str(gt_prepared.index.max().date())] if not gt_prepared.empty else []
    }
    
    summary_out_file = out_dir / f"selected_frequency_aware_preparation_summary{gt_sfx}.json"
    with open(summary_out_file, "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info("Preparation complete.")

if __name__ == "__main__":
    main()
