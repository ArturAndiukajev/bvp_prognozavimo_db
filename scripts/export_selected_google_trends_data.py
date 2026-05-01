from pathlib import Path
import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_URL = os.environ.get(
    "DB_URL",
    "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
)

OUT_DIR = PROJECT_ROOT / "data" / "selected_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_META = OUT_DIR / "selected_google_trends_metadata.csv"

OUT_OBS_RAW = OUT_DIR / "selected_google_trends_observations_raw.csv"
OUT_OBS_MONTHEND = OUT_DIR / "selected_google_trends_observations_monthend.csv"

OUT_WIDE_CSV = OUT_DIR / "selected_google_trends_wide_panel.csv"
OUT_WIDE_PARQUET = OUT_DIR / "selected_google_trends_wide_panel.parquet"

MIN_DATE = "2000-01-01"

engine = create_engine(DB_URL, future=True)


# ------------------------------------------------------------
# 1. LOAD GOOGLE TRENDS METADATA
# ------------------------------------------------------------
with engine.connect() as conn:
    gt_meta = pd.read_sql(
        text("""
            SELECT
                s.id AS series_id,
                s.key AS series_key,
                s.frequency,
                s.country,
                s.transform,
                s.unit,
                d.key AS dataset_key,
                p.name AS provider
            FROM series s
            JOIN datasets d ON d.id = s.dataset_id
            JOIN providers p ON p.id = d.provider_id
            WHERE LOWER(p.name) LIKE '%google%'
               OR LOWER(d.key) LIKE '%trend%'
               OR LOWER(s.key) LIKE '%trend%'
            ORDER BY s.id
        """),
        conn
    )

print(f"Google Trends metadata shape: {gt_meta.shape}")

if gt_meta.empty:
    raise RuntimeError("No Google Trends series found in DB.")

gt_meta.to_csv(OUT_META, index=False, encoding="utf-8-sig")
print(f"Saved Google Trends metadata to: {OUT_META}")

series_ids = gt_meta["series_id"].dropna().astype(int).unique().tolist()
print(f"Unique Google Trends series in metadata: {len(series_ids)}")


# ------------------------------------------------------------
# 2. LOAD GOOGLE TRENDS OBSERVATIONS
# ------------------------------------------------------------
with engine.connect() as conn:
    gt_obs = pd.read_sql(
        text("""
            SELECT
                o.series_id,
                o.period_date,
                o.value
            FROM observations o
            WHERE o.series_id = ANY(:series_ids)
              AND o.period_date >= :min_date
            ORDER BY o.series_id, o.period_date
        """),
        conn,
        params={
            "series_ids": series_ids,
            "min_date": MIN_DATE,
        }
    )

print(f"Google Trends raw observations shape: {gt_obs.shape}")

if gt_obs.empty:
    raise RuntimeError("No Google Trends observations found.")

gt_obs["period_date"] = pd.to_datetime(gt_obs["period_date"])
gt_obs["series_id"] = gt_obs["series_id"].astype(int)
gt_obs["value"] = pd.to_numeric(gt_obs["value"], errors="coerce")

print(f"Raw date range: {gt_obs['period_date'].min()} -> {gt_obs['period_date'].max()}")
print(f"Raw unique GT series: {gt_obs['series_id'].nunique()}")

gt_obs.to_csv(OUT_OBS_RAW, index=False, encoding="utf-8-sig")
print(f"Saved raw observations to: {OUT_OBS_RAW}")


# ------------------------------------------------------------
# 3. ALIGN DATES TO MONTH-END
# ------------------------------------------------------------
gt_obs["period_date"] = gt_obs["period_date"] + pd.offsets.MonthEnd(0)


# ------------------------------------------------------------
# 4. CHECK AND AGGREGATE DUPLICATES
# ------------------------------------------------------------
dup_count_before = gt_obs.duplicated(["series_id", "period_date"]).sum()
print(f"Duplicate series_id + period_date rows before aggregation: {dup_count_before}")

gt_monthend = (
    gt_obs
    .groupby(["series_id", "period_date"], as_index=False)["value"]
    .mean()
    .sort_values(["series_id", "period_date"])
)

dup_count_after = gt_monthend.duplicated(["series_id", "period_date"]).sum()
print(f"Duplicate series_id + period_date rows after aggregation: {dup_count_after}")

print(f"Month-end observations shape: {gt_monthend.shape}")
print(f"Month-end date range: {gt_monthend['period_date'].min()} -> {gt_monthend['period_date'].max()}")
print(f"Month-end unique GT series: {gt_monthend['series_id'].nunique()}")

gt_monthend.to_csv(OUT_OBS_MONTHEND, index=False, encoding="utf-8-sig")
print(f"Saved month-end observations to: {OUT_OBS_MONTHEND}")


# ------------------------------------------------------------
# 5. BUILD WIDE PANEL
# ------------------------------------------------------------
wide_gt = (
    gt_monthend
    .pivot(
        index="period_date",
        columns="series_id",
        values="value"
    )
    .sort_index()
)

wide_gt.index.name = "period_date"

# Prefix GT columns so they are easy to distinguish later
wide_gt.columns = [f"gt_{c}" for c in wide_gt.columns.astype(str)]

wide_gt.to_csv(OUT_WIDE_CSV, encoding="utf-8-sig")
wide_gt.to_parquet(OUT_WIDE_PARQUET)

print(f"Saved wide GT CSV to: {OUT_WIDE_CSV}")
print(f"Saved wide GT parquet to: {OUT_WIDE_PARQUET}")
print(f"Wide GT panel shape: {wide_gt.shape}")
print(f"Wide GT date range: {wide_gt.index.min()} -> {wide_gt.index.max()}")


# ------------------------------------------------------------
# 6. FINAL DIAGNOSTICS
# ------------------------------------------------------------
missing_series = set(series_ids) - set(gt_monthend["series_id"].unique())

print("\nFinal diagnostics:")
print(f"GT series in metadata: {len(series_ids)}")
print(f"GT series with observations: {gt_monthend['series_id'].nunique()}")
print(f"GT series without observations: {len(missing_series)}")
print(f"Total missing values in wide panel: {int(wide_gt.isna().sum().sum())}")
print(f"Average NaN share in wide panel: {wide_gt.isna().mean().mean():.2%}")

if missing_series:
    missing_meta = gt_meta[gt_meta["series_id"].isin(missing_series)].copy()
    missing_path = OUT_DIR / "selected_google_trends_missing_series.csv"
    missing_meta.to_csv(missing_path, index=False, encoding="utf-8-sig")
    print(f"Saved missing GT series list to: {missing_path}")

print("\nDone.")