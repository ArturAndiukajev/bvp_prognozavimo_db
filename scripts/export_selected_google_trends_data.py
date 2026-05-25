import argparse
from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------------------------------------
# CONFIG & ARGS
# ------------------------------------------------------------
load_dotenv()

parser = argparse.ArgumentParser(description="Export Google Trends data from DB to wide panel.")
parser.add_argument("--dataset-key", type=str, default=None, help="Filter by specific dataset key (e.g., google_trends or google_trends_lt)")
parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames (e.g., v1)")
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_URL = os.environ.get(
    "DB_URL",
    "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
)

OUT_DIR = PROJECT_ROOT / "data" / "selected_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# File naming with suffix
sfx = f"_{args.suffix}" if args.suffix else ""
OUT_META = OUT_DIR / f"selected_google_trends_metadata{sfx}.csv"
OUT_OBS_RAW = OUT_DIR / f"selected_google_trends_observations_raw{sfx}.csv"
OUT_OBS_MONTHEND = OUT_DIR / f"selected_google_trends_observations_monthend{sfx}.csv"
OUT_WIDE_CSV = OUT_DIR / f"selected_google_trends_wide_panel{sfx}.csv"
OUT_WIDE_PARQUET = OUT_DIR / f"selected_google_trends_wide_panel{sfx}.parquet"

MIN_DATE = "2000-01-01"

engine = create_engine(DB_URL, future=True)

# ------------------------------------------------------------
# 1. LOAD GOOGLE TRENDS METADATA
# ------------------------------------------------------------
print(f"Fetching Google Trends metadata (dataset_key={args.dataset_key})...")

where_clause = ""
if args.dataset_key:
    where_clause = "AND d.key = :dataset_key"
else:
    where_clause = """
        AND (LOWER(p.name) LIKE '%%google%%'
             OR LOWER(d.key) LIKE '%%trend%%'
             OR LOWER(s.key) LIKE '%%trend%%')
    """

query_meta = text(f"""
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
    WHERE 1=1 {where_clause}
    ORDER BY s.id
""")

with engine.connect() as conn:
    gt_meta = pd.read_sql(
        query_meta,
        conn,
        params={"dataset_key": args.dataset_key} if args.dataset_key else {}
    )

print(f"Google Trends metadata shape: {gt_meta.shape}")

if gt_meta.empty:
    raise RuntimeError(f"No Google Trends series found in DB for dataset_key={args.dataset_key}.")

gt_meta.to_csv(OUT_META, index=False, encoding="utf-8-sig")
print(f"Saved Google Trends metadata to: {OUT_META}")

series_ids = gt_meta["series_id"].dropna().astype(int).unique().tolist()
print(f"Unique Google Trends series in metadata: {len(series_ids)}")


# ------------------------------------------------------------
# 2. LOAD GOOGLE TRENDS OBSERVATIONS
# ------------------------------------------------------------
print("Fetching observations...")
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

gt_obs.to_csv(OUT_OBS_RAW, index=False, encoding="utf-8-sig")
print(f"Saved raw observations to: {OUT_OBS_RAW}")


# ------------------------------------------------------------
# 3. ALIGN DATES TO MONTH-END & AGGREGATE
# ------------------------------------------------------------
print("Aggregating to month-end...")
gt_obs["period_date"] = gt_obs["period_date"] + pd.offsets.MonthEnd(0)

gt_monthend = (
    gt_obs
    .groupby(["series_id", "period_date"], as_index=False)["value"]
    .mean()
    .sort_values(["series_id", "period_date"])
)

gt_monthend.to_csv(OUT_OBS_MONTHEND, index=False, encoding="utf-8-sig")
print(f"Saved month-end observations to: {OUT_OBS_MONTHEND}")


# ------------------------------------------------------------
# 4. BUILD WIDE PANEL
# ------------------------------------------------------------
print("Building wide panel...")
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
wide_gt.columns = [f"gt_{c}" for c in wide_gt.columns.astype(str)]

wide_gt.to_csv(OUT_WIDE_CSV, encoding="utf-8-sig")
wide_gt.to_parquet(OUT_WIDE_PARQUET)

print(f"Saved wide GT CSV to: {OUT_WIDE_CSV}")
print(f"Saved wide GT parquet to: {OUT_WIDE_PARQUET}")
print(f"Wide GT panel shape: {wide_gt.shape}")

# ------------------------------------------------------------
# 5. FINAL DIAGNOSTICS
# ------------------------------------------------------------
missing_series = set(series_ids) - set(gt_monthend["series_id"].unique())

print("\nFinal diagnostics:")
print(f"GT series in metadata: {len(series_ids)}")
print(f"GT series with observations: {gt_monthend['series_id'].nunique()}")
print(f"GT series without observations: {len(missing_series)}")

if missing_series:
    missing_meta = gt_meta[gt_meta["series_id"].isin(missing_series)].copy()
    missing_path = OUT_DIR / f"selected_google_trends_missing_series{sfx}.csv"
    missing_meta.to_csv(missing_path, index=False, encoding="utf-8-sig")
    print(f"Saved missing GT series list to: {missing_path}")

print("\nDone.")