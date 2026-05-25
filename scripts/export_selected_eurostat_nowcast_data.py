import os
from pathlib import Path
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

META_PATH = PROJECT_ROOT / "final_clean_nowcast_metadata.csv"

OUT_DIR = PROJECT_ROOT / "data" / "selected_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MAPPED_FULL = OUT_DIR / "selected_eurostat_metadata_mapped_full.csv"
OUT_META_FOR_PREP = OUT_DIR / "selected_eurostat_metadata_for_prep.csv"
OUT_OBS = OUT_DIR / "selected_eurostat_observations.csv"
OUT_MISSING = OUT_DIR / "selected_eurostat_missing_series.csv"
OUT_WIDE_CSV = OUT_DIR / "selected_eurostat_wide_panel.csv"
OUT_WIDE_PARQUET = OUT_DIR / "selected_eurostat_wide_panel.parquet"
OUT_FINAL_PARQUET = OUT_DIR / "selected_eurostat_wide_panel_with_gdp_target.parquet"

MIN_DATE = "2000-01-01"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def normalize_series_key(s: pd.Series) -> pd.Series:
    """
    Normalize Eurostat series keys so old metadata and current DB can be matched.
    """
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

def pick_col(df, *names):
    for name in names:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of these columns found: {names}")

# ------------------------------------------------------------
# 1. LOAD OLD SELECTED METADATA
# ------------------------------------------------------------
engine = create_engine(DB_URL, future=True)

print(f"Loading old metadata from {META_PATH}...")
meta_old = pd.read_csv(META_PATH)

if "series_key" not in meta_old.columns:
    raise ValueError("final_clean_nowcast_metadata.csv must contain a 'series_key' column.")

if "dataset_code" not in meta_old.columns:
    if "dataset_key" in meta_old.columns:
        meta_old["dataset_code"] = meta_old["dataset_key"].astype(str).str.lower()
    else:
        meta_old["dataset_code"] = meta_old["series_key"].astype(str).str.split(".").str[0].str.lower()

meta_old["series_key_norm"] = normalize_series_key(meta_old["series_key"])

print(f"Loaded old selected metadata: {meta_old.shape}")

# ------------------------------------------------------------
# 2. LOAD CURRENT EUROSTAT METADATA FROM DB
# ------------------------------------------------------------
print("Fetching current Eurostat metadata from DB...")
query_meta = text("""
    SELECT
        s.id AS current_series_id,
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
    WHERE LOWER(p.name) = 'eurostat'
""")

with engine.connect() as conn:
    db_meta = pd.read_sql(query_meta, conn)

if db_meta.empty:
    raise RuntimeError("No Eurostat metadata found in current DB. Check ingestion state.")

db_meta["series_key_norm"] = normalize_series_key(db_meta["series_key"])
print(f"Current DB Eurostat metadata: {db_meta.shape}")

# ------------------------------------------------------------
# 3. MAP OLD SELECTED METADATA TO CURRENT DB SERIES IDS
# ------------------------------------------------------------
print("Mapping metadata...")
mapped = meta_old.merge(
    db_meta,
    on="series_key_norm",
    how="left",
    suffixes=("_old", "_db")
)

found = mapped[mapped["current_series_id"].notna()].copy()
missing = mapped[mapped["current_series_id"].isna()].copy()

print(f"Mapped selected series: {len(found)} / {len(meta_old)}")
print(f"Missing selected series: {len(missing)}")

# Save mapping diagnostics
mapped.to_csv(OUT_MAPPED_FULL, index=False, encoding="utf-8-sig")
missing.to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")

if found.empty:
    raise RuntimeError("No selected series were mapped to current DB. Check series_key format.")

# Remove duplicates if any
found = found.drop_duplicates(subset=["series_key_norm"], keep="first").copy()
found["current_series_id"] = found["current_series_id"].astype(int)

# ------------------------------------------------------------
# 4. CREATE METADATA FILE FOR DATA PREPARATION
# ------------------------------------------------------------
print("Creating metadata for preparation...")
meta_for_prep = pd.DataFrame({
    "series_id": found["current_series_id"].astype(int),
    "series_key": pick_col(found, "series_key_db", "series_key_y", "series_key"),
    "frequency": pick_col(found, "frequency_db", "frequency_y", "frequency"),
    "country": pick_col(found, "country_db", "country_y", "country"),
    "transform": pick_col(found, "transform_db", "transform_y", "transform"),
    "unit": pick_col(found, "unit_db", "unit_y", "unit"),
    "dataset_key": pick_col(found, "dataset_key_db", "dataset_key_y", "dataset_key"),
    "provider": pick_col(found, "provider_db", "provider_y", "provider"),
})

meta_for_prep["dataset_code"] = meta_for_prep["dataset_key"].astype(str).str.lower()
meta_for_prep = meta_for_prep.drop_duplicates(subset=["series_id"]).sort_values(["dataset_code", "series_id"])

meta_for_prep.to_csv(OUT_META_FOR_PREP, index=False, encoding="utf-8-sig")
print(f"Saved metadata for preparation to: {OUT_META_FOR_PREP}")

# ------------------------------------------------------------
# 5. FETCH OBSERVATIONS FOR CURRENT SERIES IDS
# ------------------------------------------------------------
print("Fetching observations for mapped series...")
series_ids = meta_for_prep["series_id"].unique().tolist()

query_obs = text("""
    SELECT
        o.series_id,
        o.period_date,
        o.value
    FROM observations o
    WHERE o.series_id = ANY(:series_ids)
      AND o.period_date >= :min_date
    ORDER BY o.series_id, o.period_date
""")

with engine.connect() as conn:
    obs = pd.read_sql(
        query_obs,
        conn,
        params={
            "series_ids": series_ids,
            "min_date": MIN_DATE,
        }
    )

obs["period_date"] = pd.to_datetime(obs["period_date"])
print(f"Fetched {len(obs)} observations.")
obs.to_csv(OUT_OBS, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 6. CREATE WIDE PANEL
# ------------------------------------------------------------
print("Creating wide panel...")
wide = (
    obs.pivot_table(
        index="period_date",
        columns="series_id",
        values="value",
        aggfunc="last"
    )
    .sort_index()
)
wide.index.name = "period_date"
wide.columns = wide.columns.astype(str)

wide.to_csv(OUT_WIDE_CSV, encoding="utf-8-sig")
wide.to_parquet(OUT_WIDE_PARQUET)
print(f"Saved wide panel to {OUT_WIDE_PARQUET}")

# ------------------------------------------------------------
# 7. IDENTIFY GDP TARGET AND SAVE FINAL PANEL
# ------------------------------------------------------------
print("Identifying GDP Target...")
gdp_candidates = meta_for_prep[
    meta_for_prep["series_key"].str.contains("namq_10_gdp", case=False, na=False)
    & meta_for_prep["series_key"].str.contains("na_item=B1GQ", case=False, na=False)
    & meta_for_prep["series_key"].str.contains("s_adj=SCA", case=False, na=False)
]

if gdp_candidates.empty:
    print("Warning: No B1GQ GDP candidate found. Skipping final GDP target column.")
else:
    preferred = gdp_candidates[gdp_candidates["unit"].astype(str).str.upper().eq("CLV_I10")]
    gdp_row = preferred.iloc[0] if not preferred.empty else gdp_candidates.iloc[0]
    gdp_series_id = str(gdp_row["series_id"])
    
    print(f"Selected GDP target series ID: {gdp_series_id}")
    
    if gdp_series_id in wide.columns:
        wide["gdp_target"] = wide[gdp_series_id]
        wide.to_parquet(OUT_FINAL_PARQUET)
        wide.to_csv(OUT_FINAL_PARQUET.with_suffix(".csv"), encoding="utf-8-sig")
        print(f"Saved final panel with GDP target to {OUT_FINAL_PARQUET}")
    else:
        print(f"Error: GDP series {gdp_series_id} missing from wide panel columns.")

print("\nDone.")