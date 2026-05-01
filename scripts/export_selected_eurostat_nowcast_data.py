# from pathlib import Path
# import os
#
# import pandas as pd
# from sqlalchemy import create_engine, text
# from dotenv import load_dotenv
#
#
# # ------------------------------------------------------------
# # CONFIG
# # ------------------------------------------------------------
# load_dotenv()
#
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
#
# DB_URL = os.environ.get(
#     "DB_URL",
#     "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
# )
#
# META_PATH = PROJECT_ROOT / "final_clean_nowcast_metadata.csv"
#
# OUT_DIR = PROJECT_ROOT / "data" / "selected_raw"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# OUT_MAPPED_FULL = OUT_DIR / "selected_eurostat_metadata_mapped_full.csv"
# OUT_META_FOR_PREP = OUT_DIR / "selected_eurostat_metadata_for_prep.csv"
# OUT_OBS = OUT_DIR / "selected_eurostat_observations.csv"
# OUT_MISSING = OUT_DIR / "selected_eurostat_missing_series.csv"
#
# MIN_DATE = "2000-01-01"
#
#
# # ------------------------------------------------------------
# # HELPERS
# # ------------------------------------------------------------
# def normalize_series_key(s: pd.Series) -> pd.Series:
#     """
#     Normalize Eurostat series keys so old metadata and current DB can be matched.
#
#     Main issue:
#     old metadata: NAMQ_10_GDP.freq=Q...
#     current DB:   namq_10_gdp.freq=Q...
#
#     So we match lowercased keys.
#     """
#     return (
#         s.astype(str)
#         .str.strip()
#         .str.lower()
#     )
#
#
# # ------------------------------------------------------------
# # LOAD OLD SELECTED METADATA
# # ------------------------------------------------------------
# engine = create_engine(DB_URL, future=True)
#
# meta_old = pd.read_csv(META_PATH)
#
# if "series_key" not in meta_old.columns:
#     raise ValueError("final_clean_nowcast_metadata.csv must contain a 'series_key' column.")
#
# if "dataset_code" not in meta_old.columns:
#     if "dataset_key" in meta_old.columns:
#         meta_old["dataset_code"] = meta_old["dataset_key"].astype(str).str.lower()
#     else:
#         meta_old["dataset_code"] = meta_old["series_key"].astype(str).str.split(".").str[0].str.lower()
#
# meta_old["series_key_norm"] = normalize_series_key(meta_old["series_key"])
#
# print(f"Loaded old selected metadata: {meta_old.shape}")
# print(f"Old selected unique series_key: {meta_old['series_key_norm'].nunique()}")
#
#
# # ------------------------------------------------------------
# # LOAD CURRENT EUROSTAT METADATA FROM DB
# # ------------------------------------------------------------
# query_meta = text("""
#     SELECT
#         s.id AS current_series_id,
#         s.key AS series_key,
#         s.frequency,
#         s.country,
#         s.transform,
#         s.unit,
#         d.key AS dataset_key,
#         p.name AS provider
#     FROM series s
#     JOIN datasets d ON d.id = s.dataset_id
#     JOIN providers p ON p.id = d.provider_id
#     WHERE LOWER(p.name) = 'eurostat'
# """)
#
# with engine.connect() as conn:
#     db_meta = pd.read_sql(query_meta, conn)
#
# if db_meta.empty:
#     raise RuntimeError("No Eurostat metadata found in current DB. Check DB_URL or ingestion state.")
#
# db_meta["series_key_norm"] = normalize_series_key(db_meta["series_key"])
# db_meta["dataset_code_db"] = db_meta["dataset_key"].astype(str).str.lower()
#
# print(f"Current DB Eurostat metadata: {db_meta.shape}")
# print(f"Current DB unique series_key: {db_meta['series_key_norm'].nunique()}")
#
#
# # ------------------------------------------------------------
# # MAP OLD SELECTED METADATA TO CURRENT DB SERIES IDS
# # ------------------------------------------------------------
# mapped = meta_old.merge(
#     db_meta,
#     on="series_key_norm",
#     how="left",
#     suffixes=("_old", "_db")
# )
#
# found = mapped[mapped["current_series_id"].notna()].copy()
# missing = mapped[mapped["current_series_id"].isna()].copy()
#
# print(f"Mapped selected series: {len(found)} / {len(meta_old)}")
# print(f"Missing selected series: {len(missing)}")
#
#
# # ------------------------------------------------------------
# # SAVE FULL MAPPING DIAGNOSTICS
# # ------------------------------------------------------------
# mapped.to_csv(OUT_MAPPED_FULL, index=False, encoding="utf-8-sig")
# missing.to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")
#
# print(f"Saved full mapped metadata to: {OUT_MAPPED_FULL}")
# print(f"Saved missing list to: {OUT_MISSING}")
#
#
# # ------------------------------------------------------------
# # STOP IF NOTHING WAS MAPPED
# # ------------------------------------------------------------
# if found.empty:
#     print("\nNo selected series were mapped.")
#     print("Printing debug samples...")
#
#     print("\nOLD KEYS SAMPLE:")
#     print(
#         meta_old[["dataset_code", "series_key"]]
#         .head(10)
#         .to_string(index=False)
#     )
#
#     print("\nDB KEYS SAMPLE:")
#     print(
#         db_meta[["current_series_id", "dataset_key", "series_key"]]
#         .head(10)
#         .to_string(index=False)
#     )
#
#     old_codes = set(meta_old["dataset_code"].astype(str).str.lower())
#     db_codes = set(db_meta["dataset_key"].astype(str).str.lower())
#
#     print("\nOld dataset codes missing from current DB:")
#     print(sorted(old_codes - db_codes))
#
#     raise RuntimeError(
#         "No selected series were mapped to current DB series_id. "
#         "Check DB_URL, provider name, and series_key format."
#     )
#
#
# # ------------------------------------------------------------
# # REMOVE DUPLICATES IF ANY
# # ------------------------------------------------------------
# # In normal case, each old series_key should map to exactly one current DB series.
# dupes = found[found.duplicated(subset=["series_key_norm"], keep=False)].copy()
#
# if not dupes.empty:
#     dupes_path = OUT_DIR / "selected_eurostat_duplicate_mappings.csv"
#     dupes.to_csv(dupes_path, index=False, encoding="utf-8-sig")
#     print(f"Warning: duplicate mappings found. Saved to: {dupes_path}")
#
#     # Keep the first mapping per normalized key to avoid duplicated observations.
#     found = found.drop_duplicates(subset=["series_key_norm"], keep="first").copy()
#
#
# found["current_series_id"] = found["current_series_id"].astype(int)
#
#
# # ------------------------------------------------------------
# # CREATE METADATA FILE FOR DATA PREPARATION
# # ------------------------------------------------------------
#
# def pick_col(df, *names):
#     for name in names:
#         if name in df.columns:
#             return df[name]
#     raise KeyError(f"None of these columns found: {names}")
#
# meta_for_prep = pd.DataFrame({
#     "series_id": found["current_series_id"].astype(int),
#     "series_key": pick_col(found, "series_key_db", "series_key_y", "series_key"),
#     "frequency": pick_col(found, "frequency_db", "frequency_y", "frequency"),
#     "country": pick_col(found, "country_db", "country_y", "country"),
#     "transform": pick_col(found, "transform_db", "transform_y", "transform"),
#     "unit": pick_col(found, "unit_db", "unit_y", "unit"),
#     "dataset_key": pick_col(found, "dataset_key_db", "dataset_key_y", "dataset_key"),
#     "provider": pick_col(found, "provider_db", "provider_y", "provider"),
# })
#
# meta_for_prep["dataset_code"] = meta_for_prep["dataset_key"].astype(str).str.lower()
#
# # Preserve old selected info if available
# if "dataset_code_old" in found.columns:
#     meta_for_prep["old_dataset_code"] = found["dataset_code_old"]
# elif "dataset_code" in found.columns:
#     meta_for_prep["old_dataset_code"] = found["dataset_code"]
#
# if "series_id_old" in found.columns:
#     meta_for_prep["old_series_id"] = found["series_id_old"]
# elif "series_id" in found.columns:
#     meta_for_prep["old_series_id"] = found["series_id"]
#
# meta_for_prep = meta_for_prep.drop_duplicates(subset=["series_id"]).sort_values(
#     ["dataset_code", "series_id"]
# )
#
# meta_for_prep.to_csv(OUT_META_FOR_PREP, index=False, encoding="utf-8-sig")
#
# print(f"Saved metadata for preparation to: {OUT_META_FOR_PREP}")
# print(f"Metadata for prep shape: {meta_for_prep.shape}")
#
# # ------------------------------------------------------------
# # FETCH OBSERVATIONS FOR CURRENT SERIES IDS
# # ------------------------------------------------------------
# series_ids = meta_for_prep["series_id"].dropna().astype(int).unique().tolist()
#
# query_obs = text("""
#     SELECT
#         o.series_id,
#         o.period_date,
#         o.value
#     FROM observations o
#     WHERE o.series_id = ANY(:series_ids)
#       AND o.period_date >= :min_date
#     ORDER BY o.series_id, o.period_date
# """)
#
# with engine.connect() as conn:
#     obs = pd.read_sql(
#         query_obs,
#         conn,
#         params={
#             "series_ids": series_ids,
#             "min_date": MIN_DATE,
#         }
#     )
#
# obs["period_date"] = pd.to_datetime(obs["period_date"])
#
# print(f"Fetched observations: {obs.shape}")
# print(f"Date range: {obs['period_date'].min()} -> {obs['period_date'].max()}")
# print(f"Unique series in observations: {obs['series_id'].nunique()}")
#
# obs.to_csv(OUT_OBS, index=False, encoding="utf-8-sig")
#
# print(f"Saved observations to: {OUT_OBS}")
#
#
# # ------------------------------------------------------------
# # FINAL DIAGNOSTICS
# # ------------------------------------------------------------
# series_with_obs = set(obs["series_id"].dropna().astype(int).unique())
# series_expected = set(series_ids)
#
# missing_obs_ids = sorted(series_expected - series_with_obs)
#
# print("\nFinal diagnostics:")
# print(f"Expected selected series: {len(series_expected)}")
# print(f"Series with observations: {len(series_with_obs)}")
# print(f"Series without observations: {len(missing_obs_ids)}")
#
# if missing_obs_ids:
#     missing_obs = meta_for_prep[meta_for_prep["series_id"].isin(missing_obs_ids)].copy()
#     missing_obs_path = OUT_DIR / "selected_eurostat_mapped_but_no_observations.csv"
#     missing_obs.to_csv(missing_obs_path, index=False, encoding="utf-8-sig")
#     print(f"Saved mapped-but-no-observations list to: {missing_obs_path}")
#
# print("\nDataset counts in selected metadata:")
# print(meta_for_prep["dataset_code"].value_counts().to_string())
#
# print("\nDone.")

# from pathlib import Path
# import pandas as pd
# #
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# #
# OBS_PATH = PROJECT_ROOT / "data" / "selected_raw" / "selected_eurostat_observations.csv"
# OUT_DIR = PROJECT_ROOT / "data" / "selected_raw"
# OUT_WIDE_CSV = OUT_DIR / "selected_eurostat_wide_panel.csv"
# OUT_WIDE_PARQUET = OUT_DIR / "selected_eurostat_wide_panel.parquet"
# #
# # obs = pd.read_csv(OBS_PATH)
# # obs["period_date"] = pd.to_datetime(obs["period_date"])
# # obs["series_id"] = obs["series_id"].astype(str)
# #
# # wide = (
# #     obs.pivot_table(
# #         index="period_date",
# #         columns="series_id",
# #         values="value",
# #         aggfunc="last"
# #     )
# #     .sort_index()
# # )
# #
# # wide.index.name = "period_date"
# #
# # wide.to_csv(OUT_WIDE_CSV, encoding="utf-8-sig")
# # wide.to_parquet(OUT_WIDE_PARQUET)
# #
# # print("Wide panel shape:", wide.shape)
# # print("Date range:", wide.index.min(), "->", wide.index.max())
# # print("Saved CSV:", OUT_WIDE_CSV)
# # print("Saved parquet:", OUT_WIDE_PARQUET)
#
# import pandas as pd
#
# meta = pd.read_csv("bvp_prognozavimo_db/data/selected_raw/selected_eurostat_metadata_for_prep.csv")
#
# gdp_candidates = meta[
#     meta["series_key"].str.contains("namq_10_gdp", case=False, na=False)
#     & meta["series_key"].str.contains("na_item=B1GQ", case=False, na=False)
# ]
#
# print(gdp_candidates[["series_id", "series_key", "unit", "frequency"]])

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

META_PATH = PROJECT_ROOT / "data" / "selected_raw" / "selected_eurostat_metadata_for_prep.csv"
WIDE_PATH = PROJECT_ROOT / "data" / "selected_raw" / "selected_eurostat_wide_panel.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "selected_raw" / "selected_eurostat_wide_panel_with_gdp_target.parquet"

meta = pd.read_csv(META_PATH)

gdp_candidates = meta[
    meta["series_key"].str.contains("namq_10_gdp", case=False, na=False)
    & meta["series_key"].str.contains("na_item=B1GQ", case=False, na=False)
    & meta["series_key"].str.contains("s_adj=SCA", case=False, na=False)
]

print("\nGDP candidates:")
print(gdp_candidates[["series_id", "series_key", "unit", "frequency"]].to_string(index=False))

# Если кандидатов несколько, сначала смотрим, что нашлось
if gdp_candidates.empty:
    raise RuntimeError("No B1GQ GDP candidate found.")

# Лучше выбрать CLV_I10, если он есть, иначе первый найденный
preferred = gdp_candidates[gdp_candidates["unit"].astype(str).str.upper().eq("CLV_I10")]

if not preferred.empty:
    gdp_row = preferred.iloc[0]
else:
    gdp_row = gdp_candidates.iloc[0]

gdp_series_id = str(gdp_row["series_id"])

print("\nSelected GDP target:")
print(gdp_row[["series_id", "series_key", "unit", "frequency"]])

wide = pd.read_parquet(WIDE_PATH)
wide.columns = wide.columns.astype(str)

if gdp_series_id not in wide.columns:
    raise RuntimeError(f"GDP series_id {gdp_series_id} not found in wide panel columns.")

wide["gdp_target"] = wide[gdp_series_id]

wide.to_parquet(OUT_PATH)

csv_out = OUT_PATH.with_suffix(".csv")
wide.to_csv(csv_out, encoding="utf-8-sig")

print("\nSaved:")
print(OUT_PATH)
print(csv_out)
print("Shape:", wide.shape)
print("gdp_target non-null:", wide["gdp_target"].notna().sum())
print("gdp_target date range:", wide["gdp_target"].dropna().index.min(), "->", wide["gdp_target"].dropna().index.max())