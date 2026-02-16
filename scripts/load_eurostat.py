# load_eurostat.py
import json
import logging
import re
import hashlib
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
import eurostat
from sqlalchemy import create_engine, text

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eurostat")

# ----------------------------
# DB
# ----------------------------
DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

# ----------------------------
# Paths
# ----------------------------
RAW_DIR = Path("data/raw/eurostat")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Try both: repo config/ and local file next to script
CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")

# ----------------------------
# TIME parsing regex
# ----------------------------
_RE_YQ = re.compile(r"^(\d{4})[- ]?Q([1-4])$", re.I)        # 2024Q1 / 2024-Q1
_RE_YM = re.compile(r"^(\d{4})[- ]?M(\d{1,2})$", re.I)      # 2024M02 / 2024-M2
_RE_Y_MM = re.compile(r"^(\d{4})-(\d{2})$")                 # 2024-02
_RE_Y = re.compile(r"^(\d{4})$")                            # 2024
_RE_YMD = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")          # 2024-02-16


# ============================================================
# Config
# ============================================================
def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG
    if not path.exists():
        logger.error(f"Config file not found: {CONFIG_PATH} nor {FALLBACK_CONFIG}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# DB Helpers
# ============================================================
def ensure_source(conn, name: str) -> int:
    conn.execute(text("""
        INSERT INTO sources (name) VALUES (:name)
        ON CONFLICT (name) DO NOTHING
    """), {"name": name})
    return conn.execute(text("SELECT id FROM sources WHERE name=:name"), {"name": name}).scalar_one()


def ensure_series(conn,
                  source_id: int,
                  key: str,
                  country: str,
                  frequency: str,
                  transform: str,
                  unit: Optional[str],
                  name: str,
                  meta: dict) -> int:
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO series (source_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:sid, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
        ON CONFLICT (source_id, key, country, frequency, transform)
        DO UPDATE SET
            unit = EXCLUDED.unit,
            name = EXCLUDED.name,
            meta = EXCLUDED.meta
        RETURNING id
    """), {
        "sid": source_id,
        "key": key,
        "country": country,
        "freq": frequency,
        "transform": transform,
        "unit": unit,
        "name": name,
        "meta": meta_json
    }).scalar_one()


def create_release(conn,
                   source_id: int,
                   downloaded_at: datetime,
                   vintage_at: datetime,
                   description: str,
                   raw_path: Optional[str],
                   content_hash: Optional[str],
                   scope: Optional[str] = None,
                   meta: Optional[dict] = None) -> int:
    meta = meta or {}
    if scope:
        meta["scope"] = scope
    meta_json = json.dumps(meta, ensure_ascii=False)

    return conn.execute(text("""
        INSERT INTO releases (source_id, release_time, downloaded_at, vintage_at,
                              description, raw_path, content_hash, meta)
        VALUES (:sid, :rtime, :dlat, :vint, :desc, :raw, :hash, CAST(:meta AS jsonb))
        RETURNING id
    """), {
        "sid": source_id,
        "rtime": downloaded_at,     # keep compatible semantics
        "dlat": downloaded_at,
        "vint": vintage_at,
        "desc": description,
        "raw": raw_path,
        "hash": content_hash,
        "meta": meta_json
    }).scalar_one()


def last_release_hash(conn, source_id: int, scope: Optional[str] = None) -> Optional[str]:
    if scope:
        return conn.execute(text("""
            SELECT content_hash
            FROM releases
            WHERE source_id=:sid AND meta->>'scope' = :scope
            ORDER BY downloaded_at DESC
            LIMIT 1
        """), {"sid": source_id, "scope": scope}).scalar_one_or_none()

    return conn.execute(text("""
        SELECT content_hash
        FROM releases
        WHERE source_id=:sid
        ORDER BY downloaded_at DESC
        LIMIT 1
    """), {"sid": source_id}).scalar_one_or_none()


# ============================================================
# File helpers
# ============================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Time helpers
# ============================================================
def to_period_date(x: Any) -> date:
    """
    Eurostat time columns often look like:
      '2024M02', '2024Q1', '2024-02', '2024'
    Convert them into a date for period alignment:
      M -> first day of month
      Q -> first month of quarter
      Y -> Jan 1st
    """
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.date()

    s = str(x).strip()
    if not s or s.lower() == "nan":
        raise ValueError("Empty TIME_PERIOD")

    m = _RE_YMD.match(s)
    if m:
        y, mo, d = map(int, m.groups())
        return date(y, mo, d)

    m = _RE_YQ.match(s)
    if m:
        y = int(m.group(1))
        q = int(m.group(2))
        month = (q - 1) * 3 + 1
        return date(y, month, 1)

    m = _RE_YM.match(s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        return date(y, mo, 1)

    m = _RE_Y_MM.match(s)
    if m:
        y, mo = map(int, m.groups())
        return date(y, mo, 1)

    m = _RE_Y.match(s)
    if m:
        return date(int(m.group(1)), 1, 1)

    raise ValueError(f"Unsupported time label: {x!r}")


def detect_time_columns(df: pd.DataFrame) -> list[str]:
    """
    eurostat.get_data_df returns wide format:
      dimension columns + time columns like 2024M01, 2024Q1, 2024 etc.
    Detect time columns heuristically.
    """
    time_cols = []
    for c in df.columns:
        cs = str(c)
        if len(cs) >= 4 and cs[:4].isdigit():
            time_cols.append(c)
    return time_cols


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    filters:
      {"geo": ["LT"], "unit": ["CP_MEUR"], ...}
    """
    if not filters:
        return df
    out = df
    for col, allowed in filters.items():
        if col in out.columns:
            out = out[out[col].isin(allowed)]
    return out


def normalize_dim_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v)
    return s


def build_series_key(dataset_code: str, dims: Dict[str, Any], dim_cols: list[str]) -> str:
    parts = [f"{k}={normalize_dim_value(dims.get(k))}" for k in dim_cols]
    return f"{dataset_code}." + ".".join(parts)


def guess_country(dims: Dict[str, Any]) -> str:
    # Common Eurostat geo column
    for k in ("geo", "GEO"):
        if k in dims and dims[k] is not None:
            return str(dims[k])
    return "EU"


def guess_frequency(dims: Dict[str, Any], time_label: Any) -> str:
    if "freq" in dims and dims["freq"] is not None:
        return str(dims["freq"])
    s = str(time_label)
    if "Q" in s:
        return "Q"
    if "M" in s or "-" in s:
        return "M"
    return "A"


# ============================================================
# Core ingestion
# ============================================================
def fetch_dataset(dataset_code: str) -> pd.DataFrame:
    # eurostat package
    return eurostat.get_data_df(dataset_code)


def ingest_dataset(dataset_code: str, dataset_cfg: dict, mode: str = "initial"):
    """
    dataset_cfg example:
      {"filters": {"geo": ["LT"], "unit": ["CP_MEUR"]}, "name": "..."}
    """
    filters = dataset_cfg.get("filters", {}) if isinstance(dataset_cfg, dict) else {}
    dataset_name = dataset_cfg.get("name", dataset_code) if isinstance(dataset_cfg, dict) else dataset_code

    downloaded_at = datetime.now(timezone.utc)
    vintage_at = downloaded_at  # Eurostat snapshot-vintage
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")

    logger.info(f"[{mode}] Eurostat dataset={dataset_code} filters={filters}")

    # 1) Fetch
    try:
        df = fetch_dataset(dataset_code)
    except Exception as e:
        logger.error(f"Fetch failed for {dataset_code}: {e}")
        return

    if df is None or df.empty:
        logger.warning(f"{dataset_code}: empty dataset")
        return

    # 2) Filter
    df = apply_filters(df, filters)
    if df.empty:
        logger.warning(f"{dataset_code}: empty after filtering")
        return

    # 3) Save raw snapshot
    raw_path = RAW_DIR / f"{dataset_code}_{stamp}.csv"
    df.to_csv(raw_path, index=False)
    content_hash = sha256_file(raw_path)

    # 4) Scope for hash-based update skipping
    scope = f"dataset:{dataset_code}|filters:{json.dumps(filters, sort_keys=True)}"

    with engine.begin() as conn:
        source_id = ensure_source(conn, "eurostat")

        if mode == "update":
            prev_hash = last_release_hash(conn, source_id, scope=scope)
            if prev_hash == content_hash:
                logger.info(f"{dataset_code}: no changes (hash same) -> skip")
                return

        # 5) Create release
        release_id = create_release(
            conn=conn,
            source_id=source_id,
            downloaded_at=downloaded_at,
            vintage_at=vintage_at,
            description=f"Eurostat snapshot {dataset_code}",
            raw_path=str(raw_path),
            content_hash=content_hash,
            scope=scope,
            meta={"dataset_code": dataset_code, "dataset_name": dataset_name, "filters": filters, "mode": mode}
        )

        # 6) Detect time columns and melt
        time_cols = detect_time_columns(df)
        if not time_cols:
            raise RuntimeError(f"{dataset_code}: cannot detect time columns. Columns={list(df.columns)}")

        dim_cols = [c for c in df.columns if c not in time_cols]

        df_long = df.melt(
            id_vars=dim_cols,
            value_vars=time_cols,
            var_name="time_label",
            value_name="value"
        ).dropna(subset=["value"]).copy()

        inserted = 0
        failed = 0

        # 7) Cache series ids to avoid repeated ensure_series calls
        series_cache: dict[tuple, int] = {}

        for _, row in df_long.iterrows():
            dims = {c: normalize_dim_value(row[c]) for c in dim_cols}

            # Determine basic properties
            country = guess_country(dims)
            freq = guess_frequency(dims, row["time_label"])
            unit = normalize_dim_value(dims.get("unit"))

            # Build key + name
            series_key = build_series_key(dataset_code, dims, dim_cols)
            cache_key = (series_key, country, freq, "LEVEL")

            if cache_key not in series_cache:
                pretty_dims = ", ".join([f"{c}:{dims.get(c)}" for c in dim_cols])
                series_name = f"{dataset_name} ({dataset_code}) | {pretty_dims}"

                meta = {
                    "dataset_code": dataset_code,
                    "dataset_name": dataset_name,
                    "dimensions": dims,
                    "filters": filters
                }

                try:
                    sid = ensure_series(
                        conn=conn,
                        source_id=source_id,
                        key=series_key,
                        country=country,
                        frequency=freq,
                        transform="LEVEL",
                        unit=unit,
                        name=series_name,
                        meta=meta
                    )
                    series_cache[cache_key] = sid
                except Exception as e:
                    failed += 1
                    logger.warning(f"ensure_series failed for {series_key}: {e}")
                    continue

            series_id = series_cache[cache_key]

            # Parse time -> period_date
            try:
                pdate = to_period_date(row["time_label"])
                val = float(row["value"])
            except Exception as e:
                failed += 1
                logger.warning(f"Bad row time={row['time_label']}: {e}")
                continue

            # 8) Insert observation
            try:
                conn.execute(text("""
                    INSERT INTO observations (series_id, period_date, observed_at, value, release_id, meta)
                    VALUES (:sid, :pdate, :oat, :val, :rid, '{}'::jsonb)
                    ON CONFLICT (series_id, period_date, observed_at) DO NOTHING
                """), {
                    "sid": series_id,
                    "pdate": pdate,
                    "oat": vintage_at,
                    "val": val,
                    "rid": release_id
                })
                inserted += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Insert failed: {e}")

        # 9) Ingestion log
        details = {
            "dataset_code": dataset_code,
            "filters": filters,
            "mode": mode,
            "downloaded_at": downloaded_at.isoformat(),
            "vintage_at": vintage_at.isoformat(),
            "raw_path": str(raw_path),
            "hash": content_hash,
            "dim_cols": dim_cols,
            "time_cols_count": len(time_cols),
            "series_count": len(series_cache),
            "rows_long": int(len(df_long))
        }

        conn.execute(text("""
            INSERT INTO ingestion_log (source_id, status, rows_inserted, rows_failed, details)
            VALUES (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "source_id": source_id,
            "status": "ok" if failed == 0 else "ok_with_errors",
            "ins": inserted,
            "fail": failed,
            "details": json.dumps(details, ensure_ascii=False)
        })

    logger.info(f"{dataset_code}: inserted={inserted}, failed={failed}, series={len(series_cache)}")


# ============================================================
# Entry point
# ============================================================
def main(mode: str = "initial"):
    config = load_config()
    euro_cfg = config.get("eurostat", {}) if isinstance(config, dict) else {}

    if not euro_cfg:
        logger.warning("No 'eurostat' section in config.")
        return

    for dataset_code, dataset_cfg in euro_cfg.items():
        ingest_dataset(dataset_code, dataset_cfg or {}, mode=mode)


if __name__ == "__main__":
    main(mode="initial")
