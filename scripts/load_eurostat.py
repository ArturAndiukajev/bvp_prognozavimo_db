# load_eurostat.py
import os
import json
import logging
import re
import hashlib
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Tuple
import time
import pandas as pd
import yaml
import eurostat
from sqlalchemy import create_engine, text
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eurostat")

# ----------------------------
# DB
# ----------------------------
_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)
engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    connect_args={"connect_timeout": 10},
)

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
# Timer helper
# ============================================================
class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0: float | None = None

    def __enter__(self):
        self.t0 = float(time.perf_counter())
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - (self.t0 or time.perf_counter())
        logger.info(f"[TIMER] {self.label}: {dt:.2f}s")

# ============================================================
# DB Helpers
# ============================================================
def ensure_provider(conn, name: str, base_url: str | None = None) -> int:
    conn.execute(text("""
        INSERT INTO providers (name, base_url) VALUES (:name, :url)
        ON CONFLICT (name) DO NOTHING
    """), {"name": name, "url": base_url})
    return conn.execute(text("SELECT id FROM providers WHERE name=:name"), {"name": name}).scalar_one()

def ensure_dataset(conn, provider_id: int, key: str, title: str | None = None) -> int:
    conn.execute(text("""
        INSERT INTO datasets (provider_id, key, title) VALUES (:pid, :key, :title)
        ON CONFLICT (provider_id, key) DO UPDATE SET title = COALESCE(EXCLUDED.title, datasets.title)
    """), {"pid": provider_id, "key": key, "title": title})
    return conn.execute(
        text("SELECT id FROM datasets WHERE provider_id=:pid AND key=:key"),
        {"pid": provider_id, "key": key}
    ).scalar_one()

def ensure_series(conn,
                  dataset_id: int,
                  key: str,
                  country: str,
                  frequency: str,
                  transform: str,
                  unit: Optional[str],
                  name: str,
                  meta: dict) -> int:
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO series (dataset_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:did, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
        ON CONFLICT (dataset_id, key, country, frequency, transform)
        DO UPDATE SET
            unit = EXCLUDED.unit,
            name = EXCLUDED.name,
            meta = EXCLUDED.meta
        RETURNING id
    """), {
        "did": dataset_id,
        "key": key,
        "country": country,
        "freq": frequency,
        "transform": transform,
        "unit": unit,
        "name": name,
        "meta": meta_json
    }).scalar_one()


def create_release(conn,
                   dataset_id: int,
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
        INSERT INTO releases (dataset_id, release_time, downloaded_at, vintage_at,
                              description, raw_path, content_hash, meta)
        VALUES (:did, :rtime, :dlat, :vint, :desc, :raw, :hash, CAST(:meta AS jsonb))
        RETURNING id
    """), {
        "did": dataset_id,
        "rtime": downloaded_at,
        "dlat": downloaded_at,
        "vint": vintage_at,
        "desc": description,
        "raw": raw_path,
        "hash": content_hash,
        "meta": meta_json
    }).scalar_one()


def last_release_hash(conn, dataset_id: int, scope: Optional[str] = None) -> Optional[str]:
    if scope:
        return conn.execute(text("""
            SELECT content_hash
            FROM releases
            WHERE dataset_id=:did AND meta->>'scope' = :scope
            ORDER BY downloaded_at DESC
            LIMIT 1
        """), {"did": dataset_id, "scope": scope}).scalar_one_or_none()

    return conn.execute(text("""
        SELECT content_hash
        FROM releases
        WHERE dataset_id=:did
        ORDER BY downloaded_at DESC
        LIMIT 1
    """), {"did": dataset_id}).scalar_one_or_none()


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
# COPY helpers (staging temp table)
# ============================================================
def _get_psycopg2_connection(sa_conn):
    """
    SQLAlchemy Connection -> raw psycopg2 connection.
    Works with SQLAlchemy 1.4/2.0 + psycopg2.
    """
    # SQLAlchemy 2.0: sa_conn.connection is ConnectionFairy, .connection is DBAPI conn
    raw = getattr(sa_conn, "connection", None)
    if raw is None:
        raise RuntimeError("Cannot access raw DBAPI connection")

    # some envs: raw has .connection, some: it's already the DBAPI connection
    dbapi = getattr(raw, "connection", None) or getattr(raw, "driver_connection", None) or raw
    return dbapi


def copy_to_staging_and_merge(conn, rows: Iterable[Tuple[int, date, datetime, float, int]]):
    """
    rows: (series_id, period_date, observed_at, value, release_id)
    Uses TEMP staging table + COPY, then merges into observations with ON CONFLICT DO NOTHING.
    """
    conn.execute(text("""
        CREATE TEMP TABLE IF NOT EXISTS observations_staging (
            series_id   BIGINT,
            period_date DATE,
            observed_at TIMESTAMPTZ,
            value       DOUBLE PRECISION,
            release_id  BIGINT
        ) ON COMMIT DROP
    """))

    # Truncate staging (in case temp table exists in same session)
    conn.execute(text("TRUNCATE TABLE observations_staging"))

    buf = StringIO()
    # CSV without header; fastest format for COPY
    # ensure ISO strings (COPY can parse date/timestamptz from ISO)
    for sid, pdate, oat, val, rid in rows:
        buf.write(f"{sid},{pdate.isoformat()},{oat.isoformat()},{val},{rid}\n")
    buf.seek(0)

    dbapi_conn = _get_psycopg2_connection(conn)
    cur = dbapi_conn.cursor()
    cur.copy_expert(
        """
        COPY observations_staging (series_id, period_date, observed_at, value, release_id)
        FROM STDIN WITH (FORMAT csv)
        """,
        buf
    )

    # merge into main table
    conn.execute(text("""
        INSERT INTO observations (series_id, period_date, observed_at, value, release_id, meta)
        SELECT series_id, period_date, observed_at, value, release_id, '{}'::jsonb
        FROM observations_staging
        ON CONFLICT (series_id, period_date, observed_at) DO NOTHING
    """))


# ============================================================
# Core ingestion
# ============================================================
def fetch_dataset(dataset_code: str) -> pd.DataFrame:
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
    with Timer(f"Eurostat fetch {dataset_code}"):
        try:
            df = fetch_dataset(dataset_code)
        except Exception as e:
            logger.error(f"Fetch failed for {dataset_code}: {e}")
            return

    if df is None or df.empty:
        logger.warning(f"{dataset_code}: empty dataset")
        return

    # 2) Filter + save raw + hash
    with Timer(f"Eurostat filter+save {dataset_code}"):
        df = apply_filters(df, filters)
        if df.empty:
            logger.warning(f"{dataset_code}: empty after filtering")
            return

        raw_path = RAW_DIR / f"{dataset_code}_{stamp}.csv"
        df.to_csv(raw_path, index=False)
        content_hash = sha256_file(raw_path)

    # 3) Scope for hash-based update skipping
    scope = f"dataset:{dataset_code}|filters:{json.dumps(filters, sort_keys=True)}"

    # 4) Transform wide -> long (melt)
    with Timer(f"Eurostat melt {dataset_code}"):
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

    failed = 0
    series_created = 0
    rows_prepared = 0
    inserted_attempted = 0

    dim_n = len(dim_cols)

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "eurostat", base_url="https://ec.europa.eu/eurostat")
        dataset_id  = ensure_dataset(conn, provider_id, key=dataset_code, title=dataset_name)

        # update-skip
        if mode == "update":
            prev_hash = last_release_hash(conn, dataset_id, scope=scope)
            if prev_hash == content_hash:
                logger.info(f"{dataset_code}: no changes (hash same) -> skip")
                return

        # release
        release_id = create_release(
            conn=conn,
            dataset_id=dataset_id,
            downloaded_at=downloaded_at,
            vintage_at=vintage_at,
            description=f"Eurostat snapshot {dataset_code}",
            raw_path=str(raw_path),
            content_hash=content_hash,
            scope=scope,
            meta={"dataset_code": dataset_code, "dataset_name": dataset_name, "filters": filters, "mode": mode}
        )

        # Cache series ids
        series_cache: dict[tuple, int] = {}

        # Prepare COPY batches
        COPY_BATCH = 200_000  #jei yra RAM, galima didint
        copy_rows: list[Tuple[int, date, datetime, float, int]] = []

        def flush_copy_rows():
            nonlocal inserted_attempted, copy_rows
            if not copy_rows:
                return
            copy_to_staging_and_merge(conn, copy_rows)
            inserted_attempted += len(copy_rows)
            copy_rows = []

        with Timer(f"Eurostat prepare+COPY {dataset_code}"):
            #itertuples greiciau veikia nei iterrows
            #index=False kad netempt indeksa
            for t in df_long.itertuples(index=False, name=None):
                # t: (dim0, dim1, ..., time_label, value)
                dims = {dim_cols[i]: normalize_dim_value(t[i]) for i in range(dim_n)}
                time_label = t[dim_n]
                value = t[dim_n + 1]

                country = guess_country(dims)
                freq = guess_frequency(dims, time_label)
                unit = normalize_dim_value(dims.get("unit"))

                series_key = build_series_key(dataset_code, dims, dim_cols)
                cache_key = (series_key, country, freq, "LEVEL")

                if cache_key not in series_cache:
                    pretty_dims = ", ".join([f"{c}:{dims.get(c)}" for c in dim_cols])
                    series_name = f"{dataset_name} ({dataset_code}) | {pretty_dims}"
                    meta = {"dataset_code": dataset_code, "dataset_name": dataset_name, "dimensions": dims,
                            "filters": filters}

                    try:
                        sid = ensure_series(
                            conn=conn,
                            dataset_id=dataset_id,
                            key=series_key,
                            country=country,
                            frequency=freq,
                            transform="LEVEL",
                            unit=unit,
                            name=series_name,
                            meta=meta
                        )
                        series_cache[cache_key] = sid
                        series_created += 1
                    except Exception:
                        failed += 1
                        continue

                series_id = series_cache[cache_key]

                try:
                    pdate = to_period_date(time_label)
                    val = float(value)
                except Exception:
                    failed += 1
                    continue

                copy_rows.append((series_id, pdate, vintage_at, val, release_id))
                if len(copy_rows) >= COPY_BATCH:
                    flush_copy_rows()

            flush_copy_rows()

        # ingestion log
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
            "series_created": int(series_created),
            "series_cache_size": int(len(series_cache)),
            "rows_long": int(len(df_long)),
            "rows_prepared": int(rows_prepared),
            "insert_attempted": int(inserted_attempted),
        }

        conn.execute(text("""
            INSERT INTO ingestion_log (dataset_id, status, rows_inserted, rows_failed, details)
            VALUES (:dataset_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "dataset_id": dataset_id,
            "status": "ok" if failed == 0 else "ok_with_errors",
            "ins": int(inserted_attempted),
            "fail": int(failed),
            "details": json.dumps(details, ensure_ascii=False)
        })

    logger.info(
        f"{dataset_code}: attempted={inserted_attempted}, failed={failed}, "
        f"series={len(series_cache)} created={series_created}"
    )


# ============================================================
# Entry point (parallel per dataset)
# ============================================================
def main(mode: str = "initial", max_workers: int = 4):
    """
    Parallelizes ingestion across Eurostat datasets.
    max_workers=4 is usually safe. If you hit API limits or DB pressure, reduce to 2.
    """
    config = load_config()
    euro_cfg = config.get("eurostat", {}) if isinstance(config, dict) else {}

    if not euro_cfg:
        logger.warning("No 'eurostat' section in config.")
        return

    items = list(euro_cfg.items())

    logger.info(f"Eurostat main: mode={mode}, datasets={len(items)}, max_workers={max_workers}")

    #galima isjungti - max_workers=1
    if max_workers <= 1:
        for dataset_code, dataset_cfg in items:
            ingest_dataset(dataset_code, dataset_cfg or {}, mode=mode)
        return

    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(ingest_dataset, dataset_code, (dataset_cfg or {}), mode): dataset_code
            for dataset_code, dataset_cfg in items
        }

        for fut in as_completed(futures):
            code = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Dataset {code} failed: {e}")


if __name__ == "__main__":
    # initial load: main("initial", max_workers=4)
    # update run:   main("update",  max_workers=4)
    main(mode="initial", max_workers=4)
    #