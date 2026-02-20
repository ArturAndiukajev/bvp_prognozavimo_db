import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Iterable
import time
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fredmd")

# -------------------- Timer --------------------
class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None

    def __enter__(self):
        self.t0 = float(time.perf_counter())
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - (self.t0 or time.perf_counter())
        logger.info(f"[TIMER] {self.label}: {dt:.2f}s")

_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=3,
    connect_args={"connect_timeout": 10},
)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")

_DEFAULT_FREDMD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"


def load_config() -> dict:
    import yaml
    path = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_fredmd_url() -> str:
    cfg = load_config()
    return cfg.get("fredmd", {}).get("url", _DEFAULT_FREDMD_URL)

# -------------------- DB helpers --------------------
def last_release_hash(conn, dataset_id: int, scope: str | None = None) -> str | None:
    if scope:
        q = """
        SELECT content_hash
        FROM releases
        WHERE dataset_id=:did AND meta->>'scope' = :scope
        ORDER BY downloaded_at DESC
        LIMIT 1
        """
        return conn.execute(text(q), {"did": dataset_id, "scope": scope}).scalar_one_or_none()

    q = """
    SELECT content_hash
    FROM releases
    WHERE dataset_id=:did
    ORDER BY downloaded_at DESC
    LIMIT 1
    """
    return conn.execute(text(q), {"did": dataset_id}).scalar_one_or_none()

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

def ensure_series(conn, dataset_id: int, key: str, country: str, frequency: str,
                  transform: str, unit: str, name: str, meta: dict) -> int:
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO series (dataset_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:did, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
        ON CONFLICT (dataset_id, key, country, frequency, transform)
        DO UPDATE SET unit = EXCLUDED.unit,
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
        "meta": meta_json,
    }).scalar_one()

def create_release(conn, dataset_id: int, downloaded_at: datetime, vintage_at: datetime,
                   description: str, raw_path: str | None, content_hash: str | None, scope: str | None = None, meta: dict | None = None) -> int:
    meta = meta or {}
    if scope:
        meta["scope"] = scope
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO releases (dataset_id, release_time, downloaded_at, vintage_at, description, raw_path, content_hash, meta)
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


def _get_psycopg2_connection(sa_conn):
    raw = getattr(sa_conn, "connection", None)
    if raw is None:
        raise RuntimeError("Cannot access raw DBAPI connection")
    dbapi = getattr(raw, "connection", None) or getattr(raw, "driver_connection", None) or raw
    return dbapi


def copy_to_staging_and_merge(conn, rows: Iterable[Tuple[int, str, str, float, int]]):
    conn.execute(text("""
        CREATE TEMP TABLE IF NOT EXISTS observations_staging (
            series_id   BIGINT,
            period_date DATE,
            observed_at TIMESTAMPTZ,
            value       DOUBLE PRECISION,
            release_id  BIGINT
        ) ON COMMIT DROP
    """))
    conn.execute(text("TRUNCATE TABLE observations_staging"))

    buf = StringIO()
    for sid, pdate_iso, oat_iso, val, rid in rows:
        buf.write(f"{sid},{pdate_iso},{oat_iso},{val},{rid}\n")
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

    conn.execute(text("""
        INSERT INTO observations (series_id, period_date, observed_at, value, release_id, meta)
        SELECT series_id, period_date, observed_at, value, release_id, '{}'::jsonb
        FROM observations_staging
        ON CONFLICT (series_id, period_date, observed_at) DO NOTHING
    """))

# -------------------- File helpers --------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_current_fredmd_csv() -> Tuple[Path, datetime]:
    url = get_fredmd_url()
    downloaded_at = datetime.now(timezone.utc)
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")
    out_path = RAW_DIR / f"fred_md_{stamp}.csv"

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

    logger.info(f"Downloaded: {url}")
    logger.info(f"Saved to:   {out_path}")
    return out_path, downloaded_at

# -------------------- Parse --------------------

def parse_fredmd(csv_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      df_wide: index = dates, columns = variables
      meta: e.g., tcodes if present
    """
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    meta: Dict = {}

    # Detect tcodes row
    first_cell = str(df.iloc[0, 0]).strip().lower()
    if first_cell in {"transform", "tcode", "tcodes"}:
        tcodes = df.iloc[0].to_dict()
        tcodes.pop(first_col, None)
        meta["tcodes"] = tcodes
        df = df.iloc[1:].copy()

    df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
    df = df.dropna(subset=[first_col]).set_index(first_col).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, meta

# -------------------- Ingest --------------------

def ingest(df_wide: pd.DataFrame, dataset_meta: dict,
           vintage_at: datetime, downloaded_at: datetime,
           raw_path: str | None, content_hash: str | None, mode: str = "initial") -> None:
    source_name = "fredmd"
    inserted = 0
    failed = 0

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "fredmd", base_url="https://www.stlouisfed.org/")
        dataset_id  = ensure_dataset(conn, provider_id, key="fredmd_monthly_current", title="FRED-MD Monthly Current")

        release_id = create_release(
            conn=conn,
            dataset_id=dataset_id,
            downloaded_at=downloaded_at,
            vintage_at=vintage_at,
            description="FRED-MD current.csv snapshot (snapshot-vintage)",
            raw_path=raw_path,
            content_hash=content_hash,
            scope="FRED-MD",
            meta={"dataset": "FRED-MD"}
        )

        # FRED-MD is a monthly macro dataset (predictor panel)
        country = "US"
        freq = "M"
        transform = "LEVEL"
        unit = "INDEX"

        # Prepare COPY rows
        copy_rows = []
        COPY_BATCH = 100_000
        oat_iso = vintage_at.isoformat()

        with Timer("FRED-MD prepare + COPY"):
            for var in df_wide.columns:
                series_meta = {
                    "dataset": "FRED-MD",
                    "variable": var,
                    "tcode": dataset_meta.get("tcodes", {}).get(var)
                }
                series_id = ensure_series(
                    conn=conn,
                    dataset_id=dataset_id,
                    key=var,
                    country=country,
                    frequency=freq,
                    transform=transform,
                    unit=unit,
                    name=f"FRED-MD: {var}",
                    meta=series_meta
                )

                for dt, val in df_wide[var].items():
                    if pd.isna(val):
                        continue
                    copy_rows.append((
                        series_id,
                        dt.date().isoformat(),
                        oat_iso,
                        float(val),
                        release_id
                    ))
                    if len(copy_rows) >= COPY_BATCH:
                        copy_to_staging_and_merge(conn, copy_rows)
                        inserted += len(copy_rows)
                        copy_rows = []

            if copy_rows:
                copy_to_staging_and_merge(conn, copy_rows)
                inserted += len(copy_rows)

        details = {
            "downloaded_at": downloaded_at.isoformat(),
            "vintage_at": vintage_at.isoformat(),
            "raw_path": raw_path,
            "hash": content_hash,
            "columns": int(len(df_wide.columns)),
            "rows": int(len(df_wide))
        }
        conn.execute(text("""
            INSERT INTO ingestion_log (dataset_id, status, rows_inserted, rows_failed, details)
            VALUES (:dataset_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "dataset_id": dataset_id,
            "status": "ok" if failed == 0 else "ok_with_errors",
            "ins": inserted,
            "fail": failed,
            "details": json.dumps(details, ensure_ascii=False),
        })

    logger.info(f"FRED-MD inserted: {inserted}, failed: {failed}")

    # Run ANALYZE so planner stats are up to date after bulk load
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as ac:
        ac.execute(text("ANALYZE observations"))
        logger.info("ANALYZE observations: done")


def main(mode: str = "initial"):
    csv_path, downloaded_at = download_current_fredmd_csv()
    df_wide, dataset_meta = parse_fredmd(csv_path)

    vintage_at = downloaded_at
    content_hash = sha256_file(csv_path)

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "fredmd", base_url="https://www.stlouisfed.org/")
        dataset_id  = ensure_dataset(conn, provider_id, key="fredmd_monthly_current", title="FRED-MD Monthly Current")

        if mode == "update":
            prev = last_release_hash(conn, dataset_id, scope="FRED-MD")
            if prev == content_hash:
                logger.info("FRED-MD: no changes (hash same) -> skip")
                return

    ingest(
        df_wide=df_wide,
        dataset_meta=dataset_meta,
        vintage_at=vintage_at,
        downloaded_at=downloaded_at,
        raw_path=str(csv_path),
        content_hash=content_hash,
        mode=mode
    )

    # quick check
    with engine.connect() as conn:
        n_series = conn.execute(text("""
            SELECT count(*) FROM series s
            JOIN datasets d ON d.id = s.dataset_id
            JOIN providers p ON p.id = d.provider_id
            WHERE p.name = 'fredmd'
        """)).scalar_one()

        n_obs = conn.execute(text("""
            SELECT count(*)
            FROM observations o
            JOIN series s ON s.id = o.series_id
            JOIN datasets d ON d.id = s.dataset_id
            JOIN providers p ON p.id = d.provider_id
            WHERE p.name = 'fredmd'
        """)).scalar_one()

    logger.info(f"FRED-MD series in DB: {n_series}")
    logger.info(f"FRED-MD observations in DB: {n_obs}")

if __name__ == "__main__":
    main()

#