import os
import json
import logging
import hashlib
import time
import yaml
from pathlib import Path
from datetime import datetime
from io import StringIO
from typing import Optional, Iterable, Tuple, Dict, Any
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Initialize logging for helpers
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("db_helpers")

load_dotenv()

# --- Configuration & Engine ---

_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"

def get_db_url() -> str:
    return os.environ.get("DB_URL", _DEFAULT_DB_URL)

def get_engine(pool_size: int = 5, max_overflow: int = 10):
    return create_engine(
        get_db_url(),
        future=True,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        connect_args={"connect_timeout": 10},
    )

# --- Utilities ---

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

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# --- Database Operations ---

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
                  transform: str, unit: str, name: str, meta: dict | None = None) -> int:
    meta = meta or {}
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO series (dataset_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:did, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
        ON CONFLICT (dataset_id, key, country, frequency, transform)
        DO UPDATE SET name = EXCLUDED.name, meta = EXCLUDED.meta
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

def last_release_hash(conn, dataset_id: int, scope: str | None = None) -> Optional[str]:
    if scope:
        return conn.execute(text("""
            SELECT content_hash FROM releases
            WHERE dataset_id=:did AND meta->>'scope' = :scope
            ORDER BY downloaded_at DESC LIMIT 1
        """), {"did": dataset_id, "scope": scope}).scalar_one_or_none()

    return conn.execute(text("""
        SELECT content_hash FROM releases
        WHERE dataset_id=:did
        ORDER BY downloaded_at DESC LIMIT 1
    """), {"did": dataset_id}).scalar_one_or_none()

def create_release(conn, dataset_id: int, downloaded_at: datetime, vintage_at: datetime,
                   description: str, raw_path: str | None, content_hash: str | None,
                   scope: str | None = None, meta: dict | None = None) -> int:
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

def log_ingestion(conn, dataset_id: int, status: str, rows_inserted: int = 0, 
                  rows_failed: int = 0, details: dict | None = None):
    details = details or {}
    details_json = json.dumps(details, ensure_ascii=False)
    conn.execute(text("""
        INSERT INTO ingestion_log (dataset_id, status, rows_inserted, rows_failed, details)
        VALUES (:did, :status, :ins, :fail, CAST(:details AS jsonb))
    """), {
        "did": dataset_id,
        "status": status,
        "ins": rows_inserted,
        "fail": rows_failed,
        "details": details_json
    })

def _get_psycopg2_connection(sa_conn):
    raw = getattr(sa_conn, "connection", None)
    if raw is None:
        raise RuntimeError("Cannot access raw DBAPI connection")
    dbapi = getattr(raw, "connection", None) or getattr(raw, "driver_connection", None) or raw
    return dbapi

def copy_to_staging_and_merge(conn, table_name: str, rows: Iterable[Tuple[int, str, str, float, int]], 
                              columns: Tuple[str, ...] = ("series_id", "period_date", "observed_at", "value", "release_id")):
    """
    Generic COPY helper. Default columns match observations.
    """
    col_str = ", ".join(columns)
    staging_table = f"{table_name}_staging"
    
    # We assume the staging table matches the target table structure for these columns
    conn.execute(text(f"CREATE TEMP TABLE IF NOT EXISTS {staging_table} (LIKE {table_name} INCLUDING DEFAULTS) ON COMMIT DROP"))
    conn.execute(text(f"ALTER TABLE {staging_table} DROP COLUMN IF EXISTS id")) # id is usually serial
    conn.execute(text(f"ALTER TABLE {staging_table} DROP COLUMN IF EXISTS created_at"))
    conn.execute(text(f"TRUNCATE TABLE {staging_table}"))

    buf = StringIO()
    for row in rows:
        buf.write(",".join(map(str, row)) + "\n")
    buf.seek(0)

    dbapi_conn = _get_psycopg2_connection(conn)
    with dbapi_conn.cursor() as cur:
        cur.copy_from(buf, staging_table, sep=",", columns=columns)

    # Upsert from staging to main
    # This logic is specific to observations right now (UNIQUE constraint)
    if table_name == "observations":
        conn.execute(text(f"""
            INSERT INTO {table_name} ({col_str})
            SELECT {col_str} FROM {staging_table}
            ON CONFLICT (series_id, period_date, observed_at)
            DO UPDATE SET 
                value = EXCLUDED.value,
                release_id = COALESCE(EXCLUDED.release_id, {table_name}.release_id),
                meta = {table_name}.meta || EXCLUDED.meta
        """))

def analyze_table(engine, table_name: str):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        logger.info(f"Running ANALYZE on {table_name}...")
        conn.execute(text(f"ANALYZE {table_name}"))
