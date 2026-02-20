import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import time

import yaml
import pandas as pd
from sqlalchemy import create_engine, text
from pytrends.request import TrendReq
from dotenv import load_dotenv

from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Iterable, Tuple

load_dotenv()


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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DB
_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)
engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=3,
    connect_args={"connect_timeout": 10},
)

# Paths
RAW_DIR = Path("data/raw/google_trends")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


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

def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG
    if not path.exists():
        logger.error(f"Config file not found at {CONFIG_PATH} nor {FALLBACK_CONFIG}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# Maps ISO geo code -> readable country name used as dataset key
GEO_NAMES: dict[str, str] = {
    "LT": "Lithuania",
    "LV": "Latvia",
    "EE": "Estonia",
    "PL": "Poland",
    "DE": "Germany",
    "FR": "France",
    "US": "United States",
    "GB": "United Kingdom",
    "SE": "Sweden",
    "FI": "Finland",
    "NO": "Norway",
    "DK": "Denmark",
}

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


def ensure_series(conn, dataset_id: int, keyword: str, country: str) -> int:
    key = f"google_trends.{keyword}"
    return conn.execute(text("""
        INSERT INTO series (dataset_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:did, :key, :country, 'M', 'LEVEL', 'INDEX_0_100', :name, '{}'::jsonb)
        ON CONFLICT (dataset_id, key, country, frequency, transform)
        DO UPDATE SET name = EXCLUDED.name
        RETURNING id
    """), {
        "did": dataset_id,
        "key": key,
        "country": country,
        "name": f"Google Trends: {keyword}"
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


def monthly_from_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    monthly = df.resample("MS").mean()
    if not monthly.empty:
        monthly.columns = ["value"]
    return monthly


def chunk_keywords(lst, size=5):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def prepare_rows_for_copy(series_id: int, df_monthly: pd.DataFrame,
                         vintage_at: datetime, release_id: int) -> list:
    rows = []
    oat_iso = vintage_at.isoformat()
    for dt, row in df_monthly.iterrows():
        value = row["value"]
        if pd.isna(value):
            continue
        rows.append((
            series_id,
            dt.date().isoformat(),
            oat_iso,
            float(value),
            release_id
        ))
    return rows


def ingest_batch(batch, geo, geo_name, timeframe, mode, downloaded_at, vintage_at, stamp, dataset_id):
    logger.info(f"Ingesting batch: {batch}")
    pytrends = TrendReq(hl="en-US", tz=360)
    try:
        with Timer(f"GT fetch {batch}"):
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo, gprop="")
            data = pytrends.interest_over_time()

        if data is None or data.empty:
            logger.warning(f"No data for batch {batch}")
            return {"status": "empty", "inserted": 0, "failed": 0}

        if "isPartial" in data.columns:
            data = data.drop(columns=["isPartial"])

        # Save raw
        raw_path = RAW_DIR / f"gt_{geo}_{stamp}_{abs(hash(tuple(batch)))}.csv"
        data.to_csv(raw_path, index=True)
        content_hash = sha256_bytes(raw_path.read_bytes())

        scope = f"geo:{geo}|tf:{timeframe}|batch:{abs(hash(tuple(batch)))}"

        with engine.begin() as conn:
            if mode == "update":
                prev_hash = last_release_hash(conn, dataset_id, scope=scope)
                if prev_hash == content_hash:
                    logger.info(f"GT batch {batch}: no changes -> skip")
                    return {"status": "skipped", "inserted": 0, "failed": 0}

            release_id = create_release(
                conn, dataset_id,
                downloaded_at=downloaded_at,
                vintage_at=vintage_at,
                description=f"Google Trends snapshot geo={geo} batch={batch}",
                raw_path=str(raw_path),
                content_hash=content_hash,
                scope=scope,
                meta={"geo": geo, "geo_name": geo_name, "timeframe": timeframe, "batch": batch}
            )

            batch_inserted = 0
            for kw in batch:
                if kw not in data.columns:
                    continue
                df_kw = data[[kw]].copy()
                df_monthly = monthly_from_weekly(df_kw)
                if df_monthly.empty:
                    continue

                series_id = ensure_series(conn, dataset_id, kw, geo)
                rows = prepare_rows_for_copy(series_id, df_monthly, vintage_at, release_id)
                if rows:
                    copy_to_staging_and_merge(conn, rows)
                    batch_inserted += len(rows)

            return {"status": "ok", "inserted": batch_inserted, "failed": 0}

    except Exception as e:
        logger.error(f"Failed batch {batch}: {e}")
        return {"status": "error", "inserted": 0, "failed": 1}


def main(mode: str = "initial", max_workers: int = 2):
    config = load_config()
    gt_config = config.get("google_trends", {})
    if not gt_config:
        logger.warning("No 'google_trends' section in config.")
        return

    keywords = gt_config.get("keywords", [])
    geos = gt_config.get("geos", gt_config.get("geo", ["LT"]))
    if isinstance(geos, str):
        geos = [geos]
    timeframe = gt_config.get("timeframe", "today 5-y")

    if not keywords:
        logger.warning("No keywords found in google_trends config.")
        return

    downloaded_at = datetime.now(timezone.utc)
    vintage_at = downloaded_at
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")

    inserted_total = 0
    failed_batches = 0

    batches = list(chunk_keywords(keywords, 5))
    results = []

    provider_id_cache: dict[str, int] = {}

    for geo in geos:
        geo_name = GEO_NAMES.get(geo, geo)  # fallback: raw geo code
        logger.info(f"Processing geo: {geo} ({geo_name})")

        with engine.begin() as conn:
            provider_id = ensure_provider(conn, "google_trends", base_url="https://trends.google.com")
            dataset_id  = ensure_dataset(conn, provider_id, key=geo_name, title=f"Google Trends — {geo_name}")

        if max_workers <= 1:
            for b in batches:
                results.append(ingest_batch(b, geo, geo_name, timeframe, mode, downloaded_at, vintage_at, stamp, dataset_id))
                time.sleep(2)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(ingest_batch, b, geo, geo_name, timeframe, mode, downloaded_at, vintage_at, stamp, dataset_id): b
                    for b in batches
                }
                for fut in as_completed(futs):
                    results.append(fut.result())
                    time.sleep(1)  # subtle throttle between completions

    for r in results:
        inserted_total += r.get("inserted", 0)
        failed_batches += r.get("failed", 0)

    # Write one summary ingestion_log entry per geo (use last geo's dataset_id)
    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "google_trends", base_url="https://trends.google.com")
        # Use a single summary dataset for the run-level log
        dataset_id  = ensure_dataset(conn, provider_id, key="__run_summary__", title="Google Trends Run Summary")
        details = {
            "geo": geo,
            "timeframe": timeframe,
            "keywords": len(keywords),
            "downloaded_at": downloaded_at.isoformat(),
            "vintage_at": vintage_at.isoformat(),
            "inserted_total": inserted_total,
            "failed_batches": failed_batches,
        }
        conn.execute(text("""
            INSERT INTO ingestion_log (dataset_id, status, rows_inserted, rows_failed, details)
            VALUES (:dataset_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "dataset_id": dataset_id,
            "status": "ok" if failed_batches == 0 else "ok_with_errors",
            "ins": inserted_total,
            "fail": failed_batches,
            "details": json.dumps(details, ensure_ascii=False)
        })

    logger.info(f"Google Trends done: inserted={inserted_total}, failed_batches={failed_batches}")


if __name__ == "__main__":
    main(mode="initial", max_workers=2)
