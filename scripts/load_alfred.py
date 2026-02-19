import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import time
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from fredapi import Fred
from dotenv import load_dotenv
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Iterable, Tuple

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("alfred")

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

API_KEY = os.environ.get("FRED_API_KEY")
if not API_KEY:
    logger.warning("FRED_API_KEY not found in environment. ALFRED loading will fail.")
fred = Fred(api_key=API_KEY) if API_KEY else None

# Path to config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - (self.t0 or time.perf_counter())
        logger.info(f"[TIMER] {self.label}: {dt:.2f}s")
# -------------------- DB helpers --------------------
def vintage_exists(conn, series_id: int, vintage_at: datetime) -> bool:
    return conn.execute(text("""
        SELECT 1 FROM observations
        WHERE series_id=:sid AND observed_at=:oat
        LIMIT 1
    """), {"sid": series_id, "oat": vintage_at}).scalar_one_or_none() is not None

def ensure_source(conn, name: str) -> int:
    conn.execute(text("""
        INSERT INTO sources (name) VALUES (:name)
        ON CONFLICT (name) DO NOTHING
    """), {"name": name})
    return conn.execute(text("SELECT id FROM sources WHERE name=:name"), {"name": name}).scalar_one()

def ensure_series(conn, source_id: int, key: str, country: str, frequency: str,
                  transform: str, unit: Optional[str], name: str, meta: dict) -> int:
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO series (source_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:sid, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
        ON CONFLICT (source_id, key, country, frequency, transform)
        DO UPDATE SET unit = EXCLUDED.unit,
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

def create_release(conn, source_id: int, downloaded_at: datetime, vintage_at: datetime,
                   description: str, raw_path: str | None = None, content_hash: str | None = None,
                   scope: str | None = None, meta: dict | None = None) -> int:
    meta = meta or {}
    if scope:
        meta["scope"] = scope
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO releases (source_id, release_time, downloaded_at, vintage_at, description, raw_path, content_hash, meta)
        VALUES (:sid, :rtime, :dlat, :vint, :desc, :raw, :hash, CAST(:meta AS jsonb))
        RETURNING id
    """), {
        "sid": source_id,
        "rtime": downloaded_at,
        "dlat": downloaded_at,
        "vint": vintage_at,
        "desc": description,
        "raw": raw_path,
        "hash": content_hash,
        "meta": meta_json
    }).scalar_one()

def _get_dbapi_connection(sa_conn):
    # SQLAlchemy 2.x-safe
    raw = sa_conn.connection
    dbapi = getattr(raw, "driver_connection", None)
    if dbapi is None:
        # fallback for older
        dbapi = getattr(raw, "connection", None) or raw
    return dbapi


def copy_observations_via_staging(
    conn,
    rows: Iterable[Tuple[int, str, str, float, int]]
):
    """
    rows: (series_id, period_date_iso, observed_at_iso, value, release_id)
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
    conn.execute(text("TRUNCATE TABLE observations_staging"))

    buf = StringIO()
    for sid, pdate_iso, oat_iso, val, rid in rows:
        buf.write(f"{sid},{pdate_iso},{oat_iso},{val},{rid}\n")
    buf.seek(0)

    dbapi_conn = _get_dbapi_connection(conn)
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

# -------------------- Config --------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"Config not found: {CONFIG_PATH.resolve()}")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# -------------------- FRED helpers --------------------
def _ensure_tz_utc(dt) -> datetime:
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_series_info(series_key: str):
    return fred.get_series_info(series_key)


def fetch_vintage_dates(series_key: str):
    return fred.get_series_vintage_dates(series_key)


def fetch_series_vintage(series_key: str, vintage_at: datetime) -> pd.Series:
    v_str = vintage_at.strftime("%Y-%m-%d")
    return fred.get_series(series_key, realtime_start=v_str, realtime_end=v_str)


# -------------------- Core --------------------
def ingest_one_series(series_key: str, realtime: bool, max_vintages: int, mode: str) -> dict:
    """
    summary dict.
    """
    if not fred:
        return {"series": series_key, "status": "fail", "error": "fred client not initialized"}

    with Timer(f"ALFRED series total {series_key}"):
        # 1) metadata
        try:
            with Timer(f"ALFRED info {series_key}"):
                info = fetch_series_info(series_key)
        except Exception as e:
            return {"series": series_key, "status": "fail", "error": f"info failed: {e}"}

        frequency = info.get("frequency_short", "M") or "M"
        units = info.get("units") or None
        title = info.get("title") or series_key

        # 2) vintages
        target_vintages = []
        if realtime:
            try:
                with Timer(f"ALFRED vintages {series_key}"):
                    vds = fetch_vintage_dates(series_key)
                if vds is not None and len(vds) > 0:
                    target_vintages = list(vds[-max_vintages:])
            except Exception as e:
                return {"series": series_key, "status": "fail", "error": f"vintage dates failed: {e}"}

        if not target_vintages:
            target_vintages = [datetime.now(timezone.utc)]

        inserted_attempted = 0
        failed = 0
        skipped_vintages = 0
        processed_vintages = 0

        with engine.begin() as conn:
            source_id = ensure_source(conn, "alfred")

            db_series_id = ensure_series(
                conn=conn,
                source_id=source_id,
                key=series_key,
                country="US",
                frequency=frequency,
                transform="LEVEL",
                unit=units,
                name=title,
                meta={"fred_info": info.to_dict() if hasattr(info, "to_dict") else dict(info)}
            )

            #batch size for COPY
            COPY_BATCH = 200_000

            for v_date in target_vintages:
                vintage_at = _ensure_tz_utc(v_date)

                if mode == "update" and vintage_exists(conn, db_series_id, vintage_at):
                    skipped_vintages += 1
                    continue

                downloaded_at = datetime.now(timezone.utc)

                #kuriam release jei tik iterpinesim vintage'a
                release_id = create_release(
                    conn=conn,
                    source_id=source_id,
                    downloaded_at=downloaded_at,
                    vintage_at=vintage_at,
                    description=f"ALFRED vintage for {series_key} ({vintage_at.date().isoformat()})",
                    scope=f"series:{series_key}|vintage:{vintage_at.date().isoformat()}",
                    meta={"series": series_key, "vintage": vintage_at.date().isoformat(), "mode": mode}
                )

                processed_vintages += 1

                try:
                    with Timer(f"ALFRED fetch data {series_key} {vintage_at.date().isoformat()}"):
                        data = fetch_series_vintage(series_key, vintage_at)
                except Exception as e:
                    failed += 1
                    logger.error(f"{series_key}: vintage {vintage_at.date().isoformat()} fetch failed: {e}")
                    continue

                if data is None or len(data) == 0:
                    continue

                rows = []
                # data: pandas Series index=Timestamp
                for p_date, value in data.items():
                    if pd.isna(value):
                        continue
                    rows.append((
                        db_series_id,
                        p_date.date().isoformat(),
                        vintage_at.isoformat(),
                        float(value),
                        release_id
                    ))
                    if len(rows) >= COPY_BATCH:
                        copy_observations_via_staging(conn, rows)
                        inserted_attempted += len(rows)
                        rows = []

                if rows:
                    copy_observations_via_staging(conn, rows)
                    inserted_attempted += len(rows)

            #ingestion_log on series level
            conn.execute(text("""
                INSERT INTO ingestion_log (source_id, status, rows_inserted, rows_failed, details)
                VALUES (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
            """), {
                "source_id": source_id,
                "status": "ok" if failed == 0 else "ok_with_errors",
                "ins": int(inserted_attempted),
                "fail": int(failed),
                "details": json.dumps({
                    "series": series_key,
                    "mode": mode,
                    "realtime": realtime,
                    "vintages_target": len(target_vintages),
                    "vintages_processed": processed_vintages,
                    "vintages_skipped": skipped_vintages,
                    "attempted_rows": inserted_attempted,
                    "frequency": frequency,
                    "unit": units
                }, ensure_ascii=False)
            })

        return {
            "series": series_key,
            "status": "ok" if failed == 0 else "ok_with_errors",
            "attempted": inserted_attempted,
            "failed": failed,
            "vintages_target": len(target_vintages),
            "vintages_processed": processed_vintages,
            "vintages_skipped": skipped_vintages,
        }


def main(mode: str = "update", max_workers: int = 2, max_vintages: int = 10):
    config = load_config()
    alfred_cfg = config.get("alfred", {}) if isinstance(config, dict) else {}

    if not API_KEY:
        logger.error("No API key configured. Set FRED_API_KEY in .env")
        return

    if not alfred_cfg:
        logger.warning("No 'alfred' section in config.")
        return

    items = list(alfred_cfg.items())
    logger.info(f"ALFRED main: mode={mode}, series={len(items)}, max_workers={max_workers}, max_vintages={max_vintages}")

    if max_workers <= 1:
        for series_key, details in items:
            realtime = True
            if isinstance(details, dict):
                realtime = bool(details.get("realtime", True))
            res = ingest_one_series(series_key, realtime=realtime, max_vintages=max_vintages, mode=mode)
            logger.info(f"ALFRED result: {res}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for series_key, details in items:
            realtime = True
            if isinstance(details, dict):
                realtime = bool(details.get("realtime", True))
            futures[ex.submit(ingest_one_series, series_key, realtime, max_vintages, mode)] = series_key

        for fut in as_completed(futures):
            series_key = futures[fut]
            try:
                res = fut.result()
                logger.info(f"ALFRED result {series_key}: {res}")
            except Exception as e:
                logger.error(f"ALFRED series {series_key} failed: {e}")


if __name__ == "__main__":
    main(mode="update", max_workers=2, max_vintages=10)
#