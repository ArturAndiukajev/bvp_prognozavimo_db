import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from fredapi import Fred
from dotenv import load_dotenv

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
CONFIG_PATH = Path(r"C:\Users\artur\bvp_prognozavimo_db\config\datasets.yaml")  # <- поменяй, если у тебя другой путь

# -------------------- DB helpers --------------------
def last_release_hash(conn, source_id: int, scope: str | None = None) -> str | None:
    if scope:
        q = """
        SELECT content_hash
        FROM releases
        WHERE source_id=:sid AND meta->>'scope' = :scope
        ORDER BY downloaded_at DESC
        LIMIT 1
        """
        return conn.execute(text(q), {"sid": source_id, "scope": scope}).scalar_one_or_none()

    q = """
    SELECT content_hash
    FROM releases
    WHERE source_id=:sid
    ORDER BY downloaded_at DESC
    LIMIT 1
    """
    return conn.execute(text(q), {"sid": source_id}).scalar_one_or_none()

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
                  transform: str, unit: str, name: str, meta: dict) -> int:
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
                   description: str, raw_path: str | None = None, content_hash: str | None = None, scope: str | None = None, meta: dict | None = None) -> int:
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

# -------------------- Config --------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"Config not found: {CONFIG_PATH.resolve()}")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# -------------------- Core --------------------

def fetch_and_ingest_series(series_key: str, realtime: bool = True, max_vintages: int = 10, mode: str = "initial"):
    if not fred:
        logger.error("FRED client not initialized (missing API key).")
        return

    logger.info(f"Series: {series_key}")

    # 1) Series info (metadata)
    try:
        info = fred.get_series_info(series_key)
    except Exception as e:
        logger.error(f"Failed to get info for {series_key}: {e}")
        return

    frequency = info.get("frequency_short", "M") or "M"
    units = info.get("units") or None
    title = info.get("title") or series_key

    # 2) Vintage dates
    target_vintages = []
    if realtime:
        try:
            vintage_dates = fred.get_series_vintage_dates(series_key)
            # take last max_vintages to reduce workload
            target_vintages = list(vintage_dates[-max_vintages:]) if vintage_dates is not None else []
        except Exception as e:
            logger.error(f"Failed to get vintage dates for {series_key}: {e}")
            return
    if not target_vintages:
        # fallback: current only
        target_vintages = [datetime.now(timezone.utc)]

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

        total_inserted = 0
        total_failed = 0

        for v_date in target_vintages:
            # ensure tz-aware
            if getattr(v_date, "tzinfo", None) is None:
                vintage_at = v_date.replace(tzinfo=timezone.utc)
            else:
                vintage_at = v_date.astimezone(timezone.utc)

            if mode == "update" and vintage_exists(conn, db_series_id, vintage_at):
                logger.info(f"Skip vintage {vintage_at.date()} (already in DB)")
                continue

            downloaded_at = datetime.now(timezone.utc)

            # Create release per vintage
            release_id = create_release(
                conn=conn,
                source_id=source_id,
                downloaded_at=downloaded_at,
                vintage_at=vintage_at,
                description=f"ALFRED vintage for {series_key} ({vintage_at.date().isoformat()})"
            )

            v_str = vintage_at.strftime("%Y-%m-%d")
            logger.info(f"  Vintage: {v_str}")

            try:
                data = fred.get_series(series_key, realtime_start=v_str, realtime_end=v_str)
                if data is None or len(data) == 0:
                    continue

                for p_date, value in data.items():
                    if pd.isna(value):
                        continue
                    try:
                        conn.execute(text("""
                            INSERT INTO observations (series_id, period_date, observed_at, value, release_id, meta)
                            VALUES (:sid, :pdate, :oat, :val, :rid, '{}'::jsonb)
                            ON CONFLICT (series_id, period_date, observed_at) DO NOTHING
                        """), {
                            "sid": db_series_id,
                            "pdate": p_date.date(),
                            "oat": vintage_at,     # <-- real vintage
                            "val": float(value),
                            "rid": release_id
                        })
                        total_inserted += 1
                    except Exception:
                        total_failed += 1

            except Exception as e:
                total_failed += 1
                logger.error(f"Failed vintage {v_str} for {series_key}: {e}")

        conn.execute(text("""
            INSERT INTO ingestion_log (source_id, status, rows_inserted, rows_failed, details)
            VALUES (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "source_id": source_id,
            "status": "ok" if total_failed == 0 else "ok_with_errors",
            "ins": total_inserted,
            "fail": total_failed,
            "details": json.dumps({"series": series_key, "vintages": len(target_vintages)}, ensure_ascii=False)
        })

    logger.info(f"Done {series_key}: inserted={total_inserted}, failed={total_failed}")

def main():
    config = load_config()
    alfred_cfg = config.get("alfred", {}) if isinstance(config, dict) else {}

    if not API_KEY:
        logger.error("No API key configured. Set FRED_API_KEY in .env")
        return

    if not alfred_cfg:
        logger.warning("No 'alfred' section in config.")
        return

    # expected format:
    # alfred:
    #   GDPC1: {realtime: true}
    for series_key, details in alfred_cfg.items():
        realtime = True
        if isinstance(details, dict):
            realtime = bool(details.get("realtime", True))
        fetch_and_ingest_series(series_key, realtime=realtime, max_vintages=10, mode="update")

if __name__ == "__main__":
    main()
