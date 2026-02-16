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


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DB
DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

# Paths
RAW_DIR = Path("data/raw/google_trends")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


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

def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG
    if not path.exists():
        logger.error(f"Config file not found at {CONFIG_PATH} nor {FALLBACK_CONFIG}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ensure_source(conn) -> int:
    conn.execute(text("""
        INSERT INTO sources (name) VALUES ('google_trends')
        ON CONFLICT (name) DO NOTHING
    """))
    return conn.execute(text("SELECT id FROM sources WHERE name='google_trends'")).scalar_one()


def ensure_series(conn, source_id: int, keyword: str, country: str) -> int:
    key = f"google_trends.{keyword}"
    return conn.execute(text("""
        INSERT INTO series (source_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:sid, :key, :country, 'M', 'LEVEL', 'INDEX_0_100', :name, '{}'::jsonb)
        ON CONFLICT (source_id, key, country, frequency, transform)
        DO UPDATE SET name = EXCLUDED.name
        RETURNING id
    """), {
        "sid": source_id,
        "key": key,
        "country": country,
        "name": f"Google Trends: {keyword}"
    }).scalar_one()


def create_release(conn, source_id: int, downloaded_at: datetime, vintage_at: datetime,
                   description: str, raw_path: str | None, content_hash: str | None, scope: str | None = None, meta: dict | None = None) -> int:
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


def insert_observations(conn, series_id: int, df_monthly: pd.DataFrame,
                        vintage_at: datetime, release_id: int) -> int:
    inserted = 0
    for dt, row in df_monthly.iterrows():
        value = row["value"]
        if pd.isna(value):
            continue
        conn.execute(text("""
            INSERT INTO observations (series_id, period_date, observed_at, value, release_id, meta)
            VALUES (:sid, :pdate, :oat, :val, :rid, '{}'::jsonb)
            ON CONFLICT (series_id, period_date, observed_at) DO NOTHING
        """), {
            "sid": series_id,
            "pdate": dt.date(),
            "oat": vintage_at,
            "val": float(value),
            "rid": release_id
        })
        inserted += 1
    return inserted


def main(mode: str = "initial"):
    config = load_config()
    gt_config = config.get("google_trends", {})
    if not gt_config:
        logger.warning("No 'google_trends' section in config.")
        return

    keywords = gt_config.get("keywords", [])
    geo = gt_config.get("geo", "LT")
    timeframe = gt_config.get("timeframe", "today 5-y")  # можно менять в yaml

    if not keywords:
        logger.warning("No keywords found in google_trends config.")
        return

    # Query time = our snapshot vintage (Google Trends data can vary over time)
    downloaded_at = datetime.now(timezone.utc)
    vintage_at = downloaded_at
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")

    pytrends = TrendReq(hl="en-US", tz=360)

    inserted_total = 0
    failed_total = 0

    with engine.begin() as conn:
        source_id = ensure_source(conn)

        for batch in chunk_keywords(keywords, 5):
            logger.info(f"Fetching batch: {batch}")

            try:
                pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo, gprop="")
                data = pytrends.interest_over_time()

                if data is None or data.empty:
                    logger.warning(f"No data for batch {batch}")
                    continue

                if "isPartial" in data.columns:
                    data = data.drop(columns=["isPartial"])

                # Save raw for this batch
                raw_path = RAW_DIR / f"gt_{geo}_{stamp}_{abs(hash(tuple(batch)))}.csv"
                data.to_csv(raw_path, index=True)

                # Hash raw bytes (simple)
                raw_bytes = raw_path.read_bytes()
                content_hash = sha256_bytes(raw_bytes)

                scope = f"geo:{geo}|tf:{timeframe}|batch:{abs(hash(tuple(batch)))}"
                prev_hash = last_release_hash(conn, source_id, scope=scope)
                if mode == "update" and prev_hash == content_hash:
                    logger.info(f"GT batch {batch}: no changes -> skip")
                    continue

                release_id = create_release(
                    conn, source_id,
                    downloaded_at=downloaded_at,
                    vintage_at=vintage_at,
                    description=f"Google Trends snapshot geo={geo} batch={batch}",
                    raw_path=str(raw_path),
                    content_hash=content_hash,
                    scope=scope,
                    meta={"geo": geo, "timeframe": timeframe, "batch": batch}
                )

                # Insert each keyword series
                for kw in batch:
                    if kw not in data.columns:
                        continue

                    df_kw = data[[kw]].copy()
                    df_monthly = monthly_from_weekly(df_kw)
                    if df_monthly.empty:
                        continue

                    series_id = ensure_series(conn, source_id, kw, geo)
                    cnt = insert_observations(conn, series_id, df_monthly, vintage_at, release_id)
                    inserted_total += cnt
                    logger.info(f"  {kw}: inserted {cnt}")

                time.sleep(2)

            except Exception as e:
                failed_total += 1
                logger.error(f"Failed batch {batch}: {e}")

        details = {
            "geo": geo,
            "timeframe": timeframe,
            "keywords": len(keywords),
            "downloaded_at": downloaded_at.isoformat(),
            "vintage_at": vintage_at.isoformat(),
            "inserted_total": inserted_total,
            "failed_batches": failed_total,
        }
        conn.execute(text("""
            INSERT INTO ingestion_log (source_id, status, rows_inserted, rows_failed, details)
            VALUES (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "source_id": source_id,
            "status": "ok" if failed_total == 0 else "ok_with_errors",
            "ins": inserted_total,
            "fail": failed_total,
            "details": json.dumps(details, ensure_ascii=False)
        })

    logger.info(f"Google Trends done: inserted={inserted_total}, failed_batches={failed_total}")


if __name__ == "__main__":
    main(mode="initial")