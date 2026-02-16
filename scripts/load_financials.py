import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone, date

import yaml
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

RAW_DIR = Path("data/raw/yahoo_finance")
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

def last_period_date(conn, series_id: int) -> date | None:
    return conn.execute(text("""
        SELECT max(period_date) FROM observations WHERE series_id=:sid
    """), {"sid": series_id}).scalar_one_or_none()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_source(conn) -> int:
    conn.execute(text("""
        INSERT INTO sources (name) VALUES ('yahoo_finance')
        ON CONFLICT (name) DO NOTHING
    """))
    return conn.execute(text("SELECT id FROM sources WHERE name='yahoo_finance'")).scalar_one()


def ensure_series(conn, source_id: int, ticker: str, name: str) -> int:
    return conn.execute(text("""
        INSERT INTO series (source_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:sid, :key, 'Global', 'D', 'LEVEL', 'PRICE', :name, '{}'::jsonb)
        ON CONFLICT (source_id, key, country, frequency, transform)
        DO UPDATE SET name = EXCLUDED.name
        RETURNING id
    """), {
        "sid": source_id,
        "key": ticker,
        "name": name
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


def insert_observations(conn, series_id: int, df: pd.DataFrame,
                        vintage_at: datetime, release_id: int) -> int:
    inserted = 0
    for dt, row in df.iterrows():
        value = row["Close"]
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
    """
    mode:
      - "initial": full initial load (uses configured 'period' per ticker, e.g. 5y)
      - "update": incremental load (fetches only tail from last known date - tail_days)
    """
    config = load_config()
    yf_config = config.get("yahoo_finance", {})
    if not yf_config:
        logger.warning("No 'yahoo_finance' section in config.")
        return

    downloaded_at = datetime.now(timezone.utc)
    vintage_at = downloaded_at  # snapshot-vintage for Yahoo
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")

    tail_days_default = 7  # safe tail window for updates

    inserted_total = 0
    failed_total = 0
    skipped_total = 0

    with engine.begin() as conn:
        source_id = ensure_source(conn)

        for ticker, details in yf_config.items():
            # Parse config
            if isinstance(details, dict):
                name = details.get("name", ticker)
                period = details.get("period", "5y")
                tail_days = int(details.get("tail_days", tail_days_default))
            else:
                name = str(details)
                period = "5y"
                tail_days = tail_days_default

            # Ensure series exists
            try:
                series_id = ensure_series(conn, source_id, ticker, name)
            except Exception as e:
                failed_total += 1
                logger.error(f"{ticker}: ensure_series failed: {e}")
                continue

            # Decide fetch window
            start = None
            fetch_mode_label = f"period={period}"
            if mode == "update":
                last_dt = conn.execute(text("""
                    SELECT max(period_date) FROM observations WHERE series_id=:sid
                """), {"sid": series_id}).scalar_one_or_none()

                if last_dt:
                    # fetch a tail window to catch new points + minor revisions
                    start = (pd.Timestamp(last_dt) - pd.Timedelta(days=tail_days)).date()
                    fetch_mode_label = f"start={start.isoformat()} (tail_days={tail_days})"
                else:
                    # no data yet -> fallback to initial window
                    fetch_mode_label = f"fallback period={period}"

            logger.info(f"Yahoo Finance: {ticker} ({name}) [{mode}] {fetch_mode_label}")

            # Fetch data
            try:
                t = yf.Ticker(ticker)
                if start is not None:
                    hist = t.history(start=str(start))
                else:
                    hist = t.history(period=period)

                if hist is None or hist.empty:
                    logger.warning(f"{ticker}: no data returned.")
                    continue

                df = hist[["Close"]].copy()

                # Save raw snapshot per ticker per run
                raw_path = RAW_DIR / f"{ticker}_{stamp}.csv"
                df.to_csv(raw_path, index=True)
                content_hash = sha256_file(raw_path)

                # Update-skip logic (hash)
                scope = f"ticker:{ticker}|{fetch_mode_label}"
                prev_hash = conn.execute(text("""
                    SELECT content_hash
                    FROM releases
                    WHERE source_id=:sid AND meta->>'scope' = :scope
                    ORDER BY downloaded_at DESC
                    LIMIT 1
                """), {"sid": source_id, "scope": scope}).scalar_one_or_none()

                if mode == "update" and prev_hash == content_hash:
                    skipped_total += 1
                    logger.info(f"{ticker}: no changes (hash same) -> skip insert/release")
                    continue

                # Create release
                release_id = create_release(
                    conn, source_id,
                    downloaded_at=downloaded_at,
                    vintage_at=vintage_at,
                    description=f"Yahoo Finance snapshot {ticker} [{fetch_mode_label}]",
                    raw_path=str(raw_path),
                    content_hash=content_hash,
                    scope=scope,
                    meta={"ticker": ticker, "name": name, "mode": mode, "fetch": fetch_mode_label}
                )

                # Insert observations (ON CONFLICT DO NOTHING handles duplicates)
                cnt = insert_observations(conn, series_id, df, vintage_at, release_id)
                inserted_total += cnt
                logger.info(f"{ticker}: inserted {cnt} rows")

            except Exception as e:
                failed_total += 1
                logger.error(f"{ticker}: failed to fetch/ingest: {e}")

        # Log ingestion run
        details = {
            "mode": mode,
            "downloaded_at": downloaded_at.isoformat(),
            "vintage_at": vintage_at.isoformat(),
            "tickers": len(yf_config),
            "inserted_total": inserted_total,
            "skipped_total": skipped_total,
            "failed_total": failed_total
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

    logger.info(f"Yahoo Finance [{mode}] done: inserted={inserted_total}, skipped={skipped_total}, failed={failed_total}")

if __name__ == "__main__":
    main()