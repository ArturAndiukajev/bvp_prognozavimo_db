import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone, date
import time
import yaml
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Iterable, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

RAW_DIR = Path("data/raw/yahoo_finance")
RAW_DIR.mkdir(parents=True, exist_ok=True)

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

# -------------------- Config --------------------
def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG
    if not path.exists():
        logger.error(f"Config file not found at {CONFIG_PATH} nor {FALLBACK_CONFIG}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# -------------------- Hash helpers --------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
# -------------------- DB helpers --------------------
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
    """), {"sid": source_id, "key": ticker, "name": name}).scalar_one()


def last_release_hash(conn, source_id: int, scope: str | None = None) -> str | None:
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


def create_release(conn, source_id: int, downloaded_at: datetime, vintage_at: datetime,
                   description: str, raw_path: str | None, content_hash: str | None,
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
    raw = sa_conn.connection
    dbapi = getattr(raw, "driver_connection", None)
    if dbapi is None:
        dbapi = getattr(raw, "connection", None) or raw
    return dbapi


def copy_observations_via_staging(conn, rows: Iterable[Tuple[int, str, str, float, int]]):
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


def last_period_date(conn, series_id: int) -> date | None:
    return conn.execute(text("""
        SELECT max(period_date) FROM observations WHERE series_id=:sid
    """), {"sid": series_id}).scalar_one_or_none()


# -------------------- Fetch + Prepare --------------------
def fetch_yahoo_history(ticker: str, period: str, start: Optional[date]) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if start is not None:
        hist = t.history(start=str(start))
    else:
        hist = t.history(period=period)
    return hist


def prepare_rows_for_copy(series_id: int, df_close: pd.DataFrame, vintage_at: datetime, release_id: int):
    """
    df_close: index=DatetimeIndex, column Close
    darom be iterrows(), su numpy/pandas vektorizacija.
    """
    df_close = df_close.dropna(subset=["Close"]).copy()
    if df_close.empty:
        return []

    # period_date as ISO string (YYYY-MM-DD)
    pdates = df_close.index.date
    values = df_close["Close"].to_numpy(dtype="float64", copy=False)

    oat_iso = vintage_at.isoformat()
    rows = [(series_id, d.isoformat(), oat_iso, float(v), release_id) for d, v in zip(pdates, values)]
    return rows


# -------------------- One ticker ingestion --------------------
def ingest_one_ticker(ticker: str, details, mode: str, downloaded_at: datetime, vintage_at: datetime, stamp: str, tail_days_default: int) -> dict:
    # config parsing
    if isinstance(details, dict):
        name = details.get("name", ticker)
        period = details.get("period", "5y")
        tail_days = int(details.get("tail_days", tail_days_default))
    else:
        name = str(details)
        period = "5y"
        tail_days = tail_days_default

    with engine.begin() as conn:
        source_id = ensure_source(conn)
        series_id = ensure_series(conn, source_id, ticker, name)

        # decide start for update
        start = None
        fetch_mode_label = f"period={period}"
        if mode == "update":
            last_dt = last_period_date(conn, series_id)
            if last_dt:
                start = (pd.Timestamp(last_dt) - pd.Timedelta(days=tail_days)).date()
                fetch_mode_label = f"start={start.isoformat()} (tail_days={tail_days})"
            else:
                fetch_mode_label = f"fallback period={period}"

    logger.info(f"Yahoo Finance: {ticker} ({name}) [{mode}] {fetch_mode_label}")

    # fetch outside DB transaction
    try:
        with Timer(f"Yahoo fetch {ticker}"):
            hist = fetch_yahoo_history(ticker, period=period, start=start)
    except Exception as e:
        return {"ticker": ticker, "status": "fail", "error": f"fetch failed: {e}"}

    if hist is None or hist.empty:
        return {"ticker": ticker, "status": "empty", "inserted": 0, "skipped": 0}

    df = hist[["Close"]].copy()

    # save raw snapshot
    raw_path = RAW_DIR / f"{ticker}_{stamp}.csv"
    with Timer(f"Yahoo save raw {ticker}"):
        df.to_csv(raw_path, index=True)

    # hash
    with Timer(f"Yahoo hash {ticker}"):
        content_hash = sha256_file(raw_path)

    scope = f"ticker:{ticker}|{fetch_mode_label}"

    with engine.begin() as conn:
        source_id = ensure_source(conn)
        series_id = ensure_series(conn, source_id, ticker, name)

        # update skip by hash
        if mode == "update":
            prev_hash = last_release_hash(conn, source_id, scope=scope)
            if prev_hash == content_hash:
                logger.info(f"{ticker}: no changes (hash same) -> skip")
                return {"ticker": ticker, "status": "skipped", "inserted": 0, "skipped": 1}

        # create release
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

        # prepare + COPY
        with Timer(f"Yahoo prepare rows {ticker}"):
            rows = prepare_rows_for_copy(series_id, df, vintage_at, release_id)

        if not rows:
            return {"ticker": ticker, "status": "empty_after_prepare", "inserted": 0, "skipped": 0}

        COPY_BATCH = 200_000
        inserted_attempted = 0

        with Timer(f"Yahoo COPY {ticker}"):
            for i in range(0, len(rows), COPY_BATCH):
                chunk = rows[i:i + COPY_BATCH]
                copy_observations_via_staging(conn, chunk)
                inserted_attempted += len(chunk)

        return {"ticker": ticker, "status": "ok", "inserted": inserted_attempted, "skipped": 0}


# -------------------- Main --------------------
def main(mode: str = "initial", max_workers: int = 1):
    """
    mode:
      - "initial": full initial load (uses configured 'period' per ticker, e.g. 5y)
      - "update": incremental load (fetches only tail from last known date - tail_days)
    max_workers:
      - 1 = sequential
      - 2..6 gerai Yahoo
    """
    config = load_config()
    yf_config = config.get("yahoo_finance", {}) if isinstance(config, dict) else {}
    if not yf_config:
        logger.warning("No 'yahoo_finance' section in config.")
        return

    downloaded_at = datetime.now(timezone.utc)
    vintage_at = downloaded_at
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")

    tail_days_default = 7

    inserted_total = 0
    failed_total = 0
    skipped_total = 0

    items = list(yf_config.items())
    logger.info(f"Yahoo main: mode={mode}, tickers={len(items)}, max_workers={max_workers}")

    results = []

    if max_workers <= 1:
        for ticker, details in items:
            res = ingest_one_ticker(
                ticker=ticker,
                details=details,
                mode=mode,
                downloaded_at=downloaded_at,
                vintage_at=vintage_at,
                stamp=stamp,
                tail_days_default=tail_days_default
            )
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    ingest_one_ticker,
                    ticker, details, mode, downloaded_at, vintage_at, stamp, tail_days_default
                ): ticker
                for ticker, details in items
            }
            for fut in as_completed(futs):
                results.append(fut.result())

    for r in results:
        if r.get("status") in ("ok",):
            inserted_total += int(r.get("inserted", 0))
        elif r.get("status") in ("skipped",):
            skipped_total += int(r.get("skipped", 0))
        elif r.get("status") in ("empty", "empty_after_prepare"):
            pass
        else:
            failed_total += 1
            logger.error(f"Ticker failed: {r}")

    # log ingestion run (one record for whole run)
    with engine.begin() as conn:
        source_id = ensure_source(conn)
        details = {
            "mode": mode,
            "downloaded_at": downloaded_at.isoformat(),
            "vintage_at": vintage_at.isoformat(),
            "tickers": len(items),
            "inserted_total_attempted": inserted_total,
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

    logger.info(f"Yahoo Finance [{mode}] done: inserted_attempted={inserted_total}, skipped={skipped_total}, failed={failed_total}")


if __name__ == "__main__":
    main(mode="initial", max_workers=4)

#