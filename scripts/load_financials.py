import logging
from pathlib import Path
from datetime import datetime, timezone, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from scripts.db_helpers import (
    get_engine,
    Timer,
    load_config_first_existing,
    sha256_file,
    ensure_provider,
    ensure_dataset,
    ensure_series,
    last_release_hash,
    create_release,
    copy_observations_via_staging,
    last_period_date,
    log_ingestion,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("yahoo_finance")


# DB
engine = get_engine(pool_size=5, max_overflow=5)


# Paths
RAW_DIR = Path("data/raw/yahoo_finance")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


def _load_config() -> dict:
    return load_config_first_existing([CONFIG_PATH, FALLBACK_CONFIG])


# -------------------- Fetch + Prepare --------------------


def fetch_yahoo_history(ticker: str, period: str, start: Optional[date]) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if start is not None:
        return t.history(start=str(start))
    return t.history(period=period)


def prepare_rows_for_copy(series_id: int, df_close: pd.DataFrame, vintage_at: datetime, release_id: int) -> list:
    df_close = df_close.dropna(subset=["Close"]).copy()
    if df_close.empty:
        return []

    pdates = df_close.index.date
    values = df_close["Close"].to_numpy(dtype="float64", copy=False)
    oat_iso = vintage_at.isoformat()
    return [(series_id, d.isoformat(), oat_iso, float(v), release_id) for d, v in zip(pdates, values)]


# -------------------- One ticker ingestion --------------------


def ingest_one_ticker(
    ticker: str,
    details,
    mode: str,
    downloaded_at: datetime,
    vintage_at: datetime,
    stamp: str,
    tail_days_default: int,
) -> dict:
    # config parsing
    if isinstance(details, dict):
        name = details.get("name", ticker)
        period = details.get("period", "5y")
        tail_days = int(details.get("tail_days", tail_days_default))
    else:
        name = str(details)
        period = "5y"
        tail_days = tail_days_default

    # decide incremental window
    start: Optional[date] = None
    fetch_mode_label = f"period={period}"
    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "yahoo_finance", base_url="https://finance.yahoo.com")
        dataset_id = ensure_dataset(conn, provider_id, key=ticker, title=name)
        series_id = ensure_series(
            conn,
            dataset_id=dataset_id,
            key=ticker,
            country="Global",
            frequency="D",
            transform="LEVEL",
            unit="PRICE",
            name=name,
            meta={},
        )

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

    raw_path = RAW_DIR / f"{ticker}_{stamp}.csv"
    with Timer(f"Yahoo save raw {ticker}"):
        df.to_csv(raw_path, index=True)

    with Timer(f"Yahoo hash {ticker}"):
        content_hash = sha256_file(raw_path)

    scope = f"ticker:{ticker}|{fetch_mode_label}"

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "yahoo_finance", base_url="https://finance.yahoo.com")
        dataset_id = ensure_dataset(conn, provider_id, key=ticker, title=name)
        series_id = ensure_series(
            conn,
            dataset_id=dataset_id,
            key=ticker,
            country="Global",
            frequency="D",
            transform="LEVEL",
            unit="PRICE",
            name=name,
            meta={},
        )

        if mode == "update":
            prev_hash = last_release_hash(conn, dataset_id, scope=scope)
            if prev_hash == content_hash:
                logger.info(f"{ticker}: no changes (hash same) -> skip")
                return {"ticker": ticker, "status": "skipped", "inserted": 0, "skipped": 1}

        release_id = create_release(
            conn,
            dataset_id,
            downloaded_at=downloaded_at,
            vintage_at=vintage_at,
            description=f"Yahoo Finance snapshot {ticker} [{fetch_mode_label}]",
            raw_path=str(raw_path),
            content_hash=content_hash,
            scope=scope,
            meta={"ticker": ticker, "name": name, "mode": mode, "fetch": fetch_mode_label},
        )

        with Timer(f"Yahoo prepare rows {ticker}"):
            rows = prepare_rows_for_copy(series_id, df, vintage_at, release_id)

        if not rows:
            return {"ticker": ticker, "status": "empty_after_prepare", "inserted": 0, "skipped": 0}

        COPY_BATCH = 200_000
        inserted_attempted = 0
        with Timer(f"Yahoo COPY {ticker}"):
            for i in range(0, len(rows), COPY_BATCH):
                chunk = rows[i : i + COPY_BATCH]
                copy_observations_via_staging(conn, chunk)
                inserted_attempted += len(chunk)

        return {"ticker": ticker, "status": "ok", "inserted": inserted_attempted, "skipped": 0}


# -------------------- Main --------------------


def main(mode: str = "initial", max_workers: int = 4):
    """
    mode:
      - "initial": full initial load (uses configured 'period' per ticker, e.g. 5y)
      - "update": incremental load (fetches only tail from last known date - tail_days)
    """
    config = _load_config()
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
            results.append(
                ingest_one_ticker(
                    ticker=ticker,
                    details=details,
                    mode=mode,
                    downloaded_at=downloaded_at,
                    vintage_at=vintage_at,
                    stamp=stamp,
                    tail_days_default=tail_days_default,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    ingest_one_ticker,
                    ticker,
                    details,
                    mode,
                    downloaded_at,
                    vintage_at,
                    stamp,
                    tail_days_default,
                ): ticker
                for ticker, details in items
            }
            for fut in as_completed(futs):
                results.append(fut.result())

    for r in results:
        if r.get("status") == "ok":
            inserted_total += int(r.get("inserted", 0))
        elif r.get("status") == "skipped":
            skipped_total += int(r.get("skipped", 0))
        elif r.get("status") in ("empty", "empty_after_prepare"):
            pass
        else:
            failed_total += 1
            logger.error(f"Ticker failed: {r}")

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "yahoo_finance", base_url="https://finance.yahoo.com")
        dataset_id = ensure_dataset(conn, provider_id, key="__run_summary__", title="Yahoo Finance Run Summary")
        log_ingestion(
            conn,
            dataset_id=dataset_id,
            status="ok" if failed_total == 0 else "ok_with_errors",
            rows_inserted=inserted_total,
            rows_failed=failed_total,
            details={
                "mode": mode,
                "downloaded_at": downloaded_at.isoformat(),
                "vintage_at": vintage_at.isoformat(),
                "tickers": len(items),
                "inserted_total_attempted": inserted_total,
                "skipped_total": skipped_total,
                "failed_total": failed_total,
            },
        )

    logger.info(
        f"Yahoo Finance [{mode}] done: inserted_attempted={inserted_total}, skipped={skipped_total}, failed={failed_total}"
    )


if __name__ == "__main__":
    main(mode="initial", max_workers=4)
