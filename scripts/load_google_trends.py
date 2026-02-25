import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
from pytrends.request import TrendReq

from scripts.db_helpers import (
    get_engine,
    Timer,
    load_config_first_existing,
    sha256_bytes,
    ensure_provider,
    ensure_dataset,
    ensure_series,
    create_release,
    last_release_hash,
    copy_observations_via_staging,
    log_ingestion,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("google_trends")


engine = get_engine(pool_size=3, max_overflow=3)


RAW_DIR = Path("data/raw/google_trends")
RAW_DIR.mkdir(parents=True, exist_ok=True)


CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


def _load_config() -> dict:
    return load_config_first_existing([CONFIG_PATH, FALLBACK_CONFIG])


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


def monthly_from_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    monthly = df.resample("MS").mean()
    if not monthly.empty:
        monthly.columns = ["value"]
    return monthly


def chunk_keywords(lst, size=5):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def prepare_rows_for_copy(series_id: int, df_monthly: pd.DataFrame, vintage_at: datetime, release_id: int) -> list:
    oat_iso = vintage_at.isoformat()
    rows: list[tuple[int, str, str, float, int]] = []
    for dt, row in df_monthly.iterrows():
        value = row["value"]
        if pd.isna(value):
            continue
        rows.append((series_id, dt.date().isoformat(), oat_iso, float(value), release_id))
    return rows


def ingest_batch(batch: list[str], geo: str, geo_name: str, timeframe: str, mode: str,
                 downloaded_at: datetime, vintage_at: datetime, stamp: str, dataset_id: int):
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
                conn,
                dataset_id,
                downloaded_at=downloaded_at,
                vintage_at=vintage_at,
                description=f"Google Trends snapshot geo={geo} batch={batch}",
                raw_path=str(raw_path),
                content_hash=content_hash,
                scope=scope,
                meta={"geo": geo, "geo_name": geo_name, "timeframe": timeframe, "batch": batch},
            )

            batch_inserted = 0
            for kw in batch:
                if kw not in data.columns:
                    continue
                df_kw = data[[kw]].copy()
                df_monthly = monthly_from_weekly(df_kw)
                if df_monthly.empty:
                    continue

                series_id = ensure_series(
                    conn,
                    dataset_id=dataset_id,
                    key=f"google_trends.{kw}",
                    country=geo,
                    frequency="M",
                    transform="LEVEL",
                    unit="INDEX_0_100",
                    name=f"Google Trends: {kw}",
                    meta={},
                )

                rows = prepare_rows_for_copy(series_id, df_monthly, vintage_at, release_id)
                if rows:
                    copy_observations_via_staging(conn, rows)
                    batch_inserted += len(rows)

            return {"status": "ok", "inserted": batch_inserted, "failed": 0}

    except Exception as e:
        logger.error(f"Failed batch {batch}: {e}")
        return {"status": "error", "inserted": 0, "failed": 1}


def main(mode: str = "initial", max_workers: int = 2):
    config = _load_config()
    gt_config = config.get("google_trends", {}) if isinstance(config, dict) else {}
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

    for geo in geos:
        geo_name = GEO_NAMES.get(geo, geo)
        logger.info(f"Processing geo: {geo} ({geo_name})")

        with engine.begin() as conn:
            provider_id = ensure_provider(conn, "google_trends", base_url="https://trends.google.com")
            dataset_id = ensure_dataset(conn, provider_id, key=geo_name, title=f"Google Trends — {geo_name}")

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
                    time.sleep(1)

    for r in results:
        inserted_total += int(r.get("inserted", 0))
        failed_batches += int(r.get("failed", 0))

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "google_trends", base_url="https://trends.google.com")
        dataset_id = ensure_dataset(conn, provider_id, key="__run_summary__", title="Google Trends Run Summary")
        log_ingestion(
            conn,
            dataset_id=dataset_id,
            status="ok" if failed_batches == 0 else "ok_with_errors",
            rows_inserted=inserted_total,
            rows_failed=failed_batches,
            details={
                "geos": list(geos),
                "timeframe": timeframe,
                "keywords": len(keywords),
                "downloaded_at": downloaded_at.isoformat(),
                "vintage_at": vintage_at.isoformat(),
                "inserted_total": inserted_total,
                "failed_batches": failed_batches,
            },
        )

    logger.info(f"Google Trends done: inserted={inserted_total}, failed_batches={failed_batches}")


if __name__ == "__main__":
    main(mode="initial", max_workers=2)
