import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from fredapi import Fred
from sqlalchemy import text

from scripts.db_helpers import (
    get_engine,
    Timer,
    load_config_first_existing,
    ensure_provider,
    ensure_dataset,
    ensure_series,
    create_release,
    vintage_exists,
    copy_observations_via_staging,
    log_ingestion,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("alfred")


engine = get_engine(pool_size=5, max_overflow=5)


API_KEY = os.environ.get("FRED_API_KEY")
if not API_KEY:
    logger.warning("FRED_API_KEY not found in environment. ALFRED loading will fail.")
fred = Fred(api_key=API_KEY) if API_KEY else None


CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


def _load_config() -> dict:
    return load_config_first_existing([CONFIG_PATH, FALLBACK_CONFIG])


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


def ingest_one_series(series_key: str, realtime: bool, max_vintages: int, mode: str) -> dict:
    if not fred:
        return {"series": series_key, "status": "fail", "error": "fred client not initialized"}

    with Timer(f"ALFRED series total {series_key}"):
        try:
            with Timer(f"ALFRED info {series_key}"):
                info = fetch_series_info(series_key)
        except Exception as e:
            return {"series": series_key, "status": "fail", "error": f"info failed: {e}"}

        frequency = (info.get("frequency_short", "M") or "M")
        units = info.get("units") or None
        title = info.get("title") or series_key

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
            provider_id = ensure_provider(conn, "alfred", base_url="https://alfred.stlouisfed.org/")
            dataset_id = ensure_dataset(conn, provider_id, key=series_key, title=title)

            db_series_id = ensure_series(
                conn=conn,
                dataset_id=dataset_id,
                key=series_key,
                country="US",
                frequency=frequency,
                transform="LEVEL",
                unit=units,
                name=title,
                meta={"fred_info": info.to_dict() if hasattr(info, "to_dict") else dict(info)},
            )

            COPY_BATCH = 200_000

            for v_date in target_vintages:
                vintage_at = _ensure_tz_utc(v_date)
                if mode == "update" and vintage_exists(conn, db_series_id, vintage_at):
                    skipped_vintages += 1
                    continue

                downloaded_at = datetime.now(timezone.utc)

                release_id = create_release(
                    conn=conn,
                    dataset_id=dataset_id,
                    downloaded_at=downloaded_at,
                    vintage_at=vintage_at,
                    description=f"ALFRED vintage for {series_key} ({vintage_at.date().isoformat()})",
                    raw_path=None,
                    content_hash=None,
                    scope=f"series:{series_key}|vintage:{vintage_at.date().isoformat()}",
                    meta={"series": series_key, "vintage": vintage_at.date().isoformat(), "mode": mode},
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

                rows: list[tuple[int, str, str, float, int]] = []
                for p_date, value in data.items():
                    if pd.isna(value):
                        continue
                    rows.append((db_series_id, p_date.date().isoformat(), vintage_at.isoformat(), float(value), release_id))
                    if len(rows) >= COPY_BATCH:
                        copy_observations_via_staging(conn, rows)
                        inserted_attempted += len(rows)
                        rows = []

                if rows:
                    copy_observations_via_staging(conn, rows)
                    inserted_attempted += len(rows)

            log_ingestion(
                conn,
                dataset_id=dataset_id,
                status="ok" if failed == 0 else "ok_with_errors",
                rows_inserted=inserted_attempted,
                rows_failed=failed,
                details={
                    "series": series_key,
                    "mode": mode,
                    "realtime": realtime,
                    "vintages_target": len(target_vintages),
                    "vintages_processed": processed_vintages,
                    "vintages_skipped": skipped_vintages,
                    "attempted_rows": inserted_attempted,
                    "frequency": frequency,
                    "unit": units,
                },
            )

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
    config = _load_config()
    alfred_cfg = config.get("alfred", {}) if isinstance(config, dict) else {}

    if not API_KEY:
        logger.error("No API key configured. Set FRED_API_KEY in .env")
        return

    if not alfred_cfg:
        logger.warning("No 'alfred' section in config.")
        return

    items = list(alfred_cfg.items())
    logger.info(
        f"ALFRED main: mode={mode}, series={len(items)}, max_workers={max_workers}, max_vintages={max_vintages}"
    )

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
