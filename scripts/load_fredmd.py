import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict

import pandas as pd
import requests
from sqlalchemy import text

from scripts.db_helpers import (
    get_engine,
    Timer,
    load_config_first_existing,
    sha256_file,
    ensure_provider,
    ensure_dataset,
    ensure_series,
    create_release,
    last_release_hash,
    copy_observations_via_staging,
    log_ingestion,
    analyze_table,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fredmd")


engine = get_engine(pool_size=3, max_overflow=3)


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


_DEFAULT_FREDMD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"


def _load_config() -> dict:
    return load_config_first_existing([CONFIG_PATH, FALLBACK_CONFIG])


def get_fredmd_url() -> str:
    cfg = _load_config()
    return cfg.get("fredmd", {}).get("url", _DEFAULT_FREDMD_URL)


def download_current_fredmd_csv() -> Tuple[Path, datetime]:
    url = get_fredmd_url()
    downloaded_at = datetime.now(timezone.utc)
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")
    out_path = RAW_DIR / f"fred_md_{stamp}.csv"

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

    logger.info(f"Downloaded: {url}")
    logger.info(f"Saved to:   {out_path}")
    return out_path, downloaded_at


def parse_fredmd(csv_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """Returns df_wide (index=dates, columns=variables) and meta (e.g. tcodes)."""
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    meta: Dict = {}

    first_cell = str(df.iloc[0, 0]).strip().lower()
    if first_cell in {"transform", "tcode", "tcodes"}:
        tcodes = df.iloc[0].to_dict()
        tcodes.pop(first_col, None)
        meta["tcodes"] = tcodes
        df = df.iloc[1:].copy()

    df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
    df = df.dropna(subset=[first_col]).set_index(first_col).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, meta


def ingest(
    df_wide: pd.DataFrame,
    dataset_meta: dict,
    vintage_at: datetime,
    downloaded_at: datetime,
    raw_path: str | None,
    content_hash: str | None,
):
    inserted = 0
    failed = 0

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "fredmd", base_url="https://www.stlouisfed.org/")
        dataset_id = ensure_dataset(conn, provider_id, key="fredmd_monthly_current", title="FRED-MD Monthly Current")

        release_id = create_release(
            conn=conn,
            dataset_id=dataset_id,
            downloaded_at=downloaded_at,
            vintage_at=vintage_at,
            description="FRED-MD current.csv snapshot (snapshot-vintage)",
            raw_path=raw_path,
            content_hash=content_hash,
            scope="FRED-MD",
            meta={"dataset": "FRED-MD"},
        )

        country = "US"
        freq = "M"
        transform = "LEVEL"
        unit = "INDEX"

        COPY_BATCH = 100_000
        oat_iso = vintage_at.isoformat()
        copy_rows: list[tuple[int, str, str, float, int]] = []

        with Timer("FRED-MD prepare + COPY"):
            for var in df_wide.columns:
                series_meta = {
                    "dataset": "FRED-MD",
                    "variable": var,
                    "tcode": dataset_meta.get("tcodes", {}).get(var),
                }

                series_id = ensure_series(
                    conn=conn,
                    dataset_id=dataset_id,
                    key=var,
                    country=country,
                    frequency=freq,
                    transform=transform,
                    unit=unit,
                    name=f"FRED-MD: {var}",
                    meta=series_meta,
                )

                for dt, val in df_wide[var].items():
                    if pd.isna(val):
                        continue
                    copy_rows.append((series_id, dt.date().isoformat(), oat_iso, float(val), release_id))
                    if len(copy_rows) >= COPY_BATCH:
                        copy_observations_via_staging(conn, copy_rows)
                        inserted += len(copy_rows)
                        copy_rows = []

            if copy_rows:
                copy_observations_via_staging(conn, copy_rows)
                inserted += len(copy_rows)

        log_ingestion(
            conn,
            dataset_id=dataset_id,
            status="ok" if failed == 0 else "ok_with_errors",
            rows_inserted=inserted,
            rows_failed=failed,
            details={
                "downloaded_at": downloaded_at.isoformat(),
                "vintage_at": vintage_at.isoformat(),
                "raw_path": raw_path,
                "hash": content_hash,
                "columns": int(len(df_wide.columns)),
                "rows": int(len(df_wide)),
            },
        )

    logger.info(f"FRED-MD inserted: {inserted}, failed: {failed}")
    analyze_table(engine, "observations")


def main(mode: str = "initial"):
    csv_path, downloaded_at = download_current_fredmd_csv()
    df_wide, dataset_meta = parse_fredmd(csv_path)

    vintage_at = downloaded_at
    content_hash = sha256_file(csv_path)

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "fredmd", base_url="https://www.stlouisfed.org/")
        dataset_id = ensure_dataset(conn, provider_id, key="fredmd_monthly_current", title="FRED-MD Monthly Current")
        if mode == "update":
            prev = last_release_hash(conn, dataset_id, scope="FRED-MD")
            if prev == content_hash:
                logger.info("FRED-MD: no changes (hash same) -> skip")
                return

    ingest(
        df_wide=df_wide,
        dataset_meta=dataset_meta,
        vintage_at=vintage_at,
        downloaded_at=downloaded_at,
        raw_path=str(csv_path),
        content_hash=content_hash,
    )


if __name__ == "__main__":
    main(mode="initial")

