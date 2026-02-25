# load_eurostat.py
import logging
import re
import json
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import eurostat

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
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eurostat")


engine = get_engine(pool_size=5, max_overflow=5)


RAW_DIR = Path("data/raw/eurostat")
RAW_DIR.mkdir(parents=True, exist_ok=True)


CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
FALLBACK_CONFIG = Path(__file__).with_name("datasets.yaml")


def load_config() -> dict:
    return load_config_first_existing([CONFIG_PATH, FALLBACK_CONFIG])


# TIME parsing
_RE_YQ = re.compile(r"^(\d{4})[- ]?Q([1-4])$", re.I)
_RE_YM = re.compile(r"^(\d{4})[- ]?M(\d{1,2})$", re.I)
_RE_Y_MM = re.compile(r"^(\d{4})-(\d{2})$")
_RE_Y = re.compile(r"^(\d{4})$")
_RE_YMD = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def to_period_date(x: Any) -> date:
    """Convert Eurostat time labels into a date aligned to period start."""
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.date()

    s = str(x).strip()
    if not s or s.lower() == "nan":
        raise ValueError("Empty TIME_PERIOD")

    m = _RE_YMD.match(s)
    if m:
        y, mo, d = map(int, m.groups())
        return date(y, mo, d)

    m = _RE_YQ.match(s)
    if m:
        y = int(m.group(1))
        q = int(m.group(2))
        month = (q - 1) * 3 + 1
        return date(y, month, 1)

    m = _RE_YM.match(s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        return date(y, mo, 1)

    m = _RE_Y_MM.match(s)
    if m:
        y, mo = map(int, m.groups())
        return date(y, mo, 1)

    m = _RE_Y.match(s)
    if m:
        return date(int(m.group(1)), 1, 1)

    raise ValueError(f"Unsupported time label: {x!r}")


def detect_time_columns(df: pd.DataFrame) -> list[str]:
    time_cols = []
    for c in df.columns:
        cs = str(c)
        if len(cs) >= 4 and cs[:4].isdigit():
            time_cols.append(c)
    return time_cols


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if not filters:
        return df
    out = df
    for col, allowed in filters.items():
        if col in out.columns:
            out = out[out[col].isin(allowed)]
    return out


def normalize_dim_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return str(v)


def build_series_key(dataset_code: str, dims: Dict[str, Any], dim_cols: list[str]) -> str:
    parts = [f"{k}={normalize_dim_value(dims.get(k))}" for k in dim_cols]
    return f"{dataset_code}." + ".".join(parts)


def guess_country(dims: Dict[str, Any]) -> str:
    for k in ("geo", "GEO"):
        if k in dims and dims[k] is not None:
            return str(dims[k])
    return "EU"


def guess_frequency(dims: Dict[str, Any], time_label: Any) -> str:
    if "freq" in dims and dims["freq"] is not None:
        return str(dims["freq"])
    s = str(time_label)
    if "Q" in s:
        return "Q"
    if "M" in s or "-" in s:
        return "M"
    return "A"


def fetch_dataset(dataset_code: str) -> pd.DataFrame:
    return eurostat.get_data_df(dataset_code)


def ingest_dataset(dataset_code: str, dataset_cfg: dict, mode: str = "initial"):
    filters = dataset_cfg.get("filters", {}) if isinstance(dataset_cfg, dict) else {}
    dataset_name = dataset_cfg.get("name", dataset_code) if isinstance(dataset_cfg, dict) else dataset_code

    downloaded_at = datetime.now(timezone.utc)
    vintage_at = downloaded_at
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")

    logger.info(f"[{mode}] Eurostat dataset={dataset_code} filters={filters}")

    with Timer(f"Eurostat fetch {dataset_code}"):
        try:
            df = fetch_dataset(dataset_code)
        except Exception as e:
            logger.error(f"Fetch failed for {dataset_code}: {e}")
            return

    if df is None or df.empty:
        logger.warning(f"{dataset_code}: empty dataset")
        return

    with Timer(f"Eurostat filter+save {dataset_code}"):
        df = apply_filters(df, filters)
        if df.empty:
            logger.warning(f"{dataset_code}: empty after filtering")
            return

        raw_path = RAW_DIR / f"{dataset_code}_{stamp}.csv"
        df.to_csv(raw_path, index=False)
        content_hash = sha256_file(raw_path)

    scope = f"dataset:{dataset_code}|filters:{json.dumps(filters, sort_keys=True)}"

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "eurostat", base_url="https://ec.europa.eu/eurostat")
        dataset_id = ensure_dataset(conn, provider_id, key=dataset_code, title=dataset_name)
        if mode == "update":
            prev_hash = last_release_hash(conn, dataset_id, scope=scope)
            if prev_hash == content_hash:
                logger.info(f"{dataset_code}: no changes (hash same) -> skip")
                return

    time_cols = detect_time_columns(df)
    dim_cols = [c for c in df.columns if c not in time_cols]
    if not time_cols:
        logger.warning(f"{dataset_code}: no time columns detected")
        return

    with Timer(f"Eurostat melt {dataset_code}"):
        df_long = df.melt(id_vars=dim_cols, value_vars=time_cols, var_name="time_label", value_name="value")
        df_long = df_long.dropna(subset=["value"]).copy()
        if df_long.empty:
            logger.warning(f"{dataset_code}: empty after melt")
            return

    inserted_attempted = 0
    failed = 0
    rows_prepared = 0
    series_created = 0
    series_cache: dict[tuple[str, str, str, str], int] = {}

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, "eurostat", base_url="https://ec.europa.eu/eurostat")
        dataset_id = ensure_dataset(conn, provider_id, key=dataset_code, title=dataset_name)

        release_id = create_release(
            conn=conn,
            dataset_id=dataset_id,
            downloaded_at=downloaded_at,
            vintage_at=vintage_at,
            description=f"Eurostat snapshot dataset={dataset_code}",
            raw_path=str(raw_path),
            content_hash=content_hash,
            scope=scope,
            meta={"dataset_code": dataset_code, "dataset_name": dataset_name, "filters": filters, "mode": mode},
        )

        COPY_BATCH = 200_000
        copy_rows: list[tuple[int, str, str, float, int]] = []
        oat_iso = vintage_at.isoformat()

        def flush_copy_rows():
            nonlocal copy_rows, inserted_attempted
            if not copy_rows:
                return
            copy_observations_via_staging(conn, copy_rows)
            inserted_attempted += len(copy_rows)
            copy_rows = []

        for _, rec in df_long.iterrows():
            dims = {c: rec.get(c) for c in dim_cols}
            time_label = rec.get("time_label")
            value = rec.get("value")

            try:
                pdate = to_period_date(time_label)
                val = float(value)
            except Exception:
                failed += 1
                continue

            country = guess_country(dims)
            freq = guess_frequency(dims, time_label)
            series_key = build_series_key(dataset_code, dims, dim_cols)
            cache_key = (series_key, country, freq, "LEVEL")

            if cache_key not in series_cache:
                pretty_dims = ", ".join([f"{c}:{dims.get(c)}" for c in dim_cols])
                series_name = f"{dataset_name} ({dataset_code}) | {pretty_dims}"
                meta = {
                    "dataset_code": dataset_code,
                    "dataset_name": dataset_name,
                    "dimensions": dims,
                    "filters": filters,
                }
                try:
                    sid = ensure_series(
                        conn=conn,
                        dataset_id=dataset_id,
                        key=series_key,
                        country=country,
                        frequency=freq,
                        transform="LEVEL",
                        unit=str(dims.get("unit")) if dims.get("unit") is not None else None,
                        name=series_name,
                        meta=meta,
                    )
                    series_cache[cache_key] = sid
                    series_created += 1
                except Exception:
                    failed += 1
                    continue

            series_id = series_cache[cache_key]
            copy_rows.append((series_id, pdate.isoformat(), oat_iso, val, release_id))
            rows_prepared += 1
            if len(copy_rows) >= COPY_BATCH:
                flush_copy_rows()

        flush_copy_rows()

        log_ingestion(
            conn,
            dataset_id=dataset_id,
            status="ok" if failed == 0 else "ok_with_errors",
            rows_inserted=int(inserted_attempted),
            rows_failed=int(failed),
            details={
                "dataset_code": dataset_code,
                "filters": filters,
                "mode": mode,
                "downloaded_at": downloaded_at.isoformat(),
                "vintage_at": vintage_at.isoformat(),
                "raw_path": str(raw_path),
                "hash": content_hash,
                "dim_cols": dim_cols,
                "time_cols_count": len(time_cols),
                "series_created": int(series_created),
                "series_cache_size": int(len(series_cache)),
                "rows_long": int(len(df_long)),
                "rows_prepared": int(rows_prepared),
                "insert_attempted": int(inserted_attempted),
            },
        )

    logger.info(
        f"{dataset_code}: attempted={inserted_attempted}, failed={failed}, series={len(series_cache)} created={series_created}"
    )


def main(mode: str = "initial", max_workers: int = 4):
    config = load_config()
    euro_cfg = config.get("eurostat", {}) if isinstance(config, dict) else {}

    if not euro_cfg:
        logger.warning("No 'eurostat' section in config.")
        return

    items = list(euro_cfg.items())
    logger.info(f"Eurostat main: mode={mode}, datasets={len(items)}, max_workers={max_workers}")

    if max_workers <= 1:
        for dataset_code, dataset_cfg in items:
            ingest_dataset(dataset_code, dataset_cfg or {}, mode=mode)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(ingest_dataset, dataset_code, (dataset_cfg or {}), mode): dataset_code
            for dataset_code, dataset_cfg in items
        }
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Dataset {code} failed: {e}")


if __name__ == "__main__":
    main(mode="initial", max_workers=4)
