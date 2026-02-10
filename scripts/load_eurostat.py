import json
import logging
from datetime import datetime, timezone, date
from pathlib import Path
import pandas as pd
import eurostat
from sqlalchemy import create_engine, text
import re

_RE_YQ = re.compile(r"^(\d{4})[- ]?Q([1-4])$")       #2025Q3 arba 2025-Q3 arba 2025 Q3
_RE_YM = re.compile(r"^(\d{4})[- ]?M(\d{1,2})$")     #2025M01 arba 2025-M01 arba 2025 M1
_RE_Y_MM = re.compile(r"^(\d{4})-(\d{2})$")          #2025-01
_RE_Y = re.compile(r"^(\d{4})$")                     #2025
_RE_YMD = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")   #2025-01-01

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# DB
# ----------------------------
DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

# ----------------------------
# Config: datasets + filters
# ----------------------------
DATASETS = {
    "namq_10_gdp": {  # Quarterly
        "unit": ["CLV10_MNAC", "CP_MNAC"],
        "s_adj": ["SCA"],
        "na_item": ["B1GQ", "P3", "P6", "P7"],
        "geo": ["LT"],
        # можно добавить "freq": ["Q"], но в некоторых наборах freq уже зашит
    },
    "prc_hicp_midx": {  # Monthly
        "unit": ["I15"],
        "coicop": ["CP00"],
        "geo": ["LT"],
    },
    "une_rt_m": {  # Monthly
        "unit": ["PC_ACT"],
        "s_adj": ["SA"],
        "age": ["TOTAL"],
        "sex": ["T"],
        "geo": ["LT"],
    },
    "sts_inpr_m": {  # Monthly
        "unit": ["I15"],
        "s_adj": ["SCA"],
        "nace_r2": ["B-D"],
        "geo": ["LT"],
    },
}

# ----------------------------
# Helpers: DB upserts
# ----------------------------
def ensure_source(conn, name: str) -> int:
    conn.execute(
        text("""
            insert into sources (name) values (:name)
            on conflict (name) do nothing
        """),
        {"name": name},
    )
    return conn.execute(text("select id from sources where name=:name"), {"name": name}).scalar_one()


def ensure_series(
    conn,
    source_id: int,
    key: str,
    country: str,
    frequency: str,
    transform: str,
    unit: str | None,
    name: str,
    meta: dict,
) -> int:
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(
        text("""
            insert into series (source_id, key, country, frequency, transform, unit, name, meta)
            values (:sid, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
            on conflict (source_id, key, country, frequency, transform)
            do update set
                unit = excluded.unit,
                name = excluded.name,
                meta = excluded.meta
            returning id
        """),
        {
            "sid": source_id,
            "key": key,
            "country": country,
            "freq": frequency,
            "transform": transform,
            "unit": unit,
            "name": name,
            "meta": meta_json,
        },
    ).scalar_one()


# ----------------------------
# Helpers: parsing dates
# ----------------------------
def to_period_date(x) -> date:
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.date()

    s = str(x).strip()
    if not s or s.lower() == "nan":
        raise ValueError("Empty TIME_PERIOD")

    # YYYY-MM-DD
    m = _RE_YMD.match(s)
    if m:
        y, mo, d = map(int, m.groups())
        return date(y, mo, d)

    # YYYY-Qn / YYYYQn / YYYY Qn
    m = _RE_YQ.match(s)
    if m:
        y = int(m.group(1))
        q = int(m.group(2))
        month = (q - 1) * 3 + 1
        return date(y, month, 1)

    # YYYY-Mmm / YYYYMmm / YYYY Mmm
    m = _RE_YM.match(s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        return date(y, mo, 1)

    # YYYY-MM
    m = _RE_Y_MM.match(s)
    if m:
        y, mo = map(int, m.groups())
        return date(y, mo, 1)

    # YYYY
    m = _RE_Y.match(s)
    if m:
        return date(int(m.group(1)), 1, 1)

    raise ValueError(f"Unsupported TIME_PERIOD: {x!r}")


def is_time_column(col_name: str) -> bool:
    """
    Eurostat bulk df обычно имеет time-колонки:
      '2023', '2023Q1', '2023M01', '2023-01'
    """
    s = str(col_name).strip()
    if len(s) >= 4 and s[:4].isdigit():
        # очень грубая эвристика, но работает для eurostat таблиц
        return True
    return False


def detect_geo(dims: dict) -> str:
    # разные варианты названий
    for k in ("geo", "geo\\time", "geo\\TIME_PERIOD", "GEO"):
        if k in dims:
            v = dims[k]
            if pd.notna(v):
                return str(v)
    return "UNKNOWN"


def detect_freq(dims: dict, time_period_value) -> str:
    # если freq есть в dims — используем
    if "freq" in dims and pd.notna(dims["freq"]):
        return str(dims["freq"])

    # иначе пытаемся определить по времени
    tp = str(time_period_value)
    if "Q" in tp:
        return "Q"
    if "M" in tp or "-" in tp:
        return "M"
    return "A"


# ----------------------------
# Fetch + ingest
# ----------------------------
def fetch_dataset_bulk(dataset_code: str) -> pd.DataFrame:
    """
    Без deprecated SDMX. Скачиваем bulk-таблицу.
    Для LT + фильтры по нескольким переменным это обычно норм.
    """
    df = eurostat.get_data_df(dataset_code)
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for col, allowed in filters.items():
        if col in df.columns:
            df = df[df[col].isin(allowed)]
    return df


def fetch_and_ingest(dataset_code: str, filters: dict):
    logger.info(f"Fetching {dataset_code} (bulk) with filters {filters}...")

    try:
        df = fetch_dataset_bulk(dataset_code)
    except Exception as e:
        logger.exception(f"Failed to fetch bulk dataset {dataset_code}: {e}")
        return

    if df is None or df.empty:
        logger.warning(f"No data (empty) for {dataset_code}")
        return

    df = apply_filters(df, filters)

    if df.empty:
        logger.warning(f"No rows after filtering for {dataset_code}")
        return

    # Определяем колонки времени
    time_cols = [c for c in df.columns if is_time_column(c)]
    if not time_cols:
        logger.error(f"Could not detect time columns for {dataset_code}. Columns: {list(df.columns)}")
        return

    id_vars = [c for c in df.columns if c not in time_cols]

    # Приводим к long
    df_melted = df.melt(id_vars=id_vars, value_vars=time_cols, var_name="time_period", value_name="value")
    df_melted = df_melted.dropna(subset=["value"]).copy()

    logger.info(f"Filtered rows (id rows): {len(df)}; observations (after melt, non-null): {len(df_melted)}")

    observed_at = datetime.now(timezone.utc).isoformat()

    inserted = 0
    failed = 0

    with engine.begin() as conn:
        source_id = ensure_source(conn, "eurostat")

        # Чтобы не апсертить series на каждой строке — кешируем series_id по dims
        series_cache: dict[tuple, int] = {}

        for _, row in df_melted.iterrows():
            dims = {k: row[k] for k in id_vars}

            geo = detect_geo(dims)
            freq = detect_freq(dims, row["time_period"])
            unit = str(dims.get("unit")) if "unit" in dims and pd.notna(dims.get("unit")) else None

            # Серийный ключ: dataset + значения измерений (кроме времени)
            # Важно: используем фиксированный порядок ключей, чтобы ключ был стабильным
            dim_keys = sorted(id_vars)
            dim_part = ".".join([f"{k}={dims.get(k)}" for k in dim_keys])
            series_key = f"{dataset_code}.{dim_part}"

            cache_key = (series_key, geo, freq, "LEVEL")  # transform пока LEVEL

            if cache_key not in series_cache:
                series_name = f"{dataset_code} | " + ", ".join([f"{k}:{dims.get(k)}" for k in dim_keys])

                series_meta = {
                    "dataset_code": dataset_code,
                    "dimensions": {k: (None if pd.isna(dims.get(k)) else str(dims.get(k))) for k in dim_keys},
                    "filters": filters,
                }

                try:
                    sid = ensure_series(
                        conn=conn,
                        source_id=source_id,
                        key=series_key,
                        country=geo,
                        frequency=freq,
                        transform="LEVEL",
                        unit=unit,
                        name=series_name,
                        meta=series_meta,
                    )
                    series_cache[cache_key] = sid
                except Exception as e:
                    # если апсерт серии упал — логируем и пропускаем все obs этой серии
                    failed += 1
                    logger.warning(f"Failed ensure_series for {series_key}: {e}")
                    continue

            series_id = series_cache[cache_key]

            # period_date
            try:
                pdate = to_period_date(row["time_period"])
                val = float(row["value"])
            except Exception as e:
                failed += 1
                logger.warning(f"Bad row (date/value) dataset={dataset_code} time={row['time_period']}: {e}")
                continue

            # SAVEPOINT: одна плохая вставка не ломает транзакцию
            try:
                with conn.begin_nested():
                    conn.execute(
                        text("""
                            insert into observations (series_id, period_date, observed_at, value, status, meta)
                            values (:sid, :pdate, :oat, :val, null, '{}'::jsonb)
                            on conflict (series_id, period_date, observed_at) do nothing
                        """),
                        {"sid": series_id, "pdate": pdate, "oat": observed_at, "val": val},
                    )
                    inserted += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Insert observation failed: {e}")

        # ingestion_log
        details = json.dumps(
            {
                "dataset": dataset_code,
                "observed_at": observed_at,
                "rows_id": int(len(df)),
                "rows_obs": int(len(df_melted)),
                "series_created_or_seen": len(series_cache),
            },
            ensure_ascii=False,
        )

        conn.execute(
            text("""
                insert into ingestion_log (source_id, status, rows_inserted, rows_failed, details)
                values (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
            """),
            {
                "source_id": source_id,
                "status": "ok" if failed == 0 else "ok_with_errors",
                "ins": inserted,
                "fail": failed,
                "details": details,
            },
        )

    logger.info(f"Done {dataset_code}: inserted={inserted}, failed={failed}")


def main():
    for code, filters in DATASETS.items():
        fetch_and_ingest(code, filters)


if __name__ == "__main__":
    main()