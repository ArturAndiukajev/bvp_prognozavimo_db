import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"

FRED_MD_PAGE = "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(DB_URL, future=True)


def ensure_source(conn, name: str) -> int:
    conn.execute(text("""
        insert into sources (name) values (:name)
        on conflict (name) do nothing
    """), {"name": name})
    return conn.execute(text("select id from sources where name=:name"), {"name": name}).scalar_one()


def ensure_series(conn, source_id: int, key: str, country: str, frequency: str,
                  transform: str, unit: str, name: str, meta: dict) -> int:
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        insert into series (source_id, key, country, frequency, transform, unit, name, meta)
        values (:sid, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
        on conflict (source_id, key, country, frequency, transform)
        do update set
            unit = excluded.unit,
            name = excluded.name,
            meta = excluded.meta
        returning id
    """), {
        "sid": source_id,
        "key": key,
        "country": country,
        "freq": frequency,
        "transform": transform,
        "unit": unit,
        "name": name,
        "meta": meta_json,
    }).scalar_one()


def download_current_fredmd_csv() -> Path:
    url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RAW_DIR / f"fredmd_current_{ts}.csv"

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"Downloaded: {url}")
    print(f"Saved to:   {out_path}")
    return out_path



def parse_fredmd(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Возвращает:
    - df_wide: индекс = даты, колонки = переменные
    - meta: общая метаинформация (например, tcodes если они есть)
    """
    df = pd.read_csv(csv_path)

    # Обычно первая колонка sasdate
    # Иногда в FRED-MD есть строка с tcodes (трансформационные коды).
    # Попробуем детектировать: если в первой колонке лежит 'transform'/'tcode' — это строка кодов.
    first_col = df.columns[0]
    meta = {}

    if str(df.iloc[0, 0]).strip().lower() in {"transform", "tcode", "tcodes"}:
        # первая строка содержит коды трансформаций для каждой переменной
        tcodes = df.iloc[0].to_dict()
        tcodes.pop(first_col, None)
        meta["tcodes"] = tcodes
        df = df.iloc[1:].copy()

    # Парсим даты
    df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
    df = df.dropna(subset=[first_col])
    df = df.set_index(first_col).sort_index()

    # Все остальные колонки — числовые
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, meta


def ingest(df_wide: pd.DataFrame, dataset_meta: dict, observed_at: str):
    source_name = "fredmd"

    inserted = 0
    failed = 0

    with engine.begin() as conn:
        source_id = ensure_source(conn, source_name)

        # Если тебе нужны НЕ все переменные — сделай здесь фильтр:
        # keep = ["INDPRO", "UNRATE", "CPIAUCSL"]
        # df_wide = df_wide[keep]
        country = "US"
        freq = "M"
        transform = "LEVEL"
        unit = None  # в FRED-MD единицы не всегда в файле; можно оставить NULL

        for var in df_wide.columns:
            series_meta = {
                "dataset": "FRED-MD",
                "variable": var,
                "dataset_meta": dataset_meta,  # tcodes и т.п.
            }
            series_id = ensure_series(
                conn=conn,
                source_id=source_id,
                key=var,
                country=country,
                frequency=freq,
                transform=transform,
                unit=unit,
                name=f"FRED-MD {var}",
                meta=series_meta
            )

            # Вставляем наблюдения (помесячно)
            # period_date = первый день месяца (удобно и стабильно)
            for dt, val in df_wide[var].items():
                if pd.isna(val):
                    continue
                try:
                    conn.execute(text("""
                        insert into observations (series_id, period_date, observed_at, value, status, meta)
                        values (:sid, :pdate, :oat, :val, null, '{}'::jsonb)
                        on conflict do nothing
                    """), {
                        "sid": series_id,
                        "pdate": dt.date(),
                        "oat": observed_at,
                        "val": float(val),
                    })
                    inserted += 1
                except Exception:
                    failed += 1

        details = json.dumps({"observed_at": observed_at, "columns": len(df_wide.columns)}, ensure_ascii=False)
        conn.execute(text("""
            insert into ingestion_log (source_id, status, rows_inserted, rows_failed, details)
            values (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "source_id": source_id,
            "status": "ok" if failed == 0 else "ok_with_errors",
            "ins": inserted,
            "fail": failed,
            "details": details
        })

    print(f"Inserted: {inserted}, failed: {failed}")


if __name__ == "__main__":
    csv_path = download_current_fredmd_csv()
    df_wide, dataset_meta = parse_fredmd(csv_path)

    observed_at = datetime.now(timezone.utc).isoformat()
    ingest(df_wide, dataset_meta, observed_at)

    # Быстрый чек
    with engine.connect() as conn:
        n_series = conn.execute(text("select count(*) from series where source_id = (select id from sources where name='fredmd')")).scalar_one()
        n_obs = conn.execute(text("""
            select count(*)
            from observations o join series s on s.id=o.series_id
            join sources so on so.id=s.source_id
            where so.name='fredmd'
        """)).scalar_one()
    print(f"FRED-MD series in DB: {n_series}")
    print(f"FRED-MD observations in DB: {n_obs}")
