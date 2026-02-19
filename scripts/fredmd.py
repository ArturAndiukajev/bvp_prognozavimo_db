import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Optional
import time
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from contextlib import contextmanager
from io import StringIO

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(DB_URL, future=True)

FREDMD_CURRENT_CSV = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"

# -------------------- DB helpers --------------------
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

def ensure_source(conn, name: str) -> int:
    conn.execute(text("""
        INSERT INTO sources (name) VALUES (:name)
        ON CONFLICT (name) DO NOTHING
    """), {"name": name})
    return conn.execute(text("SELECT id FROM sources WHERE name=:name"), {"name": name}).scalar_one()

def ensure_series(conn, source_id: int, key: str, country: str, frequency: str,
                  transform: str, unit: str, name: str, meta: dict) -> int:
    meta_json = json.dumps(meta, ensure_ascii=False)
    return conn.execute(text("""
        INSERT INTO series (source_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:sid, :key, :country, :freq, :transform, :unit, :name, CAST(:meta AS jsonb))
        ON CONFLICT (source_id, key, country, frequency, transform)
        DO UPDATE SET unit = EXCLUDED.unit,
                      name = EXCLUDED.name,
                      meta = EXCLUDED.meta
        RETURNING id
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
        "rtime": downloaded_at,     # keep compatible meaning
        "dlat": downloaded_at,
        "vint": vintage_at,
        "desc": description,
        "raw": raw_path,
        "hash": content_hash,
        "meta": meta_json
    }).scalar_one()

# -------------------- File helpers --------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_current_fredmd_csv() -> Tuple[Path, datetime]:
    downloaded_at = datetime.now(timezone.utc)
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")
    out_path = RAW_DIR / f"fred_md_{stamp}.csv"

    r = requests.get(FREDMD_CURRENT_CSV, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

    print(f"Downloaded: {FREDMD_CURRENT_CSV}")
    print(f"Saved to:   {out_path}")
    return out_path, downloaded_at

# -------------------- Parse --------------------

def parse_fredmd(csv_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      df_wide: index = dates, columns = variables
      meta: e.g., tcodes if present
    """
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    meta: Dict = {}

    # Detect tcodes row
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

# -------------------- Ingest --------------------

def ingest(df_wide: pd.DataFrame, dataset_meta: dict,
           vintage_at: datetime, downloaded_at: datetime,
           raw_path: str | None, content_hash: str | None, mode: str = "initial") -> None:
    source_name = "fredmd"
    inserted = 0
    failed = 0

    with engine.begin() as conn:
        source_id = ensure_source(conn, source_name)

        release_id = create_release(
            conn=conn,
            source_id=source_id,
            downloaded_at=downloaded_at,
            vintage_at=vintage_at,
            description="FRED-MD current.csv snapshot (snapshot-vintage)",
            raw_path=raw_path,
            content_hash=content_hash,
            scope="FRED-MD",
            meta={"dataset": "FRED-MD"}
        )

        # FRED-MD is a monthly macro dataset (predictor panel)
        country = "US"
        freq = "M"
        transform = "LEVEL"
        unit = "INDEX"

        for var in df_wide.columns:
            series_meta = {
                "dataset": "FRED-MD",
                "variable": var,
                "tcode": dataset_meta.get("tcodes", {}).get(var)
            }
            series_id = ensure_series(
                conn=conn,
                source_id=source_id,
                key=var,
                country=country,
                frequency=freq,
                transform=transform,
                unit=unit,
                name=f"FRED-MD: {var}",
                meta=series_meta
            )

            # Insert each (period_date, value) under the same vintage_at
            for dt, val in df_wide[var].items():
                if pd.isna(val):
                    continue
                try:
                    conn.execute(text("""
                        INSERT INTO observations (series_id, period_date, observed_at, value, release_id, meta)
                        VALUES (:sid, :pdate, :oat, :val, :rid, '{}'::jsonb)
                        ON CONFLICT (series_id, period_date, observed_at) DO NOTHING
                    """), {
                        "sid": series_id,
                        "pdate": dt.date(),
                        "oat": vintage_at,     # <-- vintage/as-of time, ALFRED-like
                        "val": float(val),
                        "rid": release_id
                    })
                    inserted += 1
                except Exception:
                    failed += 1

        details = {
            "downloaded_at": downloaded_at.isoformat(),
            "vintage_at": vintage_at.isoformat(),
            "raw_path": raw_path,
            "hash": content_hash,
            "columns": int(len(df_wide.columns)),
            "rows": int(len(df_wide))
        }
        conn.execute(text("""
            INSERT INTO ingestion_log (source_id, status, rows_inserted, rows_failed, details)
            VALUES (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "source_id": source_id,
            "status": "ok" if failed == 0 else "ok_with_errors",
            "ins": inserted,
            "fail": failed,
            "details": json.dumps(details, ensure_ascii=False),
        })

    print(f"FRED-MD inserted: {inserted}, failed: {failed}")


def main(mode: str = "initial"):
    csv_path, downloaded_at = download_current_fredmd_csv()
    df_wide, dataset_meta = parse_fredmd(csv_path)

    vintage_at = downloaded_at
    content_hash = sha256_file(csv_path)

    with engine.begin() as conn:
        source_id = ensure_source(conn, "fredmd")

        # UPDATE: если hash не изменился — выходим
        if mode == "update":
            prev = last_release_hash(conn, source_id, scope="FRED-MD")
            if prev == content_hash:
                print("FRED-MD: no changes (hash same) -> skip")
                return

    ingest(
        df_wide=df_wide,
        dataset_meta=dataset_meta,
        vintage_at=vintage_at,
        downloaded_at=downloaded_at,
        raw_path=str(csv_path),
        content_hash=content_hash,
        mode=mode
    )

    # quick check
    with engine.connect() as conn:
        n_series = conn.execute(text("""
            SELECT count(*) FROM series
            WHERE source_id = (SELECT id FROM sources WHERE name='fredmd')
        """)).scalar_one()

        n_obs = conn.execute(text("""
            SELECT count(*)
            FROM observations o
            JOIN series s ON s.id=o.series_id
            JOIN sources so ON so.id=s.source_id
            WHERE so.name='fredmd'
        """)).scalar_one()

    print(f"FRED-MD series in DB: {n_series}")
    print(f"FRED-MD observations in DB: {n_obs}")

if __name__ == "__main__":
    main()
