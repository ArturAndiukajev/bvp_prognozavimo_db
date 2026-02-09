import json
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"

CSV_PATH = r"C:\Users\artur\OneDrive\Desktop\namq_10_gdp__custom_20029138_linear_2_0.csv\namq_10_gdp__custom_20029138_linear_2_0.csv"

engine = create_engine(DB_URL, future=True)


def parse_time_period(tp: str) -> str:
    """
    Eurostat TIME_PERIOD может быть:
    - '2023-Q3'  -> '2023-07-01'
    - '2023-Q4'  -> '2023-10-01'
    - '2023-01'  -> '2023-01-01'
    - '2023'     -> '2023-01-01'
    """
    tp = str(tp).strip()
    if tp == "" or tp.lower() == "nan":
        raise ValueError("Empty TIME_PERIOD")

    if "-Q" in tp:
        y, q = tp.split("-Q")
        y = int(y)
        q = int(q)
        month = 1 + (q - 1) * 3
        return f"{y:04d}-{month:02d}-01"

    # YYYY-MM
    if len(tp) == 7 and tp[4] == "-":
        y = int(tp[:4])
        m = int(tp[5:7])
        return f"{y:04d}-{m:02d}-01"

    # YYYY
    if len(tp) == 4 and tp.isdigit():
        return f"{int(tp):04d}-01-01"

    raise ValueError(f"Unsupported TIME_PERIOD format: {tp}")


def ensure_source(conn, name: str) -> int:
    conn.execute(text("""
        insert into sources (name) values (:name)
        on conflict (name) do nothing
    """), {"name": name})
    return conn.execute(text("select id from sources where name=:name"), {"name": name}).scalar_one()


def ensure_series(conn, source_id: int, key: str, country: str, freq: str,
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
        "freq": freq,
        "transform": transform,
        "unit": unit,
        "name": name,
        "meta": meta_json,   # <-- JSON eilute
    }).scalar_one()


def main():
    df = pd.read_csv(CSV_PATH)

    #paliekam tik kas mus domina  (kol kas testuojam)
    required_cols = ["freq", "unit", "s_adj", "na_item", "geo", "TIME_PERIOD", "OBS_VALUE",
                     "STRUCTURE_ID", "STRUCTURE_NAME"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV is missing columns: {missing}")

    #istrinam eilutes be reiksmiu
    df = df[df["OBS_VALUE"].notna()].copy()

    observed_at = datetime.now(timezone.utc).isoformat()

    inserted = 0
    failed = 0

    with engine.begin() as conn:
        source_id = ensure_source(conn, "eurostat")

        #Grupuojam pagal matavimus, kurie apibrezia series
        group_cols = ["STRUCTURE_ID", "freq", "unit", "s_adj", "na_item", "geo"]
        for dims, g in df.groupby(group_cols, dropna=False):
            structure_id, freq, unit, s_adj, na_item, geo = dims

            #Sugalvojam unikalu serijos rakta(galima paskui pakeisti)
            key = f"{structure_id}_{na_item}_{unit}_{s_adj}"
            country = str(geo)
            transform = "LEVEL"
            name = f"{structure_id} {na_item} {unit} {s_adj} ({country})"

            meta = {
                "structure_name": str(g["STRUCTURE_NAME"].iloc[0]),
                "s_adj": str(s_adj),
                "na_item": str(na_item),
                "unit": str(unit),
                "freq": str(freq),
            }

            series_id = ensure_series(conn, source_id, key, country, str(freq),
                                      transform, str(unit), name, meta)

            #idedam observations
            for _, row in g.iterrows():
                try:
                    period_date = parse_time_period(row["TIME_PERIOD"])
                    value = float(row["OBS_VALUE"])

                    conn.execute(text("""
                        insert into observations (series_id, period_date, observed_at, value, status, meta)
                        values (:series_id, :period_date, :observed_at, :value, :status, '{}'::jsonb)
                        on conflict do nothing
                    """), {
                        "series_id": series_id,
                        "period_date": period_date,
                        "observed_at": observed_at,
                        "value": value,
                        "status": str(row.get("OBS_FLAG")) if "OBS_FLAG" in row else None
                    })
                    inserted += 1
                except Exception:
                    failed += 1

        details_json = json.dumps({"file": CSV_PATH}, ensure_ascii=False)

        conn.execute(text("""
            insert into ingestion_log (source_id, status, rows_inserted, rows_failed, details)
            values (:source_id, :status, :ins, :fail, CAST(:details AS jsonb))
        """), {
            "source_id": source_id,
            "status": "ok_with_errors" if failed else "ok",
            "ins": inserted,
            "fail": failed,
            "details": details_json
        })

    print(f"Done. Inserted rows: {inserted}, failed rows: {failed}")

    #Parodom pora irasu
    with engine.connect() as conn:
        rows = conn.execute(text("""
            select s.key, s.country, s.frequency, o.period_date, o.observed_at, o.value
            from observations o
            join series s on s.id = o.series_id
            order by o.observed_at desc, o.period_date desc
            limit 10
        """)).mappings().all()

    print("Last 10 observations:")
    for r in rows:
        print(dict(r))


if __name__ == "__main__":
    main()