import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, date
import argparse

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

def get_vintage(vintage_date: date | str = None) -> pd.DataFrame:
    """
    Retrieves the dataset as it was known on `vintage_date`.
    If vintage_date is None, returns the latest available data.
    """
    if vintage_date is None:
        vintage_date = datetime.now().date()
    
    query = text("""
        SELECT DISTINCT ON (s.key, o.period_date)
            s.key,
            s.name,
            s.frequency,
            o.period_date,
            o.value,
            o.observed_at
        FROM observations o
        JOIN series s ON s.id = o.series_id
        WHERE o.observed_at <= :vdate
        ORDER BY s.key, o.period_date, o.observed_at DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"vdate": vintage_date})
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Get vintage data from Nowcast DB")
    parser.add_argument("--date", type=str, help="Vintage date (YYYY-MM-DD)", default=None)
    parser.add_argument("--pivot", action="store_true", help="Pivot table (date x series)")
    args = parser.parse_args()
    
    df = get_vintage(args.date)
    
    if df.empty:
        print("No data found for this vintage.")
        return

    print(f"Loaded {len(df)} rows.")
    
    if args.pivot:
        # Pivot: Index=Period, Columns=Key, Values=Value
        df_pivot = df.pivot(index="period_date", columns="key", values="value").sort_index()
        print(df_pivot.tail())
    else:
        print(df.head())

if __name__ == "__main__":
    main()
