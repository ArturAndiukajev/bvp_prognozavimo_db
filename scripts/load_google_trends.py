import logging
import yaml
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text
from pytrends.request import TrendReq

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DB
DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"Config file not found at {CONFIG_PATH}")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_source(conn):
    conn.execute(text("""
        insert into sources (name)
        values ('google_trends')
        on conflict (name) do nothing
    """))
    return conn.execute(
        text("select id from sources where name='google_trends'")
    ).scalar_one()


def ensure_series(conn, source_id, keyword, country):
    key = f"google_trends.{keyword}"

    return conn.execute(text("""
        insert into series (source_id, key, country, frequency, transform, unit, name, meta)
        values (:sid, :key, :country, 'M', 'LEVEL', 'INDEX_0_100', :name, '{}'::jsonb)
        on conflict (source_id, key, country, frequency, transform)
        do update set name = excluded.name
        returning id
    """), {
        "sid": source_id,
        "key": key,
        "country": country,
        "name": f"Google Trends: {keyword}"
    }).scalar_one()


def insert_observations(conn, series_id, df):
    observed_at = datetime.now(timezone.utc)
    
    # df index is date, column 'value' exists
    inserted = 0
    for date, row in df.iterrows():
        value = row["value"]

        if pd.isna(value):
            continue

        conn.execute(text("""
            insert into observations (series_id, period_date, observed_at, value, meta)
            values (:sid, :date, :obs, :val, '{}'::jsonb)
            on conflict (series_id, period_date, observed_at) do nothing
        """), {
            "sid": series_id,
            "date": date.date(),
            "obs": observed_at,
            "val": float(value)
        })
        inserted += 1
    return inserted


def monthly_from_weekly(df):
    df.index = pd.to_datetime(df.index)
    monthly = df.resample("MS").mean()
    # Pytrends returns dataframe where column name is the keyword. Rename to 'value'
    if not monthly.empty:
        monthly.columns = ["value"]
    return monthly


def chunk_keywords(lst, size=5):
    """Google Trends API limits to 5 keywords per request"""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def main():
    config = load_config()
    gt_config = config.get("google_trends", {})
    
    if not gt_config:
        logger.warning("No 'google_trends' section in config.")
        return

    keywords = gt_config.get("keywords", [])
    geo = gt_config.get("geo", "LT")
    
    if not keywords:
        logger.warning("No keywords found in google_trends config.")
        return

    pytrends = TrendReq(hl='en-US', tz=360)
    
    # Timeframe: Last 5 years to keep it relevant and fast? Or 'all'?
    # 'all' is good for long history.
    TIMEFRAME = "today 5-y" 

    with engine.begin() as conn:
        source_id = ensure_source(conn)
        
        # Process in chunks of 5
        for batch in chunk_keywords(keywords, 5):
            logger.info(f"Fetching batch: {batch}")
            
            try:
                pytrends.build_payload(batch, cat=0, timeframe=TIMEFRAME, geo=geo, gprop='')
                data = pytrends.interest_over_time()
                
                if data.empty:
                    logger.warning(f"No data for batch {batch}")
                    continue
                
                # Drop partial indicator if exists
                if 'isPartial' in data.columns:
                    data = data.drop(columns=['isPartial'])
                
                for keyword in batch:
                    if keyword not in data.columns:
                        continue
                        
                    # Extract single series
                    df_kw = data[[keyword]].copy()
                    
                    # Resample to Monthly (Google Trends default is weekly for 5y)
                    df_monthly = monthly_from_weekly(df_kw)
                    
                    if df_monthly.empty:
                        continue

                    series_id = ensure_series(conn, source_id, keyword, geo)
                    count = insert_observations(conn, series_id, df_monthly)
                    logger.info(f"  {keyword}: inserted {count} rows")
                    
                # Be nice to Google API
                import time
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed batch {batch}: {e}")

if __name__ == "__main__":
    main()