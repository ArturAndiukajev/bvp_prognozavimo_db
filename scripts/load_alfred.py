import os
import json
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from fredapi import Fred
from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv

# Load env vars from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database Configuration
DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

# FRED API Configuration
API_KEY = os.environ.get("FRED_API_KEY")
if not API_KEY:
    logger.warning("FRED_API_KEY environment variable not found. Script may fail if not authorized.")

try:
    fred = Fred(api_key=API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize FRED API: {e}")
    fred = None

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

def fetch_and_ingest_series(series_id: str):
    if not fred:
        logger.error("FRED client not initialized.")
        return

    logger.info(f"Processing series: {series_id}")
    
    # 1. Get Series Info
    try:
        info = fred.get_series_info(series_id)
    except Exception as e:
        logger.error(f"Failed to get info for {series_id}: {e}")
        return

    # Map FRED metadata to our schema
    frequency = info.get('frequency_short', 'M') # Default to Monthly if missing
    units = info.get('units')
    title = info.get('title')
    
    # 2. Get All Vintages (Real-time Periods)
    # We want to know when data was released.
    try:
        # This gets the dates (vintages) when the data was revised or released
        vintage_dates = fred.get_series_vintage_dates(series_id)
        logger.info(f"Found {len(vintage_dates)} vintages for {series_id}")
    except Exception as e:
        logger.error(f"Failed to get vintage dates for {series_id}: {e}")
        return

    # In a production script, we might want to filter widely to only get new vintages.
    # For now, let's fetch the last 5 vintages to test, or all if user wants.
    # To keep it simple and efficient for a "load" script, let's iterate over ALL, 
    # but maybe we can optimize by only fetching data for that vintage?
    
    # Actually, fred.get_series_all_releases() might be better but let's stick to the standard 
    # approach of "what was the data looking like on date X?"
    
    # Let's take the *latest* 5 vintages for demonstration/testing purposes
    # straightforward loop:
    
    target_vintages = vintage_dates[-5:] # Last 5 updates
    if not target_vintages:
        target_vintages = [datetime.now()] # Fallback if no vintages found (rare)

    with engine.begin() as conn:
        source_id = ensure_source(conn, "alfred")
        
        # Ensure series exists
        db_series_id = ensure_series(
            conn=conn,
            source_id=source_id,
            key=series_id,
            country="US", # FRED is mostly US
            frequency=frequency,
            transform="LEVEL", # Default
            unit=units,
            name=title,
            meta={"fred_info": info.to_dict()}
        )
        
        total_inserted = 0
        
        for v_date in target_vintages:
            v_date_str = v_date.strftime('%Y-%m-%d')
            logger.info(f"Fetching vintage: {v_date_str}")
            
            try:
                # real_time_start=v_date, real_time_end=v_date gives us the data AS IT WAS on that date.
                data = fred.get_series(series_id, realtime_start=v_date_str, realtime_end=v_date_str)
                
                if data is None or data.empty:
                    continue
                
                # Insert data points
                for p_date, value in data.items():
                    if pd.isna(value):
                        continue
                        
                    conn.execute(text("""
                        insert into observations (series_id, period_date, observed_at, value, status, meta)
                        values (:sid, :pdate, :oat, :val, null, '{}'::jsonb)
                        on conflict (series_id, period_date, observed_at) do nothing
                    """), {
                        "sid": db_series_id,
                        "pdate": p_date.date(),
                        "oat": v_date, # This is the vintage date
                        "val": float(value),
                    })
                    total_inserted += 1
                    
            except Exception as e:
                logger.error(f"Failed to fetch/insert vintage {v_date_str}: {e}")

        # Log ingestion
        conn.execute(text("""
            insert into ingestion_log (source_id, status, rows_inserted, details)
            values (:source_id, 'ok', :ins, :details)
        """), {
            "source_id": source_id,
            "ins": total_inserted,
            "details": json.dumps({"series": series_id, "vintages_count": len(target_vintages)})
        })
        
        logger.info(f"Finished {series_id}. Inserted {total_inserted} observations.")

def main():
    # Example usage: Fetch GDP and CPI
    # series_to_fetch = ["GDP", "CPIAUCSL", "UNRATE"]
    series_to_fetch = ["GDP"] # Start small
    
    if not API_KEY:
        logger.error("Stopping: No API KEY configured.")
        return

    for s in series_to_fetch:
        fetch_and_ingest_series(s)

if __name__ == "__main__":
    main()
