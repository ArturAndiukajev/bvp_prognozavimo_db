import logging
import yaml
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text

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
        values ('yahoo_finance')
        on conflict (name) do nothing
    """))
    return conn.execute(
        text("select id from sources where name='yahoo_finance'")
    ).scalar_one()

def ensure_series(conn, source_id, ticker, name):
    key = ticker
    return conn.execute(text("""
        insert into series (source_id, key, country, frequency, transform, unit, name, meta)
        values (:sid, :key, 'Global', 'D', 'LEVEL', 'PRICE', :name, '{}'::jsonb)
        on conflict (source_id, key, country, frequency, transform)
        do update set name = excluded.name
        returning id
    """), {
        "sid": source_id,
        "key": key,
        "name": name
    }).scalar_one()

def insert_observations(conn, series_id, df):
    observed_at = datetime.now(timezone.utc)
    inserted = 0
    
    # df has Date index and 'Close' column
    for date, row in df.iterrows():
        value = row["Close"]
        
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

def main():
    config = load_config()
    yf_config = config.get("yahoo_finance", {})
    
    if not yf_config:
        logger.warning("No 'yahoo_finance' section in config.")
        return

    with engine.begin() as conn:
        source_id = ensure_source(conn)
        
        for ticker, details in yf_config.items():
            name = details.get("name", ticker)
            logger.info(f"Fetching {ticker} ({name})...")
            
            try:
                # Fetch full history
                # period="max" or "5y" or "10y"
                t = yf.Ticker(ticker)
                hist = t.history(period="5y")
                
                if hist.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                
                # We only want 'Close' price usually
                df = hist[["Close"]]
                
                series_id = ensure_series(conn, source_id, ticker, name)
                count = insert_observations(conn, series_id, df)
                logger.info(f"  Inserted {count} rows")
                
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")

if __name__ == "__main__":
    main()
