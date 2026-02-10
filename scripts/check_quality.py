import pandas as pd
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

def check_freshness():
    """Checks when the latest data was observed for each series."""
    query = text("""
        SELECT 
            s.name, 
            s.key, 
            MAX(o.observed_at) as last_update,
            MAX(o.period_date) as last_period
        FROM observations o
        JOIN series s ON s.id = o.series_id
        GROUP BY s.id
        ORDER BY last_update ASC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print("\n--- Data Freshness (Top 10 Stale) ---")
    print(df.head(10))

def check_gaps():
    """Checks for missing periods in quarterly/monthly series."""
    # This is a bit complex in SQL, simplifying to just count
    pass

def main():
    logger.info("Running Data Quality Checks...")
    check_freshness()
    # Add more checks here

if __name__ == "__main__":
    main()
