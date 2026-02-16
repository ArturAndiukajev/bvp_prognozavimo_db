import logging
import argparse
from sqlalchemy import create_engine, text

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DB
DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

def reset_db(force=False):
    if not force:
        print("WARNING: This will delete ALL data from the database.")
        print("This includes all observations, series definitions, sources, and logs.")
        confirm = input("Type 'DELETE' to confirm: ")
        if confirm != "DELETE":
            print("Aborted.")
            return

    logger.info("Resetting database...")
    
    with engine.begin() as conn:
        # Disable triggers if needed, but simple truncate is usually fine
        # CASCADE ensures child tables (observations, series) are cleared when parent (sources) is cleared
        # though usually we want to clear observations first
        
        # Order matters if no CASCADE, but CASCADE handles it.
        # Clearing observations first is safer for large tables.
        
        logger.info("Truncating tables...")
        conn.execute(text("TRUNCATE TABLE observations, series, sources, ingestion_log RESTART IDENTITY CASCADE"))
        
    logger.info("Database reset complete. All tables are empty.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset the Nowcast Database")
    parser.add_argument("--force", action="store_true", help="Force deletion without prompt")
    args = parser.parse_args()
    
    reset_db(args.force)
