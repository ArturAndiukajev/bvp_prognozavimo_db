import logging
import time
from scripts import load_eurostat, load_alfred

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IngestionRunner")

def main():
    logger.info("=== Starting Master Ingestion ===")
    
    start_time = time.time()
    
    try:
        logger.info("--- Running Eurostat Loader ---")
        load_eurostat.main()
    except Exception as e:
        logger.error(f"Eurostat loader failed: {e}")

    try:
        logger.info("--- Running ALFRED Loader ---")
        load_alfred.main()
    except Exception as e:
        logger.error(f"ALFRED loader failed: {e}")

    try:
        logger.info("--- Running Google Trends Loader ---")
        from scripts import load_google_trends
        load_google_trends.main()
    except Exception as e:
        logger.error(f"Google Trends loader failed: {e}")

    try:
        logger.info("--- Running Financials Loader ---")
        from scripts import load_financials
        load_financials.main()
    except Exception as e:
        logger.error(f"Financials loader failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"=== Ingestion Complete in {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()
