import logging
import create_schema
from scripts import load_fredmd
from scripts import load_alfred
from scripts import load_eurostat
from scripts import load_google_trends
from scripts import load_financials

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FullReload")


def main():
    print("=== Full Reload Process Started ===")
    logger.info("=== Full Reload Process Started ===")
    print("Recreating schema...")
    logger.info("Recreating schema...")
    create_schema.main()

    logger.info("Loading data (each source is independent)...")

    try:
        print("--- FRED-MD ---")
        logger.info("--- FRED-MD ---")
        load_fredmd.main(mode="initial")
    except Exception as e:
        print(f"FRED-MD failed: {e}")
        logger.error(f"FRED-MD failed: {e}")

    try:
        print("--- ALFRED ---")
        logger.info("--- ALFRED ---")
        load_alfred.main(mode="initial")
    except Exception as e:
        print(f"ALFRED failed: {e}")
        logger.error(f"ALFRED failed: {e}")

    try:
        print("--- Eurostat ---")
        logger.info("--- Eurostat ---")
        load_eurostat.main(mode="initial")
    except Exception as e:
        print(f"Eurostat failed: {e}")
        logger.error(f"Eurostat failed: {e}")

    try:
        print("--- Google Trends ---")
        logger.info("--- Google Trends ---")
        load_google_trends.main(mode="initial")
    except Exception as e:
        print(f"Google Trends failed: {e}")
        logger.error(f"Google Trends failed: {e}")

    try:
        print("--- Financials ---")
        logger.info("--- Financials ---")
        load_financials.main(mode="initial")
    except Exception as e:
        print(f"Financials failed: {e}")
        logger.error(f"Financials failed: {e}")
    print("=== Full Reload Complete ===")
    logger.info("=== Full Reload Complete ===")


if __name__ == "__main__":
    main()
