import logging
from scripts import fredmd
from scripts import load_alfred
from scripts import load_eurostat
from scripts import load_google_trends
from scripts import load_financials

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("updates")

def main():
    logger.info("=== UPDATE RUN START ===")

    # Predictors panel
    fredmd.main(mode="update")

    # Key macro + vintages
    load_alfred.main(mode="update")

    # Other sources
    load_eurostat.main(mode="update")
    load_google_trends.main(mode="update")
    load_financials.main(mode="update")

    logger.info("=== UPDATE RUN END ===")

if __name__ == "__main__":
    main()