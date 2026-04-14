"""Paprastas atnaujinimas."""
import logging
from scripts import load_fredmd
from scripts import load_alfred
from scripts import load_eurostat
from scripts import load_google_trends
from scripts import load_financials
from scripts import load_statgov_all_flows

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("updates")

def main():
    logger.info("=== UPDATE RUN START ===")

    sources = {
        "FRED-MD":       lambda: load_fredmd.main(mode="update"),
        "ALFRED":        lambda: load_alfred.main(mode="update"),
        "Eurostat":      lambda: load_eurostat.main(mode="update"),
        "Google Trends": lambda: load_google_trends.main(mode="update"),
        "Financials":    lambda: load_financials.main(mode="update"),
        "StatGov":   lambda: load_statgov_all_flows.main(mode="update", workers=2),
    }

    for name, fn in sources.items():
        try:
            logger.info(f"--- Updating {name} ---")
            fn()
        except Exception as e:
            logger.error(f"Update failed for {name}: {e}", exc_info=True)

    logger.info("=== UPDATE RUN END ===")

if __name__ == "__main__":
    main()