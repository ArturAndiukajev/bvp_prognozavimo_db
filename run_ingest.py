import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts import fredmd, load_alfred, load_eurostat, load_google_trends, load_financials

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IngestionRunner")


def _run_loader(name: str, fn):
    """Wrapper that captures exceptions and logs them without stopping other loaders."""
    try:
        logger.info(f"--- Starting {name} ---")
        fn()
        logger.info(f"--- Finished {name} ---")
    except Exception as e:
        logger.error(f"{name} failed: {e}", exc_info=True)


def main(mode: str = "initial"):
    logger.info(f"=== Starting Master Ingestion (mode={mode}) ===")
    start_time = time.time()

    # -----------------------------------------------------------
    # Wave 1: Fast / lightweight loaders — run in parallel
    # fredmd (~1-2 min), alfred (~1 min), financials (~2-3 min)
    # -----------------------------------------------------------
    logger.info("Wave 1: launching fast loaders in parallel")
    wave1 = {
        "FRED-MD":    lambda: fredmd.main(mode=mode),
        "ALFRED":     lambda: load_alfred.main(mode=mode),
        "Financials": lambda: load_financials.main(mode=mode),
    }
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_run_loader, name, fn): name for name, fn in wave1.items()}
        for fut in as_completed(futs):
            fut.result()  # exceptions were already logged inside _run_loader

    # -----------------------------------------------------------
    # Wave 2: Slow / API-heavy loaders — run in parallel
    # eurostat (~30 min), google_trends (~20 min)
    # -----------------------------------------------------------
    logger.info("Wave 2: launching slow loaders in parallel")
    wave2 = {
        "Eurostat":       lambda: load_eurostat.main(mode=mode),
        "Google Trends":  lambda: load_google_trends.main(mode=mode),
    }
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_run_loader, name, fn): name for name, fn in wave2.items()}
        for fut in as_completed(futs):
            fut.result()

    elapsed = time.time() - start_time
    logger.info(f"=== Ingestion Complete in {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
