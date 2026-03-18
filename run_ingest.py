import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from scripts import load_fredmd, load_alfred, load_eurostat, load_google_trends, load_financials, build_eurostat_list

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


def _run_eurostat_dispatch(eurostat_mode: str, ingest_mode: str):
    """Dispatches the Eurostat phase to the correct loader based on CLI arg."""
    if eurostat_mode == "build":
        logger.info("Running Eurostat loader: build_eurostat_list")
        # build_eurostat_list.main doesn't take 'mode' like the others, just args
        build_eurostat_list.main()
    else:
        logger.info("Running Eurostat loader: load_eurostat")
        load_eurostat.main(mode=ingest_mode)


def main(mode: str = "initial", eurostat_mode: str = "load"):
    logger.info(f"=== Starting Master Ingestion (mode={mode}, eurostat_mode={eurostat_mode}) ===")
    start_time = time.time()

    # Wave 1: Fast / lightweight
    logger.info("Wave 1: launching fast loaders in parallel")
    wave1 = {
        "FRED-MD":    lambda: load_fredmd.main(mode=mode),
        "ALFRED":     lambda: load_alfred.main(mode=mode),
        "Financials": lambda: load_financials.main(mode=mode),
    }
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_run_loader, name, fn): name for name, fn in wave1.items()}
        for fut in as_completed(futs):
            fut.result()  # exceptions were already logged inside _run_loader

    # Wave 2: Slow / API-heavy
    logger.info("Wave 2: launching slow loaders in parallel")
    wave2 = {
        "Eurostat":       lambda: _run_eurostat_dispatch(eurostat_mode, mode),
        "Google Trends":  lambda: load_google_trends.main(mode=mode),
    }
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_run_loader, name, fn): name for name, fn in wave2.items()}
        for fut in as_completed(futs):
            fut.result()

    elapsed = time.time() - start_time
    logger.info(f"=== Ingestion Complete in {elapsed:.2f}s ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Master Ingestion Runner")
    ap.add_argument("--mode", type=str, default="initial", choices=["initial", "update"],
                    help="Ingestion mode ('initial' or 'update')")
    ap.add_argument("--eurostat-mode", type=str, default="load", choices=["load", "build"],
                    help="Choose Eurostat pipeline: 'load' (default) or 'build' (dynamic TOC scanning)")
    args = ap.parse_args()
    
    main(mode=args.mode, eurostat_mode=args.eurostat_mode)
