# import logging
# import argparse
# from scripts import load_fredmd
# from scripts import load_alfred
# from scripts import load_eurostat
# from scripts import load_google_trends
# from scripts import load_financials

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("updates")

# def main(source="all", max_workers=4):
#     logger.info("=== UPDATE RUN START ===")

#     if source in ("fredmd", "all"):
#         load_fredmd.main(mode="update")

#     if source in ("alfred", "all"):
#         load_alfred.main(mode="update")

#     if source in ("eurostat", "all"):
#         load_eurostat.main(mode="update")

#     if source in ("trends", "all"):
#         load_google_trends.main(mode="update")

#     if source in ("financials", "all"):
#         load_financials.main(mode="update")

#     logger.info("=== UPDATE RUN END ===")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--workers", type=int, default=4)
#     parser.add_argument("--source", default="all")

#     args = parser.parse_args()

#     main(source=args.source, max_workers=args.workers)


import logging
import argparse

from scripts import load_fredmd
from scripts import load_alfred
from scripts import load_eurostat
from scripts import load_google_trends
from scripts import load_financials
from scripts import load_statgov_all_flows

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("updates")


def main(source="all", max_workers=4):
    logger.info("=== UPDATE RUN START ===")

    # FRED-MD predictors
    if source in ("fredmd", "all"):
        logger.info("Updating FRED-MD...")
        load_fredmd.main(mode="update")

    # ALFRED vintages
    if source in ("alfred", "all"):
        logger.info("Updating ALFRED...")
        load_alfred.main(mode="update")

    # Eurostat
    if source in ("eurostat", "all"):
        logger.info("Updating Eurostat...")
        load_eurostat.main(mode="update")

    # Statistikos departamentas (StatGov)
    if source in ("statgov", "all"):
        logger.info("Updating StatGov (Statistikos Departamentas)...")
        load_statgov_all_flows.main()

    # Google Trends
    if source in ("trends", "all"):
        logger.info("Updating Google Trends...")
        load_google_trends.main(mode="update")

    # Financial markets (Yahoo Finance)
    if source in ("financials", "all"):
        logger.info("Updating Yahoo Finance...")
        load_financials.main(mode="update")

    logger.info("=== UPDATE RUN END ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for loaders that support parallel execution",
    )

    parser.add_argument(
        "--source",
        default="all",
        help="Run a specific source: fredmd, alfred, eurostat, statgov, trends, financials, all",
    )

    args = parser.parse_args()

    main(source=args.source, max_workers=args.workers)