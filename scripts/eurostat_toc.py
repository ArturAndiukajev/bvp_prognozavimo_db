import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import eurostat
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eurostat_toc")

RAW_DIR = Path("data/raw/eurostat")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def download_toc(lang: str = "en", agency: str = "all") -> pd.DataFrame:
    # eurostat.get_toc_df() exists in this package :contentReference[oaicite:1]{index=1}
    toc_df = eurostat.get_toc_df(agency=agency, lang=lang)
    if toc_df is None or toc_df.empty:
        raise RuntimeError("TOC is empty")
    return toc_df

def main():
    downloaded_at = datetime.now(timezone.utc)
    stamp = downloaded_at.strftime("%Y-%m-%dT%H%M%SZ")

    logger.info("Downloading Eurostat TOC...")
    toc_df = download_toc(lang="en", agency="all")

    #JSON bytes
    toc_bytes = toc_df.to_json(
        orient="records",
        force_ascii=False
    ).encode("utf-8")

    new_hash = sha256_bytes(toc_bytes)

    #Ar yra failai
    existing_files = sorted(RAW_DIR.glob("toc_*.json"))

    if existing_files:
        latest_json = existing_files[-1]
        old_bytes = latest_json.read_bytes()
        old_hash = sha256_bytes(old_bytes)

        if old_hash == new_hash:
            logger.info("TOC has not changed. Skipping save.")
            return
        else:
            logger.info("TOC changed.")
    else:
        logger.info("No previous TOC found. First download.")

    #Jei čia tai nauji duomenys

    csv_path = RAW_DIR / f"toc_{stamp}.csv"
    json_path = RAW_DIR / f"toc_{stamp}.json"

    toc_df.to_csv(csv_path, index=False)
    json_path.write_bytes(toc_bytes)

    logger.info(f"Saved TOC: {csv_path}")
    logger.info(f"Saved TOC JSON: {json_path}")
    logger.info(f"Rows: {len(toc_df)}; hash: {new_hash}")

if __name__ == "__main__":
    main()