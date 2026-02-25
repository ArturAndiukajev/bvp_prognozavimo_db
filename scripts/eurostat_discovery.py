import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import eurostat
import pandas as pd

logger = logging.getLogger("eurostat_discover")

# Directory where eurostat_toc.py stores downloaded TOC files
RAW_DIR = Path("data/raw/eurostat")


def _col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """
    Return the actual column name from df matching one of the candidate names
    (case-insensitive). Returns None if not found.

    This makes the code robust against different column capitalizations
    across eurostat package versions.
    """
    cols = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _load_latest_toc_from_disk(raw_dir: Path = RAW_DIR) -> Optional[pd.DataFrame]:
    """
    Load the most recent TOC file (JSON preferred, CSV fallback)
    from the local raw directory.

    Returns:
        DataFrame if successfully loaded,
        None if no files are available or loading failed.
    """
    if not raw_dir.exists():
        return None

    # Prefer JSON (more structured and consistent)
    json_files = sorted(raw_dir.glob("toc_*.json"))
    if json_files:
        latest = json_files[-1]
        try:
            df = pd.read_json(latest, orient="records", dtype=False)
            if df is not None and not df.empty:
                logger.info(f"Loaded TOC from disk: {latest}")
                return df
        except Exception as e:
            logger.warning(f"Failed to read TOC JSON {latest}: {e}")

    # Fallback to CSV if JSON not available
    csv_files = sorted(raw_dir.glob("toc_*.csv"))
    if csv_files:
        latest = csv_files[-1]
        try:
            df = pd.read_csv(latest)
            if df is not None and not df.empty:
                logger.info(f"Loaded TOC from disk: {latest}")
                return df
        except Exception as e:
            logger.warning(f"Failed to read TOC CSV {latest}: {e}")

    return None


def _download_toc() -> pd.DataFrame:
    """
    Download TOC (Table of Contents) directly from Eurostat API.
    """
    toc = eurostat.get_toc_df(agency="all", lang="en")
    if toc is None or toc.empty:
        raise RuntimeError("Eurostat TOC is empty")
    logger.info("Downloaded TOC from Eurostat API")
    return toc


def _toc_get(prefer_disk: bool = True) -> pd.DataFrame:
    """
    Retrieve TOC either from local disk (preferred) or from the API.

    Args:
        prefer_disk: If True, try loading from disk first.

    Returns:
        DataFrame containing TOC.
    """
    if prefer_disk:
        df = _load_latest_toc_from_disk()
        if df is not None and not df.empty:
            return df
    return _download_toc()


def _get_pars_safe(code: str) -> Optional[List[str]]:
    """
    Safely retrieve dataset parameter names (dimensions)
    for a given dataset code.

    Supports different eurostat package function names
    across versions (get_pars / get_pars_df).
    """
    for fn_name in ("get_pars", "get_pars_df"):
        fn = getattr(eurostat, fn_name, None)
        if fn:
            try:
                out = fn(code)

                # If DataFrame returned, extract parameter column
                if isinstance(out, pd.DataFrame):
                    c = _col(out, "par", "parameter", "name")
                    return (
                        out[c].astype(str).tolist()
                        if c
                        else out.astype(str).iloc[:, 0].tolist()
                    )

                # If list/tuple returned
                if isinstance(out, (list, tuple)):
                    return list(map(str, out))

            except Exception:
                return None
    return None


def _get_par_values_safe(code: str, par: str) -> Optional[List[str]]:
    """
    Safely retrieve possible values for a specific dataset parameter.

    Supports multiple eurostat function names
    (get_par_values / get_par_values_df).
    """
    for fn_name in ("get_par_values", "get_par_values_df"):
        fn = getattr(eurostat, fn_name, None)
        if fn:
            try:
                out = fn(code, par)

                if isinstance(out, pd.DataFrame):
                    c = _col(out, "value", "values", "name", "code")
                    if c:
                        return out[c].astype(str).tolist()
                    return out.astype(str).iloc[:, 0].tolist()

                if isinstance(out, (list, tuple)):
                    return list(map(str, out))

            except Exception:
                return None
    return None


def discover_geo_monthly_quarterly(
    geo: str = "LT",
    freqs: Tuple[str, ...] = ("M", "Q"),
    max_datasets: Optional[int] = None,
    prefer_disk_toc: bool = True,
) -> List[Dict[str, Any]]:
    """
    Discover Eurostat datasets that:

    - contain the specified geo dimension (e.g. "LT")
    - contain frequency dimension ("freq")
    - support at least one of the requested frequencies (e.g. M or Q)

    Args:
        geo: Country/region code (e.g. "LT").
        freqs: Tuple of acceptable frequencies.
        max_datasets: Optional limit for performance testing.
                      None means no limit.
        prefer_disk_toc: If True, load TOC from disk if available.

    Returns:
        List of dictionaries with:
            - code
            - title
            - freqs (supported frequencies intersection)
            - geo
    """

    toc = _toc_get(prefer_disk=prefer_disk_toc)

    code_col = _col(toc, "code")
    type_col = _col(toc, "type")
    title_col = _col(toc, "title")

    if not code_col:
        raise RuntimeError(f"TOC has no 'code' column. Columns={list(toc.columns)}")

    df = toc.copy()

    # Keep only dataset entries (exclude folders/categories if present)
    if type_col:
        df = df[df[type_col].astype(str).str.lower() == "dataset"]

    codes = df[code_col].astype(str).dropna().unique().tolist()

    # Optional performance limit
    if max_datasets is not None:
        codes = codes[:max_datasets]

    results: List[Dict[str, Any]] = []
    kept = 0

    for code in codes:
        pars = _get_pars_safe(code)
        if not pars:
            continue

        pars_l = {p.lower() for p in pars}

        # Must have geo dimension
        if "geo" not in pars_l:
            continue

        # Must have frequency dimension
        if "freq" not in pars_l:
            continue

        geo_vals = _get_par_values_safe(code, "geo")
        if not geo_vals or geo not in set(geo_vals):
            continue

        freq_vals = _get_par_values_safe(code, "freq")
        if not freq_vals:
            continue

        freq_set = set(map(str, freq_vals))

        # Must contain at least one required frequency
        if not any(f in freq_set for f in freqs):
            continue

        title = None
        if title_col:
            mask = df[code_col].astype(str) == str(code)
            if mask.any():
                title = df.loc[mask, title_col].astype(str).iloc[0]

        results.append(
            {
                "code": str(code),
                "title": title or str(code),
                "freqs": sorted(list(freq_set.intersection(set(freqs)))),
                "geo": geo,
            }
        )

        kept += 1

    logger.info(f"Discovery kept {kept} datasets for geo={geo} freqs={freqs}")
    return results