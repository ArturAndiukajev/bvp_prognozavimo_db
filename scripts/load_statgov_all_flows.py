import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET

from scripts import load_statgov_one_flow as loader

logger = logging.getLogger("statgov_all_flows")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATAFLOW_URL = "https://osp-rs.stat.gov.lt/rest_xml/dataflow/"

NS = {
    "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}


def fetch_dataflow_catalog_bytes(timeout: int = 90) -> bytes:
    r = requests.get(DATAFLOW_URL, timeout=timeout)
    r.raise_for_status()
    return r.content


def parse_flow_maps(catalog_xml: bytes) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """
    Returns:
      - flow_id -> agencyID
      - flow_id -> dataset name (first available Name element text)
    """
    root = ET.fromstring(catalog_xml)

    flow_agency: Dict[str, str] = {}
    flow_name: Dict[str, Optional[str]] = {}

    for df in root.findall(".//str:Dataflow", NS):
        flow_id = df.attrib.get("id")
        agency = df.attrib.get("agencyID")

        if not flow_id:
            continue

        if agency:
            flow_agency[flow_id] = agency

        # dataset name: take first available com:Name under this Dataflow
        name_el = df.find(".//com:Name", NS)
        name = name_el.text.strip() if (name_el is not None and name_el.text) else None
        flow_name[flow_id] = name

    return flow_agency, flow_name


def list_flow_ids(catalog_xml: bytes) -> List[str]:
    root = ET.fromstring(catalog_xml)
    out: List[str] = []
    for df in root.findall(".//str:Dataflow", NS):
        flow_id = df.attrib.get("id")
        if flow_id:
            out.append(flow_id)
    return out


def patch_loader_catalog_cache(catalog_xml: bytes, flow_agency: Dict[str, str]) -> None:
    """
    Speed-up: prevent loader from refetching the entire catalog for every flow.
    """

    def _cached_fetch_dataflow_catalog_xml() -> bytes:
        return catalog_xml

    loader.fetch_dataflow_catalog_xml = _cached_fetch_dataflow_catalog_xml  # type: ignore

    original_find = loader.find_agency_for_flow

    def _cached_find_agency_for_flow(flow_id: str) -> str:
        if flow_id in flow_agency:
            return flow_agency[flow_id]
        return original_find(flow_id)

    loader.find_agency_for_flow = _cached_find_agency_for_flow  # type: ignore


def run_one_flow_with_retry(
    flow_id: str,
    start: Optional[str],
    end: Optional[str],
    debug_title: bool,
    mode: str,
    retries: int,
    retry_sleep: float,
) -> Tuple[str, bool, Optional[str]]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            loader.ingest_one_flow(flow_id, start=start, end=end, debug_title=debug_title, mode=mode)
            return flow_id, True, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning(f"Flow failed flow_id={flow_id} attempt={attempt+1}/{retries+1}: {last_err}")
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    return flow_id, False, last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest ALL StatGov OSP flows using load_statgov_one_flow.ingest_one_flow()")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--debug-title", action="store_true")
    ap.add_argument("--mode", choices=["initial", "update"], default="update")

    ap.add_argument("--include", default=None, help="Regex: only ingest flows whose id matches")
    ap.add_argument("--exclude", default=None, help="Regex: skip flows whose id matches")

    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-sleep", type=float, default=2.0)

    args = ap.parse_args()

    logger.info("Fetching SDMX dataflow catalog once...")
    catalog_xml = fetch_dataflow_catalog_bytes(timeout=90)

    flow_agency, flow_name = parse_flow_maps(catalog_xml)
    flow_ids = list_flow_ids(catalog_xml)

    logger.info(f"Catalog: {len(flow_ids)} flows, {len(flow_agency)} with agencyID.")

    patch_loader_catalog_cache(catalog_xml, flow_agency)

    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    filtered: List[str] = []
    for fid in flow_ids:
        if include_re and not include_re.search(fid):
            continue
        if exclude_re and exclude_re.search(fid):
            continue
        filtered.append(fid)

    if args.offset:
        filtered = filtered[args.offset:]
    if args.limit is not None:
        filtered = filtered[: args.limit]

    if not filtered:
        logger.info("No flows to ingest after filtering.")
        return

    logger.info(f"Will ingest {len(filtered)} flows (workers={args.workers}).")

    ok_count = 0
    fail_count = 0

    # Store: (flow_id, dataset_name, error)
    failures: List[Tuple[str, Optional[str], str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                run_one_flow_with_retry,
                fid,
                args.start,
                args.end,
                args.debug_title,
                args.mode,
                args.retries,
                args.retry_sleep,
            )
            for fid in filtered
        ]

        for fut in as_completed(futs):
            fid, ok, err = fut.result()
            if ok:
                ok_count += 1
                logger.info(f"OK flow_id={fid} ({ok_count}/{len(filtered)})")
            else:
                fail_count += 1
                failures.append((fid, flow_name.get(fid), err or "unknown error"))
                logger.error(f"FAIL flow_id={fid} name={flow_name.get(fid)!r} err={err}")

    logger.info(f"DONE all flows: ok={ok_count} fail={fail_count} total={len(filtered)}")

    if failures:
        logger.info("Failed flows (first 100):")
        for fid, name, err in failures[:100]:
            name_str = name if name else "(no name in catalog)"
            logger.info(f"  {fid} - {name_str}: {err}")

        sys.exit(1)


if __name__ == "__main__":
    main()