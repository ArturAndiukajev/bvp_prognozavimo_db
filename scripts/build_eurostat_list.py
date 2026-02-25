import argparse
import json
import time
from datetime import datetime, timezone, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
import eurostat
import pandas as pd
import yaml

# --------------------
# Paths / cache
# --------------------
RAW_DIR = Path("data/raw/eurostat")
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOC_PARQUET = RAW_DIR / "toc_latest.parquet"
TOC_CSV = RAW_DIR / "toc_latest.csv"

DISCOVERY_CACHE_JSON = RAW_DIR / "discovery_cache.json"   # pars/geo/freq results
MIN_DATE_CACHE_JSON = RAW_DIR / "min_date_cache.json"     # dataset_code -> min_date (LT)

AUTOLIST_YAML = Path("config/eurostat_autolist.yaml")
AUTOLIST_YAML.parent.mkdir(parents=True, exist_ok=True)

# --------------------
# Rules
# --------------------
GEO = "LT"
FREQS_NEEDED = {"M", "Q"}
CUTOFF_DATE = date(2004, 1, 1)  #dataset must have min_date <= 2004-01-01 for LT
LIMIT_FOUND_DEFAULT = 500

# --------------------
# Cache helpers
# --------------------
def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

# --------------------
# TOC caching
# --------------------
def load_or_download_toc(refresh: bool = False) -> pd.DataFrame:
    if TOC_PARQUET.exists() and not refresh:
        print(f"[TOC] using cached: {TOC_PARQUET}")
        return pd.read_parquet(TOC_PARQUET)

    print("[TOC] downloading from Eurostat ...")
    t0 = time.perf_counter()
    toc = eurostat.get_toc_df(agency="all", lang="en")
    if toc is None or toc.empty:
        raise RuntimeError("Eurostat TOC is empty")

    toc.to_parquet(TOC_PARQUET, index=False)
    toc.to_csv(TOC_CSV, index=False)
    dt = time.perf_counter() - t0
    print(f"[TOC] downloaded rows={len(toc)} in {dt:.2f}s")
    print(f"[TOC] saved: {TOC_PARQUET} and {TOC_CSV}")
    return toc

# --------------------
# Eurostat helpers
# --------------------
def _safe_get_pars(code: str) -> Optional[List[str]]:
    try:
        return eurostat.get_pars(code)
    except Exception:
        return None

def _safe_get_par_values(code: str, par: str) -> Optional[List[str]]:
    try:
        return eurostat.get_par_values(code, par)
    except Exception:
        return None

def _detect_time_columns(df: pd.DataFrame) -> List[str]:
    # eurostat.get_data_df often returns wide format with time columns like 2004M01, 2004Q1, 2004
    out = []
    for c in df.columns:
        s = str(c)
        if len(s) >= 4 and s[:4].isdigit():
            out.append(c)
    return out

def _parse_time_label_to_date(x: Any) -> Optional[date]:
    #minimal parser for common Eurostat labels
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None

    #YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        try:
            return date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
        except Exception:
            return None

    # YYYY-Qn or YYYYQn
    if "Q" in s.upper():
        ss = s.upper().replace("-", "").replace(" ", "")
        # e.g. 2004Q1
        if len(ss) == 6 and ss[:4].isdigit() and ss[4] == "Q" and ss[5].isdigit():
            y = int(ss[:4])
            q = int(ss[5])
            m = (q - 1) * 3 + 1
            return date(y, m, 1)

    # YYYY-Mmm or YYYYMmm
    if "M" in s.upper():
        ss = s.upper().replace("-", "").replace(" ", "")
        # 2004M01
        if len(ss) in (6, 7) and ss[:4].isdigit() and ss[4] == "M":
            y = int(ss[:4])
            try:
                mo = int(ss[5:])
                return date(y, mo, 1)
            except Exception:
                return None

    # YYYY-MM
    if len(s) == 7 and s[4] == "-" and s[:4].isdigit():
        try:
            return date(int(s[:4]), int(s[5:7]), 1)
        except Exception:
            return None

    #YYYY
    if len(s) == 4 and s.isdigit():
        return date(int(s), 1, 1)

    return None

# --------------------
#Phase 1: fast eligibility check (pars + geo + freq), cached
# --------------------
def check_one_dataset_fast(
    code: str,
    title: str,
    discovery_cache: dict,
) -> Optional[Dict[str, Any]]:
    """
    geo turi LT ir freq yra M/Q
    Naudoja discovery_cache
    """
    if code in discovery_cache:
        c = discovery_cache[code]
        if not c.get("ok"):
            return None
        return {"code": code, "title": title or code, "freqs": c.get("freqs", [])}

    pars = _safe_get_pars(code)
    if not pars:
        discovery_cache[code] = {"ok": False}
        return None

    pars_l = {str(p).lower() for p in pars}
    if "geo" not in pars_l or "freq" not in pars_l:
        discovery_cache[code] = {"ok": False}
        return None

    geo_vals = _safe_get_par_values(code, "geo")
    if not geo_vals or GEO not in set(map(str, geo_vals)):
        discovery_cache[code] = {"ok": False}
        return None

    freq_vals = _safe_get_par_values(code, "freq")
    if not freq_vals:
        discovery_cache[code] = {"ok": False}
        return None

    freq_set = set(map(str, freq_vals))
    ok_freqs = sorted(list(freq_set.intersection(FREQS_NEEDED)))
    if not ok_freqs:
        discovery_cache[code] = {"ok": False}
        return None

    discovery_cache[code] = {"ok": True, "freqs": ok_freqs}
    return {"code": code, "title": title or code, "freqs": ok_freqs}

# --------------------
# Phase 2: variant-2 min_date probe for LT, cached
# --------------------
def get_min_date_for_lt(
    code: str,
    min_date_cache: dict,
    hard_timeout_s: Optional[float] = None,
) -> Optional[date]:
    """
    Siuncia dataseta, skaiciuoja min data pagal time stulpeli.
    Keshuoja rezultatus i min_date_cache
    """
    if code in min_date_cache:
        v = min_date_cache[code]
        if v is None:
            return None
        try:
            return date.fromisoformat(v)
        except Exception:
            return None

    t0 = time.perf_counter()
    try:
        df = eurostat.get_data_df(code)
    except Exception:
        min_date_cache[code] = None
        return None

    if df is None or df.empty:
        min_date_cache[code] = None
        return None

    #filter geo=LT if possible
    if "geo" in df.columns:
        df = df[df["geo"].astype(str) == GEO]
    elif "GEO" in df.columns:
        df = df[df["GEO"].astype(str) == GEO]

    if df.empty:
        min_date_cache[code] = None
        return None

    time_cols = _detect_time_columns(df)
    if not time_cols:
        #kartais buna long formatai,bet eurostat.get_data_df dazniausiai wide
        min_date_cache[code] = None
        return None

    # time columns -> min date
    mind: Optional[date] = None
    for c in time_cols:
        d = _parse_time_label_to_date(c)
        if d is None:
            continue
        if mind is None or d < mind:
            mind = d

    # optional timeout check (soft)
    if hard_timeout_s is not None:
        dt = time.perf_counter() - t0
        if dt > hard_timeout_s:
            pass

    min_date_cache[code] = mind.isoformat() if mind else None
    return mind


# --------------------
# Main builder
# --------------------
def build_autolist(
    toc: pd.DataFrame,
    max_workers_fast: int = 16,
    max_workers_probe: int = 4,
    limit_found: int = LIMIT_FOUND_DEFAULT,
    print_every: int = 10,
    include_keywords: Optional[List[str]] = None,
    refresh_discovery_cache: bool = False,
) -> Dict[str, Any]:
    cols_lower = {c.lower(): c for c in toc.columns}
    code_col = cols_lower.get("code") or list(toc.columns)[0]
    type_col = cols_lower.get("type")
    title_col = cols_lower.get("title")

    df = toc.copy()
    if type_col:
        df = df[df[type_col].astype(str).str.lower() == "dataset"]

    # keyword filter to reduce candidate list BEFORE network calls
    if include_keywords and title_col:
        pat = "|".join([k.lower() for k in include_keywords])
        before = len(df)
        df = df[df[title_col].astype(str).str.lower().str.contains(pat, na=False)]
        after = len(df)
        print(f"[FILTER] keywords reduced datasets: {before} -> {after}")

    df[code_col] = df[code_col].astype(str)
    codes = df[code_col].dropna().unique().tolist()
    total = len(codes)
    print(f"[DISCOVERY] candidates from TOC: {total}")

    # title lookup map
    title_map: Dict[str, str] = {}
    if title_col:
        for r in df[[code_col, title_col]].itertuples(index=False, name=None):
            title_map[str(r[0])] = str(r[1])
    else:
        for c in codes:
            title_map[str(c)] = str(c)

    # load caches
    discovery_cache = {} if refresh_discovery_cache else _load_json(DISCOVERY_CACHE_JSON)
    min_date_cache = _load_json(MIN_DATE_CACHE_JSON)

    # -------------------------
    # Phase 1: fast check (parallel, chunked)
    # -------------------------
    t0 = time.perf_counter()
    checked = 0
    fast_ok: List[Dict[str, Any]] = []

    print(f"[PHASE1] fast check geo={GEO}, freqs={sorted(FREQS_NEEDED)} with workers={max_workers_fast}")

    with ThreadPoolExecutor(max_workers=max_workers_fast) as ex:
        it = iter(codes)
        in_flight = {}

        #buffer
        buffer_n = max_workers_fast * 5

        for _ in range(buffer_n):
            try:
                code = next(it)
            except StopIteration:
                break
            in_flight[ex.submit(check_one_dataset_fast, code, title_map.get(code, code), discovery_cache)] = code

        while in_flight:
            for fut in as_completed(list(in_flight.keys())):
                code = in_flight.pop(fut)
                checked += 1

                try:
                    res = fut.result()
                except Exception:
                    res = None

                if res:
                    fast_ok.append(res)

                #refill
                try:
                    nxt = next(it)
                    in_flight[ex.submit(check_one_dataset_fast, nxt, title_map.get(nxt, nxt), discovery_cache)] = nxt
                except StopIteration:
                    pass

                if checked % print_every == 0:
                    dt = time.perf_counter() - t0
                    rate = checked / dt if dt > 0 else 0
                    print(f"[PHASE1] checked {checked}/{total} | fast_ok={len(fast_ok)} | {rate:.1f} ds/s | last={code}")

            # break outer while only when in_flight empty

    dt = time.perf_counter() - t0
    print(f"[PHASE1] DONE checked={checked} fast_ok={len(fast_ok)} time={dt:.1f}s")

    # save discovery cache (so next run is much faster)
    _save_json(DISCOVERY_CACHE_JSON, discovery_cache)
    print(f"[CACHE] saved discovery cache: {DISCOVERY_CACHE_JSON} entries={len(discovery_cache)}")

    # -------------------------
    # Phase 2: min_date probe
    # -------------------------
    print(f"[PHASE2] probing min_date for LT (keep if min_date <= {CUTOFF_DATE.isoformat()}) with workers={max_workers_probe}")

    accepted: List[Dict[str, Any]] = []
    probed = 0
    t1 = time.perf_counter()

    # shuffle-like stable order: by code to be deterministic
    fast_ok_sorted = sorted(fast_ok, key=lambda x: x["code"])

    # chunked probing too
    with ThreadPoolExecutor(max_workers=max_workers_probe) as ex:
        it2 = iter(fast_ok_sorted)
        in_flight2 = {}

        buffer_n2 = max_workers_probe * 2
        for _ in range(buffer_n2):
            try:
                item = next(it2)
            except StopIteration:
                break
            code = item["code"]
            in_flight2[ex.submit(get_min_date_for_lt, code, min_date_cache)] = item

        while in_flight2 and len(accepted) < limit_found:
            for fut in as_completed(list(in_flight2.keys())):
                item = in_flight2.pop(fut)
                code = item["code"]
                probed += 1

                try:
                    mind = fut.result()
                except Exception:
                    mind = None

                keep = (mind is not None and mind <= CUTOFF_DATE)
                if keep:
                    item["min_date"] = mind.isoformat()
                    accepted.append(item)

                if probed % max(20, print_every // 2) == 0:
                    dtp = time.perf_counter() - t1
                    rate = probed / dtp if dtp > 0 else 0
                    print(f"[PHASE2] probed={probed}/{len(fast_ok_sorted)} | accepted={len(accepted)} | {rate:.2f} ds/s | last={code} mind={mind}")

                # refill
                if len(accepted) >= limit_found:
                    break
                try:
                    nxt_item = next(it2)
                    nxt_code = nxt_item["code"]
                    in_flight2[ex.submit(get_min_date_for_lt, nxt_code, min_date_cache)] = nxt_item
                except StopIteration:
                    pass

                if len(accepted) >= limit_found:
                    break

    dt2 = time.perf_counter() - t1
    print(f"[PHASE2] DONE probed={probed} accepted={len(accepted)} time={dt2:.1f}s")

    # save min_date cache
    _save_json(MIN_DATE_CACHE_JSON, min_date_cache)
    print(f"[CACHE] saved min_date cache: {MIN_DATE_CACHE_JSON} entries={len(min_date_cache)}")

    # build YAML
    out_yaml = {
        "eurostat": {
            item["code"]: {
                "name": item["title"],
                "filters": {"geo": [GEO]},
                "auto": {"freqs": item.get("freqs", []), "min_date": CUTOFF_DATE.isoformat()},
                "discovery": {"min_date_lt": item.get("min_date")},
            }
            for item in accepted
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "geo": GEO,
            "freqs": sorted(list(FREQS_NEEDED)),
            "variant": "v2_min_date_le_cutoff",
            "cutoff_date": CUTOFF_DATE.isoformat(),
            "limit_found": limit_found,
            "keywords": include_keywords or [],
        },
    }
    return out_yaml


def save_autolist_yaml(obj: Dict[str, Any], path: Path = AUTOLIST_YAML) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[SAVE] autolist saved to {path} (datasets={len(obj.get('eurostat', {}))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-toc", action="store_true", help="Force re-download TOC from Eurostat")
    ap.add_argument("--workers-fast", type=int, default=16, help="Workers for fast metadata checks")
    ap.add_argument("--workers-probe", type=int, default=4, help="Workers for min_date probing")
    ap.add_argument("--limit-found", type=int, default=LIMIT_FOUND_DEFAULT, help="Stop after accepting N datasets")
    ap.add_argument("--print-every", type=int, default=100, help="Print progress every N checks")
    ap.add_argument("--refresh-discovery-cache", action="store_true", help="Ignore cached metadata and recompute")
    ap.add_argument("--keywords", type=str, default="", help="Comma-separated keywords to prefilter TOC by title")
    args = ap.parse_args()

    include_keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or [
        #makro temos pagal nutylejima, galima plesti
        "gdp", "hicp", "ppi", "unemployment", "employment", "wage", "earn",
        "retail", "industrial", "production", "trade", "export", "import",
        "confidence", "sentiment", "housing", "price", "construction"
    ]

    toc = load_or_download_toc(refresh=args.refresh_toc)

    out_yaml = build_autolist(
        toc=toc,
        max_workers_fast=max(1, args.workers_fast),
        max_workers_probe=max(1, args.workers_probe),
        limit_found=max(1, args.limit_found),
        print_every=max(10, args.print_every),
        include_keywords=include_keywords,
        refresh_discovery_cache=args.refresh_discovery_cache,
    )
    save_autolist_yaml(out_yaml)


if __name__ == "__main__":
    main()