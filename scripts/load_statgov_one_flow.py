import argparse
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone, date
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
import requests
from sqlalchemy import create_engine, text

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("statgov_loader")

# ----------------------------
# DB
# ----------------------------
DB_URL = os.environ.get("DB_URL") or "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

# ----------------------------
# OSP endpoints
# ----------------------------
BASE_JSON = "https://osp-rs.stat.gov.lt/rest_json"
BASE_XML = "https://osp-rs.stat.gov.lt/rest_xml"

# SDMX 2.1 XML namespaces
NS = {
    "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}

# ----------------------------
# Time parsing (robust)
# ----------------------------
_RE_Y = re.compile(r"^(\d{4})$")
_RE_YQ = re.compile(r"^(\d{4})-Q([1-4])$")
_RE_YM = re.compile(r"^(\d{4})-(\d{2})$")
_RE_YMD = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_RE_YM2 = re.compile(r"^(\d{4})M(\d{1,2})$")
_RE_YM3 = re.compile(r"^(\d{4})(\d{2})$")
_RE_YQ2 = re.compile(r"^(\d{4})Q([1-4])$")
_RE_YW = re.compile(r"^(\d{4})-W(\d{2})$")
_RE_YW2 = re.compile(r"^(\d{4})W(\d{2})$")
# NEW: School-year (academic year) formats: 2023-2024 / 2023–2024
_RE_SY = re.compile(r"^(\d{4})[-–](\d{4})$")

# NEW: Semi-annual (half-year) formats: 2017-S1 / 2017S1
_RE_YS = re.compile(r"^(\d{4})-S([1-2])$", re.I)
_RE_YS2 = re.compile(r"^(\d{4})S([1-2])$", re.I)
_RE_YP = re.compile(r"^(\d{4})-P([1-2])$", re.I)   # 2024-P1
_RE_YP2 = re.compile(r"^(\d{4})P([1-2])$", re.I)   # 2024P1

# NEW: Lithuanian quarter format used by StatGov UI/JSON: 2010K1, 1998K2, etc.
_RE_YQ_LT = re.compile(r"^(\d{4})K([1-4])$", re.I)


def to_period_date(tp: Any) -> date:
    s = str(tp).strip()

    # NEW: School year like 2023-2024 / 2023–2024 -> start date 2023-09-01
    m = _RE_SY.match(s)
    if m:
        y1 = int(m.group(1))
        # y2 is not strictly needed, but kept for readability/validation if you want later
        # y2 = int(m.group(2))
        return date(y1, 9, 1)
    
    m = _RE_YMD.match(s)
    if m:
        y, mo, d = map(int, m.groups())
        return date(y, mo, d)

    # NEW: Semi-annual (half-year) YYYY-S1 / YYYYS1
    m = _RE_YS.match(s) or _RE_YS2.match(s)
    if m:
        y = int(m.group(1))
        h = int(m.group(2))
        month = 1 if h == 1 else 7
        return date(y, month, 1)

    # NEW: LT quarter "2010K1"
    m = _RE_YQ_LT.match(s)
    if m:
        y = int(m.group(1))
        q = int(m.group(2))
        month = (q - 1) * 3 + 1
        return date(y, month, 1)

    m = _RE_YQ.match(s) or _RE_YQ2.match(s)
    if m:
        y = int(m.group(1))
        q = int(m.group(2))
        month = (q - 1) * 3 + 1
        return date(y, month, 1)

    m = _RE_YM.match(s)
    if m:
        y, mo = map(int, m.groups())
        return date(y, mo, 1)

    m = _RE_YM2.match(s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        return date(y, mo, 1)

    m = _RE_YM3.match(s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        if 1 <= mo <= 12:
            return date(y, mo, 1)

    m = _RE_YW.match(s) or _RE_YW2.match(s)
    if m:
        y = int(m.group(1))
        w = int(m.group(2))
        return date.fromisocalendar(y, w, 1)

    m = _RE_Y.match(s)
    if m:
        return date(int(m.group(1)), 1, 1)
    
    m = _RE_YP.match(s) or _RE_YP2.match(s)
    if m:
        y = int(m.group(1))
        h = int(m.group(2))          # P1 / P2
        month = 1 if h == 1 else 7
        return date(y, month, 1)

    raise ValueError(f"Unsupported TIME_PERIOD format: {tp!r}")


def infer_frequency_from_time(tp: Any) -> str:
    s = str(tp).strip()
    if _RE_SY.match(s):
        return "SY"
    if _RE_YMD.match(s):
        return "D"
    if _RE_YW.match(s) or _RE_YW2.match(s):
        return "W"
    if _RE_YS.match(s) or _RE_YS2.match(s):
        return "S"
    if _RE_YP.match(s) or _RE_YP2.match(s):
        return "S"   # semi-annual
    if _RE_YQ_LT.match(s):
        return "Q"
    if _RE_YQ.match(s) or _RE_YQ2.match(s):
        return "Q"
    if _RE_YM.match(s) or _RE_YM2.match(s) or (_RE_YM3.match(s) and 1 <= int(_RE_YM3.match(s).group(2)) <= 12):
        return "M"
    if _RE_Y.match(s):
        return "A"
    return "M"


# ----------------------------
# NEW: dataset attribute decoding (DS_TIME_FORMAT -> frequency)
# ----------------------------
def _decode_dataset_attributes(struct: dict, ds0: dict) -> Dict[str, Any]:
    """
    SDMX-JSON: dataSets[0].attributes is an array of indexes.
    structure.attributes.dataSet defines ids and possible values.
    We decode it into {attr_id: value_id_or_name}.
    """
    attrs_def = ((struct.get("attributes") or {}).get("dataSet") or [])
    ds_attrs = ds0.get("attributes")

    if not attrs_def or ds_attrs is None:
        return {}

    if not isinstance(ds_attrs, list):
        return {}

    out: Dict[str, Any] = {}
    for j, adef in enumerate(attrs_def):
        aid = adef.get("id")
        if not aid:
            continue
        if j >= len(ds_attrs):
            continue

        raw_idx = ds_attrs[j]
        if raw_idx is None:
            out[str(aid)] = None
            continue

        try:
            idx = int(raw_idx)
        except Exception:
            out[str(aid)] = None
            continue

        vals = adef.get("values") or []
        if vals and 0 <= idx < len(vals):
            out[str(aid)] = {
                "id": vals[idx].get("id"),
                "name": vals[idx].get("name"),
            }
        else:
            out[str(aid)] = {"id": str(idx), "name": None}

    return out


def infer_frequency_from_dataset(struct: dict, ds0: dict, df: pd.DataFrame, time_dim: str) -> str:
    """
    Prefer DS_TIME_FORMAT (Periodiškumas -> Metai/Ketvirtis/...).
    Fallback to parsing TIME_PERIOD format.
    """
    ds_attrs = _decode_dataset_attributes(struct, ds0)
    tf = ds_attrs.get("DS_TIME_FORMAT")

    if isinstance(tf, dict):
        name = (tf.get("name") or "").strip().lower()
        vid = str(tf.get("id") or "").strip().lower()

        # NEW: School year / academic year ("Mokymo metai", "Mokslo metai")
        if (
            "mokymo met" in name
            or "mokslo met" in name
            or "school year" in name
            or vid in {"4", "sy"}
        ):
            return "SY"
        if "met" in name or vid in {"3", "a", "y", "year"}:
            return "A"
        if "ketv" in name or vid in {"1", "q", "quarter"}:
            return "Q"

        # Semi-annual / half-year:
        # StatGov sometimes uses "P" (pusmetis) and TIME_PERIOD like 2024P1 / 2024P2
        if (
            "pusm" in name
            or "pusmėt" in name
            or "semi" in name
            or "half" in name
            or "semester" in name
            or vid in {
                "5", "s", "h", "p", "p1", "p2",
                "half-year", "halfyear", "semi-annual", "semiannual", "semester"
            }
        ):
            return "S"

        if "mėn" in name or "men" in name or vid in {"2", "m", "month"}:
            return "M"
        if "savait" in name or vid in {"w", "week"}:
            return "W"
        if "dien" in name or vid in {"d", "day"}:
            return "D"

    if time_dim in df.columns and not df.empty:
        return infer_frequency_from_time(df[time_dim].iloc[0])

    return "M"


# ----------------------------
# XML helpers
# ----------------------------
def _pick_lang(elements, lang: str = "en") -> Optional[str]:
    if not elements:
        return None
    for el in elements:
        if el.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == lang:
            return el.text
    return elements[0].text


def fetch_dataflow_catalog_xml() -> bytes:
    r = requests.get(f"{BASE_XML}/dataflow/", timeout=90)
    r.raise_for_status()
    return r.content


def find_agency_for_flow(flow_id: str) -> str:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(fetch_dataflow_catalog_xml())
    for df in root.findall(".//str:Dataflow", NS):
        if df.attrib.get("id") == flow_id:
            agency = df.attrib.get("agencyID")
            if agency:
                return agency
    raise ValueError(f"Could not find agencyID for flow_id={flow_id}")


def fetch_dataflow_xml(agency: str, flow_id: str) -> bytes:
    url = f"{BASE_XML}/dataflow/{agency}/{flow_id}"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content


def parse_dataflow_title_desc(df_xml: bytes) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(df_xml)
    name_els = root.findall(".//str:Dataflow//com:Name", NS)
    desc_els = root.findall(".//str:Dataflow//com:Description", NS)
    title_en = _pick_lang(name_els, "en")
    desc_en = _pick_lang(desc_els, "en")
    title_lt = _pick_lang(name_els, "lt")
    desc_lt = _pick_lang(desc_els, "lt")
    return title_en, desc_en, title_lt, desc_lt


def parse_datastructure_ref_from_dataflow_xml(df_xml: bytes) -> tuple[Optional[str], Optional[str], Optional[str]]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(df_xml)
    ref = root.find(".//str:Dataflow//str:Structure//Ref", NS)
    if ref is None:
        ref = root.find(".//str:Structure//Ref", NS)
    if ref is None:
        return None, None, None
    return ref.attrib.get("agencyID"), ref.attrib.get("id"), ref.attrib.get("version")


def fetch_datastructure_xml(agency: str, resource_id: str) -> Optional[bytes]:
    candidates = [
        f"{BASE_XML}/datastructure/{agency}/{resource_id}/latest",
        f"{BASE_XML}/datastructure/{agency}/{resource_id}",
        f"{BASE_XML}/datastructure/{agency.lower()}/{resource_id}/latest",
        f"{BASE_XML}/datastructure/{agency.lower()}/{resource_id}",
    ]
    for url in candidates:
        r = requests.get(url, timeout=90)
        if r.status_code == 200:
            return r.content
        if r.status_code not in (404, 400):
            r.raise_for_status()
    return None


def parse_datastructure_codelists(ds_xml: bytes) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(ds_xml)

    codelists: Dict[str, List[str]] = {}
    for cl in root.findall(".//str:Codelists//str:Codelist", NS):
        cl_id = cl.attrib.get("id")
        codes = [c.attrib.get("id") for c in cl.findall("./str:Code", NS)]
        if cl_id:
            codelists[cl_id] = [c for c in codes if c is not None]

    dim2cl: Dict[str, str] = {}
    dim_nodes = root.findall(".//str:DimensionList//str:Dimension", NS) + root.findall(".//str:TimeDimension", NS)
    for dim in dim_nodes:
        dim_id = dim.attrib.get("id")
        ref = dim.find(".//str:LocalRepresentation//str:Enumeration//Ref", NS)
        if dim_id and ref is not None:
            cl_id = ref.attrib.get("id")
            if cl_id:
                dim2cl[dim_id] = cl_id

    return codelists, dim2cl


# ----------------------------
# JSON fetch
# ----------------------------
def fetch_json_data(
    flow_id: str,
    start: str | None = None,
    end: str | None = None,
    lang: str | None = None,
) -> Tuple[dict, bytes, str]:
    url = f"{BASE_JSON}/data/{flow_id}"
    params = {}
    if lang:
        params["lang"] = lang
    if start:
        params["startPeriod"] = start
    if end:
        params["endPeriod"] = end

    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    raw = r.content
    h = hashlib.sha256(raw).hexdigest()
    return r.json(), raw, h


def fetch_json_data_en_first(flow_id: str, start: str | None = None, end: str | None = None) -> Tuple[dict, bytes, str, str]:
    """Fetch StatGov SDMX-JSON preferring English labels, with safe fallback.

    Some flows intermittently return HTTP 5xx when a `lang` parameter is used.
    We treat those as recoverable and fall back to LT / default.
    """

    def _try(lang: str | None) -> Optional[Tuple[dict, bytes, str, str]]:
        try:
            sdmx, raw, h = fetch_json_data(flow_id, start=start, end=end, lang=lang)
            chosen = "default" if lang is None else lang
            return sdmx, raw, h, chosen
        except requests.exceptions.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            logger.warning(f"StatGov request failed for flow={flow_id} lang={lang!r} status={status}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"StatGov request failed for flow={flow_id} lang={lang!r}: {e}")
            return None

    # 1) English
    res = _try("en")
    if res is not None:
        sdmx, raw, h, chosen = res
        name = ((sdmx.get("structure") or {}).get("name") or "").strip() if isinstance((sdmx.get("structure") or {}).get("name"), str) else ""
        if name:
            return sdmx, raw, h, chosen

    # 2) Lithuanian
    res = _try("lt")
    if res is not None:
        sdmx, raw, h, chosen = res
        name = ((sdmx.get("structure") or {}).get("name") or "").strip() if isinstance((sdmx.get("structure") or {}).get("name"), str) else ""
        if name:
            return sdmx, raw, h, chosen

    # 3) Default (no lang)
    res = _try(None)
    if res is not None:
        sdmx, raw, h, chosen = res
        return sdmx, raw, h, chosen

    # If all attempts failed, re-raise using a plain request (will raise)
    sdmx, raw, h = fetch_json_data(flow_id, start=start, end=end, lang=None)
    return sdmx, raw, h, "default"


# ----------------------------
# SDMX JSON decoding (supports both layouts)
# ----------------------------
def _dimension_values_from_defs(
    dim_defs: List[dict],
    codelists: Dict[str, List[str]],
    dim2cl: Dict[str, str],
) -> Tuple[List[str], List[List[str]]]:
    if not dim_defs:
        return [], []

    dim_ids: List[str] = []
    dim_values: List[List[str]] = []

    for d in dim_defs:
        dim_id = d.get("id")
        if not dim_id:
            raise ValueError("Some dimension IDs are missing in SDMX structure.")
        dim_id = str(dim_id)
        dim_ids.append(dim_id)

        vals = d.get("values") or []
        if vals:
            dim_values.append([str(v.get("id")) for v in vals])
            continue

        cl_id = dim2cl.get(dim_id)
        if cl_id and cl_id in codelists and codelists[cl_id]:
            dim_values.append([str(x) for x in codelists[cl_id]])
            continue

        raise ValueError(f"Dimension '{dim_id}' has no values in JSON and no codelist fallback.")

    return dim_ids, dim_values


def _decode_obs_attributes(struct: dict, arr: list) -> Dict[str, Any]:
    attrs = ((struct.get("attributes") or {}).get("observation") or [])
    if not attrs or not isinstance(arr, list) or len(arr) <= 1:
        return {}

    out: Dict[str, Any] = {}
    for j, a in enumerate(attrs):
        aid = a.get("id")
        if not aid:
            continue
        pos = j + 1
        if pos >= len(arr):
            break
        raw_idx = arr[pos]
        if raw_idx is None:
            out[str(aid)] = None
            continue
        try:
            raw_idx_int = int(raw_idx)
        except Exception:
            out[str(aid)] = None
            continue

        vals = a.get("values") or []
        if vals and 0 <= raw_idx_int < len(vals):
            out[str(aid)] = vals[raw_idx_int].get("id")
        else:
            out[str(aid)] = str(raw_idx_int)

    return out


def decode_sdmx_to_dataframe_general(
    sdmx: dict,
    codelists: Optional[Dict[str, List[str]]] = None,
    dim2cl: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    codelists = codelists or {}
    dim2cl = dim2cl or {}

    struct = sdmx.get("structure") or {}
    dims_obj = struct.get("dimensions") or {}
    dim_defs_obs = dims_obj.get("observation") or []
    dim_defs_ser = dims_obj.get("series") or []

    ds0 = (sdmx.get("dataSets") or [{}])[0]

    rows: List[Dict[str, Any]] = []

    # Case B: series layout (kept)
    if ds0.get("series"):
        series = ds0.get("series") or {}
        ser_dim_ids, ser_dim_values = _dimension_values_from_defs(dim_defs_ser, codelists, dim2cl)
        obs_dim_ids, obs_dim_values = _dimension_values_from_defs(dim_defs_obs, codelists, dim2cl)

        for ser_key, ser_obj in series.items():
            ser_idx = [int(x) for x in ser_key.split(":")] if ser_key else []
            ser_dim_map = {ser_dim_ids[i]: ser_dim_values[i][ser_idx[i]] for i in range(len(ser_idx))}

            obs = ser_obj.get("observations") or {}
            for obs_key, arr in obs.items():
                if obs_key is None:
                    continue

                parts = str(obs_key).split(":")
                try:
                    obs_idx = [int(x) for x in parts]
                except Exception:
                    continue

                if len(obs_idx) != len(obs_dim_ids):
                    if len(obs_idx) > len(obs_dim_ids):
                        obs_idx = obs_idx[-len(obs_dim_ids):]
                    else:
                        continue

                obs_dim_map: Dict[str, Any] = {}
                for i, dim_id in enumerate(obs_dim_ids):
                    obs_dim_map[dim_id] = obs_dim_values[i][obs_idx[i]]

                val = arr[0] if isinstance(arr, list) and arr else None

                row = {"value": val}
                row.update(ser_dim_map)
                row.update(obs_dim_map)
                row.update(_decode_obs_attributes(struct, arr))
                rows.append(row)

        return pd.DataFrame(rows)

    # Case A: flat layout
    obs = ds0.get("observations") or {}
    obs_dim_ids, obs_dim_values = _dimension_values_from_defs(dim_defs_obs, codelists, dim2cl)

    for key, arr in obs.items():
        if not key:
            continue

        parts = str(key).split(":")
        try:
            idx_all = [int(x) for x in parts]
        except Exception:
            continue

        n_dims = len(obs_dim_ids)

        if len(idx_all) > n_dims:
            idx = idx_all[-n_dims:]
        elif len(idx_all) < n_dims:
            continue
        else:
            idx = idx_all

        row: Dict[str, Any] = {}
        for i, dim_id in enumerate(obs_dim_ids):
            row[dim_id] = obs_dim_values[i][idx[i]]

        val = arr[0] if isinstance(arr, list) and arr else None
        row["value"] = val
        row.update(_decode_obs_attributes(struct, arr))
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Guess helpers (time/unit/geo)
# ----------------------------
def detect_time_dim(struct: dict, df: pd.DataFrame) -> str:
    time_dim = None
    dims_obj = (struct.get("dimensions") or {})
    for d in (dims_obj.get("observation") or []) + (dims_obj.get("series") or []):
        if d.get("role") == "time" or str(d.get("id", "")).upper() == "TIME_PERIOD":
            time_dim = d.get("id")
            break
    if time_dim and time_dim in df.columns:
        return str(time_dim)
    for cand in ["TIME_PERIOD", "TIME", "PERIOD", "LAIKOTARPIS"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not detect time dimension (TIME_PERIOD).")


def guess_geo_dim(columns: list[str]) -> Optional[str]:
    for cand in ["REF_AREA", "GEO", "REGION", "SAVIVALDYBE", "savivaldybe", "savivaldybesRegdb"]:
        if cand in columns:
            return cand
    return None


def guess_unit_dim(columns: list[str]) -> Optional[str]:
    for cand in ["UNIT_MEASURE", "UNIT", "MATVNT"]:
        if cand in columns:
            return cand
    return None


# ----------------------------
# DB upserts (unchanged)
# ----------------------------
def ensure_provider(conn, name: str, base_url: str | None = None, meta: dict | None = None) -> int:
    r = conn.execute(text("SELECT id FROM providers WHERE name=:n"), {"n": name}).fetchone()
    if r:
        return int(r[0])
    r = conn.execute(
        text(
            """
        INSERT INTO providers (name, base_url, meta)
        VALUES (:n, :u, CAST(:m AS jsonb))
        RETURNING id
        """
        ),
        {"n": name, "u": base_url, "m": json.dumps(meta or {}, ensure_ascii=False)},
    ).fetchone()
    return int(r[0])


def ensure_dataset(conn, provider_id: int, key: str, title: str, description: str | None, meta: dict | None) -> int:
    r = conn.execute(text("SELECT id FROM datasets WHERE provider_id=:pid AND key=:k"), {"pid": provider_id, "k": key}).fetchone()
    if r:
        did = int(r[0])
        conn.execute(
            text(
                """
            UPDATE datasets
            SET title=:t, description=:d, meta=COALESCE(meta,'{}'::jsonb) || CAST(:m AS jsonb)
            WHERE id=:id
            """
            ),
            {"t": title, "d": description, "m": json.dumps(meta or {}, ensure_ascii=False), "id": did},
        )
        return did

    r = conn.execute(
        text(
            """
        INSERT INTO datasets (provider_id, key, title, description, meta)
        VALUES (:pid, :k, :t, :d, CAST(:m AS jsonb))
        RETURNING id
        """
        ),
        {"pid": provider_id, "k": key, "t": title, "d": description, "m": json.dumps(meta or {}, ensure_ascii=False)},
    ).fetchone()
    return int(r[0])


def create_release(conn, dataset_id: int, downloaded_at: datetime, vintage_at: datetime, payload_hash: str, raw_path: str | None) -> int:
    r = conn.execute(
        text(
            """
        INSERT INTO releases (
            dataset_id,
            release_time,
            downloaded_at,
            vintage_at,
            raw_path,
            content_hash,
            meta
        )
        VALUES (
            :did,
            :rt,
            :dl,
            :vin,
            :p,
            :h,
            '{}'::jsonb
        )
        RETURNING id
        """
        ),
        {
            "did": dataset_id,
            "rt": downloaded_at,
            "dl": downloaded_at,
            "vin": vintage_at,
            "h": payload_hash,
            "p": raw_path,
        },
    ).fetchone()
    return int(r[0])


def ensure_series(
    conn,
    dataset_id: int,
    key: str,
    country: str,
    frequency: str,
    transform: str,
    unit: str | None,
    name: str | None,
    meta: dict | None,
) -> int:
    r = conn.execute(
        text(
            """
        SELECT id FROM series
        WHERE dataset_id=:did AND key=:k AND country=:c AND frequency=:f AND transform=:tr
        """
        ),
        {"did": dataset_id, "k": key, "c": country, "f": frequency, "tr": transform},
    ).fetchone()
    if r:
        sid = int(r[0])
        conn.execute(
            text(
                """
            UPDATE series
            SET unit=:u, name=:n, meta=COALESCE(meta,'{}'::jsonb) || CAST(:m AS jsonb)
            WHERE id=:id
            """
            ),
            {"u": unit, "n": name, "m": json.dumps(meta or {}, ensure_ascii=False), "id": sid},
        )
        return sid

    r = conn.execute(
        text(
            """
        INSERT INTO series (dataset_id, key, country, frequency, transform, unit, name, meta)
        VALUES (:did, :k, :c, :f, :tr, :u, :n, CAST(:m AS jsonb))
        RETURNING id
        """
        ),
        {"did": dataset_id, "k": key, "c": country, "f": frequency, "tr": transform, "u": unit, "n": name, "m": json.dumps(meta or {}, ensure_ascii=False)},
    ).fetchone()
    return int(r[0])


# ----------------------------
# Main ingest (ONLY necessary change: freq inference)
# ----------------------------
def ingest_one_flow(flow_id: str, start: str | None = None, end: str | None = None, debug_title: bool = False) -> None:
    downloaded_at = datetime.now(timezone.utc).replace(microsecond=0)
    observed_at = downloaded_at
    vintage_at = observed_at

    ZERO_STATUSES = {"dydis0"}

    logger.info(f"Fetching JSON for flow={flow_id}")
    sdmx, raw_bytes, payload_hash, chosen_lang = fetch_json_data_en_first(flow_id, start=start, end=end)

    struct = sdmx.get("structure") or {}
    ds0 = (sdmx.get("dataSets") or [{}])[0]  # needed for DS_TIME_FORMAT

    dim_value_labels: Dict[str, Dict[str, str]] = {}
    try:
        dims_obj = (struct.get("dimensions") or {})
        all_defs = (dims_obj.get("series") or []) + (dims_obj.get("observation") or [])
        for d in all_defs:
            dim_id = d.get("id")
            if not dim_id:
                continue
            vals = d.get("values") or []
            m: Dict[str, str] = {}
            for v in vals:
                vid = v.get("id")
                vname = v.get("name")
                if isinstance(vid, str) and vid and isinstance(vname, str) and vname:
                    m[vid] = vname
            if m:
                dim_value_labels[str(dim_id)] = m
    except Exception:
        dim_value_labels = {}

    json_name = struct.get("name")
    json_desc = struct.get("description")

    agency = find_agency_for_flow(flow_id)
    df_xml = fetch_dataflow_xml(agency, flow_id)
    title_en, desc_en, title_lt, desc_lt = parse_dataflow_title_desc(df_xml)

    if debug_title:
        logger.info(f"JSON structure.name={json_name!r}")
        logger.info(f"JSON structure.description={json_desc!r}")
        logger.info(f"Dataflow title_en={title_en!r}, title_lt={title_lt!r}")
        logger.info(f"Dataflow desc_en={desc_en!r}, desc_lt={desc_lt!r}")

    dataset_title = (json_name if isinstance(json_name, str) and json_name.strip() else None) or title_en or title_lt or f"OSP flow {flow_id}"
    dataset_desc = (json_desc if isinstance(json_desc, str) and json_desc.strip() else None) or desc_en or desc_lt

    dsd_ag, dsd_id, dsd_ver = parse_datastructure_ref_from_dataflow_xml(df_xml)
    codelists: Dict[str, List[str]] = {}
    dim2cl: Dict[str, str] = {}
    if dsd_id:
        ds_xml = fetch_datastructure_xml(dsd_ag or agency, dsd_id)
        if ds_xml:
            try:
                codelists, dim2cl = parse_datastructure_codelists(ds_xml)
            except Exception as e:
                logger.warning(f"Could not parse datastructure codelists: {e}")

    df = decode_sdmx_to_dataframe_general(sdmx, codelists=codelists, dim2cl=dim2cl)
    if df.empty:
        logger.warning("No decoded observations. Nothing to ingest.")
        return

    time_dim = detect_time_dim(struct, df)
    df["period_date"] = df[time_dim].apply(to_period_date)

    # Infer frequency from DS_TIME_FORMAT dataset attribute, fallback to TIME_PERIOD parsing
    freq = infer_frequency_from_dataset(struct, ds0, df, time_dim)

    geo_dim = guess_geo_dim(list(df.columns))
    unit_dim = guess_unit_dim(list(df.columns))

    obs_attr_ids = {str(a["id"]) for a in ((struct.get("attributes") or {}).get("observation") or []) if a.get("id")}

    non_series_cols = {"value", "period_date", time_dim}
    dims_for_series = [c for c in df.columns if c not in non_series_cols and c not in obs_attr_ids]
    dim_keys = sorted(dims_for_series)

    raw_dir = os.environ.get("STATGOV_RAW_DIR")
    raw_path = None
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{flow_id}_{payload_hash}.json")
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

    inserted, failed = 0, 0
    series_cache: Dict[tuple, int] = {}

    discarded_missing_by_series: Dict[str, Dict[str, int]] = {}

    provider_name = "stat_gov"

    dataset_meta = {
        "flow_id": flow_id,
        "agency": agency,
        "title_en": title_en,
        "title_lt": title_lt,
        "desc_en": desc_en,
        "desc_lt": desc_lt,
        "json_structure_name": json_name,
        "json_structure_description": json_desc,
        "datastructure": {"agency": dsd_ag, "id": dsd_id, "version": dsd_ver},
        "dim2codelist": dim2cl,
        "payload_hash": payload_hash,
        "lang": chosen_lang,
        "ds_attrs": _decode_dataset_attributes(struct, ds0),
    }

    with engine.begin() as conn:
        provider_id = ensure_provider(conn, provider_name, base_url="https://osp-rs.stat.gov.lt/", meta={"api": "osp"})
        dataset_id = ensure_dataset(conn, provider_id, key=flow_id, title=dataset_title, description=dataset_desc, meta=dataset_meta)
        release_id = create_release(conn, dataset_id, downloaded_at, vintage_at, payload_hash, raw_path)

        for _, row in df.iterrows():
            try:
                v = row.get("value")
                obs_status = row.get("OBS_STATUS")

                dims = {k: row.get(k) for k in dim_keys}
                parts = []
                parts_for_name = []
                meta_dims = {}
                for k in dim_keys:
                    dv = dims.get(k)
                    if pd.isna(dv):
                        dv = None
                    if dv is not None:
                        dv = str(dv)
                    meta_dims[k] = dv
                    parts.append(f"{k}={dv}")

                    dv_disp = None if dv is None else dim_value_labels.get(k, {}).get(dv, dv)
                    parts_for_name.append(f"{k}={dv_disp}")

                series_key = f"{flow_id}|" + "|".join(parts) if parts else flow_id

                def _note_discard(reason: str):
                    bucket = discarded_missing_by_series.setdefault(series_key, {})
                    bucket[reason] = bucket.get(reason, 0) + 1

                if isinstance(v, str):
                    v_str = v.strip()
                    if v_str == "" or v_str == "-":
                        _note_discard("blank_or_dash")
                        v = None
                    else:
                        try:
                            v = float(v_str.replace(",", "."))
                        except Exception:
                            _note_discard("unparseable_string")
                            v = None

                if pd.isna(v):
                    if isinstance(obs_status, str) and obs_status in ZERO_STATUSES:
                        v = 0.0
                    else:
                        if obs_status is None or (isinstance(obs_status, float) and pd.isna(obs_status)) or obs_status == "":
                            _note_discard("missing_value_no_status")
                        else:
                            _note_discard(f"missing_value_status={obs_status}")
                        continue

                try:
                    v = float(v)
                except Exception:
                    _note_discard("non_numeric_after_normalization")
                    continue

                country = "LT"
                if geo_dim and geo_dim in dims and pd.notna(dims[geo_dim]):
                    country = str(dims[geo_dim])

                unit_val = None
                if unit_dim and unit_dim in dims and pd.notna(dims[unit_dim]):
                    unit_val = str(dims[unit_dim])

                cache_key = (series_key, country, freq, "LEVEL")
                if cache_key not in series_cache:
                    series_name = dataset_title if not parts_for_name else f"{dataset_title} | " + ", ".join(parts_for_name)
                    sid = ensure_series(
                        conn,
                        dataset_id,
                        key=series_key,
                        country=country,
                        frequency=freq,
                        transform="LEVEL",
                        unit=unit_val,
                        name=series_name,
                        meta={"dimensions": meta_dims, "time_dim": time_dim, "flow_id": flow_id},
                    )
                    series_cache[cache_key] = sid

                series_id = series_cache[cache_key]

                status = None
                meta = {}
                for aid in obs_attr_ids:
                    av = row.get(aid)
                    if pd.isna(av):
                        continue
                    if aid == "OBS_STATUS":
                        status = str(av)
                    else:
                        meta[aid] = str(av)

                conn.execute(
                    text(
                        """
                    INSERT INTO observations (series_id, period_date, observed_at, value, status, meta, release_id)
                    VALUES (:sid, :pdate, :oat, :val, :status, CAST(:meta AS jsonb), :rid)
                    ON CONFLICT (series_id, period_date, observed_at) DO NOTHING
                    """
                    ),
                    {
                        "sid": series_id,
                        "pdate": row["period_date"],
                        "oat": observed_at,
                        "val": v,
                        "status": status,
                        "meta": json.dumps(meta, ensure_ascii=False),
                        "rid": release_id,
                    },
                )
                inserted += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Failed row insert: {e}")

        conn.execute(
            text(
                """
            INSERT INTO ingestion_log (dataset_id, status, rows_inserted, rows_failed, details)
            VALUES (:did, :status, :ins, :fail, CAST(:details AS jsonb))
            """
            ),
            {
                "did": dataset_id,
                "status": "ok" if failed == 0 else "ok_with_errors",
                "ins": inserted,
                "fail": failed,
                "details": json.dumps(
                    {
                        "flow_id": flow_id,
                        "agency": agency,
                        "dataset_title": dataset_title,
                        "json_structure_name": json_name,
                        "time_dim": time_dim,
                        "freq": freq,
                        "series_count": len(series_cache),
                        "payload_hash": payload_hash,
                        "raw_path": raw_path,
                    },
                    ensure_ascii=False,
                ),
            },
        )

    if discarded_missing_by_series:
        total_discards = sum(sum(reasons.values()) for reasons in discarded_missing_by_series.values())
        logger.info(f"DISCARDED missing observations: total={total_discards}, series_affected={len(discarded_missing_by_series)}")

        top = sorted(
            discarded_missing_by_series.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True
        )[:50]

        for sk, reasons in top:
            n = sum(reasons.values())
            reasons_str = ", ".join([f"{r}:{c}" for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True)])
            logger.info(f"DISCARDED series={sk} count={n} reasons=[{reasons_str}]")

    logger.info(f"DONE flow={flow_id} title={dataset_title!r} freq={freq} series={len(series_cache)} inserted={inserted} failed={failed}")


def main():
    parser = argparse.ArgumentParser(description="Load one StatGov OSP SDMX flow into nowcast DB")
    parser.add_argument("flow_id", help="Flow id, e.g. S3R002_M3140501")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--debug-title", action="store_true", help="Print where title/desc were obtained from")
    args = parser.parse_args()
    ingest_one_flow(args.flow_id, start=args.start, end=args.end, debug_title=args.debug_title)


if __name__ == "__main__":
    main()