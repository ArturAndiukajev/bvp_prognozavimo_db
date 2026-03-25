"""
Generic Data Ingestion Script for New Sources (v2).
Handles fetching and normalizing data from REST JSON and CSV endpoints with support for:
- Pagination
- Advanced JSON parsing (list indexing, dot-notation)
- Config-driven auth, mapping, and extraction
- Dry-run / preview mode
"""

import argparse
import json
import logging
import os
import io
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field

import pandas as pd
import requests

from scripts.db_helpers import (
    get_engine, ensure_provider, ensure_dataset, ensure_series,
    create_release, copy_observations_via_staging, log_ingestion, sha256_bytes, Timer
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generic_loader")

RAW_DIR = Path("data/raw")

# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------

@dataclass
class AuthConfig:
    type: str = "none"               # none, api_key_header, api_key_query, bearer_token, basic_auth
    key_name: str = ""               
    key_env_var: str = ""            
    key_value: str = ""              
    username_env_var: str = ""
    password_env_var: str = ""

    def get_secret(self) -> str:
        if self.key_env_var and self.key_env_var in os.environ:
            return os.environ[self.key_env_var]
        return self.key_value
        
    def get_basic_auth(self) -> Optional[Tuple[str, str]]:
        if self.type == "basic_auth":
            u = os.environ.get(self.username_env_var, "")
            p = os.environ.get(self.password_env_var, "")
            return (u, p) if u and p else None
        return None

@dataclass
class PaginationConfig:
    enabled: bool = False
    type: str = "none"               # page, offset
    param_page: str = "page"
    param_limit: str = "per_page"
    start_page: int = 1
    page_size: int = 1000
    max_pages: int = 50

@dataclass
class MappingConfig:
    date_field: str = ""
    value_field: str = ""
    series_field: Union[str, List[str]] = ""
    frequency_field: str = ""
    unit_field: str = ""
    country_field: str = ""
    date_format: str = ""

@dataclass
class ParsingConfig:
    json_records_path: str = ""
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"
    csv_skiprows: int = 0

@dataclass
class SourceConfig:
    source_name: str = ""
    source_type: str = "rest_json"
    base_url: str = ""
    endpoint: str = ""
    provider_name: str = ""
    
    query_params: Dict[str, str] = field(default_factory=dict)
    
    auth: AuthConfig = field(default_factory=AuthConfig)
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    
    country_default: str = "US"
    freq_default: str = "A"          # Default generic freq
    unit_default: str = "INDEX"
    transform_default: str = "LEVEL"

    @classmethod
    def from_dict(cls, d: dict) -> "SourceConfig":
        d = d.copy()
        auth = AuthConfig(**d.pop("auth", {}))
        pagination = PaginationConfig(**d.pop("pagination", {}))
        parsing = ParsingConfig(**d.pop("parsing", {}))
        
        mapping_data = d.pop("mapping", {})
        mapping = MappingConfig(**mapping_data)
        
        return cls(auth=auth, pagination=pagination, parsing=parsing, mapping=mapping, **d)

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        if self.auth.key_env_var:
            d["auth"]["key_value"] = "" # Hide raw secret if using env
        return d


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class GenericLoader:
    def __init__(self, config: SourceConfig):
        self.config = config
        self.downloaded_at = datetime.now(timezone.utc)
        self.stamp = self.downloaded_at.strftime("%Y%m%dT%H%M%S")
        
    def _prepare_request(self, page=None, offset=None) -> Tuple[str, dict, dict, Optional[Tuple]]:
        url = self.config.base_url.rstrip("/")
        if self.config.endpoint:
            url += "/" + self.config.endpoint.lstrip("/")
            
        headers = {}
        params = self.config.query_params.copy()
        auth_tuple = self.config.auth.get_basic_auth()
        
        # Auth injection
        auth_type = self.config.auth.type
        secret = self.config.auth.get_secret()
        
        if auth_type == "api_key_header" and secret:
            headers[self.config.auth.key_name] = secret
        elif auth_type == "api_key_query" and secret:
            params[self.config.auth.key_name] = secret
        elif auth_type == "bearer_token" and secret:
            headers["Authorization"] = f"Bearer {secret}"
            
        # Pagination injection
        if self.config.pagination.enabled:
            pag = self.config.pagination
            if pag.type == "page" and page is not None:
                params[pag.param_page] = str(page)
                if pag.param_limit:
                    params[pag.param_limit] = str(pag.page_size)
            elif pag.type == "offset" and offset is not None:
                params[pag.param_page] = str(offset) # using param_page as the offset key
                params[pag.param_limit] = str(pag.page_size)

        return url, headers, params, auth_tuple

    def fetch_page(self, page=None, offset=None) -> Tuple[bytes, str]:
        url, headers, params, auth_tuple = self._prepare_request(page, offset)
        safe_params = {k: "HIDDEN" if k == self.config.auth.key_name else v for k, v in params.items()}
        
        logger.info(f"Fetching from {url} | params: {safe_params}")
        r = requests.get(url, headers=headers, params=params, auth=auth_tuple, timeout=60)
        
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            text_preview = r.text[:500] if r.text else ""
            logger.error(f"HTTP Error {r.status_code}")
            logger.error(f"Response preview:\n{text_preview}")
            raise e
            
        ext = ".json" if self.config.source_type == "rest_json" else ".csv"
        return r.content, ext

    def parse(self, raw_data: bytes) -> pd.DataFrame:
        raise NotImplementedError
        
    def extract(self) -> Tuple[List[bytes], str, pd.DataFrame]:
        raw_payloads = []
        dfs = []
        pag = self.config.pagination
        
        if not pag.enabled:
            raw, ext = self.fetch_page()
            raw_payloads.append(raw)
            df = self.parse(raw)
            return raw_payloads, ext, df
            
        # Pagination loop
        page = pag.start_page
        offset = 0
        while True:
            raw, ext = self.fetch_page(page=page, offset=offset)
            raw_payloads.append(raw)
            try:
                df = self.parse(raw)
            except Exception as e:
                logger.error(f"Failed to parse page {page}: {e}")
                break
                
            if df is None or df.empty:
                logger.info("Empty page reached. Stopping pagination.")
                break
                
            dfs.append(df)
            
            # Simple heuristic to stop if we got fewer records than page_size
            if len(df) < pag.page_size:
                break
                
            page += 1
            offset += pag.page_size
            if page > pag.start_page + pag.max_pages - 1:
                logger.warning("Max pages reached. Stopping pagination.")
                break
                
        final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return raw_payloads, ext, final_df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies mapping to rename columns, formats dates, and builds series keys."""
        if df.empty:
            return df
            
        map_conf = self.config.mapping
        df = df.copy()
        
        # 1. Date Field
        if map_conf.date_field not in df.columns:
            raise ValueError(f"Date field '{map_conf.date_field}' not found. Available: {df.columns.tolist()}")
            
        def flexible_date_parser(s):
            if pd.isna(s): return pd.NaT
            s_str = str(s).strip()
            if map_conf.date_format:
                try: return pd.to_datetime(s_str, format=map_conf.date_format).date()
                except: pass
            if len(s_str) == 4 and s_str.isdigit():
                return datetime(int(s_str), 1, 1).date()
            if len(s_str) >= 7 and s_str[4] == '-':
                try: return datetime(int(s_str[:4]), int(s_str[5:7]), 1).date()
                except: pass
            
            dt = pd.to_datetime(s_str, errors="coerce")
            return dt.date() if not pd.isna(dt) else pd.NaT
            
        df["period_date"] = df[map_conf.date_field].apply(flexible_date_parser)
        
        # 2. Value Field
        if map_conf.value_field not in df.columns:
            raise ValueError(f"Value field '{map_conf.value_field}' not found.")
        df["value"] = pd.to_numeric(df[map_conf.value_field], errors="coerce")
        
        # Drop invalid rows early
        df = df.dropna(subset=["period_date", "value"])
        if df.empty:
            return df

        # 3. Series Field Configuration
        ser_f = map_conf.series_field
        if isinstance(ser_f, list) and ser_f:
            df["series_key"] = df.apply(lambda r: "_".join(str(r.get(f, "")) for f in ser_f), axis=1)
        elif isinstance(ser_f, str) and ser_f in df.columns:
            df["series_key"] = [str(x) for x in df[ser_f]]
        elif ser_f:
            df["series_key"] = str(ser_f)
        else:
            df["series_key"] = self.config.source_name
            
        df["series_key"] = df["series_key"].astype(str).str.replace(r"\s+", "_", regex=True)


        # 4. Optionals
        rename_map = {}
        mapped_targets = {"period_date", "value", "series_key"}
        if map_conf.frequency_field and map_conf.frequency_field in df.columns:
            rename_map[map_conf.frequency_field] = "frequency"
            mapped_targets.add("frequency")
        if map_conf.unit_field and map_conf.unit_field in df.columns:
            rename_map[map_conf.unit_field] = "unit"
            mapped_targets.add("unit")
        if map_conf.country_field and map_conf.country_field in df.columns:
            rename_map[map_conf.country_field] = "country"
            mapped_targets.add("country")
            
        if rename_map:
            df = df.rename(columns=rename_map)
            
        # Fill defaults strictly for unmapped targets or NaN
        for col, default_val in [
            ("frequency", self.config.freq_default),
            ("unit", self.config.unit_default),
            ("country", self.config.country_default),
            ("transform", self.config.transform_default),
            ("provider", self.config.provider_name or self.config.source_name)
        ]:
            if col not in mapped_targets:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val).astype(str)
                
        # Ensure all grouping targets are strictly strings to prevent unhashable object crashes
        for col in ["country", "frequency", "unit", "transform", "provider"]:
            df[col] = df[col].astype(str)
                
        # Select final schema
        return df[["period_date", "value", "series_key", "frequency", "unit", "country", "transform", "provider"]]


class RestJsonLoader(GenericLoader):
    def parse(self, raw_data: bytes) -> pd.DataFrame:
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            text_preview = raw_data[:500].decode(errors='replace')
            logger.error(f"JSON Parsing Failed: {e}")
            logger.error(f"Response preview:\n{text_preview}")
            logger.error("Hint: The endpoint may be returning HTML or XML. Did you forget a query parameter like 'format=json'?")
            raise e
            
        if self.config.parsing.json_records_path:
            for part in self.config.parsing.json_records_path.split("."):
                if not part: continue
                if isinstance(data, dict):
                    data = data.get(part, [])
                elif isinstance(data, list):
                    if part.isdigit():
                        idx = int(part)
                        if 0 <= idx < len(data): data = data[idx]
                        else: data = []
                    else:
                        raise ValueError(f"Cannot access non-numeric path part '{part}' on a list object. Current data type: list")
                        
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of records at path '{self.config.parsing.json_records_path}', got {type(data)}. Cannot convert directly to DataFrame.")
            
        return pd.DataFrame(data)


class CsvUrlLoader(GenericLoader):
    def parse(self, raw_data: bytes) -> pd.DataFrame:
        try:
            return pd.read_csv(
                io.BytesIO(raw_data), 
                sep=self.config.parsing.csv_delimiter,
                encoding=self.config.parsing.csv_encoding,
                skiprows=self.config.parsing.csv_skiprows
            )
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            raise e

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_ingestion(config: SourceConfig, dry_run: bool = False, no_db: bool = False, preview: bool = False):
    if preview:
        dry_run = True
        
    loader = RestJsonLoader(config) if config.source_type == "rest_json" else CsvUrlLoader(config)

    logger.info(f"Extracting data using {config.source_type} loader...")
    raw_payloads, ext, df_raw = loader.extract()
    
    if df_raw.empty:
        logger.warning(f"No valid data returned for {config.source_name}. Aborting.")
        if dry_run:
            print("Preview: Returned data is empty.")
        return

    logger.info(f"Normalizing {len(df_raw)} raw rows...")
    df_norm = loader.normalize(df_raw)
    
    if df_norm.empty:
        logger.warning(f"No rows remaining after normalization (check column mappings). Aborting.")
        return
        
    logger.info(f"Successfully normalized {len(df_norm)} valid observations.")

    if dry_run:
        print("\n" + "="*50)
        print(f" DRY RUN / PREVIEW: {config.source_name}")
        print("="*50)
        print(f"Raw shape:        {df_raw.shape}")
        print(f"Normalized shape: {df_norm.shape}")
        print(f"Series detected:  {df_norm['series_key'].nunique()}")
        print("\nHead of normalized data:")
        print(df_norm.head())
        print("="*50 + "\n")
        if no_db:
            return
            
    # Save Raw Backup
    source_dir = RAW_DIR / config.source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = source_dir / f"{config.source_name}_{loader.stamp}{ext}"
    if len(raw_payloads) == 1:
        raw_path.write_bytes(raw_payloads[0])
        content_hash = sha256_bytes(raw_payloads[0])
    else:
        # Multi-page save
        concat_bytes = b"\n".join(raw_payloads)
        raw_path.write_bytes(concat_bytes)
        content_hash = sha256_bytes(concat_bytes)
        
    logger.info(f"Saved raw data to {raw_path}")

    if no_db:
        logger.info("--no-db flag provided. Skipping database ingestion.")
        return

    # DB Ingestion
    logger.info("Starting database ingestion phase...")
    engine = get_engine()
    inserted = 0
    series_groups = df_norm.groupby(["series_key", "country", "frequency", "unit", "transform"])

    with engine.begin() as conn:
        provider_name = config.provider_name or config.source_name
        
        provider_id = ensure_provider(conn, provider_name, base_url=config.base_url)
        dataset_id = ensure_dataset(conn, provider_id, key=config.source_name, title=config.source_name)
        
        release_id = create_release(
            conn=conn, dataset_id=dataset_id,
            downloaded_at=loader.downloaded_at, vintage_at=loader.downloaded_at,
            description=f"Generic snapshot for {config.source_name}",
            raw_path=str(raw_path), content_hash=content_hash,
            meta={"source_type": config.source_type, "endpoint": config.endpoint}
        )

        COPY_BATCH = 50_000
        copy_rows = []
        oat_iso = loader.downloaded_at.isoformat()

        with Timer(f"Generic Insert {config.source_name}"):
            for (s_key, s_country, s_freq, s_unit, s_trans), grp in series_groups:
                series_id = ensure_series(
                    conn=conn, dataset_id=dataset_id, key=s_key,
                    country=s_country, frequency=s_freq,
                    transform=s_trans, unit=s_unit,
                    name=f"{config.source_name}: {s_key}"
                )
                for _, row in grp.iterrows():
                    copy_rows.append((series_id, row["period_date"].isoformat(), oat_iso, float(row["value"]), release_id))
                    if len(copy_rows) >= COPY_BATCH:
                        copy_observations_via_staging(conn, copy_rows)
                        inserted += len(copy_rows)
                        copy_rows = []
                        
            if copy_rows:
                copy_observations_via_staging(conn, copy_rows)
                inserted += len(copy_rows)

        log_ingestion(
            conn, dataset_id=dataset_id, status="ok", rows_inserted=inserted,
            details={"raw_path": str(raw_path), "series_count": len(series_groups)}
        )

    logger.info(f"Successfully ingested {inserted} rows for {config.source_name}")


# ---------------------------------------------------------------------------
# CLI & Interactive
# ---------------------------------------------------------------------------

def prompt_for_config() -> SourceConfig:
    print("\n--- Generic Source Onboarding ---")
    name = input("Source name (e.g. 'world_bank'): ").strip()
    stype = input("Source type [rest_json/csv_url] (default rest_json): ").strip() or "rest_json"
    base_url = input("Base URL (e.g. 'https://api.worldbank.org/v2'): ").strip()
    endpoint = input("Endpoint (e.g. 'country/all/indicator/NY.GDP.MKTP.CD' - optional): ").strip()
    
    cfg = SourceConfig(source_name=name, source_type=stype, base_url=base_url, endpoint=endpoint)

    print("\n--- Query Parameters ---")
    while True:
        qp = input("Add query param as key=value (leave blank to finish, e.g. format=json): ").strip()
        if not qp: break
        if "=" in qp:
            k, v = qp.split("=", 1)
            cfg.query_params[k] = v
        else:
            print("Invalid format. Use key=value.")

    auth_type = input("\nAuth required? [none/api_key_header/api_key_query/bearer_token] (default none): ").strip() or "none"
    cfg.auth.type = auth_type
    if auth_type != "none":
        cfg.auth.key_name = input("Auth key/header name (if applicable): ").strip()
        use_env = input("Read secret from environment variable? [y/N]: ").strip().lower()
        if use_env == 'y':
            cfg.auth.key_env_var = input("Environment variable name: ").strip()
        else:
            print("WARNING: Saving raw secrets to config is discouraged.")
            cfg.auth.key_value = input("Secret value: ").strip()

    if stype == "rest_json":
        cfg.parsing.json_records_path = input("\nJSON records path (e.g. '1' or 'data.results', leave blank if root list): ").strip()

    print("\n--- Column Mapping ---")
    cfg.mapping.date_field = input("Date field name in data: ").strip()
    cfg.mapping.value_field = input("Value field name in data: ").strip()
    
    ser_f = input("Series ID field name(s) (comma-separated if multiple): ").strip()
    if "," in ser_f:
        cfg.mapping.series_field = [x.strip() for x in ser_f.split(",")]
    else:
        cfg.mapping.series_field = ser_f
        
    cfg.freq_default = input("Default frequency (e.g. A, M, Q) (default A): ").upper().strip() or "A"

    return cfg


def main():
    ap = argparse.ArgumentParser(description="Advanced Generic Source Loader")
    ap.add_argument("--config", type=str, help="Path to JSON config file")
    
    # CLI Overrides
    ap.add_argument("--source-name", type=str)
    ap.add_argument("--source-type", type=str, choices=["rest_json", "csv_url"])
    ap.add_argument("--base-url", type=str)
    ap.add_argument("--endpoint", type=str)
    ap.add_argument("--query-param", type=str, action="append", help="key=value format. Repeatable.")
    
    # Runtime modes
    ap.add_argument("--dry-run", action="store_true", help="Fetch and parse, print stats, do not save to DB.")
    ap.add_argument("--preview", action="store_true", help="Alias for --dry-run (includes printing the preview dataframe).")
    ap.add_argument("--no-db", action="store_true", help="Download raw files, normalize, but do not insert into Database.")
    ap.add_argument("--save-config", action="store_true", help="Save the active configuration back to disk.")
    
    args = ap.parse_args()

    # 1. Load config
    if args.config:
        path = Path(args.config)
        if not path.exists():
            print(f"Error: Config file {path} not found.")
            return
        config = SourceConfig.from_dict(json.loads(path.read_text()))
    elif args.source_name and args.source_type and args.base_url:
        print("Error: Incomplete mapping via CLI args. Please use --config or interactive mode.")
        return
    else:
        config = prompt_for_config()
        if not args.save_config:
            save_ans = input("\nSave this config to JSON for future use? [Y/n]: ").strip().lower() != 'n'
            if save_ans: args.save_config = True

    # 2. Apply CLI Overrides
    if args.source_name: config.source_name = args.source_name
    if args.source_type: config.source_type = args.source_type
    if args.base_url:    config.base_url = args.base_url
    if args.endpoint:    config.endpoint = args.endpoint
    if args.query_param:
        for qp in args.query_param:
            if "=" in qp:
                k, v = qp.split("=", 1)
                config.query_params[k] = v

    # 3. Save config if requested
    if args.save_config:
        path = Path(f"config/sources/{config.source_name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config.to_dict(), indent=2))
        print(f"[Config Saved] -> {path}")

    # 4. Run
    try:
        run_ingestion(config, dry_run=(args.dry_run or args.preview), no_db=args.no_db, preview=args.preview)
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
