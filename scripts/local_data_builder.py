"""
Local data builder for GDP nowcasting pipeline.
Handles loading, joining, and vintage generation without DB dependencies.
Improved for local Lithuania/Eurostat experiments.
"""
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger("local_data_builder")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT

class LocalDataManager:
    def __init__(self, data_dir: Path = _DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.gdp_release_lag = 30 # days
        self._last_loaded_file = {} # Track actual files used
        self._column_frequencies = {} # series_id -> M/Q/GT

    def load_or_build_dataset(self, name: str, force_rebuild: bool = False) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Loads dataset if exists, otherwise tries to build it from available local files.
        Returns (X, y, source_info)
        """
        if name.startswith("final_thesis_"):
            return self.load_frequency_aware_data(name.replace("final_thesis_", ""))

        file_map = {
            "baseline_common": ["common_final_nowcast_dataset.parquet", "common_final_nowcast_dataset1.parquet"],
            "common_plus_gt": ["common_with_google_trends.parquet"],
            "gt_only": ["gt_plus_target.parquet"]
        }

        if not force_rebuild:
            try:
                path = self._find_file(file_map.get(name, []))
                logger.info(f"Using existing dataset file: {path.name}")
                X, y = self._load_and_split(path)
                return X, y, path.name
            except FileNotFoundError:
                logger.info(f"Dataset {name} not found. Attempting to build locally...")

        if name == "baseline_common":
            # Baseline must exist as it's the primary Lithuania macro source
            path = self._find_file(file_map["baseline_common"])
            X, y = self._load_and_split(path)
            return X, y, path.name
        
        elif name == "common_plus_gt":
            X, y = self.build_common_plus_gt()
            return X, y, "built:common_with_google_trends.parquet"
            
        elif name == "gt_only":
            X, y = self.build_gt_only()
            return X, y, "built:gt_plus_target.parquet"
            
        else:
            raise ValueError(f"Unknown dataset type: {name}")

    def build_common_plus_gt(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Builds common+GT by joining baseline macro and local GT data."""
        X_macro, y, _ = self.load_or_build_dataset("baseline_common")
        X_gt = self.load_gt_data()
        
        logger.info(f"Joining macro ({X_macro.shape[1]} cols) and GT ({X_gt.shape[1]} cols)...")
        # Align on index (monthly)
        X_combined = X_macro.join(X_gt, how="left")
        
        # Save for future use
        out_path = self.data_dir / "common_with_google_trends.parquet"
        pd.concat([X_combined, y], axis=1).to_parquet(out_path)
        logger.info(f"Saved rebuilt dataset to {out_path.name}")
        
        return X_combined, y

    def build_gt_only(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Builds GT-only dataset by joining GT data and GDP target."""
        _, y, _ = self.load_or_build_dataset("baseline_common")
        X_gt = self.load_gt_data()
        
        logger.info(f"Building GT-only dataset ({X_gt.shape[1]} cols)...")
        
        # Save for future use
        out_path = self.data_dir / "gt_plus_target.parquet"
        pd.concat([X_gt, y], axis=1).to_parquet(out_path)
        logger.info(f"Saved rebuilt dataset to {out_path.name}")
        
        return X_gt, y

    def load_gt_data(self) -> pd.DataFrame:
        """
        Loads Google Trends data from local raw or processed files.
        Ensures only GT features are returned, avoiding macro column leakage.
        """
        # We try to get macro columns to filter them out in case of fallback files
        try:
            # We don't use load_or_build_dataset here to avoid recursion
            path_m = self._find_file(["common_final_nowcast_dataset.parquet", "common_final_nowcast_dataset1.parquet"])
            df_m = pd.read_parquet(path_m)
            macro_cols = set(df_m.columns)
        except Exception:
            macro_cols = set()

        df = pd.DataFrame()
        # Try processed first
        try:
            path = self._find_file(["google_trends_processed.parquet", "gt_data.parquet"])
            df = pd.read_parquet(path)
        except FileNotFoundError:
            # Try to build from raw CSVs in data/raw/google_trends
            raw_dir = _PROJECT_ROOT / "data" / "raw" / "google_trends"
            if raw_dir.exists():
                logger.info(f"Aggregating raw GT CSVs from {raw_dir}...")
                csvs = list(raw_dir.glob("*.csv"))
                if not csvs:
                    raise FileNotFoundError("No GT data found (parquet or raw CSVs).")
                
                all_dfs = []
                for c in csvs:
                    # FIX: index_col instead of index_index
                    tmp = pd.read_csv(c, index_col=0, parse_dates=True)
                    all_dfs.append(tmp)
                df = pd.concat(all_dfs, axis=1)
                # Resample to monthly (end of month)
                df = df.resample("M").mean()
            else:
                # Last resort: try to extract from any file that might have it
                try:
                    path = self._find_file(["common_with_google_trends.parquet", "gt_plus_target.parquet"])
                    df = pd.read_parquet(path)
                    # This fallback file might contain macro columns, so we MUST filter
                except FileNotFoundError:
                    raise FileNotFoundError("Could not find any Google Trends data source.")

        if df.empty:
             raise ValueError("Google Trends data source is empty.")

        if "period_date" in df.columns:
            df = df.set_index("period_date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # STRICT FILTERING: keep only columns that are NOT in baseline macro and NOT the target
        # This ensures 'gt_only' mode truly only uses Google Trends features.
        final_cols = [c for c in df.columns if c not in macro_cols and c != "gdp_target"]
        df = df[final_cols]
            
        return df.select_dtypes(include=[np.number])

    def refresh_google_trends(self, keywords: List[str] = None):
        """
        Lightweight local GT refresh using pytrends.
        NOTE: Uses 'today 5-y' window which is sufficient for recent nowcasting
        but may be shorter than the full historical macro series.
        """
        logger.info("Attempting local Google Trends refresh...")
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl='en-US', tz=360)
            
            if not keywords:
                # Try to load from config
                try:
                    import yaml
                    cfg_path = _PROJECT_ROOT / "config" / "datasets.yaml"
                    if cfg_path.exists():
                        with open(cfg_path, 'r') as f:
                            config = yaml.safe_load(f)
                            keywords = config.get("google_trends", {}).get("keywords", ["Lithuania GDP", "Economy"])
                except Exception:
                    keywords = ["Lithuania GDP", "Economy"]

            if not keywords:
                 keywords = ["Lithuania GDP", "Economy"]

            logger.info(f"Fetching GT for: {keywords}")
            pytrends.build_payload(keywords, cat=0, timeframe='today 5-y', geo='LT', gprop='')
            df = pytrends.interest_over_time()
            
            if not df.empty:
                if 'isPartial' in df.columns:
                    df = df.drop(columns=['isPartial'])
                
                # Resample to Monthly
                df_m = df.resample('M').mean()
                out_path = self.data_dir / "google_trends_processed.parquet"
                df_m.to_parquet(out_path)
                logger.info(f"GT refreshed and saved to {out_path.name}")
            else:
                logger.warning("GT fetch returned empty dataframe.")
                
        except ImportError:
            logger.warning("pytrends not installed. Skipping local GT refresh.")
        except Exception as e:
            logger.error(f"Local GT refresh failed: {e}")

    def _find_file(self, candidates: List[str]) -> Path:
        for c in candidates:
            p = self.data_dir / c
            if p.exists():
                return p
        raise FileNotFoundError(f"None of the candidates found in {self.data_dir}: {candidates}")

    def _load_and_split(self, path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_parquet(path)
        
        if "period_date" in df.columns:
            df = df.set_index("period_date")
        elif "date" in df.columns:
            df = df.set_index("date")
            
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Identify target
        target_col = None
        for col in ["gdp_target", "GDP1"]:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            possible = [c for c in df.columns if "GDP" in str(c).upper() or "BVP" in str(c).upper()]
            if possible:
                target_col = possible[0]
            else:
                raise KeyError(f"Could not identify target column in {path.name}")
        
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(include=[np.number])
        
        return X, y

    def load_frequency_aware_data(self, name: str) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Loads frequency-aware datasets prepared by prepare_selected_frequency_aware_data.py."""
        proc_dir = self.data_dir / "data" / "processed"
        logger.info(f"Loading frequency-aware components for: {name}")
        
        try:
            # Load frequency map
            map_path = proc_dir / "selected_column_frequency_map.json"
            if map_path.exists():
                with open(map_path, "r") as f:
                    import json
                    self._column_frequencies = json.load(f)
                logger.info(f"Loaded frequency map with {len(self._column_frequencies)} entries.")

            y = pd.read_parquet(proc_dir / "gdp_target_quarterly.parquet")["gdp_target"]
            y.index = pd.to_datetime(y.index)
            y = y.sort_index()
            
            if name == "gt_only":
                X = pd.read_parquet(proc_dir / "selected_google_trends_monthly_prepared.parquet")
            else:
                # Eurostat Monthly
                X_m = pd.read_parquet(proc_dir / "selected_eurostat_monthly_prepared.parquet")
                # Eurostat Quarterly
                X_q = pd.read_parquet(proc_dir / "selected_eurostat_quarterly_prepared.parquet")
                
                # Join preserves all dates. Quarterly columns will have NaNs for non-quarter-end months.
                X = X_m.join(X_q, how="outer")
                
                if name == "common_plus_gt":
                    X_gt = pd.read_parquet(proc_dir / "selected_google_trends_monthly_prepared.parquet")
                    X = X.join(X_gt, how="outer")
            
            X.index = pd.to_datetime(X.index)
            X = X.sort_index()
            
            return X, y, f"frequency_aware:{name}"
            
        except FileNotFoundError as e:
            logger.error(f"Frequency-aware components missing: {e}. Please run scripts/prepare_selected_frequency_aware_data.py first.")
            raise

    def get_vintage(self, X: pd.DataFrame, y: pd.Series, cutoff_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies cutoff date filter. Separates feature cutoff and target visibility.
        """
        # 1. Feature Cutoff
        X_v = X[X.index <= cutoff_date].copy()
        
        # 2. Target Visibility (apply release lag)
        y_v = y.copy()
        release_lag_delta = pd.Timedelta(days=self.gdp_release_lag)
        y_v.loc[y_v.index + release_lag_delta > cutoff_date] = np.nan
        
        return X_v, y_v

def map_to_target_quarter(date: pd.Timestamp) -> str:
    """Maps a date to a quarter label like '2026Q1'."""
    q = (date.month - 1) // 3 + 1
    return f"{date.year}Q{q}"

def get_cutoff_dates_for_quarter(target_q_end: pd.Timestamp) -> List[Tuple[str, pd.Timestamp]]:
    """
    [DEPRECATED] 
    This function is deprecated for the bachelor thesis evaluation. 
    Use `nowcasting.data.vintage_builder.VintageBuilder` for strict pseudo-real-time 
    vintage generation and truncation.
    """
    logger.warning("get_cutoff_dates_for_quarter is deprecated. Use VintageBuilder.")
    vintages = []
    labels = ["-2", "-1", "0", "+1", "+2"]
    
    # Base is 4 months before quarter end
    base = target_q_end - pd.offsets.MonthEnd(4)
    for i, label in enumerate(labels):
        cutoff = base + pd.offsets.MonthEnd(i)
        vintages.append((label, cutoff))
        
    return vintages

