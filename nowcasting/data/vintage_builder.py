"""
Vintage simulation layer for macroeconomic nowcasting.
Handles strict pseudo-real-time truncation, ragged edge filling (AutoARIMA), 
and monthly-to-quarterly temporal aggregation.
"""
import logging
import pandas as pd
import numpy as np
import time
import hashlib
import sqlite3
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel, delayed = None, None

logger = logging.getLogger("vintage_builder")

def map_to_target_quarter(date: pd.Timestamp) -> str:
    """Helper to convert Timestamp to 'YYYYQX' format."""
    q = (date.month - 1) // 3 + 1
    return f"{date.year}Q{q}"

def normalize_vintage_label(v):
    """Standardized vintage label: '+2' -> '2', '-1' -> '-1'"""
    return str(v).replace("+", "")

def test_vintage_cutoffs():
    """Self-test for month-relative vintage cutoff logic."""
    vb = VintageBuilder(vintage_label_mode="month_relative_to_quarter_end")
    target = pd.Timestamp("2020-12-31")
    assert str(vb.get_cutoff_date(target, "-2").date()) == "2020-10-31"
    assert str(vb.get_cutoff_date(target, "-1").date()) == "2020-11-30"
    assert str(vb.get_cutoff_date(target, "0").date()) == "2020-12-31"
    assert str(vb.get_cutoff_date(target, "+1").date()) == "2021-01-31"
    assert str(vb.get_cutoff_date(target, "+2").date()) == "2021-02-28"
    logger.info("Vintage cutoff logic tests passed.")

class VintageBuilder:
    """
    Constructs pseudo-real-time vintages for GDP nowcasting.
    Ensures strict leakage prevention by simulating data availability at a specific cutoff date.
    
    Vintage Labels (month_relative_to_quarter_end mode):
      -2 : Cutoff is 2 months before the target quarter end.
      -1 : Cutoff is 1 month before the target quarter end.
      0  : Cutoff is exactly the end of the target quarter.
      +1 : Cutoff is 1 month after the target quarter end.
      +2 : Cutoff is 2 months after the target quarter end.
    """
    
    def __init__(
        self,
        gdp_release_lag_days: int = 30,
        vintage_label_mode: str = "month_relative_to_quarter_end",
        min_obs_per_series: int = 36,
        seasonal: bool = True,
        random_state: int = 123,
        dataset_name: str = "default_dataset",
        arima_fast: bool = False,
        arima_n_jobs: int = 1,
        arima_cache_path: Optional[str] = None
    ):
        self.gdp_release_lag = pd.Timedelta(days=gdp_release_lag_days)
        self.vintage_label_mode = vintage_label_mode
        self.min_obs_per_series = min_obs_per_series
        self.seasonal = seasonal
        self.random_state = random_state
        self.dataset_name = dataset_name
        self.arima_fast = arima_fast
        self.arima_n_jobs = arima_n_jobs
        self._arima_cache = {}
        
        self.arima_cache_path = Path(arima_cache_path) if arima_cache_path else Path("data/cache/arima_vintage_cache.sqlite")
        self.arima_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite_cache()

    def _init_sqlite_cache(self):
        """Initializes the SQLite cache table if it does not exist."""
        conn = sqlite3.connect(str(self.arima_cache_path))
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS arima_cache (
            cache_key TEXT PRIMARY KEY,
            dataset_name TEXT,
            series_name TEXT,
            last_valid_date TEXT,
            steps INTEGER,
            seasonal INTEGER,
            arima_fast INTEGER,
            values_hash TEXT,
            preds_json TEXT,
            status TEXT,
            created_at TEXT
        )
        ''')
        conn.commit()
        conn.close()

    def get_cutoff_date(self, target_quarter_end: pd.Timestamp, vintage_label: str) -> pd.Timestamp:
        """Calculates the exact cutoff date based on the target quarter and vintage label."""
        if self.vintage_label_mode == "month_relative_to_quarter_end":
            try:
                offset_months = int(vintage_label)
                cutoff_date = target_quarter_end + pd.offsets.MonthEnd(offset_months)
                return cutoff_date
            except ValueError:
                raise ValueError(f"Invalid vintage label '{vintage_label}' for month_relative_to_quarter_end mode.")
        else:
            raise NotImplementedError(f"Vintage label mode '{self.vintage_label_mode}' not implemented.")

    def _detect_column_frequency(self, s: pd.Series) -> str:
        """Detects if a series is 'M' (monthly) or 'Q' (quarterly) based on its non-NaN pattern."""
        valid_idx = s.dropna().index
        if len(valid_idx) < 3:
            return "M" # Default
        
        # Check if all observations fall on month 3, 6, 9, 12
        if all(m % 3 == 0 for m in valid_idx.month):
            # Check for typical quarterly spacing (approx 90 days)
            diffs = valid_idx.to_series().diff().dropna().dt.days
            if (diffs >= 80).all():
                return "Q"
        return "M"

    def _fill_series_autoarima(self, s: pd.Series, steps: int, series_name: str = None, freq: str = "M") -> Tuple[np.ndarray, str]:
        """
        Fits constrained AutoARIMA and forecasts `steps` periods ahead.

        Priority:
        1. sktime AutoARIMA
        2. LOCF fallback

        Returns:
            preds: np.ndarray
            status: str
        """
        s = s.dropna().astype(float)

        if steps <= 0:
            return np.array([]), "autoarima_no_steps"

        if s.empty:
            return np.full(steps, np.nan), "autoarima_empty_series"

        if len(s) < self.min_obs_per_series:
            return np.full(steps, s.iloc[-1]), "autoarima_too_short_locf"

        # Build stable cache key
        last_valid_date = s.index[-1].strftime('%Y-%m-%d')
        val_hash = hashlib.md5(s.values.tobytes()).hexdigest()
        
        cache_key_str = f"{self.dataset_name}_{series_name}_{last_valid_date}_{steps}_{self.seasonal}_{self.arima_fast}_{val_hash}_{freq}"

        if cache_key_str in self._arima_cache:
            return self._arima_cache[cache_key_str], "autoarima_cached_memory"
            
        # SQLite Check
        try:
            conn = sqlite3.connect(str(self.arima_cache_path), timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("SELECT preds_json, status FROM arima_cache WHERE cache_key = ?", (cache_key_str,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                preds = np.array(json.loads(row[0]), dtype=float)
                self._arima_cache[cache_key_str] = preds
                return preds, "autoarima_cached_sqlite"
        except Exception as e:
            logger.debug(f"SQLite cache read failed: {e}")

        # ---------------------------------------------------------
        # 1. Try sktime AutoARIMA first
        # ---------------------------------------------------------
        try:
            from sktime.forecasting.arima import AutoARIMA

            y = s.copy()

            # sktime works better with a regular PeriodIndex
            try:
                y.index = pd.PeriodIndex(y.index, freq=freq)
            except Exception:
                # If index conversion fails, keep original index
                pass

            if self.arima_fast:
                forecaster = AutoARIMA(
                    sp=4 if freq == "Q" else (12 if self.seasonal else 1),
                    start_p=0, start_q=0, start_P=0, start_Q=0,
                    max_p=1, max_q=1, max_P=0, max_Q=0,
                    suppress_warnings=True,
                    error_action="ignore",
                    stepwise=True,
                    n_jobs=1,
                )
            else:
                forecaster = AutoARIMA(
                    sp=4 if freq == "Q" else (12 if self.seasonal else 1),
                    start_p=0, start_q=0, start_P=0, start_Q=0,
                    max_p=2, max_q=2, max_P=1, max_Q=1,
                    suppress_warnings=True,
                    error_action="ignore",
                    stepwise=True,
                    n_jobs=1,
                )

            forecaster.fit(y)
            fh = list(range(1, steps + 1))
            pred = forecaster.predict(fh=fh)

            preds = np.asarray(pred, dtype=float)

            if len(preds) != steps or np.any(~np.isfinite(preds)):
                raise ValueError("sktime AutoARIMA returned invalid predictions.")

            self._arima_cache[cache_key_str] = preds
            
            # SQLite Insert
            try:
                conn = sqlite3.connect(str(self.arima_cache_path), timeout=10.0)
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR IGNORE INTO arima_cache 
                (cache_key, dataset_name, series_name, last_valid_date, steps, seasonal, arima_fast, values_hash, preds_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ''', (cache_key_str, self.dataset_name, str(series_name), last_valid_date, steps, 
                      int(self.seasonal), int(self.arima_fast), val_hash, json.dumps(preds.tolist()), "sktime_autoarima_ok"))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(f"SQLite cache write failed: {e}")
                
            return preds, "sktime_autoarima_ok"

        except Exception as e:
            logger.debug(f"sktime AutoARIMA failed for {series_name}: {e}. Falling back to LOCF.")

        # ---------------------------------------------------------
        # 2. Final fallback: LOCF
        # ---------------------------------------------------------
        preds = np.full(steps, s.iloc[-1])
        return preds, "sktime_autoarima_failed_locf"

    def _build_vertical_realignment_panel(self, X_visible: pd.DataFrame, target_quarter_end: pd.Timestamp, metadata: dict, col_freqs: Dict[str, str], mode: str = "calendar_blocks") -> pd.DataFrame:
        """
        Converts a monthly ragged panel into a blocked quarterly panel with vertical realignment.
        
        Modes:
        - "calendar_blocks": Converts 1 monthly feature into 3 fixed calendar months (M1, M2, M3).
          If month 3 is missing at the ragged edge, it remains NaN or is ffilled from the PREVIOUS quarter's M3.
        - "most_recent_lags": Realigns each monthly feature based on its latest available observation.
          1 monthly feature -> {feat}_most_recent, {feat}_lag1, {feat}_lag2.
          This ensures the model always sees the 'freshest' possible data in the same column position.
          
        Quarterly features are kept as single quarterly features in both modes.
        Incomplete blocks at the ragged edge are forward-filled with the last available aligned block.
        """
        # Split columns by frequency
        m_gt_cols = [c for c in X_visible.columns if col_freqs.get(str(c)) in ["M", "GT"]]
        q_cols = [c for c in X_visible.columns if col_freqs.get(str(c)) == "Q"]
        
        metadata["vertical_realignment_monthly_features"] = len(m_gt_cols)
        metadata["vertical_realignment_quarterly_features"] = len(q_cols)
        
        X_q_list = []
        
        # 1. Monthly and GT features (Blocking + Vertical Realignment)
        if m_gt_cols:
            if mode == "calendar_blocks":
                X_m_visible = X_visible[m_gt_cols]
                q_dates = X_m_visible.index.to_period("Q").end_time.normalize()
                moq = (X_m_visible.index.month - 1) % 3 + 1
                
                X_blocked = X_m_visible.copy()
                X_blocked['__q'] = q_dates
                X_blocked['__moq'] = moq
                
                # Pivot to quarterly blocks
                X_q_m = X_blocked.pivot(index='__q', columns='__moq')
                X_q_m.columns = [f"{c[0]}_m{c[1]}" for c in X_q_m.columns]
                
                # Ensure target_quarter_end is in index if not present
                if target_quarter_end not in X_q_m.index:
                    X_q_m = X_q_m.reindex(X_q_m.index.union([target_quarter_end]))
                    
                # Count missing before ffill
                missing_before = X_q_m.isna().sum().sum()
                
                # Vertical Realignment (ffill missing aligned blocks over quarters)
                X_q_m_ffilled = X_q_m.ffill()
                
                missing_after = X_q_m_ffilled.isna().sum().sum()
                
                metadata["vertical_realignment_features_total"] = len(X_q_m.columns)
                metadata["vertical_realignment_missing_before"] = int(missing_before)
                metadata["vertical_realignment_missing_after"] = int(missing_after)
                metadata["vertical_realignment_blocks_ffilled"] = int(missing_before - missing_after)
                
                X_q_list.append(X_q_m_ffilled)
            
            elif mode == "most_recent_lags":
                logger.info("Building 'most_recent_lags' realignment panel...")
                q_idx = pd.date_range(start=X_visible.index.min(), end=target_quarter_end, freq='Q').normalize()
                X_q_m = pd.DataFrame(index=q_idx)
                
                realigned_count = 0
                missing_lag_count = 0
                
                for col in m_gt_cols:
                    s_full = X_visible[col].dropna()
                    if s_full.empty:
                        continue
                        
                    res_most_recent = []
                    res_lag1 = []
                    res_lag2 = []
                    
                    for q_end in q_idx:
                        # Consider only data visible as of this quarter (respecting global cutoff)
                        # We use the full X_visible which already applied the global vintage cutoff.
                        s_vis = s_full[s_full.index <= q_end]
                        
                        if len(s_vis) >= 1:
                            res_most_recent.append(s_vis.iloc[-1])
                            realigned_count += 1
                        else:
                            res_most_recent.append(np.nan)
                            
                        if len(s_vis) >= 2:
                            res_lag1.append(s_vis.iloc[-2])
                        else:
                            res_lag1.append(np.nan)
                            missing_lag_count += 1
                            
                        if len(s_vis) >= 3:
                            res_lag2.append(s_vis.iloc[-3])
                        else:
                            res_lag2.append(np.nan)
                            missing_lag_count += 1
                            
                    X_q_m[f"{col}_most_recent"] = res_most_recent
                    X_q_m[f"{col}_lag1"] = res_lag1
                    X_q_m[f"{col}_lag2"] = res_lag2
                
                # Forward fill to handle gaps between releases if any, but only forward
                missing_before = X_q_m.isna().sum().sum()
                X_q_m_ffilled = X_q_m.ffill()
                missing_after = X_q_m_ffilled.isna().sum().sum()
                
                metadata["vertical_realignment_features_total"] = len(X_q_m.columns)
                metadata["vertical_realignment_realigned_values"] = realigned_count
                metadata["vertical_realignment_missing_lag_values"] = missing_lag_count
                metadata["vertical_realignment_missing_before"] = int(missing_before)
                metadata["vertical_realignment_missing_after"] = int(missing_after)
                metadata["vertical_realignment_blocks_ffilled"] = int(missing_before - missing_after)
                
                X_q_list.append(X_q_m_ffilled)
            
            else:
                raise ValueError(
                    f"Unknown vertical_realignment mode: {mode}. "
                    "Expected 'calendar_blocks' or 'most_recent_lags'."
                )

        # 2. Quarterly features (Direct Resampling)
        if q_cols:
            X_q_visible = X_visible[q_cols]
            # Resample to quarter end and forward fill to handle target quarter if missing
            X_q_q = X_q_visible.resample("Q").last().ffill()
            
            if target_quarter_end not in X_q_q.index:
                X_q_q = X_q_q.reindex(X_q_q.index.union([target_quarter_end])).ffill()
            
            X_q_list.append(X_q_q)
            
        if X_q_list:
            X_q = pd.concat(X_q_list, axis=1).sort_index()
        else:
            # Fallback empty dataframe
            X_q = pd.DataFrame(index=pd.date_range(start=X_visible.index.min(), end=target_quarter_end, freq='Q'))
            
        return X_q

    def _prepare_visible_panel(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_quarter_end: pd.Timestamp,
        vintage_label: str,
        macro_release_lag_months: int,
        gt_release_lag_months: int,
        quarterly_feature_release_lag_months: int,
        train_start: Optional[pd.Timestamp] = None,
        rolling_window_quarters: Optional[int] = None,
        column_frequencies: Dict[str, str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Timestamp, Dict[str, str], Dict[str, Any]]:
        """
        Shared logic to truncate X and y based on cutoff_date and release lags.
        Returns (X_visible, y_visible, cutoff_date, col_freqs, metadata_base)
        """
        if not X.index.is_monotonic_increasing:
            X = X.sort_index()
        if not y.index.is_monotonic_increasing:
            y = y.sort_index()

        cutoff_date = self.get_cutoff_date(target_quarter_end, vintage_label)
        
        metadata = {
            "target_quarter": target_quarter_end.date(),
            "vintage_label": vintage_label,
            "cutoff_date": cutoff_date.date(),
            "info": "X release calendar modeled via macro_release_lag_months and gt_release_lag_months."
        }

        # 1. Strict Truncation & Release Lag Simulation
        X_visible = X.copy()
        
        if column_frequencies:
            col_freqs = {str(k): v for k, v in column_frequencies.items()}
        else:
            col_freqs = {str(col): self._detect_column_frequency(X[col]) for col in X.columns}
            
        # Apply X release lags
        for col in X_visible.columns:
            col_str = str(col)
            freq = col_freqs.get(col_str, "M")
            
            if freq == "GT" or col_str.startswith("gt_") or "_gt" in col_str:
                lag = gt_release_lag_months
            elif freq == "Q":
                lag = quarterly_feature_release_lag_months
            else:
                lag = macro_release_lag_months
                
            if lag > 0:
                max_obs_date = cutoff_date - pd.offsets.MonthEnd(lag)
                X_visible.loc[X_visible.index > max_obs_date, col] = np.nan
        
        X_visible = X_visible[X_visible.index <= cutoff_date]
        
        y_visible = y.copy()
        y_visible.loc[y_visible.index + self.gdp_release_lag > cutoff_date] = np.nan
        
        # Leakage checks
        assert X_visible.index.max() <= cutoff_date, "Leakage: X contains data after cutoff."
        valid_y_dates = y_visible.dropna().index
        if len(valid_y_dates) > 0:
            assert valid_y_dates.max() + self.gdp_release_lag <= cutoff_date, "Leakage: y contains data released after cutoff."

        # 2. Train Window Filtering
        if train_start:
            X_visible = X_visible[X_visible.index >= train_start]
            y_visible = y_visible[y_visible.index >= train_start]
            
        if rolling_window_quarters is not None and rolling_window_quarters > 0:
            y_avail = y_visible.dropna()
            if len(y_avail) > rolling_window_quarters:
                start_cutoff = y_avail.index[-rolling_window_quarters]
                y_visible = y_visible[y_visible.index >= start_cutoff]
                X_visible = X_visible[X_visible.index >= start_cutoff]
        
        # 3. Pre-filtering: remove all-NaN or constant columns
        valid_cols = []
        for col in X_visible.columns:
            s = X_visible[col].dropna()
            if len(s) > 1 and s.nunique() > 1:
                valid_cols.append(col)
        X_visible = X_visible[valid_cols]

        m_count = sum(1 for f in col_freqs.values() if f == "M")
        q_count = sum(1 for f in col_freqs.values() if f == "Q")
        gt_count = sum(1 for f in col_freqs.values() if f == "GT")
        metadata.update({
            "detected_monthly_features": m_count,
            "detected_quarterly_features": q_count,
            "detected_gt_features": gt_count
        })

        return X_visible, y_visible, cutoff_date, col_freqs, metadata

    def build_native_vintage(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_quarter_end: pd.Timestamp,
        vintage_label: str,
        macro_release_lag_months: int = 1,
        gt_release_lag_months: int = 0,
        quarterly_feature_release_lag_months: int = 1,
        train_start: Optional[pd.Timestamp] = None,
        rolling_window_quarters: Optional[int] = None,
        allow_backcast_target_in_train: bool = False,
        column_frequencies: Dict[str, str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, float, Dict[str, str], Dict[str, Any]]:
        """
        Builds a native vintage WITHOUT external filling.
        Used for DFM and MIDAS models that handle ragged edges natively.
        Returns: (X_visible_m, y_train_q, actual_y, col_freqs, metadata)
        """
        X_visible, y_visible, cutoff_date, col_freqs, metadata = self._prepare_visible_panel(
            X, y, target_quarter_end, vintage_label,
            macro_release_lag_months, gt_release_lag_months, quarterly_feature_release_lag_months,
            train_start, rolling_window_quarters, column_frequencies
        )
        
        if allow_backcast_target_in_train and int(normalize_vintage_label(vintage_label)) > 0:
            # Allow target quarter in train for vintages +1, +2
            y_train_q = y_visible[y_visible.index <= target_quarter_end].dropna()
        else:
            y_train_q = y_visible[y_visible.index < target_quarter_end].dropna()
            
        y_actual = y.loc[target_quarter_end] if target_quarter_end in y.index else np.nan
        if isinstance(y_actual, pd.Series):
             y_actual = y_actual.iloc[0]

        metadata["fill_method"] = "native_ragged"
        metadata["col_freqs"] = col_freqs
        
        return X_visible, y_train_q, float(y_actual), col_freqs, metadata

    def build_vintage(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_quarter_end: pd.Timestamp,
        vintage_label: str,
        train_start: Optional[pd.Timestamp] = None,
        rolling_window_quarters: Optional[int] = None,
        fill_method: str = "autoarima",
        aggregation_method: str = "mean",
        vertical_realignment_mode: str = "calendar_blocks",
        preselect_top_k_before_fill: Optional[int] = None,
        debug_preselect_top_k: Optional[int] = None, # backward compatibility
        macro_release_lag_months: int = 1,
        gt_release_lag_months: int = 0,
        quarterly_feature_release_lag_months: int = 1,
        tactis2_author_config: bool = False,
        tactis2_max_epochs: int = 20,
        tactis2_epochs_phase_1: int = 20,
        tactis2_epochs_phase_2: int = 20,
        tactis2_batch_size: Optional[int] = None,
        tactis2_num_batches_per_epoch: Optional[int] = None,
        tactis2_learning_rate: float = 1e-3,
        tactis2_weight_decay: float = 1e-4,
        tactis2_maximum_learning_rate: float = 1e-3,
        tactis2_clip_gradient: float = 1e3,
        tactis2_bagging_size: Optional[int] = None,
        tactis2_skip_copula: Optional[bool] = None,
        tactis2_context_length: int = 120,
        tactis2_num_samples: Optional[int] = None,
        tactis2_device: str = "auto",
        tactis2_force_refit: bool = False,
        allow_backcast_target_in_train: bool = False,
        column_frequencies: Dict[str, str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, float, Dict[str, Any]]:
        
        X_visible, y_visible, cutoff_date, col_freqs, metadata = self._prepare_visible_panel(
            X, y, target_quarter_end, vintage_label,
            macro_release_lag_months, gt_release_lag_months, quarterly_feature_release_lag_months,
            train_start, rolling_window_quarters, column_frequencies
        )
        
        metadata.update({
            "fill_method": fill_method,
            "aggregation_method": aggregation_method,
            "series_sktime_autoarima_ok": 0,
            "series_sktime_autoarima_failed_locf": 0,
            "series_autoarima_cached": 0,
            "series_autoarima_too_short_locf": 0,
            "series_autoarima_empty_series": 0,
            "series_autoarima_no_steps": 0,
            "series_locf": 0,
            "series_rolling_mean": 0,
            "series_tactis2_failed_fallback": 0,
            "series_unfilled": 0,
            "preselect_before_fill_method": None,
            "preselect_before_fill_top_k": None,
            "preselect_before_fill_features_before": len(X_visible.columns),
            "preselect_before_fill_features_after": len(X_visible.columns)
        })

        if preselect_top_k_before_fill is None and debug_preselect_top_k is not None:
            preselect_top_k_before_fill = debug_preselect_top_k

        if (
            fill_method == "tactis2"
            and preselect_top_k_before_fill is not None 
            and preselect_top_k_before_fill < len(X_visible.columns)
        ):
            # POINT 4 & 5: Leakage-free preselection before filling
            # Use only training target (strictly before target_quarter_end)
            y_train_vis = y_visible[y_visible.index < target_quarter_end].dropna()
            
            if len(y_train_vis) > 5:
                # Aggregate visible X to quarterly for correlation check
                X_vis_q = X_visible.resample("Q").mean()
                shared_idx = y_train_vis.index.intersection(X_vis_q.index)
                
                if len(shared_idx) > 5:
                    features_before = len(X_visible.columns)
                    corrs = X_vis_q.loc[shared_idx].corrwith(y_train_vis.loc[shared_idx]).abs()
                    corrs = corrs.sort_values(ascending=False).dropna()
                    top_cols = corrs.head(preselect_top_k_before_fill).index.tolist()
                    
                    metadata["preselect_before_fill_method"] = "corr_top_n"
                    metadata["preselect_before_fill_top_k"] = preselect_top_k_before_fill
                    metadata["preselect_before_fill_features_before"] = features_before
                    
                    X_visible = X_visible[top_cols]
                    features_after = len(X_visible.columns)
                    
                    metadata["preselect_before_fill_features_after"] = features_after
                    logger.info(
                        f"[{target_quarter_end.date()} v{vintage_label}] "
                        f"Preselect before TACTiS-2: {features_before} -> {features_after} "
                        f"(top_k={preselect_top_k_before_fill})"
                    )
                else:
                    logger.warning(f"[{target_quarter_end.date()} v{vintage_label}] Not enough shared training observations for preselection.")
        elif preselect_top_k_before_fill is not None:
            logger.debug(
                f"preselect_top_k_before_fill={preselect_top_k_before_fill} ignored for fill_method={fill_method}; only applied to tactis2."
            )

        # ---------------------------------------------------------
        # 4. Ragged Edge Filling & Aggregation
        # ---------------------------------------------------------
        if fill_method == "vertical_realignment":
            logger.info(f"Applying Vertical Realignment (Mode: {vertical_realignment_mode})...")
            metadata["vertical_realignment_mode"] = vertical_realignment_mode
            X_filled_m = X_visible.copy()
            X_q = self._build_vertical_realignment_panel(X_visible, target_quarter_end, metadata, col_freqs, mode=vertical_realignment_mode)
            # Skip standard aggregation since we are already quarterly
        else:
            if target_quarter_end > X_visible.index.max():
                future_idx = pd.date_range(start=X_visible.index.max() + pd.offsets.MonthEnd(1), 
                                           end=target_quarter_end, freq='M')
                X_filled_m = X_visible.reindex(X_visible.index.union(future_idx))
            else:
                X_filled_m = X_visible[X_visible.index <= target_quarter_end].copy()
                
            if fill_method != "none":
                # POINT 3: Restructure filling for mixed frequency
                monthly_cols = [c for c in X_filled_m.columns if col_freqs.get(str(c)) in ["M", "ME", "GT"]]
                quarterly_cols = [c for c in X_filled_m.columns if col_freqs.get(str(c)) in ["Q", "QE"]]
                
                # 1. Monthly features filling
                if fill_method == "tactis2":
                    logger.info(f"TACTiS2: filling {len(monthly_cols)} monthly/GT features...")
                    t0_m = time.time()
                    try:
                        from nowcasting.fillers.tactis2_filler import TACTiS2Filler
                        filler = TACTiS2Filler(
                            author_config=tactis2_author_config,
                            context_length=tactis2_context_length,
                            max_epochs=tactis2_max_epochs,
                            epochs_phase_1=tactis2_epochs_phase_1,
                            epochs_phase_2=tactis2_epochs_phase_2,
                            batch_size=tactis2_batch_size,
                            num_batches_per_epoch=tactis2_num_batches_per_epoch,
                            learning_rate=tactis2_learning_rate,
                            weight_decay=tactis2_weight_decay,
                            maximum_learning_rate=tactis2_maximum_learning_rate,
                            clip_gradient=tactis2_clip_gradient,
                            bagging_size=tactis2_bagging_size,
                            skip_copula=tactis2_skip_copula,
                            num_samples=tactis2_num_samples,
                            device=tactis2_device,
                            random_state=self.random_state
                        )
                        X_vis_m = X_visible[monthly_cols]
                        X_filled_m_only, diagnostics = filler.fit_predict_fill(
                            X_visible_m=X_vis_m,
                            target_quarter_end=target_quarter_end,
                            cutoff_date=cutoff_date,
                            dataset_type=self.dataset_name,
                            target_quarter=map_to_target_quarter(target_quarter_end),
                            vintage_label=vintage_label,
                            force_refit=tactis2_force_refit
                        )
                        # Replace monthly part in X_filled_m
                        X_filled_m.update(X_filled_m_only)
                        metadata.update(diagnostics)
                        metadata["monthly_fill_runtime_sec"] = round(time.time() - t0_m, 2)
                    except Exception as e:
                        logger.error(f"TACTiS2 filling failed: {e}. Falling back to LOCF for monthly features.")
                        # If TACTiS2 fails, we'll let the AutoARIMA/LOCF loop handle monthly too
                        pass
                
                # 2. AutoARIMA / LOCF filling (always for Quarterly, and for Monthly if not filled by TACTiS2)
                cols_to_fill_auto = []
                for col in X_filled_m.columns:
                    # Skip if already filled by TACTiS2 (not NaN at the end)
                    if fill_method == "tactis2" and col in monthly_cols:
                         if X_filled_m.loc[target_quarter_end, col] == X_filled_m.loc[target_quarter_end, col]: # check if not nan
                             continue
                    
                    s = X_filled_m[col].dropna()
                    freq = col_freqs.get(str(col), "M")
                    
                    if len(s) < self.min_obs_per_series:
                        X_filled_m[col] = X_filled_m[col].ffill() 
                        metadata["series_unfilled"] += 1
                        continue
                        
                    last_valid_date = s.index[-1]
                    if last_valid_date < target_quarter_end:
                        if freq in ["Q", "QE"]:
                             steps = len(pd.date_range(start=last_valid_date + pd.offsets.QuarterEnd(1), 
                                                     end=target_quarter_end, freq='Q'))
                        else:
                             steps = len(X_filled_m.loc[last_valid_date + pd.offsets.MonthEnd(1) : target_quarter_end])
                        
                        if steps > 0:
                            cols_to_fill_auto.append((col, s, steps, last_valid_date, freq))

                if cols_to_fill_auto:
                    if fill_method == "autoarima":
                        logger.info(f"AutoARIMA: filling {len(cols_to_fill_auto)} series...")
                        t0_a = time.time()
                        if self.arima_n_jobs > 1 and Parallel is not None:
                            results = Parallel(n_jobs=self.arima_n_jobs)(
                                delayed(self._fill_series_autoarima)(s, steps, col, freq) 
                                for col, s, steps, _, freq in cols_to_fill_auto
                            )
                            for (col, s, steps, last_valid_date, freq), (preds, status) in zip(cols_to_fill_auto, results):
                                if freq in ["Q", "QE"]:
                                    fill_idx = pd.date_range(start=last_valid_date + pd.offsets.QuarterEnd(1), 
                                                             end=target_quarter_end, freq='Q')
                                else:
                                    fill_idx = X_filled_m.loc[last_valid_date + pd.offsets.MonthEnd(1) : target_quarter_end].index
                                X_filled_m.loc[fill_idx, col] = preds
                                metadata[f"series_{status}"] = metadata.get(f"series_{status}", 0) + 1
                        else:
                            for col, s, steps, last_valid_date, freq in cols_to_fill_auto:
                                preds, status = self._fill_series_autoarima(s, steps, col, freq)
                                if freq in ["Q", "QE"]:
                                    fill_idx = pd.date_range(start=last_valid_date + pd.offsets.QuarterEnd(1), 
                                                             end=target_quarter_end, freq='Q')
                                else:
                                    fill_idx = X_filled_m.loc[last_valid_date + pd.offsets.MonthEnd(1) : target_quarter_end].index
                                X_filled_m.loc[fill_idx, col] = preds
                                metadata[f"series_{status}"] = metadata.get(f"series_{status}", 0) + 1
                        metadata["autoarima_fill_runtime_sec"] = round(time.time() - t0_a, 2)
                    
                    elif fill_method in ["locf", "rolling_mean", "tactis2"]:
                        for col, s, steps, last_valid_date, freq in cols_to_fill_auto:
                            if fill_method == "locf":
                                preds = np.full(steps, s.iloc[-1])
                                metadata["series_locf"] += 1
                            elif fill_method == "rolling_mean":
                                val = s.tail(12).mean()
                                preds = np.full(steps, val)
                                metadata["series_rolling_mean"] += 1
                            else: # tactis2 fallback
                                preds = np.full(steps, s.iloc[-1])
                                metadata["series_tactis2_failed_fallback"] = metadata.get("series_tactis2_failed_fallback", 0) + 1
                                
                            if freq in ["Q", "QE"]:
                                fill_idx = pd.date_range(start=last_valid_date + pd.offsets.QuarterEnd(1), 
                                                         end=target_quarter_end, freq='Q')
                            else:
                                fill_idx = X_filled_m.loc[last_valid_date + pd.offsets.MonthEnd(1) : target_quarter_end].index
                            
                            if len(fill_idx) != len(preds):
                                logger.warning(f"Length mismatch for {col}: fill_idx={len(fill_idx)}, preds={len(preds)}. Truncating.")
                                X_filled_m.loc[fill_idx, col] = preds[:len(fill_idx)]
                            else:
                                X_filled_m.loc[fill_idx, col] = preds
                
                # Removed bfill to prevent leakage. Only ffill forward if absolutely necessary.
                metadata["missing_before_final_ffill"] = int(X_filled_m.isna().sum().sum())
                X_filled_m = X_filled_m.ffill()
                metadata["missing_after_final_ffill"] = int(X_filled_m.isna().sum().sum())
                metadata["values_filled_by_final_ffill"] = metadata["missing_before_final_ffill"] - metadata["missing_after_final_ffill"]
    
            # ---------------------------------------------------------
            # 5. Mixed-Frequency Aggregation
            # ---------------------------------------------------------
            m_cols = [c for c in X_filled_m.columns if col_freqs.get(str(c)) in ["M", "ME", "GT"]]
            q_cols = [c for c in X_filled_m.columns if col_freqs.get(str(c)) in ["Q", "QE"]]
            
            X_q_list = []
            if m_cols:
                X_m_filled = X_filled_m[m_cols]
                if aggregation_method == "mean":
                    X_m_q = X_m_filled.resample("Q").mean()
                elif aggregation_method == "last":
                    X_m_q = X_m_filled.resample("Q").last()
                elif aggregation_method == "sum":
                    X_m_q = X_m_filled.resample("Q").sum()
                else:
                    X_m_q = X_m_filled.resample("Q").mean()
                X_q_list.append(X_m_q)
            
            if q_cols:
                # Quarterly features: just pick the end-of-quarter values
                X_q_q = X_filled_m[q_cols].resample("Q").last()
                X_q_list.append(X_q_q)
                
            if X_q_list:
                X_q = pd.concat(X_q_list, axis=1).sort_index()
            else:
                X_q = pd.DataFrame(index=pd.date_range(start=X_filled_m.index.min(), end=target_quarter_end, freq='Q'))
        X_q.columns = X_q.columns.astype(str)
        y_q = y_visible.resample("Q").last()
        
        # ---------------------------------------------------------
        # 6. Split Train and Test Sets & Align
        # ---------------------------------------------------------
        X_train_q = X_q[X_q.index < target_quarter_end]
        
        if allow_backcast_target_in_train and int(normalize_vintage_label(vintage_label)) > 0:
            y_q_train_mask = (y_q.index <= target_quarter_end)
        else:
            y_q_train_mask = (y_q.index < target_quarter_end)
            
        y_train_q = y_q[y_q_train_mask]
        
        # Drop all rows where y_train_q is NaN and align X_train_q
        valid_train_idx = y_train_q.dropna().index
        valid_train_idx = valid_train_idx.intersection(X_train_q.index)
        
        X_train_q = X_train_q.loc[valid_train_idx].copy()
        y_train_q = y_train_q.loc[valid_train_idx].copy()
        
        X_test_q = X_q[X_q.index == target_quarter_end]
        
        y_test_actual = y.loc[target_quarter_end] if target_quarter_end in y.index else np.nan
        if isinstance(y_test_actual, pd.Series):
             y_test_actual = y_test_actual.iloc[0]

        if not X_train_q.empty:
            assert X_train_q.index.max() < target_quarter_end, "Leakage: X_train_q contains target quarter."
            
        return X_filled_m, X_train_q, y_train_q, X_test_q, float(y_test_actual), metadata

# Run the test when module is loaded
if __name__ == "__main__":
    test_vintage_cutoffs()
