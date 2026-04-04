from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nowcast_data")

# Project root = parent of the nowcasting/utils/ directory -> parent of nowcasting/ -> parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data" / "processed")
_DEFAULT_OUT_DIR  = str(_PROJECT_ROOT / "data" / "forecasts")

def validate_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "",
    min_train_rows: int = 12,
) -> bool:
    """
    Check shapes, NaN share, index alignment, and dtype.
    Returns True if inputs are usable, False if they should be rejected.
    """
    prefix = f"[validate{' ' + label if label else ''}]"

    if X.empty:
        logger.error(f"{prefix} X is empty.")
        return False

    if y.dropna().empty:
        logger.error(f"{prefix} y has no non-NaN values.")
        return False

    if len(X) < min_train_rows:
        logger.error(f"{prefix} X has only {len(X)} rows — need at least {min_train_rows}.")
        return False

    nan_share_X = X.isna().mean().mean()
    nan_share_y = y.isna().mean()

    if not isinstance(X.index, pd.DatetimeIndex):
        logger.warning(f"{prefix} X.index is not DatetimeIndex ({type(X.index).__name__}) — alignment may break.")

    if not X.index.equals(y.index):
        common = X.index.intersection(y.index)
        if len(common) < min_train_rows:
            logger.error(
                f"{prefix} X and y share only {len(common)} index dates. "
                f"X range: [{X.index.min()}..{X.index.max()}], "
                f"y range: [{y.index.min()}..{y.index.max()}]"
            )
            return False
        logger.info(f"{prefix} Aligning X ({len(X)}) and y ({len(y)}) on {len(common)} common dates.")

    logger.info(
        f"{prefix} X={X.shape} | nan%={nan_share_X:.1%} | "
        f"y_obs={y.notna().sum()} | nan%_y={nan_share_y:.1%} | "
        f"index=[{X.index.min()}..{X.index.max()}]"
    )

    if nan_share_X > 0.9:
        logger.warning(f"{prefix} X has >90% NaN — model predictions may be unreliable.")

    return True

def _load_data_with_datetime_index(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
        
    if "period_date" in df.columns:
        df = df.set_index("period_date")
    elif "month_end" in df.columns:
        df = df.set_index("month_end")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def _resolve_panel_path(data_dir: Path, panel_arg: Optional[str], default_prefix: str) -> Path:
    if panel_arg:
        p = Path(panel_arg)
        path = p if p.is_absolute() else data_dir / p
        
        if default_prefix == "mf_panel_Q" and path.name.startswith("mf_panel_M_"):
            path = path.with_name(path.name.replace("mf_panel_M_", "mf_panel_Q_"))
        elif default_prefix == "mf_panel_M" and path.name.startswith("mf_panel_Q_"):
            path = path.with_name(path.name.replace("mf_panel_Q_", "mf_panel_M_"))
            
        if not path.exists():
            raise FileNotFoundError(f"Specified panel file not found: {path} (expected prefix='{default_prefix}')")
        return path
        
    candidates = list(data_dir.glob(f"{default_prefix}*.parquet"))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        newest = candidates[0]
        logger.info(f"Auto-detected newest {default_prefix} panel file: {newest.name}")
        return newest
        
    old_default = data_dir / f"{default_prefix}.parquet"
    if old_default.exists():
        logger.info(f"Using fallback default panel file: {old_default.name}")
        return old_default
        
    if default_prefix == "panel":
        old_cf = data_dir / "panel_monthly.parquet"
        if old_cf.exists():
            logger.info(f"Using fallback legacy CF panel file: {old_cf.name}")
            return old_cf
            
    raise FileNotFoundError(
        f"No panel files matching '{default_prefix}*.parquet' found in {data_dir}. "
        "Run: python scripts/data_preparation.py"
    )

def _resolve_meta_path(data_dir: Path, panel_path: Path) -> Path:
    name = panel_path.name
    if name.startswith("panel_"):
        meta_name = name.replace("panel_", "meta_", 1).replace(".parquet", ".csv")
    elif name.startswith("mf_panel_M_") or name.startswith("mf_panel_Q_"):
        meta_name = name.replace("mf_panel_M_", "mf_meta_").replace("mf_panel_Q_", "mf_meta_").replace(".parquet", ".csv")
    else:
        meta_name = "panel_meta.csv"

    candidate = panel_path.with_name(meta_name)
    if candidate.exists():
        return candidate
        
    old_default = data_dir / "panel_meta.csv"
    if old_default.exists():
        return old_default
        
    return candidate

def _resolve_target_col(target: Optional[str], panel: pd.DataFrame, data_dir: Path, panel_path: Optional[Path] = None):
    cols = panel.columns

    def _try_match(val: str):
        if val in cols:
            return val
        try:
            as_int = int(val)
            if as_int in cols:
                return as_int
        except (ValueError, TypeError):
            pass
        return None

    def _lookup_by_key(key_pattern: str):
        if panel_path:
            meta_path = _resolve_meta_path(data_dir, panel_path)
        else:
            meta_path = data_dir / "panel_meta.csv"
            
        if not meta_path.exists():
            return None
        try:
            meta = pd.read_csv(meta_path, usecols=["series_id", "series_key"], low_memory=False)
            meta["series_id"] = pd.to_numeric(meta["series_id"], errors="coerce")
            hits = meta[meta["series_key"].str.upper().str.contains(
                key_pattern.upper(), na=False, regex=False
            )]
            for sid in hits["series_id"].dropna().astype(int):
                if sid in cols:
                    logger.info(f"Matched target: series_id={sid} (key='{key_pattern}')")
                    return sid
        except Exception:
            pass
        return None

    if target:
        resolved = _try_match(target)
        if resolved is not None:
            return resolved
        resolved = _lookup_by_key(target)
        if resolved is not None:
            return resolved
        raise KeyError(
            f"Target '{target}' not found.\n"
            f"Panel columns (first 10): {list(cols[:10])}\n"
            f"Tip: use --target GDPC1  or  --target 2"
        )

    for gdp_key in ("GDPC1", "GDP"):
        resolved = _lookup_by_key(gdp_key)
        if resolved is not None:
            return resolved

    best = panel.notna().sum().idxmax()
    logger.warning(f"No GDP target found — using most-complete column: {best}")
    return best

def load_cf_panel(data_dir: Path, target: Optional[str], panel_arg: Optional[str] = None) -> tuple[pd.DataFrame, pd.Series]:
    panel_path = _resolve_panel_path(data_dir, panel_arg, "panel")
    logger.info(f"Loading CF Panel: {panel_path}")

    panel = _load_data_with_datetime_index(panel_path)
    panel = panel.select_dtypes(include=[np.number])

    if panel.empty:
        raise ValueError("Loaded panel is empty.")

    target_col = _resolve_target_col(target, panel, data_dir, panel_path)
    logger.info(f"Target: series_id={target_col}")

    y = panel.pop(target_col)
    return panel, y

def load_mf_panels(
    data_dir: Path,
    target: Optional[str],
    lf_freq: str = "Q",
    panel_arg: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_m = pd.DataFrame()
    X_q = pd.DataFrame()

    try:
        m_path = _resolve_panel_path(data_dir, panel_arg, "mf_panel_M")
        logger.info(f"Loading Monthly MF Panel: {m_path}")
        X_m = _load_data_with_datetime_index(m_path).select_dtypes(include=[np.number])
    except FileNotFoundError as e:
        logger.warning(e)

    try:
        q_path = _resolve_panel_path(data_dir, panel_arg, "mf_panel_Q")
        logger.info(f"Loading Quarterly MF Panel: {q_path}")
        X_q = _load_data_with_datetime_index(q_path).select_dtypes(include=[np.number])
    except FileNotFoundError as e:
        logger.warning(e)

    y = pd.Series(dtype=float)
    target_col = target

    if not X_q.empty and target_col and target_col in X_q.columns:
        y = X_q.pop(target_col)
    elif not X_m.empty and target_col and target_col in X_m.columns:
        y = X_m.pop(target_col)
        y = y.resample("QE").last().dropna()
    elif not X_q.empty:
        first_col = str(X_q.columns[0])
        logger.info(f"No target specified — using first quarterly column: {first_col}")
        y = X_q.pop(first_col)
        target_col = first_col
    elif not X_m.empty:
        first_col = str(X_m.columns[0])
        logger.info(f"No target specified — using first monthly column resampled to Q: {first_col}")
        y = X_m.pop(first_col).resample("QE").last().dropna()
        target_col = first_col

    if y.empty:
        raise ValueError(
            f"Could not find target '{target_col}' in MF panels. "
            "Run: python scripts/data_preparation.py --mode mixed_frequency"
        )

    return X_m, X_q, y
