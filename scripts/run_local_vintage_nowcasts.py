"""
Local Vintage Nowcasting Historical Evaluation Pipeline
Refactored for Bachelor Thesis to use strict pseudo-real-time vintage building.

Features:
- VintageBuilder integration (AutoARIMA ragged-edge filling, temporal aggregation)
- Configurable feature selection (fit ONLY on train)
- Rolling-window evaluation
- Detailed leakage and metadata tracking
- Automated plotting
"""
import sys
import logging
import time
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.local_data_builder import LocalDataManager
from nowcasting.data.vintage_builder import VintageBuilder
from nowcasting.models.dfm import DynamicFactorNowcast
from nowcasting.models.ml_regression import ElasticNetNowcast
from nowcasting.features.selectors import PCACompressor, VarianceFilter, LassoSelector, ElasticNetSelector
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def get_compatible_freq(freq: str) -> str:
    """Helper for pandas 1.x vs 2.x frequency compatibility."""
    if freq == "ME":
        return "M"
    if freq == "QE":
        return "Q"
    return freq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vintage_eval")

class TopNCorrelationSelector(BaseEstimator, TransformerMixin):
    """Selects top N features based on absolute Pearson correlation with target."""
    def __init__(self, top_n=50):
        self.top_n = top_n
        self.selected_cols = []
    
    def fit(self, X, y=None):
        if y is None or X.empty:
            self.selected_cols = X.columns
            return self
        
        # Align indices just in case
        valid_idx = y.dropna().index.intersection(X.index)
        corrs = X.loc[valid_idx].corrwith(y.loc[valid_idx]).abs()
        corrs = corrs.sort_values(ascending=False).dropna()
        self.selected_cols = corrs.head(self.top_n).index.tolist()
        return self
        
    def transform(self, X):
        valid_cols = [c for c in self.selected_cols if c in X.columns]
        return X[valid_cols]

def apply_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                            method: str, pca_comp: int, top_n: int,
                            alpha: float = 0.1, l1_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies feature selection STRICTLY fit on X_train."""
    if method == "none" or X_train.empty:
        return X_train, X_test
        
    steps = [("var", VarianceFilter())]
    if method == "pca":
        steps.append(("pca", PCACompressor(n_components=pca_comp)))
    elif method == "corr_top_n":
        steps.append(("corr", TopNCorrelationSelector(top_n=top_n)))
    elif method == "lasso":
        steps.append(("lasso", LassoSelector(alpha=alpha)))
    elif method == "elasticnet":
        steps.append(("elasticnet", ElasticNetSelector(alpha=alpha, l1_ratio=l1_ratio)))
        
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    X_train_sel = pipeline.transform(X_train)
    X_test_sel = pipeline.transform(X_test)
    
    return X_train_sel, X_test_sel

def map_to_target_quarter(date: pd.Timestamp) -> str:
    q = (date.month - 1) // 3 + 1
    return f"{date.year}Q{q}"

def compute_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes RMSE, MAE, and Average Revision metrics."""
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    eval_df = df.dropna(subset=["prediction", "actual"]).copy()
    
    def calc_group_metrics(group):
        err = group["prediction"] - group["actual"]
        return pd.Series({
            "n_obs": len(group),
            "rmse": np.sqrt(np.mean(err**2)),
            "mae": np.mean(np.abs(err))
        })

    metrics_by_vintage = eval_df.groupby(["model", "dataset_type", "vintage_label"], as_index=False).apply(calc_group_metrics)
    metrics_overall = eval_df.groupby(["model", "dataset_type"], as_index=False).apply(calc_group_metrics)
    metrics_overall["vintage_label"] = "overall"
    metrics_final = pd.concat([metrics_by_vintage, metrics_overall], ignore_index=True)

    # Revisions
    rev_df = df.dropna(subset=["prediction"]).copy()
    rev_df["vintage_idx"] = pd.to_numeric(rev_df["vintage_label"], errors="coerce")
    rev_df = rev_df.sort_values(["model", "dataset_type", "target_quarter", "vintage_idx"])
    rev_df["prev_pred"] = rev_df.groupby(["model", "dataset_type", "target_quarter"])["prediction"].shift(1)
    rev_df["revision"] = np.abs(rev_df["prediction"] - rev_df["prev_pred"])
    rev_df["prev_vintage"] = rev_df.groupby(["model", "dataset_type", "target_quarter"])["vintage_label"].shift(1)
    rev_df["transition"] = rev_df["prev_vintage"].astype(str) + " -> " + rev_df["vintage_label"].astype(str)
    
    rev_eval = rev_df.dropna(subset=["revision"])
    rev_metrics = rev_eval.groupby(["model", "dataset_type", "transition"], as_index=False).agg(
        avg_revision=("revision", "mean"), n_revision_pairs=("revision", "count")
    )
    rev_overall = rev_eval.groupby(["model", "dataset_type"], as_index=False).agg(
        avg_revision=("revision", "mean"), n_revision_pairs=("revision", "count")
    )
    rev_overall["transition"] = "overall"
    revisions_final = pd.concat([rev_metrics, rev_overall], ignore_index=True)
    
    if "seed" in df.columns and not metrics_final.empty:
        metrics_final["seed"] = df["seed"].iloc[0]
        revisions_final["seed"] = df["seed"].iloc[0]
        
    return metrics_final, revisions_final

# Ensure plot outputs are clean
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def parse_target_quarter(tq):
    """Standardized quarter parsing: '2020Q1' -> Timestamp('2020-03-31')"""
    try:
        p = pd.Period(str(tq), freq="Q")
        return p.to_timestamp(how="end").normalize()
    except Exception:
        return pd.NaT

def normalize_vintage_label(v):
    """Standardized vintage label: '+2' -> '2', '-1' -> '-1'"""
    return str(v).replace("+", "")

def extract_prediction_for_target(pred_s, target_q_end, model_name="", target_col=None, return_mode=False):
    """
    Robustly extracts a single prediction value for target_q_end from various model outputs.
    Handles Series, DataFrame, scalar, and date alignment.
    """
    if pred_s is None:
        return (np.nan, "failed") if return_mode else np.nan
        
    # 1. Handle non-series types
    if isinstance(pred_s, (float, int)):
        return (float(pred_s), "single_value") if return_mode else float(pred_s)
    if isinstance(pred_s, (list, np.ndarray)):
        if len(pred_s) == 1:
            return (float(pred_s[0]), "single_value") if return_mode else float(pred_s[0])
        elif len(pred_s) == 0:
            return (np.nan, "failed") if return_mode else np.nan
        # If multiple values, we continue to check if it's a Series later
        pred_s = pd.Series(pred_s)

    # 2. Handle DataFrame
    if isinstance(pred_s, pd.DataFrame):
        if target_col and target_col in pred_s.columns:
            pred_s = pred_s[target_col]
        else:
            # Pick first numeric column
            numeric_cols = pred_s.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                pred_s = pred_s[numeric_cols[0]]
            else:
                return (np.nan, "failed") if return_mode else np.nan

    if not isinstance(pred_s, pd.Series):
        return (np.nan, "failed") if return_mode else np.nan

    if pred_s.empty:
        return (np.nan, "failed") if return_mode else np.nan

    # 3. Handle Index alignment
    # Ensure index is datetime-like
    if isinstance(pred_s.index, pd.PeriodIndex):
        pred_s.index = pred_s.index.to_timestamp(how="end").normalize()
    elif not isinstance(pred_s.index, pd.DatetimeIndex):
        # If it's a simple RangeIndex and length 1, assume it corresponds to target
        if len(pred_s) == 1:
            logger.info(f"[{model_name}] Extracted single value from non-datetime index.")
            return (float(pred_s.iloc[0]), "single_value") if return_mode else float(pred_s.iloc[0])
        return (np.nan, "failed") if return_mode else np.nan

    # Normalize pred index
    pred_s.index = pred_s.index.normalize()
    target_ts = pd.Timestamp(target_q_end).normalize()

    # Strategy A: Exact match
    if target_ts in pred_s.index:
        val = pred_s.loc[target_ts]
        logger.info(f"[{model_name}] Extracted prediction via exact_match at {target_ts.date()}")
        val_float = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
        return (val_float, "exact_match") if return_mode else val_float

    # Strategy B: Same Quarter match
    target_period = pd.Period(target_ts, freq="Q")
    pred_periods = pred_s.index.to_period("Q")
    mask_q = (pred_periods == target_period)
    if mask_q.any():
        val = pred_s[mask_q].iloc[-1]
        logger.info(f"[{model_name}] Extracted prediction via same_quarter ({target_period}) at {pred_s[mask_q].index[-1].date()}")
        return (float(val), "same_quarter") if return_mode else float(val)

    # Strategy C: Same Month match
    target_month = target_ts.to_period("M")
    pred_months = pred_s.index.to_period("M")
    mask_m = (pred_months == target_month)
    if mask_m.any():
        val = pred_s[mask_m].iloc[-1]
        logger.info(f"[{model_name}] Extracted prediction via same_month ({target_month}) at {pred_s[mask_m].index[-1].date()}")
        return (float(val), "same_month") if return_mode else float(val)

    # Strategy D: Length 1 fallback
    if len(pred_s) == 1:
        logger.info(f"[{model_name}] Extracted only available value at {pred_s.index[0].date()}")
        return (float(pred_s.iloc[0]), "single_value") if return_mode else float(pred_s.iloc[0])

    # Strategy E: Last value fallback (with warning)
    logger.warning(f"[{model_name}] No date match for {target_ts.date()}. Falling back to last available value at {pred_s.index[-1].date()}.")
    val_float = float(pred_s.iloc[-1])
    return (val_float, "fallback_last") if return_mode else val_float


def sanitize_filename_part(s: str) -> str:
    return (
        str(s)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "")
        .replace("=", "")
        .replace(",", "_")
        .replace(":", "-")
    )


def get_model_config_label_from_df(df: pd.DataFrame, file_path: Path | None = None) -> str:
    model = str(df["model"].iloc[0]) if "model" in df.columns else ""

    # Prefer metadata columns if available
    if model == "ElasticNet":
        if "elasticnet_use_cv" in df.columns and df["elasticnet_use_cv"].iloc[0] in [False, "False", 0, "0"]:
            alpha = df.get("elasticnet_alpha", pd.Series([""])).iloc[0]
            l1 = df.get("elasticnet_l1_ratio", pd.Series([""])).iloc[0]
            label = f"fixed_a{alpha}_l1{l1}"
        else:
            label = "cv"

        # Add selector suffix to make plot names unique.
        selector = df.get("selector", pd.Series([None])).iloc[0] if "selector" in df.columns else None

        if pd.notna(selector) and str(selector) not in {"", "none", "nan", "None"}:
            selector = str(selector)

            if selector == "pca":
                pca_components = df.get("pca_components", pd.Series([""])).iloc[
                    0] if "pca_components" in df.columns else ""
                label += f"_sel_pca{pca_components}"

            elif selector == "corr_top_n":
                top_n = df.get("top_n", pd.Series([""])).iloc[0] if "top_n" in df.columns else ""
                label += f"_sel_top{top_n}"

            elif selector == "lasso":
                sel_alpha = df.get("selector_alpha", pd.Series([""])).iloc[0] if "selector_alpha" in df.columns else ""
                label += f"_sel_lasso_a{sel_alpha}"

            elif selector == "elasticnet":
                sel_alpha = df.get("selector_alpha", pd.Series([""])).iloc[0] if "selector_alpha" in df.columns else ""
                sel_l1 = df.get("selector_l1_ratio", pd.Series([""])).iloc[
                    0] if "selector_l1_ratio" in df.columns else ""
                label += f"_sel_elasticnet_a{sel_alpha}_l1{sel_l1}"

            else:
                label += f"_sel_{selector}"

        # Append TACTiS-2 parameters if applicable
        fill_method = df.get("fill_method", pd.Series([""])).iloc[0] if "fill_method" in df.columns else ""

        if str(fill_method) == "tactis2":
            pretop = df.get("preselect_before_fill_top_k", pd.Series([None])).iloc[0] if "preselect_before_fill_top_k" in df.columns else None
            if pd.notna(pretop) and str(pretop) not in {"", "None", "nan"}:
                try:
                    pretop_clean = int(float(pretop))
                except Exception:
                    pretop_clean = pretop
                label += f"_pretop{pretop_clean}"

            epochs = df.get("tactis2_max_epochs", pd.Series([""])).iloc[0] if "tactis2_max_epochs" in df.columns else ""
            batch = df.get("tactis2_batch_size", pd.Series([""])).iloc[0] if "tactis2_batch_size" in df.columns else ""
            nb = df.get("tactis2_num_batches_per_epoch", pd.Series([""])).iloc[0] if "tactis2_num_batches_per_epoch" in df.columns else ""
            samples = df.get("tactis2_num_samples", pd.Series([""])).iloc[0] if "tactis2_num_samples" in df.columns else ""
            ctx = df.get("tactis2_context_length", pd.Series([""])).iloc[0] if "tactis2_context_length" in df.columns else ""

            skip = df.get("tactis2_skip_copula", pd.Series([None])).iloc[0] if "tactis2_skip_copula" in df.columns else None
            if str(skip).lower() in {"true", "1"}:
                cop_flag = 1
            elif str(skip).lower() in {"false", "0"}:
                cop_flag = 0
            else:
                cop_flag = "na"

            def clean_num(x):
                try:
                    if pd.isna(x):
                        return ""
                    xf = float(x)
                    return int(xf) if xf.is_integer() else xf
                except Exception:
                    return x

            label += (
                f"_tacte{clean_num(epochs)}"
                f"_b{clean_num(batch)}"
                f"_nb{clean_num(nb)}"
                f"_samp{clean_num(samples)}"
                f"_cop{cop_flag}"
                f"_ctx{clean_num(ctx)}"
            )

        return label

    if model == "MIDAS":
        parts = []
        if "midas_regression_model" in df.columns:
            parts.append(str(df["midas_regression_model"].iloc[0]))
        if "midas_n_lags" in df.columns:
            parts.append(f"lags{df['midas_n_lags'].iloc[0]}")
        if "midas_internal_fill_strategy" in df.columns:
            parts.append(f"fill{df['midas_internal_fill_strategy'].iloc[0]}")
        if parts:
            return "_".join(parts)

    if model == "MIDASML":
        parts = []
        if "midasml_regression_model" in df.columns:
            parts.append(str(df["midasml_regression_model"].iloc[0]))
        if "midas_n_lags" in df.columns:
            parts.append(f"lags{df['midas_n_lags'].iloc[0]}")
        if "midasml_cv" in df.columns:
            parts.append(f"cv{df['midasml_cv'].iloc[0]}")
        if "midasml_l1_ratio" in df.columns:
            parts.append(f"l1{df['midasml_l1_ratio'].iloc[0]}")
        if "midas_internal_fill_strategy" in df.columns:
            parts.append(f"fill{df['midas_internal_fill_strategy'].iloc[0]}")
        if parts:
            return "_".join(parts)

    if model == "DFM":
        parts = []

        if "quarterly_aggregation" in df.columns:
            parts.append(f"agg{df['quarterly_aggregation'].iloc[0]}")

        if "dfm_k_factors" in df.columns:
            parts.append(f"k{df['dfm_k_factors'].iloc[0]}")

        if "dfm_factor_order" in df.columns:
            parts.append(f"p{df['dfm_factor_order'].iloc[0]}")

        selector = None
        if "dfm_selector" in df.columns:
            selector = df["dfm_selector"].iloc[0]
            parts.append(f"sel{selector}")

        if str(selector) == "pca" and "dfm_pca_components" in df.columns:
            pca_val = df["dfm_pca_components"].iloc[0]
            if pd.notna(pca_val) and str(pca_val) not in {"", "None", "nan"}:
                try:
                    pca_val = int(float(pca_val))
                except Exception:
                    pass
                parts.append(f"pca{pca_val}")

        if str(selector) == "corr_top_n":
            top_val = None

            if "top_n" in df.columns:
                top_val = df["top_n"].iloc[0]

            if (pd.isna(top_val) or str(top_val) in {"", "None", "nan"}) and file_path is not None:
                import re
                m = re.search(r"selcorr_top_n_top(\d+)", str(file_path).lower())
                if m:
                    top_val = m.group(1)

            if pd.notna(top_val) and str(top_val) not in {"", "None", "nan"}:
                try:
                    top_val = int(float(top_val))
                except Exception:
                    pass
                parts.append(f"top{top_val}")

        # Do not append infounknown. It is not a useful plot/config label.
        if "dfm_prediction_information_set" in df.columns:
            info_val = df["dfm_prediction_information_set"].iloc[0]
            if pd.notna(info_val) and str(info_val).lower() not in {"", "none", "nan", "unknown"}:
                parts.append(f"info{info_val}")

        if parts:
            return "_".join(parts)

    if model == "DFM_MF":
        parts = []

        if "dfm_k_factors" in df.columns:
            parts.append(f"k{df['dfm_k_factors'].iloc[0]}")

        if "dfm_factor_order" in df.columns:
            parts.append(f"p{df['dfm_factor_order'].iloc[0]}")

        if "dfm_mf_selector" in df.columns:
            parts.append(f"mfsel{df['dfm_mf_selector'].iloc[0]}")

        if "dfm_mf_top_n" in df.columns:
            top_val = df["dfm_mf_top_n"].iloc[0]
            if pd.notna(top_val) and str(top_val) not in {"", "None", "nan"}:
                try:
                    top_val = int(float(top_val))
                except Exception:
                    pass
                parts.append(f"top{top_val}")

        # Do not append infounknown.
        if "dfm_prediction_information_set" in df.columns:
            info_val = df["dfm_prediction_information_set"].iloc[0]
            if pd.notna(info_val) and str(info_val).lower() not in {"", "none", "nan", "unknown"}:
                parts.append(f"info{info_val}")

        if parts:
            return "_".join(parts)

    # Fallback: infer from filename
    if file_path is not None:
        stem = file_path.stem
        # Remove leading vintage_nowcasts_ if present
        stem = stem.replace("vintage_nowcasts_", "")
        return stem

    return ""

def pretty_dataset_name(dataset_name: str) -> str:
    """Convert internal dataset names to readable Lithuanian plot labels."""
    mapping = {
        "baseline_common": "be GT",
        "final_thesis_baseline_common": "be GT",

        "common_plus_gt": "su GT",
        "final_thesis_common_plus_gt": "su GT",

        "common_plus_gt_v1": "su GT (angl.)",
        "final_thesis_common_plus_gt_v1": "su GT (angl.)",

        "common_plus_gt_lt": "su GT (liet.)",
        "final_thesis_common_plus_gt_lt": "su GT (liet.)",

        "gt_only": "tik GT",
        "final_thesis_gt_only": "tik GT",

        "gt_only_v1": "tik GT (angl.)",
        "final_thesis_gt_only_v1": "tik GT (angl.)",

        "gt_only_lt": "tik GT (liet.)",
        "final_thesis_gt_only_lt": "tik GT (liet.)",
    }
    return mapping.get(str(dataset_name), str(dataset_name))


def pretty_fill_method(fill_method: str) -> str:
    """Convert internal fill method names to readable Lithuanian plot labels."""
    mapping = {
        "vertical_realignment": "vertikalus suderinimas",
        "autoarima": "AutoARIMA",
        "locf": "paskutinė žinoma reikšmė",
        "rolling_mean": "slenkantis vidurkis",
        "tactis2": "TACTiS-2",
        "none": "be užpildymo",
        "native_ragged": "",
    }
    return mapping.get(str(fill_method), str(fill_method))


def pretty_vintage_label(vintage_label: str) -> str:
    """Format vintage label for thesis plots."""
    v = normalize_vintage_label(vintage_label)
    try:
        v_int = int(v)
        sign = "+" if v_int > 0 else ""
        return f"Prognozuotas {sign}{v_int}"
    except ValueError:
        return f"Prognozuotas {v}"


def pretty_model_display(model_name: str, config_label: str = "") -> str:
    """Convert model + config label to a cleaner title label."""
    model_name = str(model_name)
    config_label = str(config_label or "")

    if not config_label:
        return model_name

    if model_name == "ElasticNet" and config_label.startswith("fixed_a"):
        try:
            rest = config_label.replace("fixed_a", "", 1)
            alpha_part, rest2 = rest.split("_l1", 1)
            l1_part = rest2
            for sep in ["_sel", "_pretop", "_tacte", "_vr", "_calendar_blocks", "_most_recent_lags"]:
                if sep in l1_part:
                    l1_part = l1_part.split(sep, 1)[0]
            return f"{model_name} [alpha={alpha_part}, l1={l1_part}]"
        except Exception:
            return f"{model_name} [{config_label}]"

    return f"{model_name} [{config_label}]"

def plot_single_file(file_path: Path, out_dir: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    if "prediction" not in df.columns:
        return

    df = df.dropna(subset=["prediction"])
    if df.empty:
        return

    df["q_end"] = df["target_quarter"].apply(parse_target_quarter)
    df["vintage_label"] = df["vintage_label"].apply(normalize_vintage_label)
    df = df.sort_values("q_end")

    model_name = df["model"].iloc[0]
    dataset_name = df["dataset_type"].iloc[0]
    fill_method = df["fill_method"].iloc[0] if "fill_method" in df.columns else "unknown"

    config_label = get_model_config_label_from_df(df, file_path)
    config_label_clean = sanitize_filename_part(config_label)

    model_display = pretty_model_display(model_name, config_label)
    dataset_display = pretty_dataset_name(dataset_name)
    fill_display = pretty_fill_method(fill_method)

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot actual GDP
    actuals = df.dropna(subset=["actual"]).drop_duplicates("q_end")
    if not actuals.empty:
        ax.plot(
            actuals["q_end"],
            actuals["actual"],
            color="#1f77b4",
            linewidth=3.0,
            marker="o",
            markersize=5,
            label="Realus BVP",
            zorder=5,
        )

    # Plot forecasts by vintage
    vintages = sorted(
        df["vintage_label"].astype(str).unique(),
        key=lambda x: int(normalize_vintage_label(x)),
    )

    vintage_styles = {
        "-2": {"color": "#ff7f0e", "linestyle": "-", "marker": "o"},
        "-1": {"color": "#2ca02c", "linestyle": "--", "marker": "s"},
        "0": {"color": "#d62728", "linestyle": "-.", "marker": "^"},
        "1": {"color": "#9467bd", "linestyle": ":", "marker": "D"},
        "2": {"color": "#8c564b", "linestyle": (0, (5, 1)), "marker": "v"},
    }

    for v_label in vintages:
        v_label_norm = normalize_vintage_label(v_label)
        v_data = df[df["vintage_label"].astype(str) == v_label_norm]
        style = vintage_styles.get(
            v_label_norm,
            {"color": "gray", "linestyle": "-", "marker": "x"},
        )

        ax.plot(
            v_data["q_end"],
            v_data["prediction"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.2,
            markersize=4,
            alpha=0.95,
            label=pretty_vintage_label(v_label_norm),
            zorder=4,
        )

    if fill_display:
        title = (
            f"{model_display} ({fill_display}) prognozės pagal informacijos "
            f"prieinamumo momentą ({dataset_display})"
        )
    else:
        title = (
            f"{model_display} prognozės pagal informacijos "
            f"prieinamumo momentą ({dataset_display})"
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Data", fontsize=13)
    ax.set_ylabel("BVP", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if config_label_clean:
        out_name = f"plot_{fill_method}_{model_name}_{config_label_clean}_{dataset_name}.png"
    else:
        out_name = f"plot_{fill_method}_{model_name}_{dataset_name}.png"

    ax.figure.savefig(out_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_dir / out_name}")


def compare_datasets(base_dir: Path, out_dir: Path, model: str, fill_method: str, seed: int = 2234):
    """Plots datasets for the same model/fill method side-by-side or loops through and generates individual standard plots."""
    dataset_pairs = [
        ("final_thesis_baseline_common", "baseline_common"),
        ("final_thesis_common_plus_gt", "common_plus_gt"),
        ("final_thesis_gt_only", "gt_only")
    ]
    
    for preferred, fallback in dataset_pairs:
        # Try preferred first
        file_name = f"vintage_nowcasts_{fill_method}_{model}_{preferred}_s{seed}.csv"
        file_path = base_dir / file_name
        if file_path.exists():
            plot_single_file(file_path, out_dir)
            continue
            
        # Then fallback
        file_name = f"vintage_nowcasts_{fill_method}_{model}_{fallback}_s{seed}.csv"
        file_path = base_dir / file_name
        if file_path.exists():
            plot_single_file(file_path, out_dir)
        else:
            print(f"Warning: Missing data for {preferred} or {fallback}")


def compare_fills(base_dir: Path, out_dir: Path, model: str, dataset: str, vintage: str, seed: int = 2234):
    """Compares different fill methods for the SAME vintage on one plot."""
    fills = ["locf", "autoarima", "vertical_realignment", "tactis2"]
    dfs = {}

    for fill in fills:
        file_name = f"vintage_nowcasts_{fill}_{model}_{dataset}_s{seed}.csv"
        file_path = base_dir / file_name

        if file_path.exists():
            df = pd.read_csv(file_path)

            v_label_norm = normalize_vintage_label(vintage)
            df["_vintage_norm"] = df["vintage_label"].apply(normalize_vintage_label)
            v_data = df[df["_vintage_norm"] == v_label_norm].copy()

            if not v_data.empty:
                v_data["q_end"] = v_data["target_quarter"].apply(parse_target_quarter)
                dfs[fill] = v_data.sort_values("q_end")
        else:
            print(f"Warning: Missing data for {fill} -> {file_name}")

    if not dfs:
        print("No valid data found to compare fills.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot actual GDP from first available dataframe
    first_df = list(dfs.values())[0]
    actuals = first_df.dropna(subset=["actual"]).drop_duplicates("q_end")
    if not actuals.empty:
        ax.plot(
            actuals["q_end"],
            actuals["actual"],
            color="#1f77b4",
            linewidth=3.0,
            marker="o",
            markersize=5,
            label="Realus BVP",
            zorder=5,
        )

    colors = ["orange", "green", "purple", "red"]
    markers = ["s", "^", "D", "o"]

    for (fill, df_fill), color, marker in zip(dfs.items(), colors, markers):
        fill_display = pretty_fill_method(fill)

        ax.plot(
            df_fill["q_end"],
            df_fill["prediction"],
            marker=marker,
            linestyle="--",
            color=color,
            linewidth=2,
            label=fill_display,
        )

    vintage_display = pretty_vintage_label(vintage)
    dataset_display = pretty_dataset_name(dataset)

    ax.set_title(
        f"Užpildymo metodų palyginimas ({model}) | {vintage_display} | {dataset_display}",
        fontsize=16,
    )
    ax.set_xlabel("Data", fontsize=13)
    ax.set_ylabel("BVP", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_name = f"comparison_fills_{model}_{dataset}_v{normalize_vintage_label(vintage)}.png"
    ax.figure.savefig(out_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_dir / out_name}")



CHECKPOINT_COLUMNS = [
    "seed",
    "model",
    "dataset_type",
    "target_quarter",
    "vintage_label",
    "cutoff_date",
    "prediction",
    "actual",
    "runtime_sec",
    "n_features_raw",
    "n_features_sel",
    "fill_method",
    "series_sktime_autoarima_ok",
    "series_sktime_autoarima_failed_locf",
    "series_autoarima_cached_sqlite",
    "series_autoarima_cached_memory",
    "series_autoarima_too_short_locf",
    "series_tactis2_ok",
    "series_tactis2_failed_fallback",
    "tactis2_audit_entries",
    "tactis2_runtime_sec",
    "tactis2_context_length",
    "tactis2_prediction_length",
    "tactis2_max_epochs",
    "tactis2_author_config",
    "tactis2_batch_size",
    "tactis2_num_batches_per_epoch",
    "tactis2_epochs_phase_1",
    "tactis2_epochs_phase_2",
    "tactis2_learning_rate",
    "tactis2_weight_decay",
    "tactis2_maximum_learning_rate",
    "tactis2_clip_gradient",
    "tactis2_bagging_size",
    "tactis2_skip_copula",
    "tactis2_num_samples",
    "tactis2_origin_groups",
    "tactis2_values_forecasted",
    "tactis2_values_ffill_fallback",
    "tactis2_group_failures",
    "tactis2_origin_dates_used",
    "preselect_before_fill_method",
    "preselect_before_fill_top_k",
    "preselect_before_fill_features_before",
    "preselect_before_fill_features_after",
    "vertical_realignment_mode",
    "vertical_realignment_features_total",
    "vertical_realignment_blocks_ffilled",
    "vertical_realignment_realigned_values",
    "vertical_realignment_missing_lag_values",
    "selector",
    "selector_alpha",
    "selector_l1_ratio",
    "missing_before_final_ffill",
    "missing_after_final_ffill",
    "values_filled_by_final_ffill",
    "pred_extraction_mode",
    "train_q_size",
    "dfm_prediction_information_set",
    "elasticnet_use_cv",
    "elasticnet_alpha",
    "elasticnet_l1_ratio",
    "elasticnet_max_iter",
    "selector",
    "pca_components",
    "top_n",
    "selector_alpha",
    "midas_n_lags",
    "midas_regression_model",
    "midas_internal_fill_strategy",
    "midasml_regression_model",
    "midasml_cv",
    "midasml_l1_ratio",
    "midasml_max_iter",
    "quarterly_aggregation",
    "dfm_k_factors",
    "dfm_factor_order",
    "dfm_selector",
    "dfm_pca_components",
    "dfm_mf_selector",
    "dfm_mf_top_n"
]

def safe_read_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    if not checkpoint_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(checkpoint_path)
    except Exception as e:
        logger.warning(f"Checkpoint CSV is malformed: {checkpoint_path}. Error: {e}")
        backup_path = checkpoint_path.with_suffix(".corrupt.csv")
        try:
            checkpoint_path.rename(backup_path)
            logger.warning(f"Moved corrupt checkpoint to: {backup_path.name}")
        except Exception as rename_err:
            logger.error(f"Failed to rename corrupt checkpoint: {rename_err}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Bachelor Thesis GDP Nowcasting Pipeline")
    parser.add_argument("--dataset", type=str, default="baseline_common", help="Dataset to evaluate")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated list of datasets to evaluate. Overrides --dataset.")
    parser.add_argument("--model", type=str, default="ElasticNet", choices=["ElasticNet", "DFM", "DFM_MF", "MIDAS", "MIDASML"], help="Model to run")
    parser.add_argument("--start-date", type=str, default="2019-03-31", help="First target quarter to evaluate")
    parser.add_argument("--end-date", type=str, default="2026-06-30", help="Last target quarter to evaluate")
    parser.add_argument("--train-start", type=str, default="2000-01-01", help="Earliest date for training window")
    parser.add_argument("--rolling-window-quarters", type=int, default=76, help="Fixed rolling window size (in quarters of target GDP). 0 for expanding.")
    parser.add_argument("--vintages", type=str, default="-2,-1,0,1,2", help="Comma-separated list of vintages")
    parser.add_argument("--ragged-fill-method", type=str, default="autoarima", choices=["autoarima", "locf", "rolling_mean", "vertical_realignment", "tactis2", "none"], help="Ragged edge fill method (only for ElasticNet)")
    parser.add_argument("--quarterly-aggregation", type=str, default="mean", choices=["mean", "last", "sum"], help="Monthly to quarterly aggregation method")
    parser.add_argument("--selector", type=str, default="pca", choices=["none", "variance", "pca", "corr_top_n", "lasso", "elasticnet"], help="Feature selection method (primarily for ElasticNet)")
    parser.add_argument("--selector-alpha", type=float, default=0.1, help="Alpha for lasso/elasticnet selector")
    parser.add_argument("--selector-l1-ratio", type=float, default=0.5, help="L1 ratio for elasticnet selector")
    parser.add_argument("--pca-components", type=int, default=10, help="Number of PCA components")
    parser.add_argument("--dfm-maxiter", type=int, default=50, help="Max iterations for DFM fitting")
    parser.add_argument("--dfm-tolerance", type=float, default=1e-5, help="Convergence tolerance for DFM")
    parser.add_argument("--top-n", type=int, default=50, help="Top N features for corr_top_n selector")
    parser.add_argument("--seed", type=int, default=2234, help="Random seed")
    parser.add_argument("--preselect-top-k-before-fill", type=int, default=None, help="Pre-select top K features by absolute correlation with target BEFORE ragged-edge filling (primarily for TACTiS-2 performance).")
    parser.add_argument("--debug-preselect-top-k", type=int, default=None, help="Backward compatible alias for --preselect-top-k-before-fill.")
    parser.add_argument("--monthly-feature-release-lag-months", type=int, default=1, help="Simulated release lag for monthly macro features.")
    parser.add_argument("--gt-release-lag-months", type=int, default=0, help="Simulated release lag for Google Trends features.")
    parser.add_argument("--quarterly-feature-release-lag-months", type=int, default=1, help="Simulated release lag for quarterly macro features.")
    parser.add_argument("--arima-n-jobs", type=int, default=1, help="Number of parallel jobs for AutoARIMA across series.")
    parser.add_argument("--arima-fast", action="store_true", help="Use a faster (max p=1, q=1, P=0, Q=0) AutoARIMA search space.")
    parser.add_argument("--elasticnet-no-cv", action="store_true", help="Use fixed alpha/l1_ratio instead of CV-based ElasticNet.")
    parser.add_argument("--elasticnet-alpha", type=float, default=1e-5, help="Fixed alpha for ElasticNet when --elasticnet-no-cv is used.")
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.25, help="Fixed l1_ratio for ElasticNet when --elasticnet-no-cv is used.")
    parser.add_argument("--elasticnet-max-iter", type=int, default=10000, help="Max iterations for ElasticNet.")
    parser.add_argument("--arima-seasonal", type=str, default="true", choices=["true", "false"], help="Enable/disable seasonal AutoARIMA.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-quarter evaluation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and only run plots.")
    parser.add_argument("--plot-compare-datasets", action="store_true", help="Plot baseline_common vs common_plus_gt vs gt_only side-by-side.")
    parser.add_argument("--plot-compare-fills", action="store_true", help="Plot locf vs autoarima vs vertical_realignment for the specified --dataset and --plot-vintage.")
    parser.add_argument("--plot-vintage", type=str, default="0", help="Vintage label to compare across fills when using --plot-compare-fills.")
    parser.add_argument("--plot-glob", type=str, default="", help="When --skip-eval is used, plot every forecast CSV matching this glob.")
    parser.add_argument("--arima-cache-path", type=str, default=None, help="Path to SQLite arima cache database.")
    parser.add_argument("--allow-backcast-target-in-train", action="store_true", help="If True, allow y_train to include the target quarter for vintages +1/+2.")
    parser.add_argument("--vertical-realignment-mode", type=str, default="calendar_blocks", choices=["calendar_blocks", "most_recent_lags"], help="Mode for vertical realignment filling.")
    
    # DFM Parameters
    parser.add_argument("--dfm-k-factors", type=int, default=2, help="Number of factors for DFM")
    parser.add_argument("--dfm-factor-order", type=int, default=1, help="Factor AR order for DFM")
    parser.add_argument("--dfm-selector", type=str, default="none", choices=["none", "pca", "corr_top_n", "lasso", "elasticnet"], help="Selector for DFM features")
    parser.add_argument("--dfm-pca-components", type=int, default=10, help="PCA components for DFM if selector=pca")
    parser.add_argument("--dfm-mf-selector", type=str, default="corr_top_n", choices=["none", "corr_top_n"], help="Selector for DFM_MF features")
    parser.add_argument("--dfm-mf-top-n", type=int, default=80, help="Top N correlation features for DFM_MF")

    # MIDAS Parameters
    parser.add_argument("--midas-n-lags", type=int, default=4, help="Number of low-freq lags for MIDAS")
    parser.add_argument("--midas-regression-model", type=str, default="ridge", help="Regression backend for MIDAS")
    parser.add_argument("--midas-internal-fill-strategy", type=str, default="ffill_then_zero", help="Internal fill strategy for MIDAS lagged matrix")

    # MIDASML Parameters
    parser.add_argument("--midasml-regression-model", type=str, default="elasticnet", help="Regression backend for MIDASML")
    parser.add_argument("--midasml-cv", type=int, default=3, help="CV folds for MIDASML")
    parser.add_argument("--midasml-l1-ratio", type=float, default=0.5, help="L1 ratio for ElasticNet in MIDASML")
    parser.add_argument("--midasml-max-iter", type=int, default=3000, help="Max iterations for MIDASML")
    
    # Checkpoint/Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
    parser.add_argument("--checkpoint-dir", type=str, default="data/forecasts/checkpoints", help="Directory for checkpoint CSVs.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Append to checkpoint CSV every N predictions.")
    parser.add_argument("--ignore-corrupt-checkpoint", action="store_true", help="If checkpoint fails to parse, rename and start fresh.")
    
    # TACTiS-2 Parameters
    parser.add_argument("--tactis2-author-config", action="store_true", help="Use author-style heavy configuration for TACTiS2.")
    parser.add_argument("--tactis2-max-epochs", type=int, default=20, help="Max epochs for TACTiS2 training (simplified mode).")
    parser.add_argument("--tactis2-epochs-phase-1", type=int, default=20, help="Phase 1 epochs for TACTiS2 (author mode).")
    parser.add_argument("--tactis2-epochs-phase-2", type=int, default=20, help="Phase 2 epochs for TACTiS2 (author mode).")
    parser.add_argument("--tactis2-batch-size", type=int, default=None, help="Batch size for TACTiS2 (Default: 32 simplified, 256 author).")
    parser.add_argument("--tactis2-num-batches-per-epoch", type=int, default=None, help="Number of batches per epoch for TACTiS2 (Default: 32 simplified, 512 author).")
    parser.add_argument("--tactis2-learning-rate", type=float, default=1e-3, help="Learning rate for TACTiS2.")
    parser.add_argument("--tactis2-weight-decay", type=float, default=1e-4, help="Weight decay for TACTiS2.")
    parser.add_argument("--tactis2-maximum-learning-rate", type=float, default=1e-3, help="Maximum learning rate for TACTiS2.")
    parser.add_argument("--tactis2-clip-gradient", type=float, default=1e3, help="Clip gradient for TACTiS2.")
    parser.add_argument("--tactis2-bagging-size", type=int, default=None, help="Bagging size for TACTiS2 (Default: None simplified, 20 author).")
    parser.add_argument("--tactis2-skip-copula", type=str, default=None, choices=["true", "false"], help="Skip copula training in TACTiS2 (Default: true simplified, false author).")
    parser.add_argument("--tactis2-context-length", type=int, default=120, help="Context length (history window) for TACTiS2.")
    parser.add_argument("--tactis2-num-samples", type=int, default=None, help="Number of samples to generate from TACTiS2 (Default: 20 simplified, 100 author).")
    parser.add_argument("--tactis2-device", type=str, default="auto", help="Torch device for TACTiS2.")
    parser.add_argument("--tactis2-force-refit", action="store_true", help="Force retraining TACTiS2 even if cached result exists.")
    
    # Experimental Data Parameters
    parser.add_argument("--gt-suffix", type=str, default="", help="Suffix for Google Trends datasets (e.g. lt or v1)")

    args = parser.parse_args()

    # Alias handling
    if args.preselect_top_k_before_fill is None and args.debug_preselect_top_k is not None:
        args.preselect_top_k_before_fill = args.debug_preselect_top_k

    project_root = Path(__file__).resolve().parent.parent
    base_out_dir = project_root / "data" / "forecasts"
    plots_dir = base_out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine fill method for plotting (native models use 'native_ragged')
    native_models = ["DFM", "DFM_MF", "MIDAS", "MIDASML"]
    plot_fill_method = "native_ragged" if args.model in native_models else args.ragged_fill_method

    if args.skip_eval:
        if args.plot_compare_datasets:
            compare_datasets(base_out_dir, plots_dir, args.model, plot_fill_method, args.seed)
        elif args.plot_compare_fills:
            compare_fills(base_out_dir, plots_dir, args.model, args.dataset, args.plot_vintage, args.seed)
        elif args.plot_glob:
            import glob
            # Handle both absolute and relative globs
            if Path(args.plot_glob).is_absolute():
                files = [Path(f) for f in glob.glob(args.plot_glob)]
            else:
                files = sorted(Path(project_root).glob(args.plot_glob))
            
            if not files:
                logger.warning(f"No files found matching glob: {args.plot_glob}")
            else:
                logger.info(f"Plotting {len(files)} files from glob...")
                for file_path in files:
                    # If plotting files from an experiment run folder like:
                    # data/forecasts/2026.0508_test_lag0_selectors/results/*.csv
                    # save plots into the sibling:
                    # data/forecasts/2026.0508_test_lag0_selectors/plots/
                    if file_path.parent.name == "results":
                        target_plots_dir = file_path.parent.parent / "plots"
                    else:
                        target_plots_dir = plots_dir

                    target_plots_dir.mkdir(parents=True, exist_ok=True)
                    plot_single_file(file_path, target_plots_dir)
        else:
            logger.info("Skipping evaluation, but no plot arguments provided (--plot-compare-datasets, --plot-compare-fills, or --plot-glob).")
        return

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("="*60)
    logger.info(f"=== THESIS EVALUATION MODE | MODEL: {args.model} ===")
    logger.info("="*60)
    
    if args.allow_backcast_target_in_train:
        logger.info("Evaluation mode: target quarter GDP is ALLOWED in training for late vintages (+1, +2).")
    else:
        logger.info("Evaluation mode: target quarter GDP is withheld from training for all vintages, including +2. Therefore +2 is a late nowcast, not an in-sample fitted value.")
    
    # -------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------
    dm = LocalDataManager(_PROJECT_ROOT)
    
    if args.datasets:
        datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets_to_run = [args.dataset]
        
    for ds_name in datasets_to_run:
        logger.info("="*60)
        logger.info(f"=== EVALUATING DATASET: {ds_name} ===")
        logger.info("="*60)
        if ds_name.startswith("final_thesis_"):
            X, y, source_info = dm.load_frequency_aware_data(ds_name.replace("final_thesis_", ""), gt_suffix=args.gt_suffix)
        else:
            X, y, source_info = dm.load_or_build_dataset(ds_name, force_rebuild=False)
    
        # Ensure targets are Quarterly
        y = y.resample('Q').last()
    
        # -------------------------------------------------------------------
        # Configure Pipeline
        # -------------------------------------------------------------------
        is_seasonal = (args.arima_seasonal.lower() == "true")
        vb = VintageBuilder(
            vintage_label_mode="month_relative_to_quarter_end",
            min_obs_per_series=36,
            random_state=args.seed,
            dataset_name=ds_name,
            seasonal=is_seasonal,
            arima_fast=args.arima_fast,
            arima_n_jobs=args.arima_n_jobs,
            arima_cache_path=args.arima_cache_path
        )
    
        start_q = pd.Timestamp(args.start_date)
        end_q = min(pd.Timestamp(args.end_date), y.dropna().index.max() + pd.offsets.MonthEnd(6))
    
        target_quarters = pd.date_range(start_q, end_q, freq='Q')
        if args.smoke_test:
            target_quarters = target_quarters[:2] # First 2 quarters
        
        vintages_to_run = [v.strip() for v in args.vintages.split(",")]
    
        logger.info(f"Dataset: {ds_name} | Train Start: {args.train_start}")
        logger.info(f"Eval Period: {start_q.date()} to {target_quarters[-1].date()} ({len(target_quarters)} quarters)")
        logger.info(f"Vintages: {vintages_to_run} | Fill: {args.ragged_fill_method} | Agg: {args.quarterly_aggregation}")
        logger.info(f"Lags (Months): Macro={args.monthly_feature_release_lag_months}, Quarterly={args.quarterly_feature_release_lag_months}, GT={args.gt_release_lag_months}")
    
        # -------------------------------------------------------------------
        # Model Families
        # -------------------------------------------------------------------
        FILL_BASED_MODELS = ["ElasticNet"]
        NATIVE_RAGGED_MODELS = ["DFM", "DFM_MF", "MIDAS", "MIDASML"]
        
        if args.model in NATIVE_RAGGED_MODELS:
            current_fill_method = "native_ragged"
        else:
            current_fill_method = args.ragged_fill_method

        # -------------------------------------------------------------------
        # Checkpoint/Resume Setup
        # -------------------------------------------------------------------
        model_config_label = ""

        if args.model == "ElasticNet":
            if args.elasticnet_no_cv:
                model_config_label = f"_fixed_a{args.elasticnet_alpha}_l1{args.elasticnet_l1_ratio}"
            else:
                model_config_label = "_cv"

            # Clean legacy selector suffix format:
            # _sel_pca50, _sel_top50, _sel_lasso_a0.1
            if args.selector == "pca":
                model_config_label += f"_sel_pca{args.pca_components}"
            elif args.selector == "corr_top_n":
                model_config_label += f"_sel_top{args.top_n}"
            elif args.selector == "lasso":
                model_config_label += f"_sel_lasso_a{args.selector_alpha}"
            elif args.selector == "elasticnet":
                model_config_label += f"_sel_elasticnet_a{args.selector_alpha}_l1{args.selector_l1_ratio}"
            elif args.selector not in [None, "", "none"]:
                model_config_label += f"_sel_{args.selector}"

        elif args.model == "MIDAS":
            model_config_label = (
                f"_{args.midas_regression_model}"
                f"_lags{args.midas_n_lags}"
                f"_fill{args.midas_internal_fill_strategy}"
            )
        elif args.model == "MIDASML":
            model_config_label = (
                f"_{args.midasml_regression_model}"
                f"_lags{args.midas_n_lags}"
                f"_cv{args.midasml_cv}"
                f"_l1{args.midasml_l1_ratio}"
                f"_fill{args.midas_internal_fill_strategy}"
            )
        elif args.model == "DFM":
            model_config_label = (
                f"_agg{args.quarterly_aggregation}"
                f"_k{args.dfm_k_factors}"
                f"_p{args.dfm_factor_order}"
                f"_sel{args.dfm_selector}"
            )

            if args.dfm_selector == "pca":
                model_config_label += f"_pca{args.dfm_pca_components}"

            elif args.dfm_selector == "corr_top_n":
                model_config_label += f"_top{args.top_n}"
        elif args.model == "DFM_MF":
            model_config_label = (
                f"_k{args.dfm_k_factors}"
                f"_p{args.dfm_factor_order}"
                f"_mfsel{args.dfm_mf_selector}"
                f"_top{args.dfm_mf_top_n}"
            )

        if current_fill_method == "vertical_realignment":
             model_config_label += f"_{args.vertical_realignment_mode}"
        
        if current_fill_method == "tactis2" and args.preselect_top_k_before_fill is not None:
            model_config_label += f"_pretop{args.preselect_top_k_before_fill}"

        if current_fill_method == "tactis2":
            # Resolve defaults similarly to TACTiS2Filler
            if args.tactis2_author_config:
                def_batch, def_nb, def_samp, def_cop = 64, 64, 100, False
            else:
                def_batch, def_nb, def_samp, def_cop = 32, 32, 20, True

            tact_batch = args.tactis2_batch_size if args.tactis2_batch_size is not None else def_batch
            tact_nb = args.tactis2_num_batches_per_epoch if args.tactis2_num_batches_per_epoch is not None else def_nb
            tact_samples = args.tactis2_num_samples if args.tactis2_num_samples is not None else def_samp
            
            skip_copula_val = args.tactis2_skip_copula
            if skip_copula_val is None:
                cop_flag = 1 if def_cop else 0
            else:
                cop_flag = 1 if str(skip_copula_val).lower() == "true" else 0

            model_config_label += (
                f"_tacte{args.tactis2_max_epochs}"
                f"_b{tact_batch}"
                f"_nb{tact_nb}"
                f"_samp{tact_samples}"
                f"_cop{cop_flag}"
                f"_ctx{args.tactis2_context_length}"
            )
            
        checkpoint_name = f"checkpoint_{current_fill_method}_{args.model}{model_config_label}_{ds_name}_s{args.seed}.csv"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        if checkpoint_path.exists() and not args.resume:
            logger.warning(f"Removing old checkpoint (resume disabled): {checkpoint_name}")
            checkpoint_path.unlink()
        
        completed_keys = set()
        results = []
        
        if args.resume and checkpoint_path.exists():
            ckpt_df = safe_read_checkpoint(checkpoint_path)
            if not ckpt_df.empty:
                # Key: (dataset_type, model, fill_method, seed, target_quarter, vintage_label)
                for _, row in ckpt_df.iterrows():
                    key = (
                        str(row.get("dataset_type", ds_name)),
                        str(row.get("model", args.model)),
                        str(row.get("fill_method", current_fill_method)),
                        int(row.get("seed", args.seed)),
                        str(row.get("target_quarter")),
                        str(row.get("vintage_label")),
                    )
                    completed_keys.add(key)
                logger.info(f"Recovered {len(completed_keys)} completed predictions from checkpoint: {checkpoint_name}")
    
        # -------------------------------------------------------------------
        # Evaluation Loop
        # -------------------------------------------------------------------
        for target_q_end in target_quarters:
            target_label = map_to_target_quarter(target_q_end)
        
            for v_label in vintages_to_run:
                # Unique run key
                run_key = (
                    ds_name,
                    args.model,
                    current_fill_method,
                    args.seed,
                    target_label,
                    str(v_label),
                )
                if args.resume and run_key in completed_keys:
                    logger.info(f"Skipping completed checkpoint: {ds_name} {target_label} v{v_label}")
                    continue

                logger.info(f"Processing {target_label} | Vintage {v_label}")
            
                # 1. Build Vintage (Leakage-free truncation)
                try:
                    if args.model in FILL_BASED_MODELS:
                        X_filled_m, X_train_q, y_train_q, X_test_q, actual_y, meta = vb.build_vintage(
                            X=X, y=y,
                            target_quarter_end=target_q_end,
                            vintage_label=v_label,
                            train_start=pd.Timestamp(args.train_start),
                            rolling_window_quarters=args.rolling_window_quarters if args.rolling_window_quarters > 0 else None,
                            fill_method=args.ragged_fill_method,
                            aggregation_method=args.quarterly_aggregation,
                            vertical_realignment_mode=args.vertical_realignment_mode,
                            preselect_top_k_before_fill=args.preselect_top_k_before_fill,
                            macro_release_lag_months=args.monthly_feature_release_lag_months,
                            gt_release_lag_months=args.gt_release_lag_months,
                            quarterly_feature_release_lag_months=args.quarterly_feature_release_lag_months,
                            tactis2_author_config=args.tactis2_author_config,
                            tactis2_max_epochs=args.tactis2_max_epochs,
                            tactis2_epochs_phase_1=args.tactis2_epochs_phase_1,
                            tactis2_epochs_phase_2=args.tactis2_epochs_phase_2,
                            allow_backcast_target_in_train=args.allow_backcast_target_in_train,
                            tactis2_batch_size=args.tactis2_batch_size,
                            tactis2_num_batches_per_epoch=args.tactis2_num_batches_per_epoch,
                            tactis2_learning_rate=args.tactis2_learning_rate,
                            tactis2_weight_decay=args.tactis2_weight_decay,
                            tactis2_maximum_learning_rate=args.tactis2_maximum_learning_rate,
                            tactis2_clip_gradient=args.tactis2_clip_gradient,
                            tactis2_bagging_size=args.tactis2_bagging_size,
                            tactis2_skip_copula=(args.tactis2_skip_copula.lower() == "true") if args.tactis2_skip_copula is not None else None,
                            tactis2_context_length=args.tactis2_context_length,
                            tactis2_num_samples=args.tactis2_num_samples,
                            tactis2_device=args.tactis2_device,
                            tactis2_force_refit=args.tactis2_force_refit,
                            column_frequencies=dm._column_frequencies
                        )
                    else:
                        # NATIVE RAGGED PATH
                        X_visible_m, y_train_q, actual_y, col_freqs, meta = vb.build_native_vintage(
                            X=X, y=y,
                            target_quarter_end=target_q_end,
                            vintage_label=v_label,
                            train_start=pd.Timestamp(args.train_start),
                            rolling_window_quarters=args.rolling_window_quarters if args.rolling_window_quarters > 0 else None,
                            macro_release_lag_months=args.monthly_feature_release_lag_months,
                            gt_release_lag_months=args.gt_release_lag_months,
                            quarterly_feature_release_lag_months=args.quarterly_feature_release_lag_months,
                            allow_backcast_target_in_train=args.allow_backcast_target_in_train,
                            column_frequencies=dm._column_frequencies
                        )
                        X_filled_m = X_visible_m # Not filled, but for naming consistency in the loop
                        X_train_q, X_test_q = None, None # To be derived by models
                except Exception as e:
                    logger.error(f"Vintage Builder failed for {target_label} v{v_label}: {e}")
                    continue
                
                if len(y_train_q) < 5:
                    logger.warning(f"Not enough training quarters ({len(y_train_q)}) for {target_label}. Skipping.")
                    continue

                t0 = time.time()
            
                # 2. Fit Model
                pred_val = None
                n_sel = 0
                try:
                    if args.model == "ElasticNet":
                        # Feature Selection MUST be fit on train only
                        X_train_sel, X_test_sel = apply_feature_selection(
                            X_train_q, y_train_q, X_test_q, 
                            method=args.selector, 
                            pca_comp=args.pca_components, 
                            top_n=args.top_n,
                            alpha=args.selector_alpha,
                            l1_ratio=args.selector_l1_ratio
                        )
                        n_sel = X_train_sel.shape[1]
                    
                        enet_l1_ratio = args.elasticnet_l1_ratio if args.elasticnet_no_cv else None
                        model = ElasticNetNowcast(
                            target_col="gdp_target",
                            cv=3,
                            use_cv=not args.elasticnet_no_cv,
                            alpha=args.elasticnet_alpha,
                            l1_ratio=enet_l1_ratio,
                            max_iter=args.elasticnet_max_iter,
                            random_state=args.seed,
                            fill_strategy="median"
                        )
                        model.fit(X_train_sel, y_train_q)
                    
                        if not X_test_sel.empty:
                            pred_s = model.predict(X_test_sel)
                            pred_val, pred_extraction_mode = extract_prediction_for_target(
                                pred_s, target_q_end, model_name="ElasticNet", target_col="gdp_target", return_mode=True
                            )
                        
                    elif args.model in ["DFM", "DFM_MF"]:
                        from nowcasting.models.dfm import DynamicFactorNowcast

                        quarterly_cols = [
                            c for c in X_filled_m.columns
                            if col_freqs.get(str(c)) == "Q"
                        ]
                        if args.model == "DFM":
                            # Common-frequency quarterly DFM
                            # Aggregate to quarterly preserving NaNs
                            if args.quarterly_aggregation == "mean":
                                X_q_native = X_filled_m.resample("Q").mean()
                            elif args.quarterly_aggregation == "last":
                                X_q_native = X_filled_m.resample("Q").last()
                            else:
                                X_q_native = X_filled_m.resample("Q").mean()
                            
                            X_q_native.columns = X_q_native.columns.astype(str)
                            X_train_q_native = X_q_native[X_q_native.index < target_q_end]
                            X_test_q_native = X_q_native[X_q_native.index == target_q_end]
                            
                            # Feature Selection for DFM
                            X_train_sel, X_test_sel = apply_feature_selection(
                                X_train_q_native, y_train_q, X_test_q_native,
                                method=args.dfm_selector, 
                                pca_comp=args.dfm_pca_components, 
                                top_n=args.top_n,
                                alpha=args.selector_alpha,
                                l1_ratio=args.selector_l1_ratio
                            )
                            n_sel = X_train_sel.shape[1]
                            
                            model = DynamicFactorNowcast(
                                target_col="gdp_target",
                                k_factors=args.dfm_k_factors,
                                factor_order=args.dfm_factor_order,
                                mixed_frequency=False,
                                maxiter=args.dfm_maxiter,
                                tolerance=args.dfm_tolerance
                            )
                            model.fit(X_train_sel, y_train_q)
                            if not X_test_sel.empty:
                                pred_s = model.predict(X_test_sel)
                                pred_val, pred_extraction_mode = extract_prediction_for_target(
                                    pred_s, target_q_end, model_name="DFM", target_col="gdp_target", return_mode=True
                                )
                                dfm_info_set = getattr(model, "last_prediction_info", {}).get("dfm_prediction_information_set", "unknown")
                        else:
                            # Mixed-frequency DFM
                            # Reindex monthly panel up to target_q_end (M freq) to avoid truncation issues
                            native_idx = pd.date_range(start=X_filled_m.index.min(), end=target_q_end, freq="M")
                            X_native_m = X_filled_m.reindex(native_idx)
                            
                            # Treatment of GDP target as quarterly variable + string column names
                            X_native_m.columns = X_native_m.columns.astype(str)
                            
                            # For DFM_MF, we only pass the target quarter months to predict()
                            target_q_start = target_q_end.to_period("Q").start_time.normalize()
                            X_train_m_native = X_native_m[X_native_m.index < target_q_start]
                            X_test_m_native = X_native_m[
                                (X_native_m.index >= target_q_start) & 
                                (X_native_m.index <= target_q_end)
                            ]
                            
                            # Feature Selection for DFM_MF (to avoid instability with 400+ features)
                            selected_cols = list(X_native_m.columns)
                            if args.dfm_mf_selector == "corr_top_n":
                                # Aggregate training X to quarterly for correlation with y
                                X_train_q_for_corr = X_train_m_native.resample("Q").mean()
                                shared_idx = X_train_q_for_corr.index.intersection(y_train_q.index)
                                if len(shared_idx) > 5:
                                    corrs = X_train_q_for_corr.loc[shared_idx].corrwith(y_train_q.loc[shared_idx]).abs()
                                    selected_cols = corrs.sort_values(ascending=False).head(args.dfm_mf_top_n).index.tolist()
                                    logger.info(f"[DFM-MF] Selected top {len(selected_cols)} features via correlation.")
                                else:
                                    logger.warning("[DFM-MF] Not enough shared observations for feature selection. Using all features.")
                            
                            X_train_m_native = X_train_m_native[selected_cols]
                            X_test_m_native = X_test_m_native[selected_cols]
                            
                            # Update quarterly_cols consistency
                            q_cols_selected = [str(c) for c in selected_cols if col_freqs.get(str(c)) == "Q"]
                            dfm_quarterly_cols = q_cols_selected + ["gdp_target"]
                            n_sel = len(selected_cols)
                            
                            model = DynamicFactorNowcast(
                                target_col="gdp_target",
                                k_factors=args.dfm_k_factors,
                                factor_order=args.dfm_factor_order,
                                mixed_frequency=True,
                                quarterly_cols=dfm_quarterly_cols,
                                maxiter=args.dfm_maxiter,
                                tolerance=args.dfm_tolerance
                            )
                            model.fit(X_train_m_native, y_train_q)
                            if not X_test_m_native.empty:
                                pred_s = model.predict(X_test_m_native)
                                pred_val, pred_extraction_mode = extract_prediction_for_target(
                                    pred_s, target_q_end, model_name="DFM_MF", target_col="gdp_target", return_mode=True
                                )
                                dfm_info_set = getattr(model, "last_prediction_info", {}).get("dfm_prediction_information_set", "unknown")

                    elif args.model in ["MIDAS", "MIDASML"]:
                        from nowcasting.models.midas import MIDASNowcast
                        
                        # Reindex monthly panel up to target_q_end (M freq)
                        native_idx = pd.date_range(start=X_filled_m.index.min(), end=target_q_end, freq="M")
                        X_native_m = X_filled_m.reindex(native_idx)
                        
                        X_train_m_native = X_native_m[X_native_m.index < target_q_end]
                        X_test_m_native = X_native_m[X_native_m.index <= target_q_end]
                        n_sel = X_train_m_native.shape[1]
                        
                        if args.model == "MIDAS":
                            model = MIDASNowcast(
                                target_col="gdp_target",
                                freq_ratio=3,
                                n_lags=args.midas_n_lags,
                                lf_freq="Q",
                                regression_model=args.midas_regression_model,
                                fill_strategy=args.midas_internal_fill_strategy,
                                random_state=args.seed
                            )
                        else: # MIDASML
                            model = MIDASNowcast(
                                target_col="gdp_target",
                                freq_ratio=3,
                                n_lags=args.midas_n_lags,
                                lf_freq="Q",
                                regression_model=args.midasml_regression_model,
                                fill_strategy=args.midas_internal_fill_strategy,
                                regression_kwargs={
                                    "cv": args.midasml_cv,
                                    "max_iter": args.midasml_max_iter,
                                    "l1_ratio": args.midasml_l1_ratio
                                },
                                random_state=args.seed
                            )
                            
                        model.fit(X_train_m_native, y_train_q)
                        if not X_test_m_native.empty:
                            pred_s = model.predict(X_test_m_native)
                            pred_val, pred_extraction_mode = extract_prediction_for_target(
                                pred_s, target_q_end, model_name="MIDAS/MIDASML", target_col="gdp_target", return_mode=True
                            )
                                
                except Exception as e:
                    logger.error(f"Model fitting failed for {target_label} v{v_label}: {e}")
                    continue
                
                runtime = round(time.time() - t0, 2)
            
                # 3. Store Results
                results.append({
                    "seed": args.seed,
                    "model": args.model,
                    "dataset_type": ds_name,
                    "target_quarter": target_label,
                    "vintage_label": v_label,
                    "cutoff_date": meta["cutoff_date"],
                    "prediction": pred_val,
                    "actual": actual_y if pd.notna(actual_y) else None,
                    "runtime_sec": runtime + meta.get("fill_runtime_sec", 0),
                    "preselect_before_fill_method": meta.get("preselect_before_fill_method"),
                    "preselect_before_fill_top_k": meta.get("preselect_before_fill_top_k"),
                    "preselect_before_fill_features_before": meta.get("preselect_before_fill_features_before"),
                    "preselect_before_fill_features_after": meta.get("preselect_before_fill_features_after"),
                    "n_features_raw": X.shape[1],
                    "n_features_sel": n_sel,
                    "fill_method": current_fill_method,
                    "pred_extraction_mode": pred_extraction_mode if pred_val is not None else "failed",
                    "missing_before_final_ffill": meta.get("missing_before_final_ffill", 0),
                    "missing_after_final_ffill": meta.get("missing_after_final_ffill", 0),
                    "values_filled_by_final_ffill": meta.get("values_filled_by_final_ffill", 0),
                    "series_sktime_autoarima_ok": meta.get("series_sktime_autoarima_ok", 0),
                    "series_sktime_autoarima_failed_locf": meta.get("series_sktime_autoarima_failed_locf", 0),
                    "series_autoarima_cached_sqlite": meta.get("series_autoarima_cached_sqlite", 0),
                    "series_autoarima_cached_memory": meta.get("series_autoarima_cached_memory", 0),
                    "series_autoarima_too_short_locf": meta.get("series_autoarima_too_short_locf", 0),
                    "series_tactis2_ok": meta.get("series_tactis2_ok", 0),
                    "series_tactis2_failed_fallback": meta.get("series_tactis2_failed_fallback", 0),
                    "tactis2_audit_entries": meta.get("tactis2_audit_entries", 0),
                    "tactis2_runtime_sec": meta.get("tactis2_runtime_sec", 0),
                    "tactis2_context_length": meta.get("tactis2_context_length", 0),
                    "tactis2_prediction_length": meta.get("tactis2_prediction_length", 0),
                    "tactis2_author_config": meta.get("tactis2_author_config", False),
                    "tactis2_max_epochs": meta.get("tactis2_max_epochs", 0),
                    "tactis2_epochs_phase_1": meta.get("tactis2_epochs_phase_1", 0),
                    "tactis2_epochs_phase_2": meta.get("tactis2_epochs_phase_2", 0),
                    "tactis2_batch_size": meta.get("tactis2_batch_size", 0),
                    "tactis2_num_batches_per_epoch": meta.get("tactis2_num_batches_per_epoch", 0),
                    "tactis2_learning_rate": meta.get("tactis2_learning_rate", 0),
                    "tactis2_weight_decay": meta.get("tactis2_weight_decay", 0),
                    "tactis2_maximum_learning_rate": meta.get("tactis2_maximum_learning_rate", 0),
                    "tactis2_clip_gradient": meta.get("tactis2_clip_gradient", 0),
                    "tactis2_bagging_size": meta.get("tactis2_bagging_size", None),
                    "tactis2_skip_copula": meta.get("tactis2_skip_copula", True),
                    "tactis2_num_samples": meta.get("tactis2_num_samples", 0),
                    "tactis2_origin_groups": meta.get("tactis2_origin_groups", ""),
                    "tactis2_values_forecasted": meta.get("tactis2_values_forecasted", 0),
                    "tactis2_values_ffill_fallback": meta.get("tactis2_values_ffill_fallback", 0),
                    "tactis2_group_failures": meta.get("tactis2_group_failures", 0),
                    "tactis2_origin_dates_used": meta.get("tactis2_origin_dates_used", ""),
                    "vertical_realignment_mode": meta.get("vertical_realignment_mode", None),
                    "vertical_realignment_features_total": meta.get("vertical_realignment_features_total", 0),
                    "vertical_realignment_blocks_ffilled": meta.get("vertical_realignment_blocks_ffilled", 0),
                    "vertical_realignment_realigned_values": meta.get("vertical_realignment_realigned_values", 0),
                    "vertical_realignment_missing_lag_values": meta.get("vertical_realignment_missing_lag_values", 0),
                    "selector": args.selector,
                    "selector_alpha": args.selector_alpha if args.selector in ["lasso", "elasticnet"] else None,
                    "selector_l1_ratio": args.selector_l1_ratio if args.selector == "elasticnet" else None,
                    "train_q_size": len(X_train_q) if X_train_q is not None else len(y_train_q),
                    "dfm_prediction_information_set": dfm_info_set if args.model in ["DFM", "DFM_MF"] else None,
                    "elasticnet_use_cv": not args.elasticnet_no_cv,
                    "elasticnet_alpha": args.elasticnet_alpha,
                    "elasticnet_l1_ratio": args.elasticnet_l1_ratio,
                    "elasticnet_max_iter": args.elasticnet_max_iter,
                    "selector": args.selector if args.model == "ElasticNet" else None,
                    "pca_components": args.pca_components if args.model == "ElasticNet" and args.selector == "pca" else None,
                    "top_n": (args.top_n
                                if (
                                    (args.model == "ElasticNet" and args.selector == "corr_top_n")
                                    or (args.model == "DFM" and args.dfm_selector == "corr_top_n")
                                )
                                else None
                            ),
                    "selector_alpha": args.selector_alpha if args.model == "ElasticNet" and args.selector == "lasso" else None,
                    "midas_n_lags": args.midas_n_lags if args.model in ["MIDAS", "MIDASML"] else None,
                    "midas_regression_model": args.midas_regression_model if args.model == "MIDAS" else None,
                    "midas_internal_fill_strategy": args.midas_internal_fill_strategy if args.model in ["MIDAS", "MIDASML"] else None,
                    "midasml_regression_model": args.midasml_regression_model if args.model == "MIDASML" else None,
                    "midasml_cv": args.midasml_cv if args.model == "MIDASML" else None,
                    "midasml_l1_ratio": args.midasml_l1_ratio if args.model == "MIDASML" else None,
                    "midasml_max_iter": args.midasml_max_iter if args.model == "MIDASML" else None,
                    "quarterly_aggregation": args.quarterly_aggregation if args.model == "DFM" else None,
                    "dfm_k_factors": args.dfm_k_factors if args.model in ["DFM", "DFM_MF"] else None,
                    "dfm_factor_order": args.dfm_factor_order if args.model in ["DFM", "DFM_MF"] else None,
                    "dfm_selector": args.dfm_selector if args.model == "DFM" else None,
                    "dfm_pca_components": (
                                            args.dfm_pca_components
                                            if args.model == "DFM" and args.dfm_selector == "pca"
                                            else None
                                        ),
                    "dfm_mf_selector": args.dfm_mf_selector if args.model == "DFM_MF" else None,
                    "dfm_mf_top_n": args.dfm_mf_top_n if args.model == "DFM_MF" else None
                })
                
                # Checkpoint Write
                if len(results) % args.checkpoint_every == 0:
                    row_data = {col: results[-1].get(col, None) for col in CHECKPOINT_COLUMNS}
                    df_step = pd.DataFrame([row_data], columns=CHECKPOINT_COLUMNS)
                    file_exists = checkpoint_path.exists()
                    df_step.to_csv(checkpoint_path, mode='a', index=False, header=not file_exists)
                    
                    # Enhanced Logging
                    log_msg = f"Saved checkpoint row for {target_label} v{v_label}."
                    if current_fill_method == "tactis2":
                        log_msg += f" [TACTiS-2: runtime={meta.get('tactis2_runtime_sec',0)}s, forecasted={meta.get('tactis2_values_forecasted',0)}, fallback={meta.get('tactis2_values_ffill_fallback',0)}]"
                    if meta.get("preselect_before_fill_top_k") is not None:
                        log_msg += f" [Preselection: {meta.get('preselect_before_fill_features_before')} -> {meta.get('preselect_before_fill_features_after')} features]"
                    logger.info(log_msg)

        # Flush any remaining results to checkpoint if not yet saved before consolidation
        if results and len(results) % args.checkpoint_every != 0:
            row_data = {col: results[-1].get(col, None) for col in CHECKPOINT_COLUMNS}
            df_step = pd.DataFrame([row_data], columns=CHECKPOINT_COLUMNS)
            file_exists = checkpoint_path.exists()
            df_step.to_csv(checkpoint_path, mode='a', index=False, header=not file_exists)
            logger.info(f"Flushed final checkpoint results for {ds_name}")


        # -------------------------------------------------------------------
        # Outputs and Metrics
        # -------------------------------------------------------------------
        # Always try to consolidate if checkpoint exists, or if we have new results
        suffix = f"_{current_fill_method}_{args.model}{model_config_label}_{ds_name}_s{args.seed}"
        out_path = base_out_dir / f"vintage_nowcasts{suffix}.csv"
        df_final = pd.DataFrame()

        if checkpoint_path.exists():
            df_final = safe_read_checkpoint(checkpoint_path)
            if not df_final.empty:
                # Deduplicate: latest row for each run key
                key_cols = [
                    "dataset_type",
                    "model",
                    "fill_method",
                    "seed",
                    "target_quarter",
                    "vintage_label",
                    "elasticnet_alpha",
                    "elasticnet_l1_ratio",
                    "selector",
                    "pca_components",
                    "top_n",
                    "selector_alpha",
                    "midas_n_lags",
                    "midas_regression_model",
                    "midas_internal_fill_strategy",
                    "midasml_regression_model",
                    "midasml_cv",
                    "midasml_l1_ratio",
                    "quarterly_aggregation",
                    "dfm_k_factors",
                    "dfm_factor_order",
                    "dfm_selector",
                    "dfm_pca_components",
                    "dfm_mf_selector",
                    "dfm_mf_top_n",
                ]
                key_cols = [c for c in key_cols if c in df_final.columns]
                df_final["vintage_label"] = df_final["vintage_label"].astype(str)
                df_final = df_final.drop_duplicates(subset=key_cols, keep="last")
                df_final["vintage_label"] = df_final["vintage_label"].astype(str)
                df_final = df_final.drop_duplicates(subset=key_cols, keep='last')
                
                df_final.to_csv(out_path, index=False)
                logger.info(f"Consolidated results from checkpoint and saved to: {out_path.name}")
            else:
                # If checkpoint read failed or was empty, use current results
                rows = [{col: r.get(col, None) for col in CHECKPOINT_COLUMNS} for r in results]
                df_final = pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS)
                if not df_final.empty:
                    df_final.to_csv(out_path, index=False)
                    logger.info(f"Predictions saved to: {out_path.name}")
        elif results:
            rows = [{col: r.get(col, None) for col in CHECKPOINT_COLUMNS} for r in results]
            df_final = pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS)
            df_final.to_csv(out_path, index=False)
            logger.info(f"Predictions saved to: {out_path.name}")
        
        if not df_final.empty:
            metrics_df, revisions_df = compute_metrics(df_final)
            if not metrics_df.empty:
                metrics_path = base_out_dir / f"vintage_nowcasts_metrics{suffix}.csv"
                metrics_df.to_csv(metrics_path, index=False)
            
                print(f"\n=== Overall Metrics ({ds_name}) ===")
                print(metrics_df[metrics_df["vintage_label"] == "overall"].to_string(index=False))
            
                if not revisions_df.empty:
                    print(f"\n=== Average Revisions ({ds_name}) ===")
                    print(revisions_df[revisions_df["transition"] == "overall"].to_string(index=False))
                
            plot_single_file(out_path, plots_dir)
        else:
            logger.warning(f"No results generated or found in checkpoint for {ds_name}.")
            
    if args.plot_compare_datasets:
        compare_datasets(base_out_dir, plots_dir, args.model, current_fill_method, args.seed)
    if args.plot_compare_fills:
        compare_fills(base_out_dir, plots_dir, args.model, args.dataset, args.plot_vintage, args.seed)

if __name__ == "__main__":
    main()
