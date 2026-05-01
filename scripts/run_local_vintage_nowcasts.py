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
import os
import sys
import logging
import time
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.local_data_builder import LocalDataManager
from nowcasting.data.vintage_builder import VintageBuilder
from nowcasting.models.dfm import DynamicFactorNowcast
from nowcasting.models.ml_regression import ElasticNetNowcast
from nowcasting.features.selectors import PCACompressor, VarianceFilter
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
                            method: str, pca_comp: int, top_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies feature selection STRICTLY fit on X_train."""
    if method == "none" or X_train.empty:
        return X_train, X_test
        
    steps = [("var", VarianceFilter())]
    if method == "pca":
        steps.append(("pca", PCACompressor(n_components=pca_comp)))
    elif method == "corr_top_n":
        steps.append(("corr", TopNCorrelationSelector(top_n=top_n)))
        
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

def parse_target_quarter(tq_str):
    # '2020Q1' -> '2020-03-31'
    try:
        return pd.Period(tq_str, freq='Q').end_time.date()
    except Exception:
        return None


def plot_single_file(file_path: Path, out_dir: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    if 'prediction' not in df.columns:
        return
        
    df = df.dropna(subset=['prediction'])
    df['q_end'] = df['target_quarter'].apply(parse_target_quarter)
    df = df.sort_values('q_end')
    
    model_name = df['model'].iloc[0]
    dataset_name = df['dataset_type'].iloc[0]
    fill_method = df['fill_method'].iloc[0] if 'fill_method' in df.columns else 'unknown'
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot Actuals
    actuals = df.dropna(subset=['actual']).drop_duplicates('q_end')
    if not actuals.empty:
        ax.plot(actuals['q_end'], actuals['actual'], color='#1f77b4', linewidth=3.0, marker='o', markersize=5, label='Actual', zorder=5)
        
    # Plot Vintages
    vintages = sorted(df['vintage_label'].astype(str).unique(), key=lambda x: int(x))
    
    # Explicit styles for vintages
    vintage_styles = {
        "-2": {"color": "#ff7f0e", "linestyle": "-", "marker": "o"},
        "-1": {"color": "#2ca02c", "linestyle": "--", "marker": "s"},
        "0": {"color": "#d62728", "linestyle": "-.", "marker": "^"},
        "1": {"color": "#9467bd", "linestyle": ":", "marker": "D"},
        "+1": {"color": "#9467bd", "linestyle": ":", "marker": "D"},
        "2": {"color": "#8c564b", "linestyle": (0, (5, 1)), "marker": "v"},
        "+2": {"color": "#8c564b", "linestyle": (0, (5, 1)), "marker": "v"},
    }
    
    for v_label in vintages:
        v_data = df[df['vintage_label'].astype(str) == v_label]
        style = vintage_styles.get(v_label, {"color": "gray", "linestyle": "-", "marker": "x"})
        
        # Format label for legend
        try:
            v_int = int(v_label)
            v_sign = "+" if v_int > 0 else ""
            v_legend_label = f"Vintage {v_sign}{v_int}"
        except ValueError:
            v_legend_label = f"Vintage {v_label}"
            
        ax.plot(
            v_data['q_end'], v_data['prediction'], 
            marker=style["marker"], linestyle=style["linestyle"], color=style["color"], 
            linewidth=2.2, markersize=4, alpha=0.95, label=v_legend_label, zorder=4
        )
        
    # Title Mapping (Lithuanian)
    # Include both model name (prediction) and fill method
    title_map = {
        "baseline_common": f"{model_name} ({fill_method}) prognozės pagal vintage (bendras rinkinys)",
        "common_plus_gt": f"{model_name} ({fill_method}) prognozės pagal vintage(bendri + GT)",
        "gt_only": f"{model_name} ({fill_method}) prognozės pagal vintage (tik GT)",
        "final_thesis_baseline_common": f"{model_name} ({fill_method}) prognozės pagal vintage (bendras rinkinys)",
        "final_thesis_common_plus_gt": f"{model_name} ({fill_method}) prognozės pagal vintage (bendri + GT)",
        "final_thesis_gt_only": f"{model_name} ({fill_method}) prognozės pagal vintage (tik GT)",
    }
    title = title_map.get(dataset_name, f"{model_name} ({fill_method}) prognozės pagal vintage ({dataset_name})")
    
    ax.set_title(title, fontsize=17)
    ax.set_xlabel('Data', fontsize=13)
    ax.set_ylabel('BVP', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_name = f"plot_{fill_method}_{model_name}_{dataset_name}.png"
    ax.figure.savefig(out_dir / out_name, dpi=300)
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
            # Filter by vintage
            v_data = df[df['vintage_label'].astype(str) == vintage].copy()
            if not v_data.empty:
                v_data['q_end'] = v_data['target_quarter'].apply(parse_target_quarter)
                dfs[fill] = v_data.sort_values('q_end')
        else:
            print(f"Warning: Missing data for {fill} -> {file_name}")
            
    if not dfs:
        print("No valid data found to compare fills.")
        return
        
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Try to plot actuals from the first available df
    first_df = list(dfs.values())[0]
    actuals = first_df.dropna(subset=['actual']).drop_duplicates('q_end')
    if not actuals.empty:
        ax.plot(actuals['q_end'], actuals['actual'], color='#1f77b4', linewidth=3.0, marker='o', markersize=5, label='Actual', zorder=5)
        
    colors = ['orange', 'green', 'purple', 'red']
    markers = ['s', '^', 'D', 'o']
    
    for (fill, df_fill), color, marker in zip(dfs.items(), colors, markers):
        ax.plot(df_fill['q_end'], df_fill['prediction'], marker=marker, linestyle='--', color=color, linewidth=2, label=f'Method: {fill}')
        
    vintage_str = vintage if vintage.startswith('-') else f"+{vintage}" if vintage != '0' else vintage
    ax.set_title(f'Užpildymo metodų palyginimas ({model}) (Vintage {vintage_str}) | {dataset}', fontsize=17)
    ax.set_xlabel('Data', fontsize=13)
    ax.set_ylabel('BVP', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_name = f"comparison_fills_{model}_{dataset}_v{vintage}.png"
    ax.figure.savefig(out_dir / out_name, dpi=300)
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
    "vertical_realignment_features_total",
    "vertical_realignment_blocks_ffilled",
    "train_q_size"
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
    parser.add_argument("--selector", type=str, default="pca", choices=["none", "variance", "pca", "corr_top_n"], help="Feature selection method (primarily for ElasticNet)")
    parser.add_argument("--pca-components", type=int, default=10, help="Number of PCA components")
    parser.add_argument("--dfm-maxiter", type=int, default=50, help="Max iterations for DFM fitting")
    parser.add_argument("--dfm-tolerance", type=float, default=1e-5, help="Convergence tolerance for DFM")
    parser.add_argument("--top-n", type=int, default=50, help="Top N features for corr_top_n selector")
    parser.add_argument("--seed", type=int, default=2234, help="Random seed")
    parser.add_argument("--debug-preselect-top-k", type=int, default=None, help="Prefilter X for AutoARIMA using top K absolute correlations with visible target.")
    parser.add_argument("--monthly-feature-release-lag-months", type=int, default=1, help="Simulated release lag for monthly macro features.")
    parser.add_argument("--gt-release-lag-months", type=int, default=0, help="Simulated release lag for Google Trends features.")
    parser.add_argument("--quarterly-feature-release-lag-months", type=int, default=1, help="Simulated release lag for quarterly macro features.")
    parser.add_argument("--arima-n-jobs", type=int, default=1, help="Number of parallel jobs for AutoARIMA across series.")
    parser.add_argument("--arima-fast", action="store_true", help="Use a faster (max p=1, q=1, P=0, Q=0) AutoARIMA search space.")
    parser.add_argument("--arima-seasonal", type=str, default="true", choices=["true", "false"], help="Enable/disable seasonal AutoARIMA.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-quarter evaluation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and only run plots.")
    parser.add_argument("--plot-compare-datasets", action="store_true", help="Plot baseline_common vs common_plus_gt vs gt_only side-by-side.")
    parser.add_argument("--plot-compare-fills", action="store_true", help="Plot locf vs autoarima vs vertical_realignment for the specified --dataset and --plot-vintage.")
    parser.add_argument("--plot-vintage", type=str, default="0", help="Vintage label to compare across fills when using --plot-compare-fills.")
    parser.add_argument("--arima-cache-path", type=str, default=None, help="Path to SQLite arima cache database.")
    
    # DFM Parameters
    parser.add_argument("--dfm-k-factors", type=int, default=2, help="Number of factors for DFM")
    parser.add_argument("--dfm-factor-order", type=int, default=1, help="Factor AR order for DFM")
    parser.add_argument("--dfm-selector", type=str, default="none", choices=["none", "pca", "corr_top_n"], help="Selector for DFM features")
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
    
    args = parser.parse_args()

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
        else:
            logger.info("Skipping evaluation, but no plot arguments provided (--plot-compare-datasets or --plot-compare-fills).")
        return

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("="*60)
    logger.info(f"=== THESIS EVALUATION MODE | MODEL: {args.model} ===")
    logger.info("="*60)
    
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
        # We do NOT use local_data_builder's get_cutoff_dates_for_quarter anymore.
        # We load the full raw datasets and let VintageBuilder handle truncation.
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
        checkpoint_name = f"checkpoint_{current_fill_method}_{args.model}_{ds_name}_s{args.seed}.csv"
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
                        str(row.get("vintage_label"))
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
                run_key = (ds_name, args.model, current_fill_method, args.seed, target_label, str(v_label))
                
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
                            debug_preselect_top_k=args.debug_preselect_top_k,
                            macro_release_lag_months=args.monthly_feature_release_lag_months,
                            gt_release_lag_months=args.gt_release_lag_months,
                            quarterly_feature_release_lag_months=args.quarterly_feature_release_lag_months,
                            tactis2_author_config=args.tactis2_author_config,
                            tactis2_max_epochs=args.tactis2_max_epochs,
                            tactis2_epochs_phase_1=args.tactis2_epochs_phase_1,
                            tactis2_epochs_phase_2=args.tactis2_epochs_phase_2,
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
                            method=args.selector, pca_comp=args.pca_components, top_n=args.top_n
                        )
                        n_sel = X_train_sel.shape[1]
                    
                        model = ElasticNetNowcast(
                            target_col="gdp_target",
                            cv=3, max_iter=2000,
                            random_state=args.seed,
                            fill_strategy="median" # Residual missing values after ARIMA/LOCF
                        )
                        model.fit(X_train_sel, y_train_q)
                    
                        if not X_test_sel.empty:
                            pred_s = model.predict(X_test_sel)
                            pred_val = float(pred_s.iloc[0])
                        
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
                                method=args.dfm_selector, pca_comp=args.dfm_pca_components, top_n=args.top_n
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
                                pred_val = float(pred_s.iloc[0])
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
                                # Pick the target_q_end prediction
                                if target_q_end in pred_s.index:
                                    pred_val = float(pred_s.loc[target_q_end])
                                else:
                                    pred_val = float(pred_s.iloc[-1])

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
                            if target_q_end in pred_s.index:
                                pred_val = float(pred_s.loc[target_q_end])
                            else:
                                pred_val = float(pred_s.iloc[-1])
                                
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
                    "n_features_raw": X.shape[1],
                    "n_features_sel": n_sel,
                    "fill_method": current_fill_method,
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
                    "vertical_realignment_features_total": meta.get("vertical_realignment_features_total", 0),
                    "vertical_realignment_blocks_ffilled": meta.get("vertical_realignment_blocks_ffilled", 0),
                    "train_q_size": len(X_train_q) if X_train_q is not None else len(y_train_q)
                })
                
                # Checkpoint Write
                if len(results) % args.checkpoint_every == 0:
                    row_data = {col: results[-1].get(col, None) for col in CHECKPOINT_COLUMNS}
                    df_step = pd.DataFrame([row_data], columns=CHECKPOINT_COLUMNS)
                    file_exists = checkpoint_path.exists()
                    df_step.to_csv(checkpoint_path, mode='a', index=False, header=not file_exists)
                    logger.info(f"Saved checkpoint row to {checkpoint_name}")

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
        # -------------------------------------------------------------------
        # Outputs and Metrics
        # -------------------------------------------------------------------
        # Always try to consolidate if checkpoint exists, or if we have new results
        suffix = f"_{current_fill_method}_{args.model}_{ds_name}_s{args.seed}"
        out_path = base_out_dir / f"vintage_nowcasts{suffix}.csv"
        df_final = pd.DataFrame()

        if checkpoint_path.exists():
            df_final = safe_read_checkpoint(checkpoint_path)
            if not df_final.empty:
                # Deduplicate: latest row for each run key
                key_cols = ["dataset_type", "model", "fill_method", "seed", "target_quarter", "vintage_label"]
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
