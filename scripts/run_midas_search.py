"""
run_midas_search.py  —  MIDAS / U-MIDAS Grid-Search Experiment Pipeline
========================================================================

Systematic search over:
  - Feature selection / compression methods
  - MIDAS hyperparameters  (n_lags, freq_ratio, lf_freq, fill_strategy)
  - Regression backends     (linear, ridge, lasso, elasticnet)
  - Backtest settings       (train_window, step_size)

Key design guarantees
---------------------
  - No look-ahead bias: selectors / reducers are fit only on the training fold
    inside each backtesting window, then applied to the test fold.
  - Robustness: failed configurations are caught, logged, and stored in the
    results table so the search continues.
  - Reproducibility: numpy seed is set globally and per-worker.

Example (quick smoke test):
    python scripts/run_midas_search.py \\
        --selectors none,pca \\
        --pca-components 3 \\
        --n-lags 1,2 \\
        --lf-freq QE \\
        --regression-models elasticnet \\
        --train-windows 80 \\
        --search-last-n-steps 3 \\
        --seed 42

Full search:
    python scripts/run_midas_search.py \\
        --selectors none,corr_top_n,lasso,elasticnet,pca,factor_analysis \\
        --top-n 10,20,50 \\
        --lasso-alphas 0.001,0.01,0.1 \\
        --elasticnet-alphas 0.001,0.01 \\
        --elasticnet-l1-ratios 0.2,0.5,0.8 \\
        --pca-components 3,5,10 \\
        --fa-components 3,5,10 \\
        --n-lags 1,2,3,4 \\
        --freq-ratios auto,3 \\
        --lf-freq QE \\
        --fill-strategies zero,ffill_then_zero \\
        --regression-models linear,ridge,lasso,elasticnet \\
        --train-windows 80,120 \\
        --step-sizes 1,3 \\
        --search-last-n-steps 30 \\
        --n-jobs 4 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Single-thread BLAS to avoid collisions inside parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcasting.main import load_cf_panel, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.midas import MIDASNowcast

from nowcasting.features.selectors import (
    IdentitySelector,
    VarianceFilter,
    CorrTopNSelector,
    LassoSelector,
    ElasticNetSelector,
    PCACompressor,
    FactorAnalysisCompressor,
    AutoencoderCompressor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("midas_search")


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    """Replace unsafe filename characters with underscores."""
    return re.sub(r"[^\w\-.]", "_", str(s))


def _shorten(s: str, max_len: int = 50) -> str:
    """Shorten a string to max_len characters, preserving start and end."""
    s = str(s)
    if len(s) <= max_len:
        return s
    keep = max_len // 2 - 1
    return s[:keep] + "_" + s[-keep:]


def _build_suffix(
    panel_path: Optional[Path],
    seed: int,
    lf_freq: str = "",
    selector: str = "",
) -> str:
    """Build a short, safe filename suffix from key run parameters."""
    stem = _shorten(_sanitize(panel_path.stem if panel_path else "panel"), 30)
    parts = ["midas", stem]
    if lf_freq:
        parts.append(_sanitize(lf_freq).lower())
    if selector and selector != "none":
        parts.append(_sanitize(selector)[:10])
    parts.append(f"s{seed}")
    return "_".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MIDAS / U-MIDAS Grid-Search Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                    help="Directory containing processed parquet files")
    ap.add_argument("--input-panel", default=None,
                    help="Path (absolute or relative to --data-dir) to a specific parquet panel")
    ap.add_argument("--target-col", default=None, dest="target",
                    help="Target column name / series_id")

    # Feature selectors
    ap.add_argument("--selectors", default="none,corr_top_n,pca",
                    help="Comma-separated selector methods: none, variance_filter, corr_top_n, "
                         "lasso, elasticnet, pca, factor_analysis, autoencoder")
    ap.add_argument("--variance-thresholds", default="0.0,1e-6",
                    help="Variance thresholds for variance_filter (comma-separated)")
    ap.add_argument("--top-n",  default="10,20,50",
                    help="Top-N values for corr_top_n (comma-separated ints)")
    ap.add_argument("--lasso-alphas",  default="0.001,0.01,0.1",
                    help="Lasso alpha values (comma-separated)")
    ap.add_argument("--elasticnet-alphas", default="0.001,0.01",
                    help="ElasticNet alpha values (comma-separated)")
    ap.add_argument("--elasticnet-l1-ratios", default="0.2,0.5,0.8",
                    help="ElasticNet l1_ratio values (comma-separated)")
    ap.add_argument("--pca-components", default="3,5,10",
                    help="PCA component counts (comma-separated ints)")
    ap.add_argument("--fa-components", default="3,5,10",
                    help="FactorAnalysis component counts (comma-separated ints)")
    ap.add_argument("--ae-latent-dims", default="5",
                    help="Autoencoder latent dimensions (comma-separated ints)")

    # MIDAS hyperparameters
    ap.add_argument("--n-lags", default="1,2,3,4",
                    help="MIDAS lag orders (comma-separated ints)")
    ap.add_argument("--freq-ratios", default="auto",
                    help="HF/LF frequency ratios. Use 'auto' for inference. "
                         "E.g. 'auto,3,12'")
    ap.add_argument("--lf-freq", default="QE",
                    help="Low-frequency output frequency alias for pandas date_range "
                         "(e.g. QE, ME, YE). Single value applied to all experiments.")
    ap.add_argument("--fill-strategies", default="zero,ffill_then_zero",
                    help="NaN fill strategies: zero, ffill_then_zero, mean (comma-separated)")

    # Regression backends
    ap.add_argument("--regression-models", default="elasticnet,ridge",
                    help="Regression backends: linear, ridge, lasso, elasticnet")
    ap.add_argument("--regression-l1-ratios", default="0.5",
                    help="l1_ratio values for ElasticNetCV (comma-separated)")

    # Backtest settings
    ap.add_argument("--train-windows", default="80,120",
                    help="Initial training window sizes (comma-separated ints)")
    ap.add_argument("--step-sizes", default="1",
                    help="Backtest step sizes (comma-separated ints)")
    ap.add_argument("--search-last-n-steps", type=int, default=0,
                    help="Only run the last N backtest steps per config (0 = all)")
    ap.add_argument("--search-max-configs", type=int, default=0,
                    help="Randomly sub-sample at most N configs (0 = no limit)")

    # Compute
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers (1 = sequential)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global random seed")

    # Output
    ap.add_argument("--results-file", default="data/forecasts/midas_experiment_results.csv",
                    help="Path for the detailed results CSV")

    return ap


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _parse_list(s: str, dtype=str) -> list:
    return [dtype(v.strip()) for v in s.split(",") if v.strip()]


def _selector_param_grid(method: str, args: argparse.Namespace) -> List[dict]:
    """Return the list of selector parameter dicts for a given method."""
    if method == "none":
        return [{}]
    elif method == "variance_filter":
        thresholds = _parse_list(args.variance_thresholds, float)
        return [{"threshold": t} for t in thresholds]
    elif method == "corr_top_n":
        return [{"top_n": n} for n in _parse_list(args.top_n, int)]
    elif method == "lasso":
        return [{"alpha": a} for a in _parse_list(args.lasso_alphas, float)]
    elif method == "elasticnet":
        return [
            {"alpha": a, "l1_ratio": l}
            for a in _parse_list(args.elasticnet_alphas, float)
            for l in _parse_list(args.elasticnet_l1_ratios, float)
        ]
    elif method == "pca":
        return [{"n_components": n} for n in _parse_list(args.pca_components, int)]
    elif method == "factor_analysis":
        return [{"n_components": n} for n in _parse_list(args.fa_components, int)]
    elif method == "autoencoder":
        return [{"latent_dim": d} for d in _parse_list(args.ae_latent_dims, int)]
    else:
        return [{}]


def build_selector(method: str, params: dict) -> Any:
    """Instantiate a selector from method name + param dict."""
    if method == "none":
        return IdentitySelector()
    elif method == "variance_filter":
        return VarianceFilter(threshold=params.get("threshold", 1e-6))
    elif method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    elif method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    elif method == "elasticnet":
        return ElasticNetSelector(
            alpha=params.get("alpha", 0.1),
            l1_ratio=params.get("l1_ratio", 0.5),
        )
    elif method == "pca":
        return PCACompressor(n_components=params.get("n_components", 5))
    elif method == "factor_analysis":
        return FactorAnalysisCompressor(n_components=params.get("n_components", 5))
    elif method == "autoencoder":
        return AutoencoderCompressor(latent_dim=params.get("latent_dim", 5))
    else:
        raise ValueError(f"Unknown selector method: '{method}'")


def generate_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Generate all experiment configurations."""
    selectors_to_run = _parse_list(args.selectors, str)
    n_lags_list      = _parse_list(args.n_lags, int)
    freq_ratio_list  = [
        None if v.strip().lower() == "auto" else int(v.strip())
        for v in args.freq_ratios.split(",")
        if v.strip()
    ]
    fill_strategies  = _parse_list(args.fill_strategies, str)
    regression_models = _parse_list(args.regression_models, str)
    l1_ratios         = _parse_list(args.regression_l1_ratios, float)
    train_windows     = _parse_list(args.train_windows, int)
    step_sizes        = _parse_list(args.step_sizes, int)
    lf_freq           = args.lf_freq.strip()

    configs: List[Dict[str, Any]] = []

    for sel_method in selectors_to_run:
        for sel_params in _selector_param_grid(sel_method, args):
            for (nl, fr, fs, rm, tw, sz) in itertools.product(
                n_lags_list, freq_ratio_list, fill_strategies,
                regression_models, train_windows, step_sizes,
            ):
                # Build regression_kwargs based on regression model type
                reg_kw: dict = {}
                if rm == "elasticnet":
                    reg_kw = {"l1_ratio": l1_ratios}   # passed as list to ElasticNetCV
                # Other models don't need extra kwargs by default

                configs.append({
                    "selector_method":  sel_method,
                    "selector_params":  sel_params,
                    "n_lags":           nl,
                    "freq_ratio":       fr,   # None → auto-infer in MIDASNowcast
                    "lf_freq":          lf_freq,
                    "fill_strategy":    fs,
                    "regression_model": rm,
                    "regression_kwargs": reg_kw,
                    "train_window":     tw,
                    "step_size":        sz,
                })

    if args.search_max_configs > 0 and len(configs) > args.search_max_configs:
        random.shuffle(configs)
        configs = configs[:args.search_max_configs]

    return configs


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    search_last_n_steps: int,
    seed: int,
) -> dict:
    """
    Run one grid configuration and return a metrics dict.

    The selector is wrapped in a lightweight adapter inside the backtester so
    it is refitted on every training fold → no leakage.
    """
    np.random.seed(seed)   # per-worker determinism

    res: dict = {
        "model":             "MIDAS",
        "selector_method":   config["selector_method"],
        "selector_params":   json.dumps(config["selector_params"]),
        "regression_model":  config["regression_model"],
        "regression_params": json.dumps(config["regression_kwargs"]),
        "n_lags":            config["n_lags"],
        "freq_ratio":        str(config["freq_ratio"]),
        "lf_freq":           config["lf_freq"],
        "fill_strategy":     config["fill_strategy"],
        "train_window":      config["train_window"],
        "step_size":         config["step_size"],
        "n_features_input":  X_panel.shape[1],
        "n_features_used":   None,
        "rmse":              None,
        "mae":               None,
        "mape":              None,
        "n_eval_points":     None,
        "n_folds":           None,
        "runtime_sec":       None,
        "seed":              seed,
        "window_type":       "expanding",
        "eval_mode":         "mixed_frequency",
        "status":            "running",
        "error_message":     "",
    }

    t0 = time.time()
    try:
        init_train = config["train_window"]
        step_sz    = config["step_size"]
        if search_last_n_steps > 0:
            required_start = len(y_target) - search_last_n_steps
            init_train = max(init_train, required_start)

        bt = RollingBacktester(
            initial_train_periods=init_train,
            step_size=step_sz,
            window_type="expanding",
            eval_mode="mixed_frequency",
        )

        # Build the selector instance
        selector = build_selector(config["selector_method"], config["selector_params"])

        # Build MIDAS model
        reg_kw = dict(config["regression_kwargs"])   # copy so pop() doesn't mutate
        model = MIDASNowcast(
            target_col=str(y_target.name),
            freq_ratio=config["freq_ratio"],
            n_lags=config["n_lags"],
            lf_freq=config["lf_freq"],
            regression_model=config["regression_model"],
            regression_kwargs=reg_kw,
            fill_strategy=config["fill_strategy"],
            seed=seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_df = bt.backtest(
                model=model,
                X_panel=X_panel,
                y_target=y_target,
                transformer=None,
                feature_selector=selector,
            )

        res["runtime_sec"] = round(time.time() - t0, 2)

        if eval_df.empty:
            res["status"] = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            return res

        metrics = compute_metrics(eval_df, model_name="MIDAS")
        res["rmse"]          = float(metrics["RMSE"].iloc[0])
        res["mae"]           = float(metrics["MAE"].iloc[0])
        res["mape"]          = float(metrics["MAPE"].iloc[0]) if "MAPE" in metrics.columns else float("nan")
        res["n_eval_points"] = len(eval_df)
        res["n_folds"]       = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)

        # Try to capture how many features were actually used post-selection
        if hasattr(selector, "selected_cols_"):
            res["n_features_used"] = len(selector.selected_cols_)
        elif hasattr(selector, "pca") and hasattr(selector, "pca") and selector.pca is not None:
            res["n_features_used"] = getattr(selector.pca, "n_components_", None)
        elif hasattr(selector, "fa") and selector.fa is not None:
            res["n_features_used"] = getattr(selector.fa, "n_components", None)
        else:
            res["n_features_used"] = X_panel.shape[1]

        res["status"] = "success"

    except Exception as exc:
        res["status"]       = "failed"
        res["error_message"] = str(exc)
        res["runtime_sec"]   = round(time.time() - t0, 2)
        logger.debug(f"Config failed: {exc}", exc_info=True)

    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    # Global seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("  MIDAS / U-MIDAS Grid-Search Experiment Runner")
    logger.info(f"  seed={args.seed}  n_jobs={args.n_jobs}")
    logger.info("=" * 60)

    # 1. Load data
    data_dir = Path(args.data_dir)
    logger.info("Loading panel ...")
    X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
    logger.info(f"Panel shape: {X_panel.shape}  |  Target: {y_target.name}")

    # Detect panel path for output naming
    if args.input_panel:
        _ip = Path(args.input_panel)
        panel_path: Optional[Path] = _ip if _ip.exists() else Path(args.data_dir) / _ip
    else:
        panel_path = None

    dataset_stem = panel_path.stem if panel_path else "panel"

    # 2. Build grid
    grid = generate_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    if not grid:
        logger.warning("Empty grid — check your CLI arguments. Exiting.")
        return

    # 3. Run experiments
    logger.info(f"Running experiments  (parallel={args.n_jobs > 1}, workers={args.n_jobs})")
    tstart = time.time()

    run_kwargs = dict(
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        seed=args.seed,
    )

    if args.n_jobs > 1:
        results = Parallel(n_jobs=args.n_jobs, verbose=5, prefer="processes")(
            delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
        )
    else:
        results = []
        for i, cfg in enumerate(grid, 1):
            logger.info(
                f"[{i}/{len(grid)}] "
                f"sel={cfg['selector_method']:<16s}  "
                f"reg={cfg['regression_model']:<11s}  "
                f"lags={cfg['n_lags']}  "
                f"tw={cfg['train_window']}"
            )
            results.append(run_single_experiment(cfg, **run_kwargs))

    total_time = time.time() - tstart

    # 4. Collect and sort
    df_res = pd.DataFrame(results)
    df_res["dataset_stem"] = dataset_stem
    df_res["dataset_file"] = str(panel_path or args.input_panel or args.data_dir)

    success      = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)

    logger.info(
        f"\nFinished in {total_time / 60:.1f} min  |  "
        f"Success: {len(success)}  |  Failed: {failed_count}"
    )

    # 5. Save results
    out_path = Path(args.results_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not success.empty:
        df_res = df_res.sort_values(["rmse", "mae"], ascending=True).reset_index(drop=True)

        # Console summary
        display_cols = [c for c in [
            "selector_method", "regression_model", "n_lags", "lf_freq",
            "train_window", "rmse", "mae", "runtime_sec"
        ] if c in df_res.columns]
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        print(df_res.head(5)[display_cols].to_string(index=False))

    # Always save the full table (includes failed runs for diagnosis)
    df_res.to_csv(out_path, index=False)
    logger.info(f"\nDetailed results → {out_path.absolute()}")

    # 6. Best configuration JSON
    if not success.empty:
        best = df_res.iloc[0].to_dict()
        for json_col in ("selector_params", "regression_params"):
            try:
                best[json_col] = json.loads(best[json_col])
            except Exception:
                pass

        # Parameter-aware filename
        suffix = _build_suffix(
            panel_path=panel_path,
            seed=args.seed,
            lf_freq=args.lf_freq,
            selector=best.get("selector_method", ""),
        )
        best_path = out_path.parent / f"best_config_{suffix}.json"
        with open(best_path, "w") as fh:
            json.dump(best, fh, indent=4, default=str)
        logger.info(f"Best configuration → {best_path}")
    else:
        logger.error("ALL experiments failed — check error_message column in results CSV.")


if __name__ == "__main__":
    main()
