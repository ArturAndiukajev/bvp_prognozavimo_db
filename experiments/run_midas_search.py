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
        --seed 123

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
        --seed 123
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
import optuna
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

from nowcasting.evaluation.checkpoint import prune_grid, run_chunks, build_run_signature
from nowcasting.utils.data_loader import load_mf_panels, load_cf_panel, _DEFAULT_DATA_DIR
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
    FastScreeningFilter,
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
    panel_stem = _sanitize(panel_path.stem if panel_path else "panel")[:30]
    parts = ["midas", panel_stem]
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
    ap.add_argument("--selectors", default="none,fast_screen,corr_top_n,pca",
                    help="Comma-separated selector methods: none, variance_filter, corr_top_n, "
                         "lasso, elasticnet, pca, factor_analysis, autoencoder, fast_screen")
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
    ap.add_argument("--fast-screen-top-k", default="50,100",
                    help="Fast screen top K features (comma-separated ints)")

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
    ap.add_argument("--search-strategy", type=str, default="full", choices=["full", "staged"],
                    help="Use 'staged' for a fast 1-pass coarse search to filter options.")
    ap.add_argument("--search-top-k", type=int, default=5,
                    help="Top configs to promote to final pass in staged search.")

    # ---------- Optuna ----------
    ap.add_argument("--engine", choices=["optuna", "grid", "staged"], default=None,
                    help="Search engine to use: optuna (Bayesian), grid (exhaustive), staged (coarse-to-fine)")
    ap.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna engine")
    ap.add_argument("--study-storage", type=str, default=None, 
                    help="Optuna storage URI (e.g. sqlite:///midas.db). Defaults to sqlite in results directory.")
    ap.add_argument("--study-name", type=str, default=None, help="Optuna study name for persistence and resuming.")
    ap.add_argument("--save-every-n-trials", type=int, default=5, help="Save summary CSV every N trials")

    # Compute
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers (1 = sequential)")
    ap.add_argument("--seed", type=int, default=123,
                    help="Global random seed")

    # Output
    ap.add_argument("--resume", action="store_true", help="Resume from an interrupted search using existing results CSV.")
    ap.add_argument("--checkpoint-chunk-size", type=int, default=10, help="Save to CSV after completing this many configurations.")
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
    elif method == "fast_screen":
        return [{"top_n": n} for n in _parse_list(args.fast_screen_top_k, int)]
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
    elif method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_n", 50))
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
            
        # Track feature counts seamlessly natively provided by the backtester updates
        res["n_raw_features"]    = int(eval_df["n_raw_features"].median()) if "n_raw_features" in eval_df else None
        res["n_trans_features"]  = int(eval_df["n_trans_features"].median()) if "n_trans_features" in eval_df else None
        res["n_sel_features"]    = int(eval_df["n_sel_features"].median()) if "n_sel_features" in eval_df else None
        res["n_model_used_vars"] = int(eval_df["n_model_used_features"].median()) if "n_model_used_features" in eval_df else None

        res["status"] = "success"
        res["_eval_df"] = eval_df

    except Exception as exc:
        res["status"]       = "failed"
        res["error_message"] = str(exc)
        res["runtime_sec"]   = round(time.time() - t0, 2)
        logger.debug(f"Config failed: {exc}", exc_info=True)

    return res


# ---------------------------------------------------------------------------
# Optuna Engine
# ---------------------------------------------------------------------------

class OptunaBestTracker:
    def __init__(self, out_dir: Path, suffix: str, initial_best: float = float("inf")):
        self.best_rmse = initial_best
        self.out_dir = out_dir
        self.suffix = suffix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
def optuna_objective(
    trial: optuna.Trial, 
    X_panel: pd.DataFrame, 
    y_target: pd.Series, 
    args: argparse.Namespace, 
    tracker: OptunaBestTracker
) -> float:
    selector_name = trial.suggest_categorical("selector", ["none", "corr_top_n", "fast_screen", "variance_filter", "lasso", "elasticnet", "pca", "factor_analysis"])
    selector_params = {}
    
    if selector_name == "variance_filter":
        selector_params["threshold"] = trial.suggest_float("variance_threshold", 1e-6, 1e-3, log=True)
    elif selector_name == "corr_top_n":
        selector_params["top_n"] = trial.suggest_int("corr_top_n_k", 5, 50)
    elif selector_name == "lasso":
        selector_params["alpha"] = trial.suggest_float("lasso_alpha", 1e-4, 1.0, log=True)
    elif selector_name == "elasticnet":
        selector_params["alpha"] = trial.suggest_float("elasticnet_selector_alpha", 1e-4, 1.0, log=True)
        selector_params["l1_ratio"] = trial.suggest_float("elasticnet_selector_l1", 0.1, 0.9)
    elif selector_name == "fast_screen":
        selector_params["top_n"] = trial.suggest_int("fast_screen_k", 20, 100)
    elif selector_name == "pca":
        selector_params["n_components"] = trial.suggest_int("pca_comp", 2, 20)
    elif selector_name == "factor_analysis":
        selector_params["n_components"] = trial.suggest_int("fa_comp", 2, 20)

    reg_model = trial.suggest_categorical("regression_model", ["linear", "ridge", "lasso", "elasticnet"])
    reg_params = {}
    
    if reg_model in ("ridge", "lasso"):
        reg_params["alpha"] = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
    elif reg_model == "elasticnet":
        reg_params["alpha"] = trial.suggest_float("reg_enet_alpha", 1e-4, 10.0, log=True)
        reg_params["l1_ratio"] = trial.suggest_float("reg_enet_l1", 0.1, 0.9)

    n_lags = trial.suggest_int("n_lags", 1, 12)
    fill_strategy = trial.suggest_categorical("fill_strategy", ["zero", "ffill_then_zero", "mean"])
    train_window = trial.suggest_categorical("train_window", [60, 80, 100, 120])
    step_size = trial.suggest_categorical("step_size", [1, 3])

    freq_choices = [v.strip() for v in str(args.freq_ratios).split(",") if v.strip()]
    freq_choice = trial.suggest_categorical("freq_ratio", freq_choices)
    freq_ratio = freq_choice if freq_choice == "auto" else int(freq_choice)

    config = {
        "selector_method": selector_name,
        "selector_params": selector_params,
        "regression_model": reg_model,
        "regression_kwargs": reg_params,
        "n_lags": n_lags,
        "freq_ratio": freq_ratio,
        "lf_freq": args.lf_freq,
        "fill_strategy": fill_strategy,
        "train_window": train_window,
        "step_size": step_size,
    }
    
    res = run_single_experiment(
        config=config,
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        seed=args.seed
    )
    
    for k, v in res.items():
        if k != "_eval_df":
            trial.set_user_attr(k, v)
            
    if res.get("status") != "success":
        raise optuna.TrialPruned(f"Run failed: {res.get('error_message', 'Unknown error')}")
        
    rmse = res.get("rmse")
    if rmse is None or pd.isna(rmse):
        raise optuna.TrialPruned("RMSE was NaN or None")
        
    if rmse < tracker.best_rmse:
        tracker.best_rmse = rmse
        if "_eval_df" in res and not res["_eval_df"].empty:
            preds_path = tracker.out_dir / f"midas_optuna_best_preds_{tracker.suffix}.csv"
            res["_eval_df"].to_csv(preds_path, index=False)
            logger.info(f"New best RMSE ({rmse:.4f}) -> {preds_path}")
            
    return rmse


def run_optuna_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    logger.info("Starting OPTUNA engine.")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    study_name = args.study_name or f"midas_{suffix}"
    storage_url = args.study_storage or f"sqlite:///{out_dir.absolute()}/midas_optuna.db"
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )
    
    tracker = OptunaBestTracker(out_dir, suffix)
    
    if len(study.trials) > 0:
        logger.info(f"Resuming study '{study_name}' with {len(study.trials)} existing trials.")
        try:
            tracker.best_rmse = study.best_value
        except ValueError:
            pass

    out_csv = out_dir / f"optuna_trials_{suffix}.csv"
    
    def save_callback(study_cb: optuna.Study, trial_cb: optuna.trial.FrozenTrial):
        if trial_cb.number % args.save_every_n_trials == 0:
            df_trials = study_cb.trials_dataframe()
            df_trials.to_csv(out_csv, index=False)
            try:
                best_trial = study_cb.best_trial
                best_dict = best_trial.user_attrs.copy()
                best_dict["rmse"] = study_cb.best_value
                with open(out_dir / f"optuna_best_config_{suffix}.json", "w") as f:
                    json.dump(best_dict, f, indent=4, default=str)
            except ValueError:
                pass

    study.optimize(
        lambda t: optuna_objective(t, X_panel, y_target, args, tracker),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        catch=(Exception,),
        callbacks=[save_callback]
    )
    
    df_trials = study.trials_dataframe()
    df_trials.to_csv(out_csv, index=False)
    logger.info(f"Optuna trials saved to {out_csv}")
    
    try:
        best_trial = study.best_trial
        logger.info(f"Optuna Best RMSE: {best_trial.value:.4f}")
        logger.info(f"Optuna Best Params: {best_trial.params}")
        best_dict = best_trial.user_attrs.copy()
        with open(out_dir / f"optuna_best_config_{suffix}.json", "w") as f:
            json.dump(best_dict, f, indent=4, default=str)
    except ValueError:
        logger.warning("No successful trials completed.")


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

    # 3. Output directory setup
    out_path = Path(args.results_file)
    suffix_base = _build_suffix(panel_path, args.seed)

    # 4. Engine Branching
    active_engine = args.engine
    if active_engine is None:
        active_engine = "staged" if args.search_strategy == "staged" else "grid"

    if active_engine == "optuna":
        run_optuna_engine(X_panel, y_target, args, out_path.parent, suffix_base)
        return

    # Fallback to standard grid/staged below...
    grid = generate_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    if not grid:
        logger.warning("Empty grid — check your CLI arguments. Exiting.")
        return

    logger.info(f"Running experiments  (parallel={args.n_jobs > 1}, workers={args.n_jobs})")
    tstart = time.time()

    run_kwargs = dict(
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        seed=args.seed,
    )
    
    out_path = Path(args.results_file)

    if active_engine == "staged":
        logger.info(f"--- Stage 1: Coarse Search ({len(grid)} configs, 3 steps) ---")
        stage1_kwargs = dict(run_kwargs)
        stage1_steps = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3
        stage1_kwargs["search_last_n_steps"] = stage1_steps
        
        run_signature = build_run_signature(
            script_name="run_midas_search.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            lf_freq=args.lf_freq
        )
        grid_s1 = prune_grid(grid, stage1_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        results_s1 = run_chunks(grid_s1, run_single_experiment, stage1_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 1")

        if out_path.exists():
            try:
                full_s1_df = pd.read_csv(out_path)
                full_s1_df = full_s1_df[(full_s1_df["status"] == "success")]
                best_configs = full_s1_df.sort_values("rmse").head(args.search_top_k).to_dict("records")
            except Exception:
                df_s1 = pd.DataFrame([r for r in results_s1 if r.get("status") == "success"])
                best_configs = df_s1.sort_values("rmse", ascending=True).head(args.search_top_k).to_dict("records") if not df_s1.empty else []
        else:
            df_s1 = pd.DataFrame([r for r in results_s1 if r.get("status") == "success"])
            best_configs = df_s1.sort_values("rmse", ascending=True).head(args.search_top_k).to_dict("records") if not df_s1.empty else []

        if not best_configs:
            logger.error("Stage 1 failed completely.")
            return

        logger.info(f"--- Stage 2: Fine Search (Top {len(best_configs)} configs, {args.search_last_n_steps} steps) ---")
        clean_keys = set(grid[0].keys())
        staged_grid = [{k: v for k, v in cfg.items() if k in clean_keys} for cfg in best_configs]
        staged_grid = prune_grid(staged_grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)

        results = run_chunks(staged_grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 2")
    else:
        run_signature = build_run_signature(
            script_name="run_midas_search.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            lf_freq=args.lf_freq
        )
        grid = prune_grid(grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        results = run_chunks(grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10))

    total_time = time.time() - tstart

    # 4. Collect and sort
    if out_path.exists():
        df_res = pd.read_csv(out_path)
    else:
        df_res = pd.DataFrame(results)
        
    if "dataset_stem" not in df_res.columns:
        df_res["dataset_stem"] = dataset_stem
    if "dataset_file" not in df_res.columns:
        df_res["dataset_file"] = str(panel_path or args.input_panel or args.data_dir)

    success      = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)

    logger.info(
        f"\nFinished in {total_time / 60:.1f} min  |  "
        f"Success: {len(success)}  |  Failed: {failed_count}"
    )

    # 5. Save results
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
    suffix = _build_suffix(panel_path, args.seed)
    
    best_eval_df = pd.DataFrame()
    if not success.empty:
        best_cfg = success.iloc[0].to_dict()
        memory_res = [r for r in results if r.get("_config_id") == best_cfg.get("_config_id")]
        
        if memory_res and "_eval_df" in memory_res[0]:
            best_eval_df = memory_res[0]["_eval_df"]
        else:
            logger.info("Re-evaluating best configuration to regenerate predictions output...")
            b_cfg = {k:v for k,v in best_cfg.items() if not str(k).startswith("_")}
            if isinstance(b_cfg.get('selector_params'), str):
                b_cfg['selector_params'] = json.loads(b_cfg['selector_params'])
            if isinstance(b_cfg.get('regression_params'), str):
                b_cfg['regression_params'] = json.loads(b_cfg['regression_params'])
            b_res = run_single_experiment(b_cfg, **run_kwargs)
            best_eval_df = b_res.get("_eval_df", pd.DataFrame())

        if not best_eval_df.empty:
            preds_path = out_path.parent / f"predictions_{suffix}.csv"
            best_eval_df.to_csv(preds_path, index=False)
            logger.info(f"MIDAS best preds → {preds_path}")

    if "_eval_df" in df_res.columns:
        df_res = df_res.drop(columns=["_eval_df"])

    df_res.to_csv(out_path, index=False)
    logger.info(f"\nDetailed results → {out_path.absolute()}")

    # 6. Best configuration JSON
    if not success.empty:
        best = success.sort_values("rmse").iloc[0].to_dict()
        if "_eval_df" in best:
            del best["_eval_df"]
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
