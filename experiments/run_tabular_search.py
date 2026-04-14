"""
Systematic search over:
  - Feature selection / compression methods  (none, corr_top_n, variance_filter,
    lasso, elasticnet, pca, factor_analysis)
  - ElasticNet hyperparameters  (cv, max_iter, l1_ratio, fill_strategy)
  - LightGBM hyperparameters    (n_estimators, learning_rate, num_leaves, …)
  - Backtest settings            (train_window, step_size)

Key design guarantees
---------------------
  • No look-ahead bias: all selectors / scalers fit only on the training fold
    and are then applied to the corresponding test fold.
  • Robust: failed configs are caught and stored; the search continues.
  • Reproducible: numpy and per-model seeds are set consistently.
  • Parallel-safe: LightGBM n_jobs=1 inside each worker; BLAS thread limits set.

Quick start
-----------
    python scripts/run_tabular_search.py \\
        --models elasticnet,lightgbm \\
        --selectors none,pca --pca-components 3 \\
        --enet-l1-ratios 0.5 --enet-cv 3 \\
        --lgbm-n-estimators 50 --lgbm-learning-rates 0.05 \\
        --train-windows 80 --search-last-n-steps 3 --seed 123

Full search
-----------
    python scripts/run_tabular_search.py \\
        --models elasticnet,lightgbm \\
        --selectors none,corr_top_n,variance_filter,lasso,elasticnet,pca,factor_analysis \\
        --top-n 10,20,50 \\
        --variance-thresholds 1e-6 \\
        --lasso-alphas 0.001,0.01,0.1 \\
        --elasticnet-selector-alphas 0.001,0.01 \\
        --elasticnet-selector-l1-ratios 0.2,0.5,0.8 \\
        --pca-components 3,5,10 \\
        --fa-components 3,5,10 \\
        --enet-l1-ratios 0.1,0.3,0.5,0.7,0.9 \\
        --enet-max-iter 2000,5000 \\
        --enet-cv 3,5 \\
        --enet-fill-strategies zero,mean \\
        --lgbm-n-estimators 100,300 \\
        --lgbm-learning-rates 0.01,0.05,0.1 \\
        --lgbm-num-leaves 15,31,63 \\
        --lgbm-max-depth -1,5,10 \\
        --lgbm-min-child-samples 5,10,20 \\
        --lgbm-subsample 0.8,1.0 \\
        --lgbm-colsample-bytree 0.8,1.0 \\
        --lgbm-reg-alpha 0.0,0.1 \\
        --lgbm-reg-lambda 0.0,0.1 \\
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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import optuna

# Single-thread BLAS to avoid nested-parallelism collisions
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcasting.evaluation.checkpoint import prune_grid, run_chunks, build_run_signature
from nowcasting.utils.data_loader import load_cf_panel, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.ml_regression import ElasticNetNowcast, LightGBMNowcast

from nowcasting.features.selectors import (
    IdentitySelector,
    VarianceFilter,
    CorrTopNSelector,
    LassoSelector,
    ElasticNetSelector,
    PCACompressor,
    FactorAnalysisCompressor,
    FastScreeningFilter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tabular_search")


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(s))



def _build_suffix(panel_path: Optional[Path], seed: int, model: str = "") -> str:
    stem = _sanitize(panel_path.stem if panel_path else "panel")[:30]
    parts = [model, stem, f"s{seed}"] if model else [stem, f"s{seed}"]
    return "_".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="ElasticNet & LightGBM Grid-Search Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    ap.add_argument("--input-panel", default=None)
    ap.add_argument("--target-col", default=None, dest="target")

    # Models
    ap.add_argument("--models", default="elasticnet,lightgbm",
                    help="Comma-separated: elasticnet, lightgbm")

    # ---------- Feature selectors ----------
    ap.add_argument("--selectors", default="none,fast_screen,corr_top_n,pca",
                    help="none, fast_screen, variance_filter, corr_top_n, lasso, elasticnet, pca, factor_analysis")
    ap.add_argument("--variance-thresholds", default="0.0,1e-6")
    ap.add_argument("--top-n",  default="10,20,50")
    ap.add_argument("--fast-screen-top-k", default="50,100")
    ap.add_argument("--lasso-alphas",  default="0.001,0.01,0.1",
                    help="Alpha values for the Lasso *selector*")
    ap.add_argument("--elasticnet-selector-alphas", default="0.001,0.01",
                    help="Alpha values for the ElasticNet *selector*")
    ap.add_argument("--elasticnet-selector-l1-ratios", default="0.2,0.5,0.8",
                    help="l1_ratio values for the ElasticNet *selector*")
    ap.add_argument("--pca-components", default="3,5,10")
    ap.add_argument("--fa-components",  default="3,5,10")

    # ---------- ElasticNetNowcast hyperparams ----------
    ap.add_argument("--enet-l1-ratios",  default="0.1,0.5,0.9",
                    help="l1_ratio values for ElasticNetCV model")
    ap.add_argument("--enet-cv",         default="3,5")
    ap.add_argument("--enet-max-iter",   default="2000")
    ap.add_argument("--enet-fill-strategies", default="zero",
                    help="NaN fill for ElasticNet: zero, mean, median, ffill_then_zero")

    # ---------- LightGBMNowcast hyperparams ----------
    ap.add_argument("--lgbm-n-estimators",     default="100,300")
    ap.add_argument("--lgbm-learning-rates",   default="0.01,0.05,0.1")
    ap.add_argument("--lgbm-num-leaves",       default="31")
    ap.add_argument("--lgbm-max-depth",        default="-1")
    ap.add_argument("--lgbm-min-child-samples",default="20")
    ap.add_argument("--lgbm-subsample",        default="1.0")
    ap.add_argument("--lgbm-colsample-bytree", default="1.0")
    ap.add_argument("--lgbm-reg-alpha",        default="0.0")
    ap.add_argument("--lgbm-reg-lambda",       default="0.0")

    # ---------- Backtest ----------
    ap.add_argument("--train-windows", default="80,120")
    ap.add_argument("--step-sizes",    default="1")
    ap.add_argument("--search-last-n-steps", type=int, default=0)
    ap.add_argument("--search-max-configs",  type=int, default=0)
    ap.add_argument("--search-strategy", type=str, default="full", choices=["full", "staged"])
    ap.add_argument("--search-top-k", type=int, default=5)

    # ---------- Optuna ----------
    ap.add_argument("--engine", choices=["optuna", "grid", "staged", "evaluate"], default=None,
                    help="Search engine to use: optuna, grid, staged, or evaluate (single run from config)")
    ap.add_argument("--config-path", type=str, default=None, help="Path to a JSON configuration file for '--engine evaluate'")
    ap.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna engine")
    ap.add_argument("--study-storage", type=str, default=None, 
                    help="Optuna storage URI (e.g. sqlite:///tabular.db). Defaults to sqlite in results directory.")
    ap.add_argument("--study-name", type=str, default=None, help="Optuna study name for persistence and resuming.")
    ap.add_argument("--save-every-n-trials", type=int, default=5, help="Save summary CSV every N trials")

    # ---------- Compute ----------
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed",   type=int, default=123)

    # ---------- Output ----------
    ap.add_argument("--resume", action="store_true", help="Resume from an interrupted search using existing results CSV.")
    ap.add_argument("--checkpoint-chunk-size", type=int, default=10, help="Save to CSV after completing this many configurations.")
    ap.add_argument("--results-file", default="data/forecasts/tabular_experiment_results.csv")

    return ap


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _plist(s: str, dtype=str) -> list:
    return [dtype(v.strip()) for v in s.split(",") if v.strip()]


def _selector_param_grid(method: str, args: argparse.Namespace) -> List[dict]:
    if method == "none":
        return [{}]
    elif method == "variance_filter":
        return [{"threshold": t} for t in _plist(args.variance_thresholds, float)]
    elif method == "corr_top_n":
        return [{"top_n": n} for n in _plist(args.top_n, int)]
    elif method == "lasso":
        return [{"alpha": a} for a in _plist(args.lasso_alphas, float)]
    elif method == "elasticnet":
        return [
            {"alpha": a, "l1_ratio": l}
            for a in _plist(args.elasticnet_selector_alphas, float)
            for l in _plist(args.elasticnet_selector_l1_ratios, float)
        ]
    elif method == "pca":
        return [{"n_components": n} for n in _plist(args.pca_components, int)]
    elif method == "factor_analysis":
        return [{"n_components": n} for n in _plist(args.fa_components, int)]
    elif method == "fast_screen":
        return [{"top_n": n} for n in _plist(args.fast_screen_top_k, int)]
    return [{}]


def build_selector(method: str, params: dict) -> Any:
    if method == "none":
        return IdentitySelector()
    elif method == "variance_filter":
        return VarianceFilter(threshold=params.get("threshold", 1e-6))
    elif method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    elif method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    elif method == "elasticnet":
        return ElasticNetSelector(alpha=params.get("alpha", 0.1),
                                  l1_ratio=params.get("l1_ratio", 0.5))
    elif method == "pca":
        return PCACompressor(n_components=params.get("n_components", 5))
    elif method == "factor_analysis":
        return FactorAnalysisCompressor(n_components=params.get("n_components", 5))
    elif method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_n", 50))
    raise ValueError(f"Unknown selector: {method}")


def _enet_model_grid(args: argparse.Namespace) -> List[dict]:
    return [
        {"l1_ratio": l, "cv": c, "max_iter": mi, "fill_strategy": fs}
        for l  in _plist(args.enet_l1_ratios, float)
        for c  in _plist(args.enet_cv, int)
        for mi in _plist(args.enet_max_iter, int)
        for fs in _plist(args.enet_fill_strategies, str)
    ]


def _lgbm_model_grid(args: argparse.Namespace) -> List[dict]:
    return [
        {
            "n_estimators":      ne,
            "learning_rate":     lr,
            "num_leaves":        nl,
            "max_depth":         md,
            "min_child_samples": mc,
            "subsample":         ss,
            "colsample_bytree":  cs,
            "reg_alpha":         ra,
            "reg_lambda":        rl,
        }
        for ne in _plist(args.lgbm_n_estimators, int)
        for lr in _plist(args.lgbm_learning_rates, float)
        for nl in _plist(args.lgbm_num_leaves, int)
        for md in _plist(args.lgbm_max_depth, int)
        for mc in _plist(args.lgbm_min_child_samples, int)
        for ss in _plist(args.lgbm_subsample, float)
        for cs in _plist(args.lgbm_colsample_bytree, float)
        for ra in _plist(args.lgbm_reg_alpha, float)
        for rl in _plist(args.lgbm_reg_lambda, float)
    ]


def generate_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    models_to_run    = _plist(args.models, str)
    selectors_to_run = _plist(args.selectors, str)
    train_windows    = _plist(args.train_windows, int)
    step_sizes       = _plist(args.step_sizes, int)

    configs: List[Dict[str, Any]] = []

    for model_name in models_to_run:
        if model_name == "elasticnet":
            model_param_grid = _enet_model_grid(args)
        elif model_name == "lightgbm":
            model_param_grid = _lgbm_model_grid(args)
        else:
            logger.warning(f"Unknown model '{model_name}' — skipping.")
            continue

        for sel_method in selectors_to_run:
            for sel_params in _selector_param_grid(sel_method, args):
                for model_params, tw, sz in itertools.product(model_param_grid, train_windows, step_sizes):
                    configs.append({
                        "model":           model_name,
                        "selector_method": sel_method,
                        "selector_params": sel_params,
                        "model_params":    model_params,
                        "train_window":    tw,
                        "step_size":       sz,
                    })

    if args.search_max_configs > 0 and len(configs) > args.search_max_configs:
        random.shuffle(configs)
        configs = configs[:args.search_max_configs]

    return configs


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    search_last_n_steps: int,
    seed: int,
) -> dict:
    """Run one configuration and return a result dict. Never raises."""
    np.random.seed(seed)

    model_name = config["model"]
    res: dict = {
        "model":            model_name,
        "selector_method":  config["selector_method"],
        "selector_params":  json.dumps(config["selector_params"]),
        "model_params":     json.dumps(config["model_params"]),
        "train_window":     config["train_window"],
        "step_size":        config["step_size"],
        "window_type":      "expanding",
        "eval_mode":        "common_frequency",
        "n_features_input": X_panel.shape[1],
        "n_features_used":  None,
        "rmse":             None,
        "mae":              None,
        "mape":             None,
        "n_eval_points":    None,
        "n_folds":          None,
        "runtime_sec":      None,
        "seed":             seed,
        "status":           "running",
        "error_message":    "",
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
            eval_mode="common_frequency"
        )

        selector = build_selector(config["selector_method"], config["selector_params"])
        mp = dict(config["model_params"])   # copy — might be mutated downstream

        if model_name == "elasticnet":
            model = ElasticNetNowcast(
                target_col=str(y_target.name),
                seed=seed,
                l1_ratio=mp.pop("l1_ratio", None),
                cv=mp.pop("cv", 5),
                max_iter=mp.pop("max_iter", 2000),
                fill_strategy=mp.pop("fill_strategy", "zero"),
            )
        elif model_name == "lightgbm":
            model = LightGBMNowcast(
                target_col=str(y_target.name),
                seed=seed,
                **mp,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

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
            res["status"]        = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            return res

        metrics = compute_metrics(eval_df, model_name=model_name)
        res["rmse"]         = float(metrics["RMSE"].iloc[0])
        res["mae"]          = float(metrics["MAE"].iloc[0])
        res["mape"]         = float(metrics["MAPE"].iloc[0]) if "MAPE" in metrics.columns else float("nan")
        res["n_eval_points"]= len(eval_df)
        res["n_folds"]      = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)

        # Capture actual feature dimension after selection
        if hasattr(selector, "selected_cols_") and selector.selected_cols_ is not None:
            res["n_features_used"] = len(selector.selected_cols_)
        elif hasattr(selector, "pca") and getattr(selector, "pca", None) is not None:
            res["n_features_used"] = getattr(selector.pca, "n_components_", None)
        elif hasattr(selector, "fa") and getattr(selector, "fa", None) is not None:
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
        res["runtime_sec"]  = round(time.time() - t0, 2)
        logger.debug(f"Config failed [{model_name}]: {exc}", exc_info=True)

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
        selector_params["alpha"] = trial.suggest_float("elasticnet_alpha", 1e-4, 1.0, log=True)
        selector_params["l1_ratio"] = trial.suggest_float("elasticnet_l1", 0.1, 0.9)
    elif selector_name == "fast_screen":
        selector_params["top_n"] = trial.suggest_int("fast_screen_k", 20, 100)
    elif selector_name == "pca":
        selector_params["n_components"] = trial.suggest_int("pca_comp", 2, 20)
    elif selector_name == "factor_analysis":
        selector_params["n_components"] = trial.suggest_int("fa_comp", 2, 20)

    model_name = trial.suggest_categorical("model", ["elasticnet", "lightgbm"])
    model_params = {}
    
    if model_name == "elasticnet":
        model_params["l1_ratio"] = trial.suggest_float("enet_l1_ratio", 0.1, 0.9)
        model_params["cv"] = trial.suggest_categorical("enet_cv", [3, 5])
        model_params["max_iter"] = trial.suggest_categorical("enet_max_iter", [1000, 2000, 5000])
        model_params["fill_strategy"] = trial.suggest_categorical("enet_fill", ["zero", "mean", "median"])
    elif model_name == "lightgbm":
        model_params["n_estimators"] = trial.suggest_int("lgbm_n_estimators", 50, 500)
        model_params["learning_rate"] = trial.suggest_float("lgbm_lr", 1e-3, 0.2, log=True)
        model_params["num_leaves"] = trial.suggest_int("lgbm_num_leaves", 15, 63)
        model_params["max_depth"] = trial.suggest_categorical("lgbm_max_depth", [-1, 5, 10])
        model_params["min_child_samples"] = trial.suggest_int("lgbm_min_child", 5, 50)
        model_params["subsample"] = trial.suggest_float("lgbm_subsample", 0.5, 1.0)
        model_params["colsample_bytree"] = trial.suggest_float("lgbm_colsample", 0.5, 1.0)

        model_params["reg_alpha"] = trial.suggest_float("lgbm_alpha", 1e-4, 1.0, log=True)
        model_params["reg_lambda"] = trial.suggest_float("lgbm_lambda", 1e-4, 1.0, log=True)

    train_window = trial.suggest_categorical("train_window", [60, 80, 100, 120])
    step_size = trial.suggest_categorical("step_size", [1, 3])

    config = {
        "model": model_name,
        "selector_method": selector_name,
        "selector_params": selector_params,
        "model_params": model_params,
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
            preds_path = tracker.out_dir / f"{model_name}_optuna_best_preds_{tracker.suffix}.csv"
            res["_eval_df"].to_csv(preds_path, index=False)
            logger.info(f"New best RMSE ({rmse:.4f}) -> {preds_path}")
            
    return rmse


def run_optuna_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    logger.info("Starting OPTUNA engine.")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    study_name = args.study_name or f"tabular_{suffix}"
    storage_url = args.study_storage or f"sqlite:///{out_dir.absolute()}/tabular_optuna.db"
    
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


def run_evaluation_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    if not args.config_path:
        logger.error("Evaluation mode requires '--config-path'.")
        return

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Robust parsing of nested JSON strings (e.g. from results CSV exports)
    def _parse_if_str(val):
        if isinstance(val, str):
            try: return json.loads(val)
            except: return val
        return val

    # Extract configuration parts
    eval_config = {}
    for k, v in config.items():
        if k not in ["rmse", "mae", "mape", "status", "runtime_sec", "n_eval_points", "n_folds", "error_message", "n_features_used", "n_raw_features", "n_trans_features", "n_sel_features", "n_model_used_vars"]:
            eval_config[k] = _parse_if_str(v)

    logger.info(f"Evaluating Tabular with config from {args.config_path}")
    
    # Wrap in single-item grid for checkpoint system
    search_grid = [eval_config]
    
    run_signature = build_run_signature(
        script_name="run_tabular_search.py",
        input_panel=str(args.input_panel or args.data_dir),
        target=str(args.target),
        seed=args.seed,
        search_last_n_steps=args.search_last_n_steps,
        search_strategy=args.search_strategy,
        models=args.models,
        engine="evaluate"
    )
    
    out_path = Path(args.results_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pruned_grid = prune_grid(search_grid, args.search_last_n_steps, out_path, resume=args.resume, run_signature=run_signature)
    
    if not pruned_grid:
        logger.info("Evaluation already completed and found in results CSV. Skipping.")
        return

    run_kwargs = dict(
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        seed=args.seed,
    )

    results = run_chunks(pruned_grid, run_single_experiment, run_kwargs, out_path, n_jobs=1, chunk_sz=1, stage_name="Evaluation")
    
    if results and results[0]["status"] == "success":
        res = results[0]
        logger.info(f"Evaluation Success | RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f}")
        eval_df = res.pop("_eval_df", pd.DataFrame())
        if not eval_df.empty:
            out_preds = out_dir / f"tabular_eval_preds_{suffix}.csv"
            eval_df.to_csv(out_preds, index=False)
            logger.info(f"Saved predictions to {out_preds}")
        
        out_json = out_dir / f"tabular_eval_results_{suffix}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items() if k != "_eval_df"}, f, indent=4, default=str)
        logger.info(f"Saved results to {out_json}")
    else:
        err = results[0].get("error_message") if results else "Unknown error"
        logger.error(f"Evaluation failed: {err}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("  Tabular ML Grid-Search  (ElasticNet + LightGBM)")
    logger.info(f"  seed={args.seed}  |  n_jobs={args.n_jobs}")
    logger.info("=" * 60)

    # 1. Load panel
    data_dir = Path(args.data_dir)
    logger.info("Loading panel ...")
    X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
    logger.info(f"Panel: {X_panel.shape}  |  Target: {y_target.name}")

    # Resolve panel stem for output filenames
    if args.input_panel:
        _ip = Path(args.input_panel)
        panel_path: Optional[Path] = _ip if _ip.exists() else Path(args.data_dir) / _ip
    else:
        panel_path = None
    dataset_stem = panel_path.stem if panel_path else "panel"

    # 2. Extract out_path directory
    out_path = Path(args.results_file)
    suffix_base = _build_suffix(panel_path, args.seed)

    # 3. Branch routing by Engine
    active_engine = args.engine
    if active_engine is None:
        active_engine = "staged" if args.search_strategy == "staged" else "grid"

    if active_engine == "optuna":
        run_optuna_engine(X_panel, y_target, args, out_path.parent, suffix_base)
        return

    if active_engine == "evaluate":
        run_evaluation_engine(X_panel, y_target, args, out_path.parent, suffix_base)
        return

    grid = generate_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    if not grid:
        logger.warning("Empty grid — check CLI arguments. Exiting.")
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
            script_name="run_tabular_search.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            models=args.models
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
            script_name="run_tabular_search.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            models=args.models
        )
        grid = prune_grid(grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        results = run_chunks(grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10))

    total_time = time.time() - tstart

    # 4. Collect + sort
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

    out_base = Path(args.results_file)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if not success.empty:
        df_res = df_res.sort_values(["rmse", "mae"], ascending=True).reset_index(drop=True)

        display_cols = [c for c in [
            "model", "selector_method", "train_window", "rmse", "mae",
            "n_features_used", "runtime_sec"
        ] if c in df_res.columns]
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        print(df_res.head(5)[display_cols].to_string(index=False))

    # 5. Save per-model CSVs + best configs
    for model_name in df_res["model"].unique():
        model_df = df_res[df_res["model"] == model_name].copy()
        suffix   = _build_suffix(panel_path, args.seed, model_name)

        model_success = model_df[model_df["status"] == "success"]
        if not model_success.empty:
            best_idx = model_success["rmse"].idxmin()
            
            best_cfg = model_success.loc[best_idx].to_dict()
            memory_res = [r for r in results if r.get("_config_id") == best_cfg.get("_config_id")]
            if memory_res and "_eval_df" in memory_res[0]:
                best_eval_df = memory_res[0]["_eval_df"]
            else:
                logger.info(f"Re-evaluating best configuration for {model_name} to regenerate predictions output...")
                b_cfg = {k:v for k,v in best_cfg.items() if not str(k).startswith("_")}
                if isinstance(b_cfg.get('selector_params'), str):
                    b_cfg['selector_params'] = json.loads(b_cfg['selector_params'])
                if isinstance(b_cfg.get('model_params'), str):
                    b_cfg['model_params'] = json.loads(b_cfg['model_params'])
                b_res = run_single_experiment(b_cfg, **run_kwargs)
                best_eval_df = b_res.get("_eval_df", pd.DataFrame())
                
            if not best_eval_df.empty:
                preds_path = out_base.parent / f"predictions_{suffix}.csv"
                best_eval_df.to_csv(preds_path, index=False)
                logger.info(f"{model_name} best preds → {preds_path}")

        if "_eval_df" in model_df.columns:
            model_df = model_df.drop(columns=["_eval_df"])

        csv_path = out_base.parent / f"gridsearch_{suffix}.csv"
        model_df.to_csv(csv_path, index=False)
        logger.info(f"{model_name} results  → {csv_path}")

        if not model_success.empty:
            best = model_success.sort_values("rmse").iloc[0].to_dict()
            if "_eval_df" in best:
                del best["_eval_df"]
            for jcol in ("selector_params", "model_params"):
                try:
                    best[jcol] = json.loads(best[jcol])
                except Exception:
                    pass
            best_path = out_base.parent / f"best_config_{suffix}.json"
            with open(best_path, "w") as fh:
                json.dump(best, fh, indent=4, default=str)
            logger.info(f"{model_name} best cfg → {best_path}")

    # Also write the combined table
    if "_eval_df" in df_res.columns:
        df_res = df_res.drop(columns=["_eval_df"])
    df_res.to_csv(out_base, index=False)
    logger.info(f"\nCombined results     → {out_base.absolute()}")


if __name__ == "__main__":
    main()
