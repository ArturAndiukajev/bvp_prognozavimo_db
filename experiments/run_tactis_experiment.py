"""
Tai yra TACTiS Grid ir Optuna eksperimentų paleidimo skriptas
Sujungia hiperparametrų paieškos metodus (Grid, Staged ir Optuna) į vieną skriptą
su bendru vertinimo ciklu `eval_tactis_config()`.
Paieškos režimai:
--engine optuna : paieška per didelę parametrų erdvę naudojant Optuna.
--engine grid   : Pilnas visų nurodytų parametrų kombinacijų perrinkimas.
--engine staged : Dviejų etapų paieška (pirma grubus filtravimas, tada detalesnė analizė).
"""

from __future__ import annotations

import argparse
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
from nowcasting.models.tactis_wrapper import TACTiSNowcastWrapper

from nowcasting.features.selectors import (
    IdentitySelector,
    VarianceFilter,
    CorrTopNSelector,
    LassoSelector,
    ElasticNetSelector,
    FastScreeningFilter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tactis_unified_search")


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified TACTiS Grid & Optuna Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    ap.add_argument("--engine", choices=["optuna", "grid", "staged", "evaluate"], default="optuna",
                    help="Search engine to use: optuna, grid, staged, or evaluate (single run from config)")
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    ap.add_argument("--input-panel", default=None)
    ap.add_argument("--target-col", default=None, dest="target")
    ap.add_argument("--train-window", type=int, default=80)
    ap.add_argument("--step-size", type=int, default=1)
    
    # Compute
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for Grid/Staged")
    ap.add_argument("--resume", action="store_true", help="Resume from an interrupted search using existing results CSV.")
    ap.add_argument("--checkpoint-chunk-size", type=int, default=10, help="Save to CSV after completing this many configurations.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--results-dir", default="data/forecasts/tactis/", help="Directory to save experiment results")
    ap.add_argument("--config-path", type=str, default=None, help="Path to a JSON configuration file for '--engine evaluate'")
    
    # Optuna Specific
    ap.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna engine")
    ap.add_argument("--study-storage", type=str, default=None, 
                    help="Optuna storage URI (e.g. sqlite:///tactis.db). Defaults to sqlite in results-dir.")
    ap.add_argument("--study-name", type=str, default=None, help="Optuna study name for persistence and resuming.")
    ap.add_argument("--save-every-n-trials", type=int, default=5, help="Save summary CSV every N trials")

    # Grid/Staged Specific
    ap.add_argument("--selectors", default="none,corr_top_n", help="Selectors for grid engine")
    ap.add_argument("--top-n", default="10,20")
    ap.add_argument("--variance-thresholds", default="1e-6")
    ap.add_argument("--lasso-alphas", default="0.001,0.01")
    ap.add_argument("--elasticnet-alphas", default="0.001,0.01")
    ap.add_argument("--elasticnet-l1-ratios", default="0.5")
    ap.add_argument("--fast-screen-top-k", default="50,100")
    ap.add_argument("--history-lengths", default="12,24")
    ap.add_argument("--epochs-list", default="5,10")
    ap.add_argument("--batch-sizes", default="16,32")
    ap.add_argument("--learning-rates", default="1e-4,1e-3")
    ap.add_argument("--num-samples-list", default="20,50")
    ap.add_argument("--skip-copula-options", default="false,true")
    ap.add_argument("--search-last-n-steps", type=int, default=0, help="Faster evaluation on the tail for grid")
    ap.add_argument("--search-max-configs", type=int, default=0, help="Random sample of grid size")
    ap.add_argument("--search-top-k", type=int, default=5, help="Top configs to promote in staged search")

    return ap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(s))

def _plist(s: str, dtype=str) -> list:
    return [dtype(v.strip()) for v in s.split(",") if v.strip()]

def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes")

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
        return ElasticNetSelector(alpha=params.get("alpha", 0.1), l1_ratio=params.get("l1_ratio", 0.5))
    elif method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_n", 50))
    raise ValueError(f"Unknown selector: {method}")


# ---------------------------------------------------------------------------
# Evaluation Pipeline (Shared across Optuna and Grid engines)
# ---------------------------------------------------------------------------

def eval_tactis_config(
    tactis_params: Dict[str, Any],
    selector_method: str,
    selector_params: Dict[str, Any],
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    train_window: int,
    step_size: int,
    seed: int,
    search_last_n_steps: int = 0,
    step_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Evaluates a specific configuration using RollingBacktester.
    Returns (metrics_dict, eval_df).
    Does NOT throw exceptions (catches and returns failed dict).
    """
    np.random.seed(seed)
    
    res: dict = {
        "model": "TACTiS",
        "selector_method": selector_method,
        "selector_params": json.dumps(selector_params),
        "tactis_params": json.dumps(tactis_params),
        "train_window": train_window,
        "step_size": step_size,
        "n_features_input": X_panel.shape[1],
        "rmse": float("nan"),
        "mae": float("nan"),
        "runtime_sec": 0.0,
        "seed": seed,
        "status": "running",
        "error_message": "",
    }
    
    # Expose individual tactis params for easier pandas querying later
    for k, v in tactis_params.items():
        res[k] = v

    t0 = time.time()
    eval_df = pd.DataFrame()
    
    try:
        init_train = train_window
        if search_last_n_steps > 0:
            required_start = len(y_target) - search_last_n_steps
            init_train = max(init_train, required_start)

        bt = RollingBacktester(
            initial_train_periods=init_train, 
            step_size=step_size,
            window_type="expanding",
            eval_mode="common_frequency"
        )
        selector = build_selector(selector_method, selector_params)
        model = TACTiSNowcastWrapper(target_col=str(y_target.name), seed=seed, **tactis_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_df = bt.backtest(
                model=model,
                X_panel=X_panel,
                y_target=y_target,
                transformer=None,
                feature_selector=selector,
                step_callback=step_callback,
            )

        res["runtime_sec"] = round(time.time() - t0, 2)

        if eval_df.empty:
            res["status"] = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            res["_eval_df"] = eval_df
            return res

        metrics = compute_metrics(eval_df, model_name="TACTiS")
        res["rmse"] = float(metrics["RMSE"].iloc[0])
        res["mae"] = float(metrics["MAE"].iloc[0])
        res["mape"] = float(metrics["MAPE"].iloc[0]) if "MAPE" in metrics.columns else float("nan")
        
        if hasattr(selector, "selected_cols_") and selector.selected_cols_ is not None:
            res["n_features_used"] = len(selector.selected_cols_)
        else:
            res["n_features_used"] = X_panel.shape[1]
            
        res["status"] = "success"

    except optuna.TrialPruned as pr:
        # Pass the prune exception back up
        raise optuna.TrialPruned(f"Pruned early: {str(pr)}")
        
    except Exception as exc:
        res["status"] = "failed"
        res["error_message"] = str(exc)
        res["runtime_sec"] = round(time.time() - t0, 2)
        logger.debug(f"Config failed [TACTiS]: {exc}", exc_info=True)

    res["_eval_df"] = eval_df
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
    # 1. Selector Hyperparameters
    selector_name = trial.suggest_categorical("selector", ["none", "corr_top_n", "fast_screen", "variance_filter", "lasso", "elasticnet"])
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

    # 2. TACTiS Hyperparameters
    tactis_params = {
        "history_length": trial.suggest_categorical("history_length", [12, 18, 24]),
        "epochs": trial.suggest_int("epochs", 5, 20),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "num_samples": trial.suggest_categorical("num_samples", [20, 50, 100]),
        "skip_copula": trial.suggest_categorical("skip_copula", [True, False]),
        # Architecture parameters (can be exposed here)
        "flow_layers": trial.suggest_int("flow_layers", 1, 3),
        "flow_hid_dim": trial.suggest_categorical("flow_hid_dim", [24, 48, 64]),
    }

    def _pruning_callback(running_rmse: float, step: int):
        trial.report(running_rmse, step)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Optuna pruned at step {step} with RMSE: {running_rmse:.4f}")

    metrics = eval_tactis_config(
        tactis_params=tactis_params,
        selector_method=selector_name,
        selector_params=selector_params,
        X_panel=X_panel,
        y_target=y_target,
        train_window=args.train_window,
        step_size=args.step_size,
        seed=args.seed,
        step_callback=_pruning_callback
    )
    eval_df = metrics.get("_eval_df", pd.DataFrame())

    if metrics["status"] != "success":
        logger.debug(f"Trial failed: {metrics['error_message']}")
        raise optuna.TrialPruned(f"Failed: {metrics['error_message']}")

    current_rmse = metrics["rmse"]
    
    # Save per-trial artifacts automatically (parallel safe)
    preds_path = tracker.out_dir / f"tactis_eval_df_trial_{trial.number}.csv"
    eval_df.to_csv(preds_path, index=False)
    
    cfg_path = tracker.out_dir / f"tactis_trial_{trial.number}.json"
    with open(cfg_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    if current_rmse < tracker.best_rmse:
        tracker.best_rmse = current_rmse
        logger.info(f"Iteration {trial.number} | New Best RMSE: {current_rmse:.4f}")
        
    return current_rmse

def run_optuna_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    study_name = args.study_name or f"tactis_study_{suffix}"
    if args.study_storage:
        storage_url = args.study_storage
    else:
        db_path = out_dir / "tactis_optuna.db"
        storage_url = f"sqlite:///{db_path.absolute().as_posix()}"
        
    logger.info(f"Starting OPTUNA engine for {args.n_trials} trials.")
    logger.info(f"Study Name: {study_name}")
    logger.info(f"Storage: {storage_url}")
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize", 
        sampler=sampler, 
        pruner=pruner
    )

    initial_best = float("inf")
    try:
        if len(study.best_trials) > 0:
            initial_best = study.best_value
            logger.info(f"Resuming existing study. Current best RMSE: {initial_best:.4f}")
    except ValueError:
        pass # No completed trials yet
        
    # NOTE ON PARALLEL WORKERS:
    # Instead of fighting locks for the `best` tracker concurrently, each
    # process writes its own independent trial output. Tracking the `tracker.best_rmse`
    # is merely used for lightweight logging indicators.
    tracker = OptunaBestTracker(out_dir, suffix, initial_best)
    
    def periodic_save_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.number % args.save_every_n_trials == 0:
            try:
                trials_df = study.trials_dataframe()
                trials_filepath = out_dir / f"tactis_optuna_trials_{suffix}.csv"
                trials_df.to_csv(trials_filepath, index=False)
            except Exception:
                pass # Gracefully handle OS locking collisions from parallel nodes

            try:
                best_value = study.best_value
                best_params = study.best_params
            except ValueError:
                best_value = None
                best_params = None
                
            status = {
                "study_name": study_name,
                "completed_trials": len(study.trials),
                "best_value": best_value,
                "best_params": best_params,
                "timestamp": time.time(),
                "input_panel": str(args.input_panel)
            }
            status_path = out_dir / f"latest_status_{study_name}.json"
            with open(status_path, "w") as f:
                json.dump(status, f, indent=4)
                
    t0 = time.time()
    try:
        study.optimize(
            lambda t: optuna_objective(t, X_panel, y_target, args, tracker),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            catch=(Exception,),
            callbacks=[periodic_save_callback]
        )
    except KeyboardInterrupt:
        logger.warning("\n" + "="*50)
        logger.warning("KeyboardInterrupt detected. Saving current Optuna progress cleanly...")
        logger.warning("="*50)
    
    logger.info(f"Optuna completed in {(time.time()-t0)/60:.1f} mins.")
    
    if len(study.trials) == 0:
        logger.error("No trials started.")
        return
        
    try:
        if len(study.best_trials) == 0:
            logger.error("No trials finished successfully.")
            return
            
        best_trial = study.best_trial
        logger.info(f"Global Best RMSE: {best_trial.value:.4f}")
        
        # Determine the global best safely and extract artifacts
        best_preds_src = out_dir / f"tactis_eval_df_trial_{best_trial.number}.csv"
        best_cfg_src = out_dir / f"tactis_trial_{best_trial.number}.json"
        
        if best_preds_src.exists():
            import shutil
            shutil.copy(best_preds_src, out_dir / f"tactis_optuna_best_preds_{suffix}.csv")
            shutil.copy(best_cfg_src, out_dir / f"tactis_optuna_best_config_{suffix}.json")
            logger.info("Copied true best trial artifacts to global representation.")
            
        # Cleanup temporary parallel files to save space
        for p in out_dir.glob("tactis_eval_df_trial_*.csv"):
            try: p.unlink()
            except Exception: pass
            
        for p in out_dir.glob("tactis_trial_*.json"):
            try: p.unlink()
            except Exception: pass
            
    except ValueError:
        logger.error("Could not retrieve best trial (maybe none succeeded).")
    
    # Final save
    trials_df = study.trials_dataframe()
    trials_filepath = out_dir / f"tactis_optuna_trials_{suffix}.csv"
    trials_df.to_csv(trials_filepath, index=False)


def run_evaluation_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    if not args.config_path:
        logger.error("Evaluation mode requires '--config-path'.")
        return

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Robust parsing of nested JSON strings if present (often happens in results CSV exports)
    def _parse_if_str(val):
        if isinstance(val, str):
            try: return json.loads(val)
            except: return val
        return val

    # Extract configuration parts
    eval_config = {}
    for k, v in config.items():
        if k not in ["rmse", "mae", "mape", "status", "runtime_sec", "n_eval_points", "n_folds", "error_message", "n_raw_features", "n_trans_features", "n_sel_features", "n_model_used_vars"]:
            eval_config[k] = _parse_if_str(v)
    
    # Standardize result file path
    out_path = out_dir / "tactis_experiment_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating TACTiS with config from {args.config_path}")
    
    # Wrap in single-item grid for checkpoint system
    search_grid = [eval_config]
    
    run_signature = build_run_signature(
        script_name="run_tactis_experiment.py",
        input_panel=str(args.input_panel or args.data_dir),
        target=str(y_target.name),
        seed=args.seed,
        search_last_n_steps=args.search_last_n_steps,
        engine="evaluate"
    )
    
    # Reuse shared partial evaluator
    def eval_tactis_config_partial(cfg: dict, **kwargs):
        t_params = json.loads(cfg["tactis_params"]) if isinstance(cfg["tactis_params"], str) else cfg["tactis_params"]
        s_method = cfg.get("selector_method", "none")
        s_params = json.loads(cfg["selector_params"]) if isinstance(cfg["selector_params"], str) else cfg.get("selector_params", {})
        
        return eval_tactis_config(
            tactis_params=t_params,
            selector_method=s_method,
            selector_params=s_params,
            **kwargs
        )
    
    run_kwargs = dict(
        X_panel=X_panel,
        y_target=y_target,
        train_window=args.train_window,
        step_size=args.step_size,
        seed=args.seed,
        search_last_n_steps=args.search_last_n_steps
    )
    
    pruned_grid = prune_grid(search_grid, args.search_last_n_steps, out_path, resume=args.resume, run_signature=run_signature)
    
    if not pruned_grid:
        logger.info("Evaluation already completed and found in results CSV. Skipping.")
        return

    results = run_chunks(pruned_grid, eval_tactis_config_partial, run_kwargs, out_path, n_jobs=1, chunk_sz=1, stage_name="Evaluation")
    
    if results and results[0]["status"] == "success":
        res = results[0]
        logger.info(f"Evaluation Success | RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f}")
        eval_df = res.pop("_eval_df", pd.DataFrame())
        if not eval_df.empty:
            out_preds = out_dir / f"tactis_eval_preds_{suffix}.csv"
            eval_df.to_csv(out_preds, index=False)
            logger.info(f"Saved predictions to {out_preds}")
        
        # Also save separate JSON for convenience
        out_json = out_dir / f"tactis_eval_results_{suffix}.json"
        with open(out_json, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "_eval_df"}, f, indent=4)
        logger.info(f"Saved results to {out_json}")
    else:
        err = results[0].get("error_message") if results else "Unknown error"
        logger.error(f"Evaluation failed: {err}")


# ---------------------------------------------------------------------------
# Grid / Staged Engines
# ---------------------------------------------------------------------------

def _build_grid_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    tactis_grid = [
        {
            "history_length": hl, "epochs": ep, "batch_size": bs, 
            "lr": lr, "num_samples": ns, "skip_copula": sc,
        }
        for hl in _plist(args.history_lengths, int)
        for ep in _plist(args.epochs_list, int)
        for bs in _plist(args.batch_sizes, int)
        for lr in _plist(args.learning_rates, float)
        for ns in _plist(args.num_samples_list, int)
        for sc in [_parse_bool(v) for v in _plist(args.skip_copula_options, str)]
    ]
    
    def _sel_params(method):
        if method == "variance_filter": return [{"threshold": t} for t in _plist(args.variance_thresholds, float)]
        elif method == "corr_top_n": return [{"top_n": n} for n in _plist(args.top_n, int)]
        elif method == "lasso": return [{"alpha": a} for a in _plist(args.lasso_alphas, float)]
        elif method == "elasticnet": return [{"alpha": a, "l1_ratio": l} for a in _plist(args.elasticnet_alphas, float) for l in _plist(args.elasticnet_l1_ratios, float)]
        elif method == "fast_screen": return [{"top_n": n} for n in _plist(args.fast_screen_top_k, int)]
        return [{}]

    configs = []
    for sel_method in _plist(args.selectors, str):
        for sel_p in _sel_params(sel_method):
            for t_p in tactis_grid:
                configs.append({
                    "selector_method": sel_method,
                    "selector_params": sel_p,
                    "tactis_params": t_p
                })
                
    if args.search_max_configs > 0 and len(configs) > args.search_max_configs:
        random.shuffle(configs)
        configs = configs[:args.search_max_configs]
        
    return configs

def run_grid_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    logger.info("Starting GRID engine.")
    configs = _build_grid_configs(args)
    logger.info(f"Generated {len(configs)} configurations.")
    
    run_kwargs = dict(X_panel=X_panel, y_target=y_target, train_window=args.train_window, 
                      step_size=args.step_size, seed=args.seed, search_last_n_steps=args.search_last_n_steps)
    
    out_path = out_dir / f"tactis_grid_results_{suffix}.csv"
    run_signature = build_run_signature(
        script_name="run_tactis_experiment.py",
        engine="grid",
        input_panel=str(args.input_panel or args.data_dir),
        target=str(args.target),
        seed=args.seed,
        search_last_n_steps=args.search_last_n_steps
    )
    configs = prune_grid(configs, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
    
    wrapper = lambda cfg, **kwargs: eval_tactis_config(**cfg, **kwargs)
    results = run_chunks(configs, wrapper, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10))
            
    _process_grid_results(results, args, out_path, suffix, "grid", X_panel, y_target)

def run_staged_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    logger.info("Starting STAGED engine.")
    configs = _build_grid_configs(args)
    
    out_path = out_dir / f"tactis_staged_results_{suffix}.csv"
    run_signature = build_run_signature(
        script_name="run_tactis_experiment.py",
        engine="staged",
        input_panel=str(args.input_panel or args.data_dir),
        target=str(args.target),
        seed=args.seed,
        search_last_n_steps=args.search_last_n_steps
    )
    st1_steps = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3
    run_kwargs = dict(X_panel=X_panel, y_target=y_target, train_window=args.train_window, 
                      step_size=args.step_size, seed=args.seed)

    logger.info(f"--- Stage 1: Coarse Search ({len(configs)} configs, {st1_steps} steps) ---")
    st1_kwargs = dict(run_kwargs)
    st1_kwargs["search_last_n_steps"] = st1_steps
    
    configs_s1 = prune_grid(configs, st1_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
    wrapper = lambda cfg, **kwargs: eval_tactis_config(**cfg, **kwargs)
    
    st1_res = run_chunks(configs_s1, wrapper, st1_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 1")

    if out_path.exists():
        try:
            full_s1_df = pd.read_csv(out_path)
            full_s1_df = full_s1_df[(full_s1_df["status"] == "success")]
            best_dict_list = full_s1_df.sort_values("rmse").head(args.search_top_k).to_dict("records")
            best_configs = []
            for b in best_dict_list:
                best_configs.append({
                    "selector_method": b.get("selector_method"),
                    "selector_params": json.loads(b.get("selector_params", "{}")) if isinstance(b.get("selector_params"), str) else b.get("selector_params"),
                    "tactis_params": json.loads(b.get("tactis_params", "{}")) if isinstance(b.get("tactis_params"), str) else b.get("tactis_params")
                })
        except Exception:
            df_st1 = pd.DataFrame([r for r in st1_res if r.get("status") == "success"])
            if not df_st1.empty:
                best_indices = df_st1.sort_values("rmse").index[:args.search_top_k]
                best_configs = [configs[i] for i in best_indices]
            else:
                best_configs = []
    else:
        df_st1 = pd.DataFrame([r for r in st1_res if r.get("status") == "success"])
        if not df_st1.empty:
            best_indices = df_st1.sort_values("rmse").index[:args.search_top_k]
            best_configs = [configs[i] for i in best_indices]
        else:
            best_configs = []

    if not best_configs:
        logger.error("Stage 1 failed completely.")
        return

    logger.info(f"--- Stage 2: Fine Search (Top {len(best_configs)} configs, {args.search_last_n_steps} steps) ---")
    st2_kwargs = dict(run_kwargs)
    st2_kwargs["search_last_n_steps"] = args.search_last_n_steps
    best_configs = prune_grid(best_configs, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
    st2_res = run_chunks(best_configs, wrapper, st2_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 2")

    _process_grid_results(st2_res, args, out_path, suffix, "staged", X_panel, y_target)

def _process_grid_results(results: List[dict], args: argparse.Namespace, out_path: Path, suffix: str, engine_name: str, X_panel: pd.DataFrame, y_target: pd.Series):
    if out_path.exists():
        df_res = pd.read_csv(out_path)
    else:
        df_res = pd.DataFrame(results)

    success = df_res[df_res["status"] == "success"]
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not success.empty:
        df_res = df_res.sort_values("rmse").reset_index(drop=True)
        
        best_cfg = df_res.iloc[0].to_dict()
        memory_res = [r for r in results if r.get("_config_id") == best_cfg.get("_config_id")]
        if memory_res and "_eval_df" in memory_res[0]:
            best_eval_df = memory_res[0]["_eval_df"]
        else:
            logger.info("Re-evaluating best configuration to regenerate predictions output...")
            wrapper = lambda cfg, **kwargs: eval_tactis_config(**cfg, **kwargs)
            try:
                b_run_kwargs = dict(X_panel=X_panel, y_target=y_target, train_window=args.train_window, step_size=args.step_size, seed=args.seed, search_last_n_steps=args.search_last_n_steps)
                sp = best_cfg.get("selector_params", "{}")
                tp = best_cfg.get("tactis_params", "{}")
                bc = {
                    "selector_method": best_cfg.get("selector_method"),
                    "selector_params": json.loads(sp) if isinstance(sp, str) else sp,
                    "tactis_params": json.loads(tp) if isinstance(tp, str) else tp
                }
                b_res = wrapper(bc, **b_run_kwargs)
                best_eval_df = b_res.get("_eval_df", pd.DataFrame())
            except Exception as e:
                logger.error(f"Failed to cleanly re-evaluate best preds for {engine_name}: {e}")
                best_eval_df = pd.DataFrame()
        
        if not best_eval_df.empty:
            preds_path = out_dir / f"tactis_{engine_name}_best_preds_{suffix}.csv"
            best_eval_df.to_csv(preds_path, index=False)
            
        cfg_path = out_dir / f"tactis_{engine_name}_best_config_{suffix}.json"
        with open(cfg_path, 'w') as f:
            bd = dict(best_cfg)
            for json_col in ("selector_params", "tactis_params"):
                if isinstance(bd.get(json_col), str):
                    try: bd[json_col] = json.loads(bd[json_col])
                    except: pass
            bd.pop("_eval_df", None)
            json.dump(bd, f, indent=4)
        
        logger.info(f"Top {engine_name.capitalize()} Result RMSE: {best_cfg['rmse']:.4f}")
        logger.info(f"Saved artifacts to {out_dir}")
        
    else:
        logger.error(f"All {engine_name} configs failed.")
        
    if "_eval_df" in df_res.columns:
        df_res = df_res.drop(columns=["_eval_df"])
    df_res.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info(f"  Unified TACTiS Runner | Engine: {args.engine.upper()}")
    logger.info(f"  seed={args.seed} | n_jobs={args.n_jobs}")
    logger.info("=" * 60)

    data_dir = Path(args.data_dir)
    try:
        X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    _ip = Path(args.input_panel) if args.input_panel else None
    panel_stem = _sanitize(_ip.stem if _ip else "panel")[:30]
    suffix = f"{panel_stem}_s{args.seed}"
    out_dir = Path(args.results_dir)

    if args.engine == "optuna":
        run_optuna_engine(X_panel, y_target, args, out_dir, suffix)
    elif args.engine == "grid":
        run_grid_engine(X_panel, y_target, args, out_dir, suffix)
    elif args.engine == "staged":
        run_staged_engine(X_panel, y_target, args, out_dir, suffix)
    elif args.engine == "evaluate":
        run_evaluation_engine(X_panel, y_target, args, out_dir, suffix)



if __name__ == "__main__":
    main()
