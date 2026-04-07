"""
run_bvar_experiment.py - BVAR & VAR Grid Search Experiment Pipeline
====================================================================

Runs a systematic grid search over VAR/BVAR hyperparameters and feature
selection strategies. Supports optional parallel execution via joblib.

Example (quick smoke test):
    python scripts/run_bvar_experiment.py \\
        --modes bvar,var \\
        --lags 1,2,3 \\
        --lambda1 0.01,0.1,1.0 \\
        --lambda2 0.5,1.0 \\
        --lambda3 1,2 \\
        --selectors none,pca \\
        --pca-components 5,10 \\
        --train-windows 80,120 \\
        --n-jobs 4 \\
        --seed 123

Full run replacing all defaults:
    python scripts/run_bvar_experiment.py \\
        --modes bvar,var \\
        --lags 1,2,3,4,6 \\
        --lambda1 0.01,0.1,1.0 \\
        --lambda2 0.1,0.5,1.0 \\
        --lambda3 1,2 \\
        --selectors none,pca,corr_top_n,lasso,elasticnet \\
        --pca-components 3,5,10 \\
        --corr-top-n 10,20,50 \\
        --lasso-alphas 0.01,0.1 \\
        --elasticnet-alphas 0.01,0.1 \\
        --elasticnet-l1-ratios 0.2,0.5,0.8 \\
        --train-windows 60,80,120 \\
        --step-sizes 1,3 \\
        --n-jobs 4 \\
        --seed 123
"""

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed

# Force single-thread BLAS in parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure nowcasting package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcasting.utils.data_loader import load_cf_panel, _DEFAULT_DATA_DIR
from nowcasting.evaluation.checkpoint import prune_grid, run_chunks, build_run_signature
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.bvar import BVARNowcast

from nowcasting.features.selectors import (
    IdentitySelector, VarianceFilter, PCACompressor, FactorAnalysisCompressor,
    LassoSelector, ElasticNetSelector, CorrTopNSelector, FastScreeningFilter
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bvar_experiment")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_experiment_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BVAR / VAR Feature Selection Experiment Runner")

    # Data
    ap.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--input-panel", type=str, default=None)

    # Mode
    ap.add_argument("--modes", type=str, default="bvar,var",
                    help="Comma-separated: bvar, var")

    # VAR / BVAR hyperparams
    ap.add_argument("--lags",    type=str, default="1,2,3,4",
                    help="Comma-separated lag orders")
    ap.add_argument("--lambda1", type=str, default="0.01,0.1,1.0",
                    help="Overall shrinkage (BVAR only)")
    ap.add_argument("--lambda2", type=str, default="0.1,0.5,1.0",
                    help="Cross-variable shrinkage (BVAR only)")
    ap.add_argument("--lambda3", type=str, default="1,2",
                    help="Lag-decay exponents (BVAR only)")
    ap.add_argument("--max-vars", type=int, default=30,
                    help="Maximum variables in the VAR system (dimensionality cap)")

    # Feature selectors
    ap.add_argument("--selectors", type=str, default="none,fast_screen,pca,corr_top_n",
                    help="Comma-separated: none, fast_screen, pca, corr_top_n, lasso, elasticnet, factor_analysis")
    ap.add_argument("--pca-components",        type=str, default="3,5,10")
    ap.add_argument("--factor-analysis-components", type=str, default="3,5")
    ap.add_argument("--corr-top-n",            type=str, default="10,20,50")
    ap.add_argument("--lasso-alphas",          type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-alphas",     type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-l1-ratios",  type=str, default="0.5")
    ap.add_argument("--fast-screen-top-k",     type=str, default="50,100")


    # Training settings
    ap.add_argument("--train-windows", type=str, default="60,80,120",
                    help="Initial training window sizes (comma-separated)")
    ap.add_argument("--step-sizes",    type=str, default="1",
                    help="Backtest step sizes (comma-separated)")

    # Search controls
    ap.add_argument("--search-last-n-steps", type=int, default=0,
                    help="Only run the last N backtest steps (0 = all).")
    ap.add_argument("--search-max-configs",  type=int, default=0,
                    help="Randomly sub-sample at most this many configs (0 = no limit).")
    ap.add_argument("--search-strategy", type=str, default="full", choices=["full", "staged"],
                    help="Use 'staged' for a fast 1-pass coarse search to filter options.")
    ap.add_argument("--search-top-k", type=int, default=5,
                    help="Top configs to promote to final pass in staged search.")

    # ---------- Optuna ----------
    ap.add_argument("--engine", choices=["optuna", "grid", "staged"], default=None,
                    help="Search engine to use: optuna (Bayesian), grid (exhaustive), staged (coarse-to-fine)")
    ap.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna engine")
    ap.add_argument("--study-storage", type=str, default=None, 
                    help="Optuna storage URI. Defaults to sqlite in results directory.")
    ap.add_argument("--study-name", type=str, default=None, help="Optuna study name for persistence and resuming.")
    ap.add_argument("--save-every-n-trials", type=int, default=5, help="Save summary CSV every N trials")

    # Compute
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed",   type=int, default=123,
                    help="Global random seed for reproducibility.")

    # Output
    ap.add_argument("--resume", action="store_true", help="Resume from an interrupted search using existing results CSV.")
    ap.add_argument("--checkpoint-chunk-size", type=int, default=10, help="Save to CSV after completing this many configurations.")
    ap.add_argument("--out-file", type=str,
                    default="data/forecasts/bvar_experiment_results.csv")

    return ap


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_list(val: str, dtype=str) -> list:
    return [dtype(v.strip()) for v in val.split(",") if v.strip()]


def build_selector(method: str, params: dict):
    """Construct a feature selector / compressor from method name + params."""
    if method == "none":
        return IdentitySelector()
    elif method == "pca":
        return PCACompressor(n_components=params.get("n_components", 5))
    elif method == "factor_analysis":
        return FactorAnalysisCompressor(n_components=params.get("n_components", 5))
    elif method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    elif method == "elasticnet":
        return ElasticNetSelector(alpha=params.get("alpha", 0.1),
                                  l1_ratio=params.get("l1_ratio", 0.5))
    elif method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    elif method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_n", 50))
    else:
        raise ValueError(f"Unknown selector method: '{method}'")


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    modes            = parse_list(args.modes, str)
    lags_list        = parse_list(args.lags, int)
    lambda1_list     = parse_list(args.lambda1, float)
    lambda2_list     = parse_list(args.lambda2, float)
    lambda3_list     = parse_list(args.lambda3, float)
    train_windows    = parse_list(args.train_windows, int)
    step_sizes       = parse_list(args.step_sizes, int)
    selectors_to_run = parse_list(args.selectors, str)

    grids: List[Dict[str, Any]] = []

    for mode in modes:
        # Minnesota parameters only apply to BVAR
        if mode == "bvar":
            prior_combos = list(itertools.product(lambda1_list, lambda2_list, lambda3_list))
        else:
            prior_combos = [(None, None, None)]   # one placeholder row for VAR

        for sel in selectors_to_run:
            param_variations = [{}]
            if sel == "pca":
                param_variations = [{"n_components": c}
                                    for c in parse_list(args.pca_components, int)]
            elif sel == "factor_analysis":
                param_variations = [{"n_components": c}
                                    for c in parse_list(args.factor_analysis_components, int)]
            elif sel == "corr_top_n":
                param_variations = [{"top_n": n}
                                    for n in parse_list(args.corr_top_n, int)]
            elif sel == "lasso":
                param_variations = [{"alpha": a}
                                    for a in parse_list(args.lasso_alphas, float)]
            elif sel == "elasticnet":
                param_variations = [
                    {"alpha": a, "l1_ratio": l}
                    for a in parse_list(args.elasticnet_alphas, float)
                    for l in parse_list(args.elasticnet_l1_ratios, float)
                ]
            elif sel == "fast_screen":
                param_variations = [{"top_n": n}
                                    for n in parse_list(args.fast_screen_top_k, int)]

            for p_var in param_variations:
                for (l1, l2, l3) in prior_combos:
                    for (lags, tw, sz) in itertools.product(lags_list, train_windows, step_sizes):
                        grids.append({
                            "mode":             mode,
                            "lags":             lags,
                            "lambda1":          l1,
                            "lambda2":          l2,
                            "lambda3":          l3,
                            "selector_method":  sel,
                            "selector_params":  p_var,
                            "train_window":     tw,
                            "step_size":        sz,
                        })

    # Optional random sub-sampling
    if args.search_max_configs > 0 and len(grids) > args.search_max_configs:
        random.shuffle(grids)
        grids = grids[:args.search_max_configs]

    return grids


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    search_last_n_steps: int,
    max_vars: int,
    seed: int,
) -> dict:
    """Execute one grid configuration and return a metrics dict."""
    res = {**config, "status": "running"}
    # Serialise selector_params for JSON safety
    res["selector_params"] = json.dumps(config["selector_params"])
    res["window_type"] = "expanding"
    res["eval_mode"] = "common_frequency"
    res["n_folds"] = None

    np.random.seed(seed)   # worker-level determinism

    try:
        t0 = time.time()

        # --- Backtester ---
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

        # --- Selector pipeline ---
        from sklearn.pipeline import make_pipeline
        base_sel = build_selector(config["selector_method"], config["selector_params"])
        selector = make_pipeline(VarianceFilter(), base_sel)

        # --- Model ---
        mode = config["mode"]
        lags = config["lags"]

        if mode == "bvar":
            model = BVARNowcast(
                target_col=str(y_target.name),
                mode="bvar",
                lags=lags,
                lambda1=config["lambda1"],
                lambda2=config["lambda2"],
                lambda3=config["lambda3"],
                max_vars=max_vars,
                seed=seed,
            )
        else:
            model = BVARNowcast(
                target_col=str(y_target.name),
                mode="var",
                lags=lags,
                max_vars=max_vars,
                seed=seed,
            )

        # --- Backtest ---
        import warnings
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

        metrics = compute_metrics(eval_df, model_name=f"BVAR_{mode}")
        res["rmse"]          = float(metrics["RMSE"].iloc[0])
        res["mae"]           = float(metrics["MAE"].iloc[0])
        res["mape"]          = float(metrics["MAPE"].iloc[0]) if "MAPE" in metrics.columns else float("nan")
        res["n_eval_points"] = len(eval_df)
        res["n_folds"]       = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)
        
        # Track feature counts seamlessly natively provided by the backtester updates
        res["n_raw_features"]    = int(eval_df["n_raw_features"].median()) if "n_raw_features" in eval_df else None
        res["n_trans_features"]  = int(eval_df["n_trans_features"].median()) if "n_trans_features" in eval_df else None
        res["n_sel_features"]    = int(eval_df["n_sel_features"].median()) if "n_sel_features" in eval_df else None
        res["n_model_used_vars"] = int(eval_df["n_model_used_features"].median()) if "n_model_used_features" in eval_df else None
        
        res["status"]        = "success"
        res["_eval_df"]      = eval_df

    except Exception as exc:
        logger.warning(f"Experiment failed: {config} → {exc}")
        res["status"]        = "failed"
        res["error_message"] = str(exc)
        res.setdefault("runtime_sec", round(time.time() - t0, 2))

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
    selector_name = trial.suggest_categorical("selector", ["none", "corr_top_n", "fast_screen", "lasso", "elasticnet", "pca", "factor_analysis"])
    selector_params = {}
    
    if selector_name == "corr_top_n":
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

    mode = trial.suggest_categorical("mode", ["bvar", "var"])
    lags = trial.suggest_int("lags", 1, 12)
    
    l1, l2, l3 = None, None, None
    if mode == "bvar":
        l1 = trial.suggest_float("lambda1", 1e-3, 10.0, log=True)
        l2 = trial.suggest_float("lambda2", 0.1, 1.0)
        l3 = trial.suggest_int("lambda3", 1, 2)

    train_window = trial.suggest_categorical("train_window", [60, 80, 100, 120])
    step_size = trial.suggest_categorical("step_size", [1, 3])

    config = {
        "mode": mode,
        "lags": lags,
        "lambda1": l1,
        "lambda2": l2,
        "lambda3": l3,
        "selector_method": selector_name,
        "selector_params": selector_params,
        "train_window": train_window,
        "step_size": step_size,
    }
    
    res = run_single_experiment(
        config=config,
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        max_vars=args.max_vars,
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
            preds_path = tracker.out_dir / f"bvar_optuna_best_preds_{tracker.suffix}.csv"
            res["_eval_df"].to_csv(preds_path, index=False)
            logger.info(f"New best RMSE ({rmse:.4f}) -> {preds_path}")
            
    return rmse


def run_optuna_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    logger.info("Starting OPTUNA engine.")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    study_name = args.study_name or f"bvar_{suffix}"
    storage_url = args.study_storage or f"sqlite:///{out_dir.absolute()}/bvar_optuna.db"
    
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
    ap   = build_experiment_parser()
    args = ap.parse_args()

    # Seed everything at the top level
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=== BVAR / VAR Experiment Search Runner ===")
    logger.info(f"Seed: {args.seed}  |  n_jobs: {args.n_jobs}")

    # 1. Load data
    data_dir = Path(args.data_dir)
    logger.info("Loading panel ...")
    X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
    logger.info(f"Panel shape: {X_panel.shape}  |  Target: {y_target.name}")

    # 2. Build grid
    grid = generate_experiment_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    if not grid:
        logger.warning("Empty grid — exiting.")
        return

    # 3. Run
    logger.info(f"Running experiments ...")
    tstart = time.time()

    run_kwargs = dict(
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        max_vars=args.max_vars,
        seed=args.seed,
    )
    
    out_path = Path(args.out_file)
    suffix_base = (Path(args.input_panel).stem if args.input_panel else "panel")[:30] + f"_s{args.seed}"
    
    active_engine = args.engine
    if active_engine is None:
        active_engine = "staged" if args.search_strategy == "staged" else "grid"

    if active_engine == "optuna":
        run_optuna_engine(X_panel, y_target, args, out_path.parent, suffix_base)
        return

    if active_engine == "staged":
        logger.info(f"--- Stage 1: Coarse Search ({len(grid)} configs, 3 steps) ---")
        stage1_kwargs = dict(run_kwargs)
        stage1_steps = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3
        stage1_kwargs["search_last_n_steps"] = stage1_steps
        
        run_signature = build_run_signature(
            script_name="run_bvar_experiment.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            modes=args.modes
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
            script_name="run_bvar_experiment.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            modes=args.modes
        )
        grid = prune_grid(grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        results = run_chunks(grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10))

    total_time = time.time() - tstart

    # 4. Collect results
    if out_path.exists():
        df_res = pd.read_csv(out_path)
    else:
        df_res = pd.DataFrame(results)
        
    if "dataset" not in df_res.columns:
        df_res["dataset"] = str(args.input_panel or args.data_dir)
    if "seed" not in df_res.columns:
        df_res["seed"]    = args.seed

    success      = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)

    logger.info(f"\nExperiment complete in {total_time / 60:.1f} min  |  "
                f"Success: {len(success)}  |  Failed: {failed_count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not success.empty:
        df_res = df_res.sort_values("rmse", ascending=True).reset_index(drop=True)
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        display_cols = [c for c in ["mode", "lags", "lambda1", "lambda2", "selector_method",
                                     "train_window", "rmse", "mae", "runtime_sec"]
                        if c in df_res.columns]
        print(df_res.head(5)[display_cols].to_string(index=False))

        best_eval_df = pd.DataFrame()
        best_cfg = success.iloc[0].to_dict()
        memory_res = [r for r in results if r.get("_config_id") == best_cfg.get("_config_id")]
        
        if memory_res and "_eval_df" in memory_res[0]:
            best_eval_df = memory_res[0]["_eval_df"]
        else:
            logger.info("Re-evaluating best configuration to regenerate predictions output...")
            b_cfg = {k:v for k,v in best_cfg.items() if not str(k).startswith("_")}
            if isinstance(b_cfg.get('selector_params'), str):
                b_cfg['selector_params'] = json.loads(b_cfg['selector_params'])
            b_res = run_single_experiment(b_cfg, **run_kwargs)
            best_eval_df = b_res.get("_eval_df", pd.DataFrame())

        if not best_eval_df.empty:
            preds_path = out_path.parent / f"predictions_bvar_{args.seed}.csv"
            best_eval_df.to_csv(preds_path, index=False)
            logger.info(f"BVAR best preds → {preds_path}")

        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])

        df_res.to_csv(out_path, index=False)
        logger.info(f"\nDetailed results saved to: {out_path.absolute()}")

        # Best configuration JSON
        best = df_res.iloc[0].to_dict()
        if "_eval_df" in best:
            del best["_eval_df"]
        try:
            best["selector_params"] = json.loads(best["selector_params"])
        except Exception:
            pass
        best_path = out_path.parent / "best_bvar_configuration.json"
        with open(best_path, "w") as fh:
            json.dump(best, fh, indent=4, default=str)
        logger.info(f"Best configuration saved to: {best_path}")
    else:
        logger.error("ALL experiments failed. Saving failure log.")
        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])
        df_res.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
