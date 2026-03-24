"""
run_bridge_search.py - Bridge Equation Grid Search & Feature Selection Experiment Pipeline
========================================================================================

Executes a grid search over Bridge Equation hyperparameters, regression models, 
and feature selection strategies. Parallelizes outer loops to speed up backtesting. 
Resulting combinations are sorted by RMSE/MAE and exported to CSV.

Example:
    python scripts/run_bridge_search.py --selectors none,pca,lasso,elasticnet \
           --ar-lags 1,2,3 \
           --agg-rules mean,last \
           --regression-models linear,ridge,lasso,elasticnet \
           --pca-components 3,5 \
           --lasso-alphas 0.05,0.1 \
           --n-jobs 4 --search-mode quick
"""

import argparse
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

# Force single-thread linear algebra to prevent collision in parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We import pipeline bits from nowcasting package
from nowcasting.main import build_arg_parser as build_main_parser, load_cf_panel, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.bridge_equation import BridgeEquationNowcast

from nowcasting.features.selectors import (
    IdentitySelector, VarianceFilter, PCACompressor, FactorAnalysisCompressor,
    LassoSelector, ElasticNetSelector, CorrTopNSelector, AutoencoderCompressor, FastScreeningFilter
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bridge_experiment")


def build_experiment_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Bridge Feature Selection Experiment Runner")
    
    # Data arguments 
    ap.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--input-panel", type=str, default=None)
    
    # Grid Search Spaces (Comma-separated lists)
    ap.add_argument("--selectors", type=str, default="none,fast_screen,pca,lasso,elasticnet,corr_top_n",
                    help="Comma-separated selectors to test: none,fast_screen,pca,lasso,elasticnet,corr_top_n,factor_analysis,autoencoder")
    
    # Bridge specific
    ap.add_argument("--ar-lags", type=str, default="1,2,3")
    ap.add_argument("--agg-rules", type=str, default="mean,last,sum")
    ap.add_argument("--regression-models", type=str, default="linear,ridge,lasso,elasticnet")
    
    # Feature selector hyperparams
    ap.add_argument("--pca-components", type=str, default="3,5,10")
    ap.add_argument("--factor-analysis-components", type=str, default="3,5")
    ap.add_argument("--lasso-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-l1-ratios", type=str, default="0.5")
    ap.add_argument("--corr-top-n", type=str, default="10,20,50")
    ap.add_argument("--ae-latent-dims", type=str, default="3,5")
    ap.add_argument("--fast-screen-top-k", type=str, default="50,100")

    # Time rules
    ap.add_argument("--train-windows", type=str, default="120", 
                    help="Initial train window sizes (e.g., 80,120)")
    ap.add_argument("--step-sizes", type=str, default="1", 
                    help="Backtest step sizes (e.g., 1,3)")
    
    # Experiment controls
    ap.add_argument("--search-mode", type=str, choices=["quick", "full"], default="full",
                    help="Quick mode reduces maxiter and uses fewer backtest steps.")
    ap.add_argument("--search-last-n-steps", type=int, default=0,
                    help="Only run the backtest for the last N steps to save time.")
    ap.add_argument("--search-max-configs", type=int, default=0,
                    help="Limit the number of random configurations tested.")
    ap.add_argument("--search-strategy", type=str, default="full", choices=["full", "staged"],
                    help="Use 'staged' for a fast 1-pass coarse search to filter options.")
    ap.add_argument("--search-top-k", type=int, default=5,
                    help="Top configs to promote to final pass in staged search.")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Number of parallel workers for joblib.")
    
    ap.add_argument("--out-file", type=str, default="data/forecasts/bridge_search_results.csv")

    return ap


def parse_list(val_str: str, dtype=str):
    return [dtype(v.strip()) for v in val_str.split(",") if v.strip()]


def build_selector(method: str, params: dict):
    if method == "none":
        return IdentitySelector()
    elif method == "pca":
        return PCACompressor(n_components=params.get("n_components", 5))
    elif method == "factor_analysis":
        return FactorAnalysisCompressor(n_components=params.get("n_components", 5))
    elif method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    elif method == "elasticnet":
        return ElasticNetSelector(alpha=params.get("alpha", 0.1), l1_ratio=params.get("l1_ratio", 0.5))
    elif method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    elif method == "autoencoder":
        return AutoencoderCompressor(latent_dim=params.get("latent_dim", 5), epochs=10)
    elif method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_n", 50))
    else:
        raise ValueError(f"Unknown selector method: {method}")


def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame, 
    y_target: pd.Series,
    search_mode: str,
    search_last_n_steps: int
) -> dict:
    """Runs a single combination grid configuration and returns metrics."""
    res = {**config}
    res["status"] = "running"
    res["window_type"] = "expanding"
    res["eval_mode"] = "mixed_frequency"
    res["n_folds"] = None
    
    try:
        t0 = time.time()
        
        # 1. Backtester initialization
        init_train = config["train_window"]
        step_sz = config["step_size"]
        
        if search_last_n_steps > 0:
            # Shift the initial_train so it only runs the last N steps.
            required_start_index = len(y_target) - search_last_n_steps
            init_train = max(init_train, required_start_index)
            
        bt = RollingBacktester(
            initial_train_periods=init_train, 
            step_size=step_sz,
            window_type="expanding",
            eval_mode="mixed_frequency"
        )

        
        # 2. Selector Initialization
        from sklearn.pipeline import make_pipeline
        base_sel = build_selector(config["selector_method"], config["selector_params"])
        selector = make_pipeline(VarianceFilter(), base_sel)
        
        # 3. Model Initialization
        lags = config["ar_lags"]
        rule = config["agg_rule"]
        reg_model = config["regression_model"]
        
        mod_kws = {}
        if search_mode == "quick" and reg_model in ["lasso", "elasticnet"]:
            mod_kws["max_iter"] = 1000  # Faster fits for quick mode
            
        model = BridgeEquationNowcast(
            target_col=str(y_target.name), 
            ar_lags=lags, 
            agg_rule=rule,
            regression_model=reg_model,
            regression_kwargs=mod_kws
        )
        
        # 4. Run loop
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_df = bt.backtest(
                model=model,
                X_panel=X_panel,
                y_target=y_target,
                transformer=None,
                feature_selector=selector
            )
            
        runt = time.time() - t0
        res["runtime_sec"] = round(runt, 2)
        
        if eval_df.empty:
            res["status"] = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            return res
            
        res["rmse"] = metrics["RMSE"].iloc[0]
        res["mae"] = metrics["MAE"].iloc[0]
        res["status"] = "success"
        res["_eval_df"] = eval_df
        res["n_eval_points"] = len(eval_df)
        res["n_folds"]       = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)
        
        # Track feature counts seamlessly natively provided by the backtester updates
        res["n_raw_features"]    = int(eval_df["n_raw_features"].median()) if "n_raw_features" in eval_df else None
        res["n_trans_features"]  = int(eval_df["n_trans_features"].median()) if "n_trans_features" in eval_df else None
        res["n_sel_features"]    = int(eval_df["n_sel_features"].median()) if "n_sel_features" in eval_df else None
        res["n_model_used_vars"] = int(eval_df["n_model_used_features"].median()) if "n_model_used_features" in eval_df else None
    except Exception as e:
        logger.warning(f"Experiment failed: {config} -> {e}")
        res["status"] = "failed"
        res["error_message"] = str(e)
        
    return res


def generate_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selectors_to_run = parse_list(args.selectors, str)
    
    ar_lags_list = parse_list(args.ar_lags, int)
    agg_rules = parse_list(args.agg_rules, str)
    regression_models = parse_list(args.regression_models, str)
    
    train_windows = parse_list(args.train_windows, int)
    step_sizes = parse_list(args.step_sizes, int)
    
    grids = []
    
    for sel in selectors_to_run:
        param_variations = [{}]
        
        if sel == "pca":
            param_variations = [{"n_components": c} for c in parse_list(args.pca_components, int)]
        elif sel == "factor_analysis":
            param_variations = [{"n_components": c} for c in parse_list(args.factor_analysis_components, int)]
        elif sel == "lasso":
            param_variations = [{"alpha": a} for a in parse_list(args.lasso_alphas, float)]
        elif sel == "elasticnet":
            param_variations = [
                {"alpha": a, "l1_ratio": l} 
                for a in parse_list(args.elasticnet_alphas, float)
                for l in parse_list(args.elasticnet_l1_ratios, float)
            ]
        elif sel == "corr_top_n":
            param_variations = [{"top_n": n} for n in parse_list(args.corr_top_n, int)]
        elif sel == "autoencoder":
            param_variations = [{"latent_dim": n} for n in parse_list(args.ae_latent_dims, int)]
        elif sel == "fast_screen":
            param_variations = [{"top_n": n} for n in parse_list(args.fast_screen_top_k, int)]
            
        for p_var in param_variations:
            for (lags, agg, reg_m, tw, sz) in itertools.product(
                ar_lags_list, agg_rules, regression_models, train_windows, step_sizes
            ):
                grids.append({
                    "selector_method": sel,
                    "selector_params": p_var,
                    "ar_lags": lags,
                    "agg_rule": agg,
                    "regression_model": reg_m,
                    "train_window": tw,
                    "step_size": sz
                })
                
    if args.search_max_configs > 0 and len(grids) > args.search_max_configs:
        import random
        random.shuffle(grids)
        grids = grids[:args.search_max_configs]
        
    return grids


def main():
    ap = build_experiment_parser()
    args = ap.parse_args()
    
    logger.info("=== Bridge Equation Experiment Search Runner ===")
    
    data_dir = Path(args.data_dir)
    logger.info(f"Loading panel ...")
    X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
    
    logger.info(f"Panel shape: {X_panel.shape}, Target shape: {y_target.shape}")
    
    grid = generate_experiment_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    
    if len(grid) == 0:
        logger.warning("Empty grid. Exiting.")
        return

    logger.info(f"Running experiments (n_jobs={args.n_jobs}, mode={args.search_mode}) ...")
    tstart = time.time()
    
    run_kwargs = {
        "X_panel": X_panel,
        "y_target": y_target,
        "search_mode": args.search_mode,
        "search_last_n_steps": args.search_last_n_steps
    }
    
    if args.search_strategy == "staged":
        logger.info(f"--- Stage 1: Coarse Search ({len(grid)} configs, 3 steps) ---")
        run_kwargs["search_last_n_steps"] = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3
        if args.n_jobs > 1:
            results_s1 = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
            )
        else:
            results_s1 = []
            for i, cfg in enumerate(grid, 1):
                logger.info(f"[{i}/{len(grid)}] Stage 1 Running: {cfg}")
                results_s1.append(run_single_experiment(cfg, **run_kwargs))

        df_s1 = pd.DataFrame([r for r in results_s1 if r["status"] == "success"])
        if df_s1.empty:
            logger.error("Stage 1 failed completely.")
            return

        df_s1 = df_s1.sort_values("rmse")
        top_k = min(len(df_s1), args.search_top_k)
        best_configs = df_s1.head(top_k).to_dict("records")

        logger.info(f"--- Stage 2: Fine Search (Top {top_k} configs, {args.search_last_n_steps} steps) ---")
        clean_keys = set(grid[0].keys())
        staged_grid = [{k: v for k, v in cfg.items() if k in clean_keys} for cfg in best_configs]
        
        run_kwargs["search_last_n_steps"] = args.search_last_n_steps
        if args.n_jobs > 1:
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in staged_grid
            )
        else:
            results = []
            for i, cfg in enumerate(staged_grid, 1):
                logger.info(f"[{i}/{len(staged_grid)}] Stage 2 Running: {cfg}")
                results.append(run_single_experiment(cfg, **run_kwargs))
    else:
        if args.n_jobs > 1:
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
            )
        else:
            results = []
            for i, cfg in enumerate(grid, 1):
                logger.info(f"[{i}/{len(grid)}] Running: {cfg}")
                results.append(run_single_experiment(cfg, **run_kwargs))
            
    total_time = time.time() - tstart
    df_res = pd.DataFrame(results)
    
    df_res['selector_params'] = df_res['selector_params'].apply(lambda x: json.dumps(x))
    
    success = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)
    
    if not success.empty:
        df_res = df_res.sort_values("rmse", ascending=True).reset_index(drop=True)
        
        logger.info(f"\nExperiment complete in {total_time/60:.1f} minutes.")
        logger.info(f"Success: {len(success)}, Failed: {failed_count}")
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        
        print(df_res.head(5)[["selector_method", "selector_params", "ar_lags", "agg_rule", "regression_model", "rmse", "mae", "runtime_sec"]].to_string())
        
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        best_idx = success["rmse"].idxmin()
        best_eval_df = df_res.loc[best_idx, "_eval_df"]
        preds_path = out_path.parent / f"predictions_bridge_{args.seed}.csv"
        best_eval_df.to_csv(preds_path, index=False)
        logger.info(f"Bridge best preds → {preds_path}")

        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])

        df_res.to_csv(out_path, index=False)
        logger.info(f"\nDetailed results saved to {out_path.absolute()}")
        
        # Save best configuration to JSON
        best_config = df_res.iloc[0].to_dict()
        if "_eval_df" in best_config:
            del best_config["_eval_df"]
        import ast
        try:
            best_config['selector_params'] = ast.literal_eval(best_config['selector_params'])
        except:
            pass
            
        best_cfg_path = out_path.parent / "best_bridge_configuration.json"
        with open(best_cfg_path, "w") as f:
            json.dump(best_config, f, indent=4)
        logger.info(f"Best configuration saved to {best_cfg_path}")
    else:
        logger.error("ALL experiments failed. Check logs.")
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])
        df_res.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
